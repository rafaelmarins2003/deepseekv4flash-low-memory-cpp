from __future__ import annotations

import json
import struct
import sys
import unittest
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_tensor_inventory import (  # noqa: E402
    InventoryError,
    _budgets,
    _classify,
    _parse_header,
)
from probe_weight_ranges import _decode, _sample_spans  # noqa: E402


def header_bytes(entries: dict[str, object]) -> bytes:
    return json.dumps(entries).encode("utf-8")


def tensor_entry(dtype: str, shape: list[int], start: int, end: int) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}


def artifact_size(raw: bytes, payload: int) -> int:
    return 8 + len(raw) + payload


class ClassifyTests(unittest.TestCase):
    def test_assigns_scope_index_and_role(self) -> None:
        cases = {
            "layers.7.ffn.experts.42.w1.weight": ("layer", 7, "routed_expert"),
            "layers.7.ffn.experts.42.w2.scale": ("layer", 7, "routed_expert"),
            "layers.0.ffn.shared_experts.w3.weight": ("layer", 0, "shared_expert"),
            "layers.0.ffn.gate.tid2eid": ("layer", 0, "router"),
            "layers.5.ffn.gate.bias": ("layer", 5, "router"),
            "layers.5.attn.wq_b.weight": ("layer", 5, "attention_core"),
            "layers.5.attn.attn_sink": ("layer", 5, "attention_core"),
            "layers.6.attn.indexer.wq_b.scale": ("layer", 6, "attention_indexer"),
            "layers.6.attn.compressor.ape": ("layer", 6, "attention_compressor"),
            "layers.6.hc_ffn_fn": ("layer", 6, "hyper_connection"),
            "layers.6.ffn_norm.weight": ("layer", 6, "norm"),
            "mtp.0.main_proj.weight": ("mtp", 0, "mtp_input_projection"),
            "mtp.2.markov_head.markov_w2.weight": ("mtp", 2, "mtp_markov_head"),
            "mtp.2.confidence_head.proj.weight": ("mtp", 2, "mtp_confidence_head"),
            "mtp.1.ffn.experts.3.w1.weight": ("mtp", 1, "routed_expert"),
            "embed.weight": ("global", None, "embedding"),
            "head.weight": ("global", None, "output_head"),
            "norm.weight": ("global", None, "norm"),
            "hc_head_base": ("global", None, "hyper_connection"),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(_classify(name), expected)

    def test_indexer_is_not_absorbed_by_attention_core(self) -> None:
        scope, _, role = _classify("layers.2.attn.indexer.compressor.wkv.weight")
        self.assertEqual((scope, role), ("layer", "attention_indexer"))

    def test_unknown_name_is_rejected(self) -> None:
        with self.assertRaises(InventoryError):
            _classify("layers.3.attn.wq_c.weight")


class ParseHeaderTests(unittest.TestCase):
    def test_accepts_a_fully_described_shard(self) -> None:
        raw = header_bytes(
            {
                "__metadata__": {"format": "pt"},
                "embed.weight": tensor_entry("BF16", [4, 8], 0, 64),
                "norm.weight": tensor_entry("F32", [8], 64, 96),
            }
        )
        tensors = _parse_header("shard", artifact_size(raw, 96), raw)
        self.assertEqual(len(tensors), 2)
        self.assertEqual(sum(tensor["bytes"] for tensor in tensors), 96)
        self.assertEqual({tensor["role"] for tensor in tensors}, {"embedding", "norm"})

    def test_rejects_payload_shorter_than_shape_and_dtype(self) -> None:
        raw = header_bytes({"embed.weight": tensor_entry("BF16", [4, 8], 0, 32)})
        with self.assertRaisesRegex(InventoryError, "does not match shape and dtype"):
            _parse_header("shard", artifact_size(raw, 32), raw)

    def test_rejects_unknown_dtype(self) -> None:
        raw = header_bytes({"embed.weight": tensor_entry("F16", [4], 0, 8)})
        with self.assertRaisesRegex(InventoryError, "unsupported dtype"):
            _parse_header("shard", artifact_size(raw, 8), raw)

    def test_rejects_overlapping_payloads(self) -> None:
        raw = header_bytes(
            {
                "embed.weight": tensor_entry("BF16", [4], 0, 8),
                "head.weight": tensor_entry("BF16", [4], 4, 12),
            }
        )
        with self.assertRaisesRegex(InventoryError, "overlapping"):
            _parse_header("shard", artifact_size(raw, 12), raw)

    def test_rejects_undescribed_trailing_bytes(self) -> None:
        raw = header_bytes({"embed.weight": tensor_entry("BF16", [4], 0, 8)})
        with self.assertRaisesRegex(InventoryError, "not fully described"):
            _parse_header("shard", artifact_size(raw, 4096), raw)

    def test_rejects_payload_escaping_the_shard(self) -> None:
        raw = header_bytes({"embed.weight": tensor_entry("BF16", [4], 0, 8)})
        with self.assertRaisesRegex(InventoryError, "escapes its shard"):
            _parse_header("shard", artifact_size(raw, 4), raw)

    def test_rejects_header_larger_than_artifact(self) -> None:
        raw = header_bytes({"embed.weight": tensor_entry("BF16", [4], 0, 8)})
        with self.assertRaisesRegex(InventoryError, "exceeds declared artifact size"):
            _parse_header("shard", 8, raw)

    def test_rejects_zero_dimension(self) -> None:
        raw = header_bytes({"embed.weight": tensor_entry("BF16", [0], 0, 0)})
        with self.assertRaisesRegex(InventoryError, "invalid shape"):
            _parse_header("shard", artifact_size(raw, 0), raw)


class BudgetTests(unittest.TestCase):
    CONFIG = {"num_hidden_layers": 2, "n_routed_experts": 2, "num_experts_per_tok": 1}

    def tensors(self, expert_bytes: tuple[int, ...] = (100, 100, 100, 100)) -> list[dict]:
        records = []
        for layer in range(2):
            for expert in range(2):
                records.append(
                    {
                        "name": f"layers.{layer}.ffn.experts.{expert}.w1.weight",
                        "scope": "layer",
                        "scope_index": layer,
                        "role": "routed_expert",
                        "expert_index": expert,
                        "bytes": expert_bytes[layer * 2 + expert],
                    }
                )
            records.append(
                {
                    "name": f"layers.{layer}.attn.wq_a.weight",
                    "scope": "layer",
                    "scope_index": layer,
                    "role": "attention_core",
                    "expert_index": None,
                    "bytes": 7,
                }
            )
        records.append(
            {
                "name": "embed.weight",
                "scope": "global",
                "scope_index": None,
                "role": "embedding",
                "expert_index": None,
                "bytes": 11,
            }
        )
        return records

    def test_derives_traffic_from_uniform_experts(self) -> None:
        budgets = _budgets(self.tensors(), self.CONFIG)
        self.assertEqual(budgets["routed_expert_unit_bytes"], 100)
        self.assertEqual(budgets["routed_expert_bytes_all_layers"], 400)
        self.assertEqual(budgets["decode_routed_expert_bytes_per_token"], 200)
        self.assertEqual(budgets["non_expert_layer_bytes_all_layers"], 14)
        self.assertEqual(budgets["global_bytes"]["embedding"], 11)

    def test_rejects_non_uniform_expert_sizes(self) -> None:
        with self.assertRaisesRegex(InventoryError, "not uniformly sized"):
            _budgets(self.tensors((100, 100, 100, 99)), self.CONFIG)

    def test_rejects_layer_count_disagreement(self) -> None:
        with self.assertRaisesRegex(InventoryError, "config declares"):
            _budgets(self.tensors(), {**self.CONFIG, "num_hidden_layers": 3})

    def test_rejects_expert_count_disagreement(self) -> None:
        with self.assertRaisesRegex(InventoryError, "routed-expert count"):
            _budgets(self.tensors(), {**self.CONFIG, "n_routed_experts": 3})


class DecodeTests(unittest.TestCase):
    def test_bf16_widening_matches_float32_truncation(self) -> None:
        values = [0.0, 1.0, -2.375, 17.125, 2.0**-24, -0.0]
        raw = b"".join(struct.pack("<f", value)[2:] for value in values)
        self.assertEqual(list(_decode(raw, "BF16")), values)

    def test_f32_roundtrip(self) -> None:
        values = array("f", [1.5, -0.25, 3.0e-8])
        self.assertEqual(list(_decode(values.tobytes(), "F32")), list(values))

    def test_rejects_quantized_dtype(self) -> None:
        with self.assertRaisesRegex(InventoryError, "only defined for BF16 and F32"):
            _decode(b"\x00", "F8_E4M3")


class SampleSpanTests(unittest.TestCase):
    def test_small_tensor_is_read_completely(self) -> None:
        self.assertEqual(_sample_spans(1000, 2, 4096, 8), [(0, 1000)])

    def test_large_tensor_is_strided_aligned_and_bounded(self) -> None:
        size, element, budget, chunks = 1_000_000, 2, 8192, 8
        spans = _sample_spans(size, element, budget, chunks)
        self.assertEqual(len(spans), chunks)
        self.assertLessEqual(sum(end - start for start, end in spans), budget)
        for start, end in spans:
            self.assertEqual(start % element, 0)
            self.assertEqual((end - start) % element, 0)
            self.assertLessEqual(end, size)
        self.assertEqual(spans, sorted(spans))
        self.assertGreater(spans[-1][0], spans[0][0])


if __name__ == "__main__":
    unittest.main()
