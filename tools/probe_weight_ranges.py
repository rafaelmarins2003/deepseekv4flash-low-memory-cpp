#!/usr/bin/env python3
"""Measure the numeric range of selected non-expert tensors.

The primary GPU target is Turing `sm_75`, which has no BF16 arithmetic. Every
BF16 tensor in the checkpoint must therefore be executed as FP16 or FP32, and
that choice changes the VRAM budget by gigabytes. FP16 is only admissible where
the stored values stay inside its representable range, so the decision needs
measurement rather than assumption.

This tool reads selected tensors by byte range and reports their value range
against FP16 limits. Tensors within the byte budget are measured completely;
larger ones are sampled at evenly spaced strides and reported as samples. A
range violation found in a sample is conclusive; the absence of one in a sample
is not, so `fp16_representable` is null for incompletely measured tensors.

It never downloads expert payloads, never writes weight bytes to disk, and is
not a substitute for the RAF-12 materialization gate. Weight range says nothing
about activation or accumulation range.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from array import array
from pathlib import Path
from typing import Any, Iterable

from build_tensor_inventory import (
    DTYPE_ELEMENT_BYTES,
    SAFETENSORS_LENGTH_FIELD_BYTES,
    InventoryError,
    _request_bytes,
)
from verify_model_manifest import ManifestValidationError, load_manifest


FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0**-14  # 6.103515625e-05
FP16_MIN_SUBNORMAL = 2.0**-24  # 5.960464477539063e-08
DEFAULT_MAX_TENSOR_BYTES = 16 * 1024 * 1024
DEFAULT_SAMPLE_CHUNKS = 8

DEFAULT_TENSORS = (
    "embed.weight",
    "head.weight",
    "layers.0.ffn.gate.weight",
    "layers.20.ffn.gate.weight",
    "layers.2.attn.compressor.wkv.weight",
    "layers.20.attn.indexer.weights_proj.weight",
    "layers.20.hc_attn_fn",
    "mtp.2.markov_head.markov_w1.weight",
)


def _decode(data: bytes, dtype: str) -> array:
    """Decode BF16 or F32 payload bytes into a native float array."""

    if dtype == "F32":
        values = array("f")
        values.frombytes(data)
        return values
    if dtype == "BF16":
        # BF16 is the leading 16 bits of FP32, so widening is a byte placement.
        widened = bytearray(len(data) * 2)
        widened[2::4] = data[0::2]
        widened[3::4] = data[1::2]
        values = array("f")
        values.frombytes(bytes(widened))
        return values
    raise InventoryError(f"range probing is only defined for BF16 and F32, got {dtype}")


def _sample_spans(
    size: int, element_bytes: int, budget: int, chunks: int
) -> list[tuple[int, int]]:
    """Return element-aligned [start, end) spans covering at most `budget` bytes.

    A tensor larger than the budget is sampled at evenly spaced strides rather
    than from its prefix, so the measurement is not dominated by one region of
    the matrix. The result is still a sample and is reported as such.
    """

    if size <= budget:
        return [(0, size)]
    chunk = max(element_bytes, (budget // chunks) - (budget // chunks) % element_bytes)
    stride = (size - chunk) // (chunks - 1)
    stride -= stride % element_bytes
    spans = []
    for index in range(chunks):
        start = min(index * stride, size - chunk)
        start -= start % element_bytes
        spans.append((start, start + chunk))
    return spans


def _shard_header(url: str, artifact_size: int) -> dict[str, Any]:
    prefix = _request_bytes(url, 0, SAFETENSORS_LENGTH_FIELD_BYTES - 1)
    (header_bytes,) = struct.unpack("<Q", prefix)
    if not 0 < header_bytes <= artifact_size:
        raise InventoryError(f"implausible safetensors header length: {header_bytes}")
    raw = _request_bytes(
        url,
        SAFETENSORS_LENGTH_FIELD_BYTES,
        SAFETENSORS_LENGTH_FIELD_BYTES + header_bytes - 1,
    )
    return {"header": json.loads(raw), "payload_origin": SAFETENSORS_LENGTH_FIELD_BYTES + header_bytes}


def probe(
    manifest: dict[str, Any], names: list[str], max_tensor_bytes: int
) -> list[dict[str, Any]]:
    index_artifact = next(
        item for item in manifest["artifacts"] if item["path"] == "model.safetensors.index.json"
    )
    index = json.loads(
        _request_bytes(index_artifact["source_url"], 0, index_artifact["size_bytes"] - 1)
    )
    weight_map = index["weight_map"]
    shards = {item["path"]: item for item in manifest["artifacts"] if item["role"] == "weight_shard"}

    headers: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for name in names:
        if "experts." in name:
            raise InventoryError(f"refusing to probe expert payload: {name}")
        if name not in weight_map:
            raise InventoryError(f"tensor is not in the pinned weight index: {name}")
        shard_path = weight_map[name]
        shard = shards[shard_path]
        if shard_path not in headers:
            headers[shard_path] = _shard_header(shard["source_url"], shard["size_bytes"])
        header = headers[shard_path]
        entry = header["header"][name]
        start, end = entry["data_offsets"]
        size = end - start
        if entry["dtype"] not in DTYPE_ELEMENT_BYTES:
            raise InventoryError(f"unsupported dtype for {name}: {entry['dtype']}")

        origin = header["payload_origin"]
        spans = _sample_spans(
            size, DTYPE_ELEMENT_BYTES[entry["dtype"]], max_tensor_bytes, DEFAULT_SAMPLE_CHUNKS
        )
        values = array("f")
        for span_start, span_end in spans:
            data = _request_bytes(
                shard["source_url"], origin + start + span_start, origin + start + span_end - 1
            )
            values.extend(_decode(data, entry["dtype"]))
        read_bytes = sum(span_end - span_start for span_start, span_end in spans)
        complete = read_bytes == size

        finite = [value for value in values if value == value and abs(value) != float("inf")]
        if len(finite) != len(values):
            raise InventoryError(f"tensor contains non-finite values: {name}")
        magnitudes = [abs(value) for value in finite if value != 0.0]
        violates = any(value > FP16_MAX for value in magnitudes) or any(
            value < FP16_MIN_SUBNORMAL for value in magnitudes
        )
        results.append(
            {
                "name": name,
                "shard": shard_path,
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "bytes": size,
                "coverage": "complete" if complete else "sampled_strided",
                "measured_bytes": read_bytes,
                "min": min(finite),
                "max": max(finite),
                "max_abs": max(magnitudes) if magnitudes else 0.0,
                "min_abs_nonzero": min(magnitudes) if magnitudes else 0.0,
                "zero_count": len(finite) - len(magnitudes),
                "above_fp16_max": sum(1 for value in magnitudes if value > FP16_MAX),
                "below_fp16_min_normal": sum(1 for value in magnitudes if value < FP16_MIN_NORMAL),
                "below_fp16_min_subnormal": sum(
                    1 for value in magnitudes if value < FP16_MIN_SUBNORMAL
                ),
                # A violation is conclusive from any sample; safety is only
                # established when the whole tensor was measured.
                "fp16_representable": False if violates else (True if complete else None),
            }
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/deepseek-v4-flash-0731.json")
    )
    parser.add_argument(
        "--tensor",
        action="append",
        dest="tensors",
        help="Tensor name to probe; repeatable. Defaults to the BF16/F32 tensors on the "
        "constrained-VRAM critical path.",
    )
    parser.add_argument("--max-tensor-bytes", type=int, default=DEFAULT_MAX_TENSOR_BYTES)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        results = probe(
            manifest, list(args.tensors or DEFAULT_TENSORS), args.max_tensor_bytes
        )
    except (InventoryError, ManifestValidationError, OSError, ValueError, KeyError) as exc:
        print(f"PROBE ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "fp16_limits": {
                    "max": FP16_MAX,
                    "min_normal": FP16_MIN_NORMAL,
                    "min_subnormal": FP16_MIN_SUBNORMAL,
                },
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 4 if any(result["fp16_representable"] is False for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
