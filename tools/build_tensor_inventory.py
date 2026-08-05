#!/usr/bin/env python3
"""Derive the DeepSeek-V4-Flash-0731 tensor inventory from pinned shard headers.

The immutable S1.1 manifest establishes which artifacts define the model. This
tool answers the next question RAF-6 needs: what is actually inside the 48
weight shards, and how do those bytes divide into components an engine must
place in VRAM, RAM, or NVMe.

Only safetensors headers are read. Each shard's header is a JSON prefix, so the
complete tensor inventory is obtained with two HTTP range requests per shard and
without downloading any weight payload. Full-checkpoint materialization remains
RAF-12's responsibility.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from verify_model_manifest import (
    ManifestValidationError,
    load_manifest,
    verify_digest_file,
)


SCHEMA_VERSION = "1.0.0"
USER_AGENT = "deepseekv4flash-low-memory-engine-tensor-inventory/1.0"
MAX_HEADER_BYTES = 64 * 1024 * 1024
SAFETENSORS_LENGTH_FIELD_BYTES = 8

# Element sizes for every dtype the pinned checkpoint actually uses. An unknown
# dtype is a specification event, not a value to guess.
DTYPE_ELEMENT_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I64": 8,
}

# Exhaustive tensor-name grammar for the normative revision. Each pattern maps a
# name to (scope, role). Anything unmatched aborts the run.
_LAYER = r"layers\.(?P<index>\d+)\."
_MTP = r"mtp\.(?P<index>\d+)\."
_SUFFIX = r"(?:weight|scale)"

_ROLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"ffn\.experts\.\d+\.w[123]\.{_SUFFIX}$"), "routed_expert"),
    (re.compile(rf"ffn\.shared_experts\.w[123]\.{_SUFFIX}$"), "shared_expert"),
    (re.compile(r"ffn\.gate\.(?:weight|bias|tid2eid)$"), "router"),
    (re.compile(r"ffn_norm\.weight$|attn_norm\.weight$"), "norm"),
    (re.compile(r"attn\.indexer\."), "attention_indexer"),
    (re.compile(r"attn\.compressor\."), "attention_compressor"),
    (
        re.compile(
            rf"attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)\.{_SUFFIX}$"
            r"|attn\.(?:q_norm|kv_norm)\.weight$"
            r"|attn\.attn_sink$"
        ),
        "attention_core",
    ),
    (re.compile(r"hc_(?:attn|ffn|head)_(?:base|fn|scale)$"), "hyper_connection"),
    (re.compile(rf"main_proj\.{_SUFFIX}$|main_norm\.weight$"), "mtp_input_projection"),
    (re.compile(r"markov_head\.markov_w[12]\.weight$"), "mtp_markov_head"),
    (re.compile(r"confidence_head\.proj\.weight$"), "mtp_confidence_head"),
    (re.compile(r"^norm\.weight$"), "norm"),
    (re.compile(r"^embed\.weight$"), "embedding"),
    (re.compile(r"^head\.weight$"), "output_head"),
]


class InventoryError(RuntimeError):
    """Official evidence does not satisfy the derived-inventory contract."""


def _request_bytes(url: str, first: int, last: int) -> bytes:
    """Fetch an inclusive byte range, retrying transient transport failures."""

    request = Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "Range": f"bytes={first}-{last}",
            "User-Agent": USER_AGENT,
        },
    )
    expected = last - first + 1
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise InventoryError(
                        f"server ignored the range request for {url}: HTTP {response.status}"
                    )
                data = response.read()
            if len(data) != expected:
                raise InventoryError(
                    f"short range read for {url}: expected {expected}, got {len(data)}"
                )
            return data
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    raise InventoryError(f"cannot fetch official source {url}: {last_error}")


def _classify(name: str) -> tuple[str, int | None, str]:
    """Return (scope, scope_index, role) for a normative tensor name."""

    scope = "global"
    scope_index: int | None = None
    remainder = name
    layer = re.match(_LAYER, name)
    mtp = re.match(_MTP, name)
    if layer is not None:
        scope, scope_index, remainder = "layer", int(layer["index"]), name[layer.end() :]
    elif mtp is not None:
        scope, scope_index, remainder = "mtp", int(mtp["index"]), name[mtp.end() :]

    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(remainder if scope != "global" else name):
            return scope, scope_index, role
    raise InventoryError(f"tensor name has no explicit semantic role: {name}")


def _expert_index(name: str) -> int | None:
    match = re.search(r"ffn\.experts\.(\d+)\.", name)
    return int(match[1]) if match else None


def _parse_header(shard_path: str, artifact_size: int, raw: bytes) -> list[dict[str, Any]]:
    """Validate one shard header and return its tensor records."""

    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InventoryError(f"shard header is not JSON: {shard_path}") from exc
    if not isinstance(header, dict):
        raise InventoryError(f"shard header is not an object: {shard_path}")

    payload_origin = SAFETENSORS_LENGTH_FIELD_BYTES + len(raw)
    payload_capacity = artifact_size - payload_origin
    if payload_capacity < 0:
        raise InventoryError(f"shard header exceeds declared artifact size: {shard_path}")

    tensors: list[dict[str, Any]] = []
    spans: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise InventoryError(f"tensor entry is not an object: {shard_path}:{name}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in DTYPE_ELEMENT_BYTES:
            raise InventoryError(f"unsupported dtype {dtype!r} in {shard_path}:{name}")
        if not isinstance(shape, list) or not shape or not all(
            isinstance(dim, int) and dim > 0 for dim in shape
        ):
            raise InventoryError(f"tensor has an invalid shape: {shard_path}:{name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise InventoryError(f"tensor has invalid data_offsets: {shard_path}:{name}")

        start, end = offsets
        size = end - start
        elements = 1
        for dim in shape:
            elements *= dim
        expected = elements * DTYPE_ELEMENT_BYTES[dtype]
        if size != expected:
            raise InventoryError(
                f"declared payload does not match shape and dtype for {shard_path}:{name}: "
                f"header={size}, computed={expected}"
            )
        if start < 0 or end > payload_capacity or start > end:
            raise InventoryError(f"tensor payload escapes its shard: {shard_path}:{name}")

        spans.append((start, end, name))
        scope, scope_index, role = _classify(name)
        tensors.append(
            {
                "name": name,
                "shard": shard_path,
                "dtype": dtype,
                "shape": shape,
                "bytes": size,
                "scope": scope,
                "scope_index": scope_index,
                "role": role,
                "expert_index": _expert_index(name),
            }
        )

    spans.sort()
    cursor = 0
    for start, end, name in spans:
        if start < cursor:
            raise InventoryError(f"overlapping tensor payloads at {shard_path}:{name}")
        cursor = end
    if cursor != payload_capacity:
        raise InventoryError(
            f"shard payload is not fully described by its header: {shard_path}: "
            f"described={cursor}, capacity={payload_capacity}"
        )
    return tensors


def _read_shard(artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    url = artifact["source_url"]
    prefix = _request_bytes(url, 0, SAFETENSORS_LENGTH_FIELD_BYTES - 1)
    (header_bytes,) = struct.unpack("<Q", prefix)
    if not 0 < header_bytes <= MAX_HEADER_BYTES:
        raise InventoryError(
            f"refusing implausible header length {header_bytes} for {artifact['path']}"
        )
    raw = _request_bytes(
        url,
        SAFETENSORS_LENGTH_FIELD_BYTES,
        SAFETENSORS_LENGTH_FIELD_BYTES + header_bytes - 1,
    )
    tensors = _parse_header(artifact["path"], artifact["size_bytes"], raw)
    overhead = SAFETENSORS_LENGTH_FIELD_BYTES + header_bytes
    return tensors, overhead


def _fetch_verified_artifact(
    manifest: dict[str, Any], path: str, max_bytes: int
) -> bytes:
    """Download one manifest artifact and require its pinned SHA-256."""

    from hashlib import sha256

    artifact = next(
        (item for item in manifest["artifacts"] if item["path"] == path), None
    )
    if artifact is None:
        raise InventoryError(f"manifest does not contain required artifact: {path}")
    if artifact["size_bytes"] > max_bytes:
        raise InventoryError(f"refusing oversized artifact {path}: {artifact['size_bytes']}")
    data = _request_bytes(artifact["source_url"], 0, artifact["size_bytes"] - 1)
    digest = sha256(data).hexdigest()
    if digest != artifact["sha256"]:
        raise InventoryError(
            f"artifact bytes do not match the immutable manifest for {path}: "
            f"manifest={artifact['sha256']}, local={digest}"
        )
    return data


def _cross_check_index(manifest: dict[str, Any], tensors: list[dict[str, Any]]) -> None:
    """Require the weight index and the shard containers to agree exactly."""

    index = json.loads(
        _fetch_verified_artifact(manifest, "model.safetensors.index.json", 32 * 1024 * 1024)
    )
    weight_map = index["weight_map"]
    container_map = {tensor["name"]: tensor["shard"] for tensor in tensors}
    if weight_map != container_map:
        missing = sorted(set(weight_map) - set(container_map))
        extra = sorted(set(container_map) - set(weight_map))
        moved = sorted(
            name
            for name in set(weight_map) & set(container_map)
            if weight_map[name] != container_map[name]
        )
        raise InventoryError(
            "weight index and shard headers disagree: "
            f"missing={missing[:5]}, extra={extra[:5]}, moved={moved[:5]}"
        )


def _rollup(tensors: list[dict[str, Any]], key: Any) -> list[dict[str, Any]]:
    grouped: dict[Any, dict[str, int]] = {}
    for tensor in tensors:
        bucket = grouped.setdefault(key(tensor), {"tensor_count": 0, "bytes": 0})
        bucket["tensor_count"] += 1
        bucket["bytes"] += tensor["bytes"]
    return [
        {"key": name, **values} for name, values in sorted(grouped.items(), key=lambda kv: str(kv[0]))
    ]


def _budgets(tensors: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Derive the placement quantities the engine's memory plan depends on."""

    layers = config["num_hidden_layers"]
    routed_experts = config["n_routed_experts"]
    active_experts = config["num_experts_per_tok"]

    layer_tensors = [tensor for tensor in tensors if tensor["scope"] == "layer"]
    layer_indices = {tensor["scope_index"] for tensor in layer_tensors}
    if layer_indices != set(range(layers)):
        raise InventoryError(
            f"shard headers describe layers {sorted(layer_indices)[:5]}…, config declares {layers}"
        )

    expert_units: dict[tuple[int, int], int] = {}
    for tensor in layer_tensors:
        if tensor["role"] != "routed_expert":
            continue
        unit = (tensor["scope_index"], tensor["expert_index"])
        expert_units[unit] = expert_units.get(unit, 0) + tensor["bytes"]
    unit_sizes = set(expert_units.values())
    if len(unit_sizes) != 1:
        raise InventoryError(f"routed experts are not uniformly sized: {sorted(unit_sizes)}")
    expert_bytes = unit_sizes.pop()
    per_layer_experts = {index: 0 for index in range(layers)}
    for (layer_index, _), _ in expert_units.items():
        per_layer_experts[layer_index] += 1
    if set(per_layer_experts.values()) != {routed_experts}:
        raise InventoryError("layers do not all carry the configured routed-expert count")

    non_expert_layer_bytes = sum(
        tensor["bytes"] for tensor in layer_tensors if tensor["role"] != "routed_expert"
    )
    global_bytes = {
        role: sum(
            tensor["bytes"]
            for tensor in tensors
            if tensor["scope"] == "global" and tensor["role"] == role
        )
        for role in ("embedding", "output_head", "norm", "hyper_connection")
    }
    mtp_bytes = sum(tensor["bytes"] for tensor in tensors if tensor["scope"] == "mtp")

    return {
        "routed_expert_unit_bytes": expert_bytes,
        "routed_experts_per_layer": routed_experts,
        "active_routed_experts_per_token": active_experts,
        "routed_expert_bytes_per_layer": expert_bytes * routed_experts,
        "routed_expert_bytes_all_layers": expert_bytes * routed_experts * layers,
        "non_expert_layer_bytes_all_layers": non_expert_layer_bytes,
        "global_bytes": global_bytes,
        "mtp_bytes": mtp_bytes,
        "decode_routed_expert_bytes_per_token": expert_bytes * active_experts * layers,
        "prefill_routed_expert_bytes_per_full_union_sweep": (
            expert_bytes * routed_experts * layers
        ),
        "formulas": {
            "decode_routed_expert_bytes_per_token": (
                "routed_expert_unit_bytes * num_experts_per_tok * num_hidden_layers; "
                "assumes a cold expert cache and no reuse between consecutive tokens"
            ),
            "prefill_routed_expert_bytes_per_full_union_sweep": (
                "routed_expert_unit_bytes * n_routed_experts * num_hidden_layers; "
                "upper bound reached once a prefill chunk's routing union covers every expert"
            ),
            "non_expert_layer_bytes_all_layers": (
                "sum of every layer-scope tensor that is not a routed expert; "
                "the candidate permanently resident VRAM working set"
            ),
        },
    }


def build_inventory(
    manifest: dict[str, Any], manifest_sha256: str, generated_at_utc: str, workers: int
) -> dict[str, Any]:
    shards = [
        artifact for artifact in manifest["artifacts"] if artifact["role"] == "weight_shard"
    ]
    if not shards:
        raise InventoryError("manifest declares no weight shards")

    tensors: list[dict[str, Any]] = []
    container_overhead = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for shard_tensors, overhead in pool.map(_read_shard, shards):
            tensors.extend(shard_tensors)
            container_overhead += overhead

    names = [tensor["name"] for tensor in tensors]
    if len(names) != len(set(names)):
        raise InventoryError("shard headers declare duplicate tensor names")
    tensors.sort(key=lambda tensor: tensor["name"])

    totals = manifest["totals"]
    payload_bytes = sum(tensor["bytes"] for tensor in tensors)
    if len(tensors) != totals["weight_tensor_count"]:
        raise InventoryError(
            f"tensor count disagrees with the immutable manifest: "
            f"headers={len(tensors)}, manifest={totals['weight_tensor_count']}"
        )
    if payload_bytes != totals["weight_tensor_bytes"]:
        raise InventoryError(
            f"tensor payload bytes disagree with the immutable manifest: "
            f"headers={payload_bytes}, manifest={totals['weight_tensor_bytes']}"
        )
    if container_overhead != totals["weight_container_overhead_bytes"]:
        raise InventoryError(
            f"container overhead disagrees with the immutable manifest: "
            f"headers={container_overhead}, manifest={totals['weight_container_overhead_bytes']}"
        )

    _cross_check_index(manifest, tensors)
    config = json.loads(_fetch_verified_artifact(manifest, "config.json", 1024 * 1024))

    return {
        "schema_version": SCHEMA_VERSION,
        "derived_from_manifest_sha256": manifest_sha256,
        "model": dict(manifest["model"]),
        "generated_at_utc": generated_at_utc,
        "verification": {
            "method": "safetensors_header_range_reads",
            "weight_payload_downloaded": False,
            "shards_read": len(shards),
            "manifest_totals_cross_check": "pass",
            "weight_index_cross_check": "pass",
            "config_sha256_cross_check": "pass",
        },
        "totals": {
            "tensor_count": len(tensors),
            "tensor_payload_bytes": payload_bytes,
            "container_overhead_bytes": container_overhead,
            "shard_bytes": totals["weight_shard_bytes"],
        },
        "config_facts": {
            key: config[key]
            for key in (
                "num_hidden_layers",
                "num_hash_layers",
                "n_routed_experts",
                "n_shared_experts",
                "num_experts_per_tok",
                "hidden_size",
                "moe_intermediate_size",
                "expert_dtype",
                "num_nextn_predict_layers",
            )
        },
        "by_dtype": _rollup(tensors, lambda tensor: tensor["dtype"]),
        "by_scope_role": _rollup(
            tensors, lambda tensor: f"{tensor['scope']}/{tensor['role']}"
        ),
        "budgets": _budgets(tensors, config),
    }


def _write_output(path: Path, data: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return
        if not force:
            raise InventoryError(f"refusing to replace changed output without --force: {path}")
    path.write_bytes(data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-at-utc",
        required=True,
        help="Fixed whole-second ISO-8601 UTC timestamp used in deterministic output",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/deepseek-v4-flash-0731.json")
    )
    parser.add_argument(
        "--manifest-digest",
        type=Path,
        default=Path("manifests/deepseek-v4-flash-0731.sha256"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("manifests/derived/tensor-inventory-0731.json")
    )
    parser.add_argument(
        "--digest", type=Path, default=Path("manifests/derived/tensor-inventory-0731.sha256")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", args.generated_at_utc):
            raise InventoryError("--generated-at-utc must be whole-second UTC, e.g. 2026-08-05T12:00:00Z")
        if args.workers < 1:
            raise InventoryError("--workers must be at least 1")
        manifest = load_manifest(args.manifest)
        manifest_sha256 = verify_digest_file(args.manifest, args.manifest_digest)
        inventory = build_inventory(
            manifest, manifest_sha256, args.generated_at_utc, args.workers
        )
        payload = (
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        from hashlib import sha256

        digest = sha256(payload).hexdigest()
        _write_output(args.output, payload, force=args.force)
        _write_output(
            args.digest, f"{digest}  {args.output.name}\n".encode("ascii"), force=args.force
        )
    except (InventoryError, ManifestValidationError, OSError, ValueError, KeyError) as exc:
        print(f"INVENTORY ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "derived_from_manifest_sha256": manifest_sha256,
                "inventory": str(args.output),
                "inventory_sha256": digest,
                "tensor_count": inventory["totals"]["tensor_count"],
                "tensor_payload_bytes": inventory["totals"]["tensor_payload_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
