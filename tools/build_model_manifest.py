#!/usr/bin/env python3
"""Build the pinned DeepSeek-V4-Flash-0731 manifest from official sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from verify_model_manifest import validate_manifest, verify_artifacts


MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
NORMATIVE_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
DOCUMENTATION_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
WEIGHT_UPLOAD_REVISION = "7da4bfdd329d1baa9e18691121327b082e809454"
INITIAL_REVISION = "a5afaad1bafd888c90a8b6f58719204a7753416e"

PREVIEW_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
PREVIEW_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
DSPARK_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
DSPARK_REVISION = "62af8fffb2f7030cac4de2f0169f5b8d1101b646"

EXPECTED_ARTIFACT_COUNT = 74
EXPECTED_WEIGHT_SHARD_COUNT = 48
EXPECTED_NON_WEIGHT_COUNT = 26
EXPECTED_WEIGHT_SHARD_BYTES = 166_886_535_336
MAX_NON_WEIGHT_FILE_BYTES = 32 * 1024 * 1024
USER_AGENT = "deepseekv4flash-low-memory-engine-s1.1/1.0"
WEIGHT_RE = re.compile(r"^model-(\d{5})-of-00048[.]safetensors$")
ANY_WEIGHT_RE = re.compile(r"^model-\d{5}-of-\d{5}[.]safetensors$")

EXPECTED_LINEAGE = [
    {
        "role": "documentation_snapshot",
        "revision": DOCUMENTATION_REVISION,
        "parent_revision": NORMATIVE_REVISION,
        "title": "add sglang cookbook to model card (#20)",
        "author_date_utc": "2026-08-01T03:07:41Z",
        "committer_date_utc": "2026-08-01T03:07:41Z",
    },
    {
        "role": "normative_release",
        "revision": NORMATIVE_REVISION,
        "parent_revision": WEIGHT_UPLOAD_REVISION,
        "title": "Release DeepSeek-V4-Flash-0731",
        "author_date_utc": "2026-07-31T12:02:14Z",
        "committer_date_utc": "2026-07-31T12:02:14Z",
    },
    {
        "role": "weight_upload",
        "revision": WEIGHT_UPLOAD_REVISION,
        "parent_revision": INITIAL_REVISION,
        "title": "Add files using upload-large-folder tool",
        "author_date_utc": "2026-07-31T09:54:07Z",
        "committer_date_utc": "2026-07-31T09:54:07Z",
    },
    {
        "role": "initial_commit",
        "revision": INITIAL_REVISION,
        "parent_revision": None,
        "title": "initial commit",
        "author_date_utc": "2026-07-31T07:30:24Z",
        "committer_date_utc": "2026-07-31T07:30:24Z",
    },
]


class AuditError(RuntimeError):
    """Official evidence does not satisfy the pinned audit contract."""


def _request_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/json, application/octet-stream;q=0.9, */*;q=0.8",
            "User-Agent": USER_AGENT,
        },
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    raise AuditError(f"cannot fetch official source {url}: {last_error}")


def _request_json(url: str) -> Any:
    try:
        return json.loads(_request_bytes(url))
    except json.JSONDecodeError as exc:
        raise AuditError(f"official source did not return JSON: {url}") from exc


def _tree_url(model_id: str, revision: str) -> str:
    query = urlencode({"recursive": "true", "expand": "false", "blobs": "true"})
    return (
        f"https://huggingface.co/api/models/{quote(model_id, safe='/')}/tree/"
        f"{revision}?{query}"
    )


def _resolve_url(model_id: str, revision: str, path: str) -> str:
    return (
        f"https://huggingface.co/{quote(model_id, safe='/')}/resolve/"
        f"{revision}/{quote(path, safe='/')}"
    )


def _commit_url(model_id: str, revision: str) -> str:
    return f"https://huggingface.co/{quote(model_id, safe='/')}/commit/{revision}"


def _normalize_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditError(f"invalid upstream timestamp: {value}") from exc
    return (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_audit_timestamp(value: str) -> str:
    normalized = _normalize_utc(value)
    if normalized != value:
        raise AuditError(
            f"--audited-at-utc must be normalized to whole-second UTC, expected {normalized}"
        )
    return value


def _fetch_tree(model_id: str, revision: str) -> dict[str, dict[str, Any]]:
    payload = _request_json(_tree_url(model_id, revision))
    if not isinstance(payload, list):
        raise AuditError(f"tree response is not a list for {model_id}@{revision}")
    files: dict[str, dict[str, Any]] = {}
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise AuditError(f"tree contains an invalid file path for {model_id}@{revision}")
        if path in files:
            raise AuditError(f"tree contains duplicate path: {path}")
        size = entry.get("size")
        oid = entry.get("oid")
        if not isinstance(size, int) or size < 0 or not isinstance(oid, str):
            raise AuditError(f"tree entry lacks size or Git oid: {path}")
        files[path] = entry
    return files


def _entry_fingerprint(entry: dict[str, Any]) -> tuple[Any, ...]:
    lfs = entry.get("lfs") or {}
    return (entry.get("oid"), entry.get("size"), lfs.get("oid"), entry.get("xetHash"))


def _changed_paths(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> list[str]:
    return sorted(
        path
        for path in set(left) | set(right)
        if path not in left
        or path not in right
        or _entry_fingerprint(left[path]) != _entry_fingerprint(right[path])
    )


def _is_weight(path: str) -> bool:
    return WEIGHT_RE.fullmatch(path) is not None


def _weight_paths(tree: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(path for path in tree if ANY_WEIGHT_RE.fullmatch(path))


def _artifact_role(path: str) -> str:
    if _is_weight(path):
        return "weight_shard"
    exact_roles = {
        ".gitattributes": "repository_metadata",
        "LICENSE": "license",
        "README.md": "model_card",
        "config.json": "model_config",
        "generation_config.json": "generation_config",
        "model.safetensors.index.json": "weight_index",
        "tokenizer.json": "tokenizer",
        "tokenizer_config.json": "tokenizer_config",
        "encoding/README.md": "encoding_documentation",
        "encoding/encoding_dsv4.py": "encoding_reference_code",
        "encoding/test_encoding_dsv4.py": "encoding_test",
        "inference/README.md": "inference_documentation",
        "inference/config.json": "inference_config",
        "inference/requirements.txt": "inference_requirements",
    }
    if path in exact_roles:
        return exact_roles[path]
    if re.fullmatch(r"encoding/tests/test_input_\d+[.]json", path):
        return "encoding_test_input"
    if re.fullmatch(r"encoding/tests/test_output_\d+[.]txt", path):
        return "encoding_test_output"
    if path.startswith("inference/") and path.endswith(".py"):
        return "inference_reference_code"
    raise AuditError(f"artifact has no explicit semantic role: {path}")


def _upstream_objects(entry: dict[str, Any]) -> list[dict[str, str]]:
    objects = [
        {"kind": "git_blob", "algorithm": "sha1", "value": entry["oid"]}
    ]
    lfs = entry.get("lfs")
    if lfs:
        oid = lfs.get("oid")
        if not isinstance(oid, str):
            raise AuditError(f"LFS entry has no oid: {entry['path']}")
        objects.append({"kind": "lfs", "algorithm": "sha256", "value": oid})
    xet_hash = entry.get("xetHash")
    if xet_hash:
        objects.append({"kind": "xet", "algorithm": "sha256", "value": xet_hash})
    return objects


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _verified_content_sha256(
    entry: dict[str, Any], data: bytes, path: str
) -> str:
    if len(data) != entry["size"]:
        raise AuditError(
            f"downloaded size mismatch for {path}: "
            f"tree={entry['size']}, local={len(data)}"
        )

    local_git_oid = _git_blob_sha1(data)
    if local_git_oid != entry["oid"]:
        raise AuditError(
            f"downloaded Git blob mismatch for {path}: "
            f"tree={entry['oid']}, local={local_git_oid}"
        )
    return _sha256(data)


def _fetch_release_lineage() -> list[dict[str, Any]]:
    url = (
        f"https://huggingface.co/api/models/{quote(MODEL_ID, safe='/')}/commits/"
        f"{DOCUMENTATION_REVISION}"
    )
    payload = _request_json(url)
    if not isinstance(payload, list):
        raise AuditError("commit history response is not a list")
    actual = payload[: len(EXPECTED_LINEAGE)]
    actual_revisions = [entry.get("id") for entry in actual]
    expected_revisions = [entry["revision"] for entry in EXPECTED_LINEAGE]
    if actual_revisions != expected_revisions:
        raise AuditError(
            f"release lineage mismatch: expected {expected_revisions}, got {actual_revisions}"
        )
    lineage = []
    for expected, upstream in zip(EXPECTED_LINEAGE, actual, strict=True):
        if upstream.get("title") != expected["title"]:
            raise AuditError(
                f"commit title mismatch for {expected['revision']}: {upstream.get('title')!r}"
            )
        upstream_date = upstream.get("date")
        if not isinstance(upstream_date, str):
            raise AuditError(f"commit date missing for {expected['revision']}")
        if _normalize_utc(upstream_date) != expected["committer_date_utc"]:
            raise AuditError(f"commit date mismatch for {expected['revision']}")
        lineage.append(
            {
                **expected,
                "source_url": _commit_url(MODEL_ID, expected["revision"]),
            }
        )
    return lineage


def _prepare_cache(path: Path) -> Path:
    if path.exists():
        if not path.is_dir():
            raise AuditError(f"non-weight cache is not a directory: {path}")
        if any(path.iterdir()):
            raise AuditError(f"non-weight cache must be empty: {path}")
    else:
        path.mkdir(parents=True)
    return path.resolve()


def _download_non_weights(
    tree: dict[str, dict[str, Any]], cache_root: Path
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    artifacts: list[dict[str, Any]] = []
    local_bytes: dict[str, bytes] = {}
    for path in sorted(tree):
        entry = tree[path]
        role = _artifact_role(path)
        if role == "weight_shard":
            lfs = entry.get("lfs")
            if not isinstance(lfs, dict) or not isinstance(lfs.get("oid"), str):
                raise AuditError(f"weight shard lacks typed LFS SHA-256: {path}")
            artifacts.append(
                {
                    "path": path,
                    "role": role,
                    "size_bytes": entry["size"],
                    "sha256": lfs["oid"],
                    "verification_level": "upstream_metadata",
                    "upstream_objects": _upstream_objects(entry),
                    "source_url": _resolve_url(MODEL_ID, NORMATIVE_REVISION, path),
                }
            )
            continue

        if entry["size"] > MAX_NON_WEIGHT_FILE_BYTES:
            raise AuditError(
                f"refusing unexpected large non-weight artifact {path}: {entry['size']} bytes"
            )
        data = _request_bytes(_resolve_url(MODEL_ID, NORMATIVE_REVISION, path))
        content_sha256 = _verified_content_sha256(entry, data, path)
        target = cache_root.joinpath(*path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                handle.write(data)
        except FileExistsError as exc:
            raise AuditError(f"refusing to overwrite cached artifact: {target}") from exc
        local_bytes[path] = data
        artifacts.append(
            {
                "path": path,
                "role": role,
                "size_bytes": entry["size"],
                "sha256": content_sha256,
                "verification_level": "local_bytes",
                "upstream_objects": _upstream_objects(entry),
                "source_url": _resolve_url(MODEL_ID, NORMATIVE_REVISION, path),
            }
        )
    return artifacts, local_bytes


def _json_file(local_bytes: dict[str, bytes], path: str) -> dict[str, Any]:
    try:
        value = json.loads(local_bytes[path])
    except (KeyError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot parse required JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"required JSON artifact is not an object: {path}")
    return value


def _known_divergences(local_bytes: dict[str, bytes]) -> list[dict[str, Any]]:
    config = _json_file(local_bytes, "config.json")
    generation = _json_file(local_bytes, "generation_config.json")
    inference = _json_file(local_bytes, "inference/config.json")
    tokenizer = _json_file(local_bytes, "tokenizer_config.json")
    requirements = local_bytes["inference/requirements.txt"].decode("utf-8").splitlines()
    readme = local_bytes["README.md"].decode("utf-8")

    expected_requirements = {
        "torch>=2.10.0",
        "transformers>=5.0.0",
        "safetensors>=0.7.0",
        "fast_hadamard_transform",
        "tilelang==0.1.8",
    }
    if set(requirements) != expected_requirements:
        raise AuditError(f"unexpected inference requirements: {requirements}")
    if "chat_template" in tokenizer:
        raise AuditError("tokenizer_config.json unexpectedly contains chat_template")
    if "does not include a Jinja-format chat template" not in readme:
        raise AuditError("normative model card no longer states the expected Jinja absence")

    return [
        {
            "id": "transformers_version_metadata",
            "summary": "Official files name three different Transformers version constraints or producer versions.",
            "status": "unresolved_semantic_mapping",
            "handoff": ["RAF-6", "RAF-8"],
            "evidence": [
                {
                    "path": "config.json",
                    "selector": "/transformers_version",
                    "value": config.get("transformers_version"),
                },
                {
                    "path": "generation_config.json",
                    "selector": "/transformers_version",
                    "value": generation.get("transformers_version"),
                },
                {
                    "path": "inference/requirements.txt",
                    "selector": "line matching transformers",
                    "value": next(
                        (line for line in requirements if line.startswith("transformers")), None
                    ),
                },
            ],
        },
        {
            "id": "next_token_prediction_layer_counts",
            "summary": "Top-level and reference-inference configs expose differently named predictive-layer counts.",
            "status": "unresolved_semantic_mapping",
            "handoff": ["RAF-6", "RAF-8"],
            "evidence": [
                {
                    "path": "config.json",
                    "selector": "/num_nextn_predict_layers",
                    "value": config.get("num_nextn_predict_layers"),
                },
                {
                    "path": "inference/config.json",
                    "selector": "/n_mtp_layers",
                    "value": inference.get("n_mtp_layers"),
                },
            ],
        },
        {
            "id": "quantization_scope_representation",
            "summary": "Expert tensors are identified as FP4 while the broader released quantization metadata and reference dtype identify FP8 scopes.",
            "status": "distinct_scopes_preserved",
            "handoff": ["RAF-6", "RAF-8"],
            "evidence": [
                {
                    "path": "config.json",
                    "selector": "/expert_dtype",
                    "value": config.get("expert_dtype"),
                },
                {
                    "path": "config.json",
                    "selector": "/quantization_config/quant_method",
                    "value": (config.get("quantization_config") or {}).get("quant_method"),
                },
                {
                    "path": "inference/config.json",
                    "selector": "/dtype",
                    "value": inference.get("dtype"),
                },
                {
                    "path": "inference/config.json",
                    "selector": "/expert_dtype",
                    "value": inference.get("expert_dtype"),
                },
            ],
        },
    ]


def _weight_index_summary(
    local_bytes: dict[str, bytes], expected_weight_paths: list[str], shard_bytes: int
) -> dict[str, int]:
    index = _json_file(local_bytes, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AuditError("model.safetensors.index.json has no non-empty weight_map")
    if not isinstance(metadata, dict):
        raise AuditError("model.safetensors.index.json has no metadata object")
    shard_values = list(weight_map.values())
    if not all(isinstance(path, str) for path in shard_values):
        raise AuditError("weight index contains a non-string shard path")
    referenced_shards = sorted(set(shard_values))
    if referenced_shards != expected_weight_paths:
        raise AuditError(
            "weight index shard references do not exactly match the normative tree"
        )
    tensor_bytes = metadata.get("total_size")
    if not isinstance(tensor_bytes, int) or tensor_bytes < 0:
        raise AuditError("weight index metadata.total_size is not a non-negative integer")
    container_overhead = shard_bytes - tensor_bytes
    if container_overhead < 0:
        raise AuditError("weight tensor bytes exceed physical shard bytes")
    return {
        "weight_tensor_count": len(weight_map),
        "weight_tensor_bytes": tensor_bytes,
        "weight_container_overhead_bytes": container_overhead,
    }


def _tree_totals(tree: dict[str, dict[str, Any]]) -> dict[str, int]:
    weights = _weight_paths(tree)
    return {
        "artifact_count": len(tree),
        "weight_shard_count": len(weights),
        "weight_shard_bytes": sum(tree[path]["size"] for path in weights),
    }


def _prior_release_comparison(
    normative_tree: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    preview_tree = _fetch_tree(PREVIEW_MODEL_ID, PREVIEW_REVISION)
    dspark_tree = _fetch_tree(DSPARK_MODEL_ID, DSPARK_REVISION)
    preview = _tree_totals(preview_tree)
    dspark = _tree_totals(dspark_tree)

    normative_weights = _weight_paths(normative_tree)
    dspark_weights = _weight_paths(dspark_tree)
    common_weights = sorted(set(normative_weights) & set(dspark_weights))
    matching_sha256 = sum(
        1
        for path in common_weights
        if (normative_tree[path].get("lfs") or {}).get("oid")
        == (dspark_tree[path].get("lfs") or {}).get("oid")
    )
    matching_size = sum(
        1
        for path in common_weights
        if normative_tree[path]["size"] == dspark_tree[path]["size"]
    )
    all_paths = set(normative_tree) | set(dspark_tree)
    non_weight_changed = sorted(
        path
        for path in all_paths
        if not ANY_WEIGHT_RE.fullmatch(path)
        and (
            path not in normative_tree
            or path not in dspark_tree
            or _entry_fingerprint(normative_tree[path]) != _entry_fingerprint(dspark_tree[path])
        )
    )

    return {
        "preview": {
            "model_id": PREVIEW_MODEL_ID,
            "revision": PREVIEW_REVISION,
            **preview,
            "summary": "The official 0731 release supersedes this earlier 46-shard preview snapshot.",
        },
        "dspark": {
            "model_id": DSPARK_MODEL_ID,
            "revision": DSPARK_REVISION,
            **dspark,
            "config_matches_normative": _entry_fingerprint(normative_tree["config.json"])
            == _entry_fingerprint(dspark_tree["config.json"]),
            "matching_weight_sha256_count": matching_sha256,
            "matching_weight_size_count": matching_size,
            "non_weight_changed_paths": non_weight_changed,
            "summary": "The 0731 release preserves the DSpark-capable structure while changing weight identity and selected encoding/model-card artifacts.",
        },
    }


def _assert_normative_tree(tree: dict[str, dict[str, Any]]) -> None:
    weights = _weight_paths(tree)
    expected_weight_paths = [
        f"model-{index:05d}-of-00048.safetensors"
        for index in range(1, EXPECTED_WEIGHT_SHARD_COUNT + 1)
    ]
    failures = []
    if len(tree) != EXPECTED_ARTIFACT_COUNT:
        failures.append(f"artifact_count={len(tree)}")
    if weights != expected_weight_paths:
        failures.append("weight shard names are incomplete or unexpected")
    if len(tree) - len(weights) != EXPECTED_NON_WEIGHT_COUNT:
        failures.append(f"non_weight_count={len(tree) - len(weights)}")
    weight_bytes = sum(tree[path]["size"] for path in weights)
    if weight_bytes != EXPECTED_WEIGHT_SHARD_BYTES:
        failures.append(f"weight_shard_bytes={weight_bytes}")
    if failures:
        raise AuditError(f"normative tree contract failed: {', '.join(failures)}")


def _write_output(path: Path, data: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = path.read_bytes()
        if current == data:
            return
        if not force:
            raise AuditError(f"refusing to replace changed output without --force: {path}")
    path.write_bytes(data)


def build_manifest(audited_at_utc: str, cache_root: Path) -> dict[str, Any]:
    normative_tree = _fetch_tree(MODEL_ID, NORMATIVE_REVISION)
    documentation_tree = _fetch_tree(MODEL_ID, DOCUMENTATION_REVISION)
    _assert_normative_tree(normative_tree)
    changed_paths = _changed_paths(normative_tree, documentation_tree)
    if changed_paths != ["README.md"]:
        raise AuditError(
            f"documentation snapshot changed unexpected paths: {changed_paths}"
        )

    artifacts, local_bytes = _download_non_weights(normative_tree, cache_root)
    documentation_readme = _request_bytes(
        _resolve_url(MODEL_ID, DOCUMENTATION_REVISION, "README.md")
    )
    documentation_readme_sha256 = _verified_content_sha256(
        documentation_tree["README.md"], documentation_readme, "README.md"
    )

    weights = [artifact for artifact in artifacts if artifact["role"] == "weight_shard"]
    weight_paths = [artifact["path"] for artifact in weights]
    weight_shard_bytes = sum(artifact["size_bytes"] for artifact in weights)
    weight_index = _weight_index_summary(
        local_bytes, weight_paths, weight_shard_bytes
    )
    manifest = {
        "schema_version": "1.0.0",
        "model": {
            "id": MODEL_ID,
            "normative_revision": NORMATIVE_REVISION,
        },
        "documentation_snapshot": {
            "revision": DOCUMENTATION_REVISION,
            "scope": "informative_only",
            "changed_paths": changed_paths,
            "readme_sha256": documentation_readme_sha256,
            "readme_size_bytes": len(documentation_readme),
            "verification_level": "local_bytes",
        },
        "audited_at_utc": audited_at_utc,
        "source_urls": {
            "documentation_commit": _commit_url(MODEL_ID, DOCUMENTATION_REVISION),
            "documentation_tree": (
                f"https://huggingface.co/{MODEL_ID}/tree/{DOCUMENTATION_REVISION}"
            ),
            "dspark_tree": (
                f"https://huggingface.co/{DSPARK_MODEL_ID}/tree/{DSPARK_REVISION}"
            ),
            "normative_commit": _commit_url(MODEL_ID, NORMATIVE_REVISION),
            "normative_tree": (
                f"https://huggingface.co/{MODEL_ID}/tree/{NORMATIVE_REVISION}"
            ),
            "preview_tree": (
                f"https://huggingface.co/{PREVIEW_MODEL_ID}/tree/{PREVIEW_REVISION}"
            ),
        },
        "release_lineage": _fetch_release_lineage(),
        "known_upstream_divergences": _known_divergences(local_bytes),
        "prior_release_comparison": _prior_release_comparison(normative_tree),
        "totals": {
            "artifact_count": len(artifacts),
            "weight_shard_count": len(weights),
            "non_weight_artifact_count": len(artifacts) - len(weights),
            "weight_shard_bytes": weight_shard_bytes,
            **weight_index,
            "total_bytes": sum(artifact["size_bytes"] for artifact in artifacts),
        },
        "artifacts": artifacts,
    }
    validate_manifest(manifest)
    verified = verify_artifacts(manifest, cache_root, scope="local-bytes", strict=True)
    if len(verified) != EXPECTED_NON_WEIGHT_COUNT:
        raise AuditError(f"verified {len(verified)} non-weight files, expected 26")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audited-at-utc",
        required=True,
        help="Fixed whole-second ISO-8601 UTC timestamp used in deterministic output",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/deepseek-v4-flash-0731.json"),
    )
    parser.add_argument(
        "--digest",
        type=Path,
        default=Path("manifests/deepseek-v4-flash-0731.sha256"),
    )
    parser.add_argument(
        "--non-weight-dir",
        type=Path,
        help="Optional empty directory that retains the 26 locally verified non-weight files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace changed manifest outputs; never overwrites cached source artifacts",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        audited_at = _validate_audit_timestamp(args.audited_at_utc)
        if args.non_weight_dir:
            cache_context = nullcontext(_prepare_cache(args.non_weight_dir))
        else:
            cache_context = tempfile.TemporaryDirectory(prefix="deepseek-v4-flash-0731-")
        with cache_context as cache_value:
            cache_root = Path(cache_value)
            manifest = build_manifest(audited_at, cache_root)

        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        digest = _sha256(manifest_bytes)
        digest_bytes = f"{digest}  {args.manifest.name}\n".encode("ascii")
        _write_output(args.manifest, manifest_bytes, force=args.force)
        _write_output(args.digest, digest_bytes, force=args.force)
    except (AuditError, OSError, ValueError) as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "artifact_count": manifest["totals"]["artifact_count"],
        "documentation_changed_paths": manifest["documentation_snapshot"][
            "changed_paths"
        ],
        "locally_verified_non_weight_count": manifest["totals"][
            "non_weight_artifact_count"
        ],
        "manifest": str(args.manifest),
        "manifest_sha256": digest,
        "weight_shard_bytes": manifest["totals"]["weight_shard_bytes"],
        "weight_shard_count": manifest["totals"]["weight_shard_count"],
        "weight_tensor_bytes": manifest["totals"]["weight_tensor_bytes"],
        "weight_tensor_count": manifest["totals"]["weight_tensor_count"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
