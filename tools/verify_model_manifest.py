#!/usr/bin/env python3
"""Validate a model manifest and, optionally, the artifact bytes it names."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote


SCHEMA_VERSION = "1.0.0"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_ID_RE = re.compile(r"^[^/]+/[^/]+$")
IMMUTABLE_HUGGING_FACE_URL_RE = re.compile(
    r"^https://huggingface[.]co/[^/?#]+/[^/?#]+/"
    r"(?:commit|resolve|tree)/[0-9a-f]{40}(?:/[^?#]+)?$"
)
CHUNK_SIZE = 8 * 1024 * 1024

TOP_LEVEL_KEYS = {
    "artifacts",
    "audited_at_utc",
    "documentation_snapshot",
    "known_upstream_divergences",
    "model",
    "prior_release_comparison",
    "release_lineage",
    "schema_version",
    "source_urls",
    "totals",
}
MODEL_KEYS = {"id", "normative_revision"}
DOCUMENTATION_KEYS = {
    "changed_paths",
    "readme_sha256",
    "readme_size_bytes",
    "revision",
    "scope",
    "verification_level",
}
TOTAL_KEYS = {
    "artifact_count",
    "non_weight_artifact_count",
    "total_bytes",
    "weight_container_overhead_bytes",
    "weight_shard_bytes",
    "weight_shard_count",
    "weight_tensor_bytes",
    "weight_tensor_count",
}
ARTIFACT_KEYS = {
    "path",
    "role",
    "sha256",
    "size_bytes",
    "source_url",
    "upstream_objects",
    "verification_level",
}
UPSTREAM_OBJECT_KEYS = {"algorithm", "kind", "value"}
RELEASE_KEYS = {
    "author_date_utc",
    "committer_date_utc",
    "parent_revision",
    "revision",
    "role",
    "source_url",
    "title",
}
DIVERGENCE_KEYS = {"evidence", "handoff", "id", "status", "summary"}
EVIDENCE_KEYS = {"path", "selector", "value"}
SOURCE_URL_KEYS = {
    "documentation_commit",
    "documentation_tree",
    "dspark_tree",
    "normative_commit",
    "normative_tree",
    "preview_tree",
}


class ManifestValidationError(ValueError):
    """The manifest does not satisfy its structural contract."""


class ArtifactVerificationError(ValueError):
    """Local artifact bytes do not satisfy the manifest."""


def _require_type(value: Any, expected: type, context: str) -> None:
    if not isinstance(value, expected):
        raise ManifestValidationError(
            f"{context} must be {expected.__name__}, got {type(value).__name__}"
        )


def _require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ManifestValidationError(f"{context} has invalid keys: {', '.join(details)}")


def _require_sha(value: Any, algorithm: str, context: str) -> None:
    _require_type(value, str, context)
    pattern = SHA1_RE if algorithm == "sha1" else SHA256_RE
    if not pattern.fullmatch(value):
        raise ManifestValidationError(f"{context} is not a lowercase {algorithm}")


def _require_immutable_url(value: Any, context: str) -> None:
    _require_type(value, str, context)
    if not IMMUTABLE_HUGGING_FACE_URL_RE.fullmatch(value):
        raise ManifestValidationError(
            f"{context} must pin a full 40-character revision in an HTTPS "
            "Hugging Face commit, resolve, or tree URL"
        )


def _hugging_face_url(
    model_id: str, action: str, revision: str, path: str | None = None
) -> str:
    url = f"https://huggingface.co/{quote(model_id, safe='/')}/{action}/{revision}"
    if path is not None:
        url += f"/{quote(path, safe='/')}"
    return url


def _require_exact_url(value: Any, expected: str, context: str) -> None:
    _require_immutable_url(value, context)
    if value != expected:
        raise ManifestValidationError(
            f"{context} does not match its pinned authority: expected {expected}, got {value}"
        )


def _require_utc_timestamp(value: Any, context: str) -> None:
    _require_type(value, str, context)
    if not value.endswith("Z"):
        raise ManifestValidationError(f"{context} must use a UTC Z suffix")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ManifestValidationError(f"{context} is not ISO-8601: {value}") from exc


def _validate_relative_path(value: Any, context: str) -> str:
    _require_type(value, str, context)
    if not value or "\x00" in value or "\\" in value:
        raise ManifestValidationError(f"{context} is not a normalized POSIX path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestValidationError(f"{context} is unsafe: {value!r}")
    if path.as_posix() != value:
        raise ManifestValidationError(f"{context} is not normalized: {value!r}")
    return value


def _validate_documentation_snapshot(value: Any) -> None:
    _require_type(value, dict, "documentation_snapshot")
    _require_exact_keys(value, DOCUMENTATION_KEYS, "documentation_snapshot")
    _require_sha(value["revision"], "sha1", "documentation_snapshot.revision")
    if value["scope"] != "informative_only":
        raise ManifestValidationError("documentation_snapshot.scope must be informative_only")
    if value["verification_level"] != "local_bytes":
        raise ManifestValidationError(
            "documentation_snapshot.verification_level must be local_bytes"
        )
    _require_sha(value["readme_sha256"], "sha256", "documentation_snapshot.readme_sha256")
    if not isinstance(value["readme_size_bytes"], int) or value["readme_size_bytes"] < 0:
        raise ManifestValidationError(
            "documentation_snapshot.readme_size_bytes must be a non-negative integer"
        )
    _require_type(value["changed_paths"], list, "documentation_snapshot.changed_paths")
    changed_paths = [
        _validate_relative_path(path, "documentation_snapshot.changed_paths[]")
        for path in value["changed_paths"]
    ]
    if changed_paths != sorted(set(changed_paths)):
        raise ManifestValidationError(
            "documentation_snapshot.changed_paths must be sorted and unique"
        )


def _validate_release_lineage(value: Any, model_id: str) -> None:
    _require_type(value, list, "release_lineage")
    if not value:
        raise ManifestValidationError("release_lineage must not be empty")
    revisions: set[str] = set()
    for index, entry in enumerate(value):
        context = f"release_lineage[{index}]"
        _require_type(entry, dict, context)
        _require_exact_keys(entry, RELEASE_KEYS, context)
        _require_sha(entry["revision"], "sha1", f"{context}.revision")
        if entry["revision"] in revisions:
            raise ManifestValidationError(f"duplicate release revision: {entry['revision']}")
        revisions.add(entry["revision"])
        parent = entry["parent_revision"]
        if parent is not None:
            _require_sha(parent, "sha1", f"{context}.parent_revision")
        for key in ("author_date_utc", "committer_date_utc"):
            _require_utc_timestamp(entry[key], f"{context}.{key}")
        for key in ("role", "title"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise ManifestValidationError(f"{context}.{key} must be a non-empty string")
        _require_exact_url(
            entry["source_url"],
            _hugging_face_url(model_id, "commit", entry["revision"]),
            f"{context}.source_url",
        )


def _validate_divergences(value: Any) -> None:
    _require_type(value, list, "known_upstream_divergences")
    identifiers: set[str] = set()
    for index, entry in enumerate(value):
        context = f"known_upstream_divergences[{index}]"
        _require_type(entry, dict, context)
        _require_exact_keys(entry, DIVERGENCE_KEYS, context)
        identifier = entry["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ManifestValidationError(f"{context}.id must be a non-empty string")
        if identifier in identifiers:
            raise ManifestValidationError(f"duplicate divergence id: {identifier}")
        identifiers.add(identifier)
        for key in ("status", "summary"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise ManifestValidationError(f"{context}.{key} must be a non-empty string")
        _require_type(entry["handoff"], list, f"{context}.handoff")
        if entry["handoff"] != sorted(set(entry["handoff"])):
            raise ManifestValidationError(f"{context}.handoff must be sorted and unique")
        _require_type(entry["evidence"], list, f"{context}.evidence")
        if not entry["evidence"]:
            raise ManifestValidationError(f"{context}.evidence must not be empty")
        for evidence_index, evidence in enumerate(entry["evidence"]):
            evidence_context = f"{context}.evidence[{evidence_index}]"
            _require_type(evidence, dict, evidence_context)
            _require_exact_keys(evidence, EVIDENCE_KEYS, evidence_context)
            _validate_relative_path(evidence["path"], f"{evidence_context}.path")
            if not isinstance(evidence["selector"], str) or not evidence["selector"]:
                raise ManifestValidationError(
                    f"{evidence_context}.selector must be a non-empty string"
                )


def _validate_prior_release_comparison(value: Any) -> None:
    _require_type(value, dict, "prior_release_comparison")
    if set(value) != {"dspark", "preview"}:
        raise ManifestValidationError(
            "prior_release_comparison must contain exactly preview and dspark"
        )
    for name, comparison in value.items():
        context = f"prior_release_comparison.{name}"
        _require_type(comparison, dict, context)
        required = {
            "artifact_count",
            "model_id",
            "revision",
            "summary",
            "weight_shard_bytes",
            "weight_shard_count",
        }
        if name == "dspark":
            required |= {
                "config_matches_normative",
                "matching_weight_sha256_count",
                "matching_weight_size_count",
                "non_weight_changed_paths",
            }
        _require_exact_keys(comparison, required, context)
        _require_sha(comparison["revision"], "sha1", f"{context}.revision")
        for key in ("artifact_count", "weight_shard_bytes", "weight_shard_count"):
            if not isinstance(comparison[key], int) or comparison[key] < 0:
                raise ManifestValidationError(f"{context}.{key} must be non-negative integer")
        if not isinstance(comparison["model_id"], str) or not comparison["model_id"]:
            raise ManifestValidationError(f"{context}.model_id must be non-empty string")
        if not isinstance(comparison["summary"], str) or not comparison["summary"]:
            raise ManifestValidationError(f"{context}.summary must be non-empty string")
        if name == "dspark":
            if not isinstance(comparison["config_matches_normative"], bool):
                raise ManifestValidationError(
                    f"{context}.config_matches_normative must be boolean"
                )
            for key in ("matching_weight_sha256_count", "matching_weight_size_count"):
                if not isinstance(comparison[key], int) or comparison[key] < 0:
                    raise ManifestValidationError(f"{context}.{key} must be non-negative integer")
            paths = comparison["non_weight_changed_paths"]
            _require_type(paths, list, f"{context}.non_weight_changed_paths")
            normalized = [
                _validate_relative_path(path, f"{context}.non_weight_changed_paths[]")
                for path in paths
            ]
            if normalized != sorted(set(normalized)):
                raise ManifestValidationError(
                    f"{context}.non_weight_changed_paths must be sorted and unique"
                )


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Validate manifest structure and cross-field totals."""

    _require_type(manifest, dict, "manifest")
    _require_exact_keys(manifest, TOP_LEVEL_KEYS, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestValidationError(
            f"unsupported schema_version: {manifest['schema_version']!r}"
        )
    _require_utc_timestamp(manifest["audited_at_utc"], "audited_at_utc")

    model = manifest["model"]
    _require_type(model, dict, "model")
    _require_exact_keys(model, MODEL_KEYS, "model")
    if not isinstance(model["id"], str) or not MODEL_ID_RE.fullmatch(model["id"]):
        raise ManifestValidationError("model.id must be a non-empty owner/name identifier")
    _require_sha(model["normative_revision"], "sha1", "model.normative_revision")

    documentation = manifest["documentation_snapshot"]
    _validate_documentation_snapshot(documentation)
    if documentation["revision"] == model["normative_revision"]:
        raise ManifestValidationError(
            "documentation snapshot must be separate from normative revision"
        )

    _validate_release_lineage(manifest["release_lineage"], model["id"])
    _validate_divergences(manifest["known_upstream_divergences"])
    comparisons = manifest["prior_release_comparison"]
    _validate_prior_release_comparison(comparisons)

    source_urls = manifest["source_urls"]
    _require_type(source_urls, dict, "source_urls")
    _require_exact_keys(source_urls, SOURCE_URL_KEYS, "source_urls")
    expected_source_urls = {
        "documentation_commit": _hugging_face_url(
            model["id"], "commit", documentation["revision"]
        ),
        "documentation_tree": _hugging_face_url(
            model["id"], "tree", documentation["revision"]
        ),
        "dspark_tree": _hugging_face_url(
            comparisons["dspark"]["model_id"],
            "tree",
            comparisons["dspark"]["revision"],
        ),
        "normative_commit": _hugging_face_url(
            model["id"], "commit", model["normative_revision"]
        ),
        "normative_tree": _hugging_face_url(
            model["id"], "tree", model["normative_revision"]
        ),
        "preview_tree": _hugging_face_url(
            comparisons["preview"]["model_id"],
            "tree",
            comparisons["preview"]["revision"],
        ),
    }
    for key, expected in expected_source_urls.items():
        _require_exact_url(source_urls[key], expected, f"source_urls.{key}")

    artifacts = manifest["artifacts"]
    _require_type(artifacts, list, "artifacts")
    if not artifacts:
        raise ManifestValidationError("artifacts must not be empty")

    paths: list[str] = []
    total_bytes = 0
    weight_shard_bytes = 0
    weight_shard_count = 0
    for index, artifact in enumerate(artifacts):
        context = f"artifacts[{index}]"
        _require_type(artifact, dict, context)
        _require_exact_keys(artifact, ARTIFACT_KEYS, context)
        path = _validate_relative_path(artifact["path"], f"{context}.path")
        paths.append(path)
        if not isinstance(artifact["role"], str) or not artifact["role"]:
            raise ManifestValidationError(f"{context}.role must be a non-empty string")
        size = artifact["size_bytes"]
        if not isinstance(size, int) or size < 0:
            raise ManifestValidationError(f"{context}.size_bytes must be non-negative integer")
        total_bytes += size
        _require_sha(artifact["sha256"], "sha256", f"{context}.sha256")
        if artifact["verification_level"] not in {"local_bytes", "upstream_metadata"}:
            raise ManifestValidationError(
                f"{context}.verification_level must be local_bytes or upstream_metadata"
            )
        _require_exact_url(
            artifact["source_url"],
            _hugging_face_url(
                model["id"], "resolve", model["normative_revision"], path
            ),
            f"{context}.source_url",
        )

        upstream_objects = artifact["upstream_objects"]
        _require_type(upstream_objects, list, f"{context}.upstream_objects")
        if not upstream_objects:
            raise ManifestValidationError(f"{context}.upstream_objects must not be empty")
        seen_objects: set[tuple[str, str, str]] = set()
        for object_index, upstream_object in enumerate(upstream_objects):
            object_context = f"{context}.upstream_objects[{object_index}]"
            _require_type(upstream_object, dict, object_context)
            _require_exact_keys(upstream_object, UPSTREAM_OBJECT_KEYS, object_context)
            algorithm = upstream_object["algorithm"]
            if algorithm not in {"sha1", "sha256"}:
                raise ManifestValidationError(f"{object_context}.algorithm is unsupported")
            if not isinstance(upstream_object["kind"], str) or not upstream_object["kind"]:
                raise ManifestValidationError(f"{object_context}.kind must be non-empty string")
            _require_sha(upstream_object["value"], algorithm, f"{object_context}.value")
            identity = (upstream_object["kind"], algorithm, upstream_object["value"])
            if identity in seen_objects:
                raise ManifestValidationError(f"duplicate upstream object in {context}: {identity}")
            seen_objects.add(identity)

        if artifact["role"] == "weight_shard":
            weight_shard_count += 1
            weight_shard_bytes += size
            lfs_hashes = {
                item["value"]
                for item in upstream_objects
                if item["kind"] == "lfs" and item["algorithm"] == "sha256"
            }
            if artifact["sha256"] not in lfs_hashes:
                raise ManifestValidationError(
                    f"{context}.sha256 must match its typed LFS SHA-256"
                )
            if artifact["verification_level"] != "upstream_metadata":
                raise ManifestValidationError(
                    f"{context} weight shard must remain upstream_metadata in the immutable manifest"
                )

    if len(paths) != len(set(paths)):
        raise ManifestValidationError("artifact paths must be unique")
    if paths != sorted(paths):
        raise ManifestValidationError("artifact paths must be POSIX-sorted")

    totals = manifest["totals"]
    _require_type(totals, dict, "totals")
    _require_exact_keys(totals, TOTAL_KEYS, "totals")
    calculated_totals = {
        "artifact_count": len(artifacts),
        "non_weight_artifact_count": len(artifacts) - weight_shard_count,
        "total_bytes": total_bytes,
        "weight_shard_bytes": weight_shard_bytes,
        "weight_shard_count": weight_shard_count,
    }
    mismatches = {
        key: {"declared": totals[key], "calculated": value}
        for key, value in calculated_totals.items()
        if totals[key] != value
    }
    if mismatches:
        raise ManifestValidationError(f"totals mismatch: {mismatches}")
    for key in (
        "weight_container_overhead_bytes",
        "weight_tensor_bytes",
        "weight_tensor_count",
    ):
        if not isinstance(totals[key], int) or totals[key] < 0:
            raise ManifestValidationError(f"totals.{key} must be a non-negative integer")
    if totals["weight_tensor_bytes"] + totals["weight_container_overhead_bytes"] != totals[
        "weight_shard_bytes"
    ]:
        raise ManifestValidationError(
            "weight tensor bytes plus container overhead must equal physical shard bytes"
        )
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot read manifest {path}: {exc}") from exc
    return validate_manifest(manifest)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(path: Path) -> str:
    return sha256_file(path)


def verify_digest_file(manifest_path: Path, digest_path: Path) -> str:
    try:
        fields = digest_path.read_text(encoding="ascii").strip().split()
    except OSError as exc:
        raise ManifestValidationError(f"cannot read digest file {digest_path}: {exc}") from exc
    if len(fields) != 2 or not SHA256_RE.fullmatch(fields[0]):
        raise ManifestValidationError(
            f"digest file must contain '<sha256>  <filename>': {digest_path}"
        )
    if fields[1] != manifest_path.name:
        raise ManifestValidationError(
            f"digest filename mismatch: expected {manifest_path.name}, got {fields[1]}"
        )
    actual = manifest_digest(manifest_path)
    if fields[0] != actual:
        raise ManifestValidationError(
            f"manifest digest mismatch: expected {fields[0]}, got {actual}"
        )
    return actual


def _selected_artifacts(
    manifest: dict[str, Any], scope: str
) -> list[dict[str, Any]]:
    if scope == "all":
        return list(manifest["artifacts"])
    if scope == "local-bytes":
        return [
            artifact
            for artifact in manifest["artifacts"]
            if artifact["verification_level"] == "local_bytes"
        ]
    raise ValueError(f"unsupported verification scope: {scope}")


def _ensure_no_symlink(root: Path, relative_path: PurePosixPath) -> Path:
    candidate = root
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ArtifactVerificationError(
                f"symlink is not allowed in pristine artifact tree: {relative_path}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactVerificationError(f"missing artifact: {relative_path}") from exc
    if root not in resolved.parents:
        raise ArtifactVerificationError(f"artifact escapes root: {relative_path}")
    return candidate


def verify_artifacts(
    manifest: dict[str, Any], root: Path, *, scope: str = "all", strict: bool = False
) -> list[str]:
    """Verify selected artifact bytes and return their relative paths."""

    try:
        root = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactVerificationError(f"artifact root does not exist: {root}") from exc
    if not root.is_dir():
        raise ArtifactVerificationError(f"artifact root is not a directory: {root}")

    selected = _selected_artifacts(manifest, scope)
    expected_paths = {artifact["path"] for artifact in selected}
    verified: list[str] = []
    for artifact in selected:
        relative = PurePosixPath(artifact["path"])
        candidate = _ensure_no_symlink(root, relative)
        if not candidate.is_file():
            raise ArtifactVerificationError(f"artifact is not a regular file: {relative}")
        actual_size = candidate.stat().st_size
        if actual_size != artifact["size_bytes"]:
            raise ArtifactVerificationError(
                f"size mismatch for {relative}: expected {artifact['size_bytes']}, got {actual_size}"
            )
        actual_sha256 = sha256_file(candidate)
        if actual_sha256 != artifact["sha256"]:
            raise ArtifactVerificationError(
                f"SHA-256 mismatch for {relative}: expected {artifact['sha256']}, got {actual_sha256}"
            )
        verified.append(artifact["path"])

    if strict:
        actual_paths: set[str] = set()
        for candidate in root.rglob("*"):
            relative = candidate.relative_to(root)
            if candidate.is_symlink():
                raise ArtifactVerificationError(
                    f"symlink is not allowed in strict artifact tree: {relative.as_posix()}"
                )
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ArtifactVerificationError(
                    "unsupported filesystem entry in strict artifact tree: "
                    f"{relative.as_posix()}"
                )
            actual_paths.add(relative.as_posix())
        unexpected = sorted(actual_paths - expected_paths)
        if unexpected:
            raise ArtifactVerificationError(f"unexpected artifacts: {unexpected}")
    return verified


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to model-manifest JSON")
    parser.add_argument(
        "--digest-file", type=Path, help="Optional sha256sum-style manifest digest file"
    )
    parser.add_argument(
        "--artifacts-root", type=Path, help="Root directory containing artifact bytes"
    )
    parser.add_argument(
        "--scope",
        choices=("manifest", "local-bytes", "all"),
        default="manifest",
        help="Validation scope; artifact scopes require --artifacts-root",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Reject unexpected files in artifacts root"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.scope == "manifest" and args.artifacts_root is not None:
        print("ERROR: --artifacts-root requires --scope local-bytes or all", file=sys.stderr)
        return 2
    if args.scope != "manifest" and args.artifacts_root is None:
        print("ERROR: artifact verification scope requires --artifacts-root", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.manifest)
        digest = None
        if args.digest_file:
            digest = verify_digest_file(args.manifest, args.digest_file)
        verified: list[str] = []
        if args.scope != "manifest":
            verified = verify_artifacts(
                manifest, args.artifacts_root, scope=args.scope, strict=args.strict
            )
    except ManifestValidationError as exc:
        print(f"MANIFEST ERROR: {exc}", file=sys.stderr)
        return 2
    except ArtifactVerificationError as exc:
        print(f"ARTIFACT ERROR: {exc}", file=sys.stderr)
        return 3

    message = f"manifest valid: {len(manifest['artifacts'])} artifacts"
    if digest:
        message += f", digest={digest}"
    if args.scope != "manifest":
        message += f", verified_local_files={len(verified)}, scope={args.scope}"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
