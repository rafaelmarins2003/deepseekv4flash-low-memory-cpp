from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_model_manifest import (  # noqa: E402
    AuditError,
    _verified_content_sha256,
)
from verify_model_manifest import (  # noqa: E402
    ArtifactVerificationError,
    ManifestValidationError,
    load_manifest,
    main as verify_main,
    validate_manifest,
    verify_artifacts,
    verify_digest_file,
)


REVISION = "1" * 40
DOCUMENTATION_REVISION = "2" * 40
PREVIEW_REVISION = "3" * 40
DSPARK_REVISION = "4" * 40
EXPECTED_MANIFEST_DIGEST = (
    "83f634c8dfab073636bc59cd77cca63695ee5894d2a6f4bacf0c5e24fd57f183"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def artifact(path: str, data: bytes) -> dict[str, object]:
    return {
        "path": path,
        "role": "test_data",
        "size_bytes": len(data),
        "sha256": sha256(data),
        "verification_level": "local_bytes",
        "upstream_objects": [
            {
                "kind": "git_blob",
                "algorithm": "sha1",
                "value": "a" * 40,
            }
        ],
        "source_url": f"https://huggingface.co/test/tiny/resolve/{REVISION}/{path}",
    }


def manifest_for(artifacts: list[dict[str, object]]) -> dict[str, object]:
    weight_artifacts = [item for item in artifacts if item["role"] == "weight_shard"]
    return {
        "schema_version": "1.0.0",
        "model": {
            "id": "test/tiny",
            "normative_revision": REVISION,
        },
        "documentation_snapshot": {
            "revision": DOCUMENTATION_REVISION,
            "scope": "informative_only",
            "changed_paths": ["README.md"],
            "readme_sha256": "b" * 64,
            "readme_size_bytes": 1,
            "verification_level": "local_bytes",
        },
        "audited_at_utc": "2026-08-05T00:00:00Z",
        "source_urls": {
            "documentation_commit": (
                f"https://huggingface.co/test/tiny/commit/{DOCUMENTATION_REVISION}"
            ),
            "normative_tree": f"https://huggingface.co/test/tiny/tree/{REVISION}",
            "documentation_tree": (
                f"https://huggingface.co/test/tiny/tree/{DOCUMENTATION_REVISION}"
            ),
            "dspark_tree": (
                f"https://huggingface.co/test/dspark/tree/{DSPARK_REVISION}"
            ),
            "normative_commit": (
                f"https://huggingface.co/test/tiny/commit/{REVISION}"
            ),
            "preview_tree": (
                f"https://huggingface.co/test/preview/tree/{PREVIEW_REVISION}"
            ),
        },
        "release_lineage": [
            {
                "role": "normative_release",
                "revision": REVISION,
                "parent_revision": None,
                "title": "test release",
                "author_date_utc": "2026-08-05T00:00:00Z",
                "committer_date_utc": "2026-08-05T00:00:00Z",
                "source_url": f"https://huggingface.co/test/tiny/commit/{REVISION}",
            }
        ],
        "known_upstream_divergences": [
            {
                "id": "test_divergence",
                "summary": "Synthetic evidence for verifier tests.",
                "status": "resolved",
                "handoff": ["TEST-1"],
                "evidence": [
                    {
                        "path": "config.json",
                        "selector": "/value",
                        "value": 1,
                    }
                ],
            }
        ],
        "prior_release_comparison": {
            "preview": {
                "model_id": "test/preview",
                "revision": PREVIEW_REVISION,
                "artifact_count": 0,
                "weight_shard_count": 0,
                "weight_shard_bytes": 0,
                "summary": "Synthetic preview.",
            },
            "dspark": {
                "model_id": "test/dspark",
                "revision": DSPARK_REVISION,
                "artifact_count": 0,
                "weight_shard_count": 0,
                "weight_shard_bytes": 0,
                "config_matches_normative": False,
                "matching_weight_sha256_count": 0,
                "matching_weight_size_count": 0,
                "non_weight_changed_paths": [],
                "summary": "Synthetic DSpark comparison.",
            },
        },
        "totals": {
            "artifact_count": len(artifacts),
            "weight_shard_count": len(weight_artifacts),
            "non_weight_artifact_count": len(artifacts) - len(weight_artifacts),
            "weight_shard_bytes": sum(
                int(item["size_bytes"]) for item in weight_artifacts
            ),
            "weight_tensor_bytes": 0,
            "weight_tensor_count": 0,
            "weight_container_overhead_bytes": sum(
                int(item["size_bytes"]) for item in weight_artifacts
            ),
            "total_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        },
        "artifacts": artifacts,
    }


def write_artifact(root: Path, relative_path: str, data: bytes) -> None:
    target = root.joinpath(*relative_path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


class BuilderVerificationTests(unittest.TestCase):
    def test_downloaded_git_blob_is_verified(self) -> None:
        data = b"official artifact bytes"
        entry = {
            "oid": git_blob_sha1(data),
            "path": "artifact.bin",
            "size": len(data),
        }
        self.assertEqual(
            _verified_content_sha256(entry, data, "artifact.bin"),
            sha256(data),
        )

    def test_downloaded_git_blob_mismatch_is_rejected(self) -> None:
        data = b"modified artifact bytes"
        entry = {
            "oid": "a" * 40,
            "path": "artifact.bin",
            "size": len(data),
        }
        with self.assertRaisesRegex(AuditError, "Git blob mismatch"):
            _verified_content_sha256(entry, data, "artifact.bin")


class ManifestStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {"a.bin": b"alpha", "nested/b.bin": b"bravo"}
        self.manifest = manifest_for(
            [artifact(path, data) for path, data in sorted(self.data.items())]
        )

    def test_valid_manifest_and_artifacts(self) -> None:
        validate_manifest(self.manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path, data in self.data.items():
                write_artifact(root, path, data)
            verified = verify_artifacts(self.manifest, root, scope="all", strict=True)
        self.assertEqual(verified, sorted(self.data))

    def test_missing_file_hard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, "a.bin", self.data["a.bin"])
            with self.assertRaisesRegex(ArtifactVerificationError, "missing artifact"):
                verify_artifacts(self.manifest, root, scope="all")

    def test_renamed_file_hard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, "renamed.bin", self.data["a.bin"])
            write_artifact(root, "nested/b.bin", self.data["nested/b.bin"])
            with self.assertRaisesRegex(ArtifactVerificationError, "missing artifact"):
                verify_artifacts(self.manifest, root, scope="all", strict=True)

    def test_truncated_file_hard_fails_on_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, "a.bin", b"a")
            write_artifact(root, "nested/b.bin", self.data["nested/b.bin"])
            with self.assertRaisesRegex(ArtifactVerificationError, "size mismatch"):
                verify_artifacts(self.manifest, root, scope="all")

    def test_same_size_modification_hard_fails_on_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, "a.bin", b"ALPHA")
            write_artifact(root, "nested/b.bin", self.data["nested/b.bin"])
            with self.assertRaisesRegex(ArtifactVerificationError, "SHA-256 mismatch"):
                verify_artifacts(self.manifest, root, scope="all")

    def test_strict_mode_rejects_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path, data in self.data.items():
                write_artifact(root, path, data)
            write_artifact(root, "extra.bin", b"unexpected")
            with self.assertRaisesRegex(ArtifactVerificationError, "unexpected artifacts"):
                verify_artifacts(self.manifest, root, scope="all", strict=True)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO test requires POSIX")
    def test_strict_mode_rejects_special_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path, data in self.data.items():
                write_artifact(root, path, data)
            os.mkfifo(root / "unexpected.fifo")
            with self.assertRaisesRegex(
                ArtifactVerificationError, "unsupported filesystem entry"
            ):
                verify_artifacts(self.manifest, root, scope="all", strict=True)

    def test_symlink_hard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, "target.bin", self.data["a.bin"])
            (root / "a.bin").symlink_to(root / "target.bin")
            write_artifact(root, "nested/b.bin", self.data["nested/b.bin"])
            with self.assertRaisesRegex(ArtifactVerificationError, "symlink"):
                verify_artifacts(self.manifest, root, scope="all")

    def test_duplicate_path_is_invalid(self) -> None:
        duplicate = copy.deepcopy(self.manifest)
        duplicate["artifacts"].append(copy.deepcopy(duplicate["artifacts"][0]))
        duplicate["totals"]["artifact_count"] += 1
        duplicate["totals"]["non_weight_artifact_count"] += 1
        duplicate["totals"]["total_bytes"] += duplicate["artifacts"][0]["size_bytes"]
        with self.assertRaisesRegex(ManifestValidationError, "unique"):
            validate_manifest(duplicate)

    def test_unsorted_paths_are_invalid(self) -> None:
        unsorted = copy.deepcopy(self.manifest)
        unsorted["artifacts"].reverse()
        with self.assertRaisesRegex(ManifestValidationError, "POSIX-sorted"):
            validate_manifest(unsorted)

    def test_path_traversal_is_invalid(self) -> None:
        unsafe = copy.deepcopy(self.manifest)
        unsafe["artifacts"][0]["path"] = "../escape.bin"
        with self.assertRaisesRegex(ManifestValidationError, "unsafe"):
            validate_manifest(unsafe)

    def test_absolute_path_is_invalid(self) -> None:
        unsafe = copy.deepcopy(self.manifest)
        unsafe["artifacts"][0]["path"] = "/absolute.bin"
        with self.assertRaisesRegex(ManifestValidationError, "unsafe"):
            validate_manifest(unsafe)

    def test_non_normalized_paths_are_invalid(self) -> None:
        for path in ("a//b.bin", "a/./b.bin", "a\\b.bin"):
            with self.subTest(path=path):
                unsafe = copy.deepcopy(self.manifest)
                unsafe["artifacts"][0]["path"] = path
                with self.assertRaises(ManifestValidationError):
                    validate_manifest(unsafe)

    def test_invalid_sha256_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.manifest)
        invalid["artifacts"][0]["sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ManifestValidationError, "lowercase sha256"):
            validate_manifest(invalid)

    def test_mutable_main_url_is_rejected(self) -> None:
        mutable = copy.deepcopy(self.manifest)
        mutable["source_urls"]["normative_tree"] = (
            "https://huggingface.co/test/tiny/tree/main"
        )
        with self.assertRaisesRegex(ManifestValidationError, "40-character revision"):
            validate_manifest(mutable)

    def test_mutable_non_main_revision_url_is_rejected(self) -> None:
        mutable = copy.deepcopy(self.manifest)
        mutable["source_urls"]["normative_tree"] = (
            "https://huggingface.co/test/tiny/tree/dev"
        )
        with self.assertRaisesRegex(ManifestValidationError, "40-character revision"):
            validate_manifest(mutable)

    def test_documentation_revision_artifact_url_is_rejected(self) -> None:
        mixed = copy.deepcopy(self.manifest)
        mixed["artifacts"][0]["source_url"] = (
            "https://huggingface.co/test/tiny/resolve/"
            f"{DOCUMENTATION_REVISION}/a.bin"
        )
        with self.assertRaisesRegex(ManifestValidationError, "pinned authority"):
            validate_manifest(mixed)

    def test_artifact_source_path_must_match_manifest_path(self) -> None:
        mismatched = copy.deepcopy(self.manifest)
        mismatched["artifacts"][0]["source_url"] = (
            f"https://huggingface.co/test/tiny/resolve/{REVISION}/other.bin"
        )
        with self.assertRaisesRegex(ManifestValidationError, "pinned authority"):
            validate_manifest(mismatched)

    def test_manifest_digest_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_bytes = (
                json.dumps(self.manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            manifest_path.write_bytes(manifest_bytes)
            digest = sha256(manifest_bytes)
            digest_path = root / "manifest.sha256"
            digest_path.write_text(
                f"{digest}  {manifest_path.name}\n", encoding="ascii"
            )
            self.assertEqual(
                verify_digest_file(manifest_path, digest_path),
                digest,
            )
            manifest_path.write_bytes(manifest_bytes + b"\n")
            with self.assertRaisesRegex(ManifestValidationError, "digest mismatch"):
                verify_digest_file(manifest_path, digest_path)

    def test_cli_returns_manifest_error_code(self) -> None:
        invalid = copy.deepcopy(self.manifest)
        invalid["source_urls"]["normative_tree"] = (
            "https://huggingface.co/test/tiny/tree/dev"
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = verify_main([str(manifest_path)])
        self.assertEqual(result, 2)

    def test_cli_returns_artifact_error_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            artifact_root = root / "artifacts"
            artifact_root.mkdir()
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = verify_main(
                    [
                        str(manifest_path),
                        "--artifacts-root",
                        str(artifact_root),
                        "--scope",
                        "all",
                    ]
                )
        self.assertEqual(result, 3)


class CheckedInManifestTests(unittest.TestCase):
    def test_schema_is_valid_json(self) -> None:
        schema_path = ROOT / "manifests" / "model-manifest.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0.0")

    def test_pinned_manifest_contract(self) -> None:
        manifest_path = ROOT / "manifests" / "deepseek-v4-flash-0731.json"
        digest_path = ROOT / "manifests" / "deepseek-v4-flash-0731.sha256"
        manifest = load_manifest(manifest_path)
        digest = verify_digest_file(manifest_path, digest_path)

        self.assertEqual(
            manifest["model"],
            {
                "id": "deepseek-ai/DeepSeek-V4-Flash-0731",
                "normative_revision": "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
            },
        )
        self.assertEqual(
            manifest["documentation_snapshot"]["revision"],
            "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        )
        self.assertEqual(manifest["documentation_snapshot"]["changed_paths"], ["README.md"])
        self.assertEqual(manifest["totals"]["artifact_count"], 74)
        self.assertEqual(manifest["totals"]["weight_shard_count"], 48)
        self.assertEqual(manifest["totals"]["non_weight_artifact_count"], 26)
        self.assertEqual(manifest["totals"]["weight_shard_bytes"], 166_886_535_336)
        self.assertEqual(manifest["totals"]["weight_tensor_bytes"], 166_878_536_440)
        self.assertEqual(manifest["totals"]["weight_tensor_count"], 72_317)
        self.assertEqual(manifest["totals"]["weight_container_overhead_bytes"], 7_998_896)
        self.assertEqual(digest, EXPECTED_MANIFEST_DIGEST)

        weights = [
            item for item in manifest["artifacts"] if item["role"] == "weight_shard"
        ]
        local = [
            item
            for item in manifest["artifacts"]
            if item["verification_level"] == "local_bytes"
        ]
        self.assertEqual(len(weights), 48)
        self.assertEqual(len(local), 26)
        self.assertEqual(
            [item["path"] for item in weights],
            [
                f"model-{index:05d}-of-00048.safetensors"
                for index in range(1, 49)
            ],
        )


if __name__ == "__main__":
    unittest.main()
