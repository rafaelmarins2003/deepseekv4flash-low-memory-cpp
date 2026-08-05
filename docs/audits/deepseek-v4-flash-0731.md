# DeepSeek-V4-Flash-0731 Provenance Audit

## Status

S1.1 provenance evidence is complete. The immutable model identity and artifact set are frozen; full local verification of the 48 weight-shard bytes is intentionally deferred to RAF-12.

- Audit timestamp: `2026-08-05T01:07:40Z`
- Manifest: `manifests/deepseek-v4-flash-0731.json`
- Manifest SHA-256: `83f634c8dfab073636bc59cd77cca63695ee5894d2a6f4bacf0c5e24fd57f183`
- Schema version: `1.0.0`

## Mental model: identity versus possession

This audit answers two different questions separately:

1. **Identity:** Which immutable upstream revision and exact artifact hashes define the model?
2. **Possession:** Does a particular local directory contain every one of those bytes?

S1.1 proves identity. It also downloads and hashes every non-weight artifact because they are small enough to verify immediately. RAF-12 later proves possession of all weight bytes without mutating the S1.1 manifest or its digest.

## Immutable authority

| Role | Model/revision | Scope |
|---|---|---|
| Normative model | [`deepseek-ai/DeepSeek-V4-Flash-0731@9e165c3`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/commit/9e165c30e2704aec5d9d593cce3eebd58bbef1cb) | Weights, configuration, tokenizer, encoding, inference reference, and artifact membership |
| Informative documentation | [`7872f01`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/commit/7872f01b1d1fe23eabc4c98b48bffcef5a386062) | Later model-card guidance only |
| Preview comparison | [`deepseek-ai/DeepSeek-V4-Flash@60d8d70`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1) | Prior official Preview evidence |
| DSpark comparison | [`deepseek-ai/DeepSeek-V4-Flash-DSpark@62af8ff`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/tree/62af8fffb2f7030cac4de2f0169f5b8d1101b646) | Prior official speculative-decoding evidence |

No mutable `main` reference is used as artifact, test, fixture, download, or semantic authority.

## Release lineage

The Git parent chain and author/committer dates were inspected without checking out files or downloading LFS payloads.

| Role | Revision | Parent | Author and committer date (UTC) |
|---|---|---|---|
| Initial commit | `a5afaad1bafd888c90a8b6f58719204a7753416e` | None | `2026-07-31T07:30:24Z` |
| Weight upload | `7da4bfdd329d1baa9e18691121327b082e809454` | `a5afaad…` | `2026-07-31T09:54:07Z` |
| Normative release | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` | `7da4bfd…` | `2026-07-31T12:02:14Z` |
| Documentation snapshot | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | `9e165c3…` | `2026-08-01T03:07:41Z` |

Verified fact: the Git diff from `9e165c3…` to `7872f01…` contains exactly one change: `M README.md`.

## Normative artifact inventory

The recursive Hub API returns 77 tree entries: 74 files and three directory nodes (`encoding`, `encoding/tests`, and `inference`). Only files are manifest artifacts.

| Classification | Count | Bytes | Verification level |
|---|---:|---:|---|
| Weight shards | 48 | 166,886,535,336 | Upstream LFS SHA-256 and exact size |
| Non-weight artifacts | 26 | 12,124,994 | Downloaded and locally SHA-256 hashed |
| Total | 74 | 166,898,660,330 | Mixed, explicitly labeled per artifact |

The shard names are the complete contiguous sequence `model-00001-of-00048.safetensors` through `model-00048-of-00048.safetensors`.

The locally verified `model.safetensors.index.json` maps 72,317 tensors to exactly those 48 shard names. Its `metadata.total_size` is 166,878,536,440 tensor-payload bytes. The difference from physical shard size is 7,998,896 bytes of Safetensors container overhead; these quantities are recorded separately rather than treated as a mismatch.

Each manifest entry records:

- normalized POSIX path;
- semantic role;
- exact byte size;
- content SHA-256;
- verification level;
- typed upstream identifiers;
- immutable source URL.

Typed identifiers matter. A regular Git blob SHA-1, an LFS content SHA-256, and a Xet hash are different identifiers and are never silently treated as interchangeable.

For every downloaded ordinary Git artifact, the builder recomputes the canonical
`SHA1("blob " + decimal_size + NUL + content)` object ID and compares it with the
pinned tree entry before accepting the local SHA-256. Weight shards are not
downloaded here; their typed LFS content IDs remain upstream metadata for RAF-12.

Artifact source URLs are also bound contextually: every normative artifact must
resolve from the manifest's exact model ID, normative revision, and artifact path.
A different full commit, including the documentation snapshot, is rejected even
though it is independently immutable.

## Documentation snapshot separation

| README | Size | Local SHA-256 | Scope |
|---|---:|---|---|
| Normative `9e165c3…` | 6,494 | `3e66a85d35215e658011f0d8ad4fe7d725c4beacb095892ef5aa96f369f2fa20` | Member of normative manifest |
| Informative `7872f01…` | 7,238 | `252acafdc9204d0dba3fde1b0a93d71cd1664a4ceadfe222b60117ed0ccc56ff` | Separate documentation evidence |

The documentation commit adds SGLang cookbook guidance. Its README hash does not replace the normative README entry.

## Encoding authority

Verified facts:

- the release has no Jinja-format chat template;
- `tokenizer_config.json` contains no `chat_template` field;
- the normative protocol is the complete 11-file `encoding/` directory;
- that directory includes implementation, documentation, four inputs, four expected outputs, and a test driver.

Project decision: downstream tokenizer and parity work must preserve the official encoding implementation and vectors rather than inventing an equivalent-looking Jinja template.

## Prior-release comparison

### Preview

The pinned Preview has 73 artifacts, 46 shards, and 159,617,149,040 shard bytes. The official 0731 release has 48 shards and 7,269,386,296 additional shard bytes.

Verified upstream statement: the 0731 model card identifies this release as superseding Preview.

### DSpark

The pinned DSpark comparison has 74 artifacts, 48 shards, and the same 166,886,535,336 shard-byte total as 0731.

Verified facts:

- `config.json` has the same Git object identity in both snapshots;
- all 48 corresponding shard sizes match;
- zero of the 48 corresponding LFS content SHA-256 values match;
- the changed non-weight paths are `README.md`, `encoding/README.md`, and `encoding/encoding_dsv4.py`.

Inference: 0731 retains the DSpark-capable structural contract but is not a rename or documentation-only republish; its weights and selected encoding behavior have new identities.

## Known upstream representations requiring handoff

| ID | Official evidence | Disposition |
|---|---|---|
| `transformers_version_metadata` | `config.json`: `4.57.1`; `generation_config.json`: `4.46.3`; requirements: `transformers>=5.0.0` | RAF-6/RAF-8 must determine role; do not choose one silently |
| `next_token_prediction_layer_counts` | `num_nextn_predict_layers=1`; `n_mtp_layers=3` | Treat as unresolved semantic mapping until tensor/execution evidence exists |
| `quantization_scope_representation` | Expert dtype `fp4`; general/reference dtype metadata `fp8` | Preserve as distinct scopes rather than a contradiction |

## Manifest digest design

The SHA-256 is stored in `deepseek-v4-flash-0731.sha256`, not inside the manifest. Embedding a file's own digest in that file would create a circular definition. The sidecar hashes the exact checked-in UTF-8 bytes, including formatting and final newline.

Downstream artifacts cite this digest through `derived_from_manifest_sha256`; they do not copy or mutate the source manifest.

## Reproduction on Linux

Versions used for this audit:

- Python `3.12.13` for the reproducibility run and tests;
- Python `3.14.4` for the independent initial generation;
- Git `2.54.0`;
- Hugging Face CLI `1.19.0` for independent dry-run inventory confirmation;
- GNU `sha256sum` `9.7`;
- `uv` `0.11.3` for the locked Python project environment.

Create or synchronize the project environment without changing the lockfile:

```bash
uv sync --locked
```

Generate the exact manifest while retaining the 26 non-weight files in a fresh temporary directory:

```bash
checkpoint_audit_dir="$(mktemp -d /tmp/deepseek-v4-flash-0731.XXXXXX)"
uv run --locked python -B tools/build_model_manifest.py \
  --audited-at-utc 2026-08-05T01:07:40Z \
  --non-weight-dir "$checkpoint_audit_dir"
```

The cache directory must be empty. The builder refuses to overwrite cached source artifacts or changed outputs without an explicit `--force`.

Validate the manifest digest and all retained non-weight bytes:

```bash
uv run --locked python -B tools/verify_model_manifest.py \
  manifests/deepseek-v4-flash-0731.json \
  --digest-file manifests/deepseek-v4-flash-0731.sha256 \
  --artifacts-root "$checkpoint_audit_dir" \
  --scope local-bytes \
  --strict
```

Validate the sidecar independently:

```bash
cd manifests
sha256sum -c deepseek-v4-flash-0731.sha256
```

Run synthetic positive and corruption tests:

```bash
uv run --locked python -B -m unittest discover -s tests -v
```

Inspect Git metadata without downloading LFS weights:

```bash
git -c filter.lfs.smudge= -c filter.lfs.required=false clone \
  --filter=blob:none --no-checkout \
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 \
  /tmp/deepseek-v4-flash-0731-git-metadata

git -C /tmp/deepseek-v4-flash-0731-git-metadata show -s \
  --format='%H%x09%P%x09%aI%x09%cI%x09%s' \
  7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  7da4bfdd329d1baa9e18691121327b082e809454 \
  a5afaad1bafd888c90a8b6f58719204a7753416e

git -C /tmp/deepseek-v4-flash-0731-git-metadata diff --name-status \
  9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  7872f01b1d1fe23eabc4c98b48bffcef5a386062
```

## Verification results

- Manifest structural validation: pass.
- Manifest sidecar check: pass.
- Strict local verification of 26 non-weight artifacts: pass.
- Local-byte comparison with all 26 normative Git blob IDs: pass.
- Byte-identical generation under Python 3.14.4 and Python 3.12.13: pass.
- Clean locked-environment regeneration after provenance hardening: same manifest SHA-256, pass.
- Synthetic test suite: 25 tests, pass.
- Covered failures: missing, renamed, truncated, same-size modified, unexpected regular or special filesystem entries, symlinked, duplicated, unsorted, absolute, escaping, or non-normalized paths, malformed SHA-256, mutable revisions including `main` and `dev`, documentation-revision mixing, source-path mismatch, manifest-digest mismatch, Git blob mismatch, and non-zero CLI manifest/artifact error codes.

## Boundary for RAF-12

This audit does not claim that 166,886,535,336 local weight bytes exist or were hashed. RAF-12 must:

1. materialize the exact normative tree;
2. verify all 74 local files against this immutable manifest;
3. emit a separate dated verification report that cites manifest SHA-256 `83f634c8dfab073636bc59cd77cca63695ee5894d2a6f4bacf0c5e24fd57f183`;
4. leave this manifest unchanged.
