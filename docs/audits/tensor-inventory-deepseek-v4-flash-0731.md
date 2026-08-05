# DeepSeek-V4-Flash-0731 Tensor Inventory and Placement Budget

## Status

Derived evidence for RAF-6. The complete tensor anatomy of the pinned checkpoint
is frozen and cross-checked against the immutable S1.1 manifest. No weight
payload was materialized; RAF-12 remains the possession gate.

- Audit timestamp: `2026-08-05T18:00:00Z`
- Derived artifact: `manifests/derived/tensor-inventory-0731.json`
- Derived artifact SHA-256: `9198ac018b24faee9408164c1bd12d57c90be86c47b25d9c434bbf298802101e`
- `derived_from_manifest_sha256`: `83f634c8dfab073636bc59cd77cca63695ee5894d2a6f4bacf0c5e24fd57f183`
- Normative model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`

## Mental model: identity, anatomy, possession

The provenance audit answers *which bytes define the model*. This audit answers
*what those bytes are made of*: how many tensors exist, what each one is for,
and how they divide into the budgets an engine must place across NVMe, RAM, and
VRAM. It still does not answer *do we hold them* — that remains RAF-12.

The anatomy is obtained without downloading weights. A safetensors file begins
with an 8-byte little-endian header length followed by a JSON header describing
every tensor's dtype, shape, and payload span. Two range requests per shard
therefore expose the entire inventory at a cost of 8 MB instead of 167 GB.

## Verification performed

| Check | Result |
|---|---|
| Shard headers read | 48 of 48, range requests only |
| Tensor count vs immutable manifest | 72,317 = 72,317, pass |
| Tensor payload bytes vs immutable manifest | 166,878,536,440 = 166,878,536,440, pass |
| Container overhead vs immutable manifest | 7,998,896 = 7,998,896, pass |
| Every shard fully described by its header | pass, no gaps or overlaps |
| Declared payload vs shape × dtype, per tensor | pass for all 72,317 |
| `model.safetensors.index.json` name→shard map vs containers | exact match, pass |
| `config.json` bytes vs manifest SHA-256 | pass |
| Tensor names covered by the explicit role grammar | 72,317 of 72,317, no fallback |

Verified fact: the S1.1 totals, which were read from `model.safetensors.index.json`,
are now independently confirmed by the shard containers themselves. The
7,998,896-byte difference between physical and payload bytes is exactly the sum
of the 48 header regions.

## Inventory by component

| Scope | Role | Tensors | Bytes | GB |
|---|---|---:|---:|---:|
| layer | routed_expert | 66,048 | 147,169,738,752 | 147.17 |
| layer | attention_core | 559 | 4,599,478,144 | 4.60 |
| layer | shared_expert | 258 | 1,082,196,480 | 1.08 |
| layer | attention_compressor | 164 | 525,722,624 | 0.53 |
| layer | attention_indexer | 147 | 275,353,344 | 0.28 |
| layer | hyper_connection | 258 | 135,275,592 | 0.14 |
| layer | router | 86 | 108,834,816 | 0.11 |
| layer | norm | 86 | 704,512 | 0.00 |
| global | embedding | 1 | 1,059,061,760 | 1.06 |
| global | output_head | 1 | 1,059,061,760 | 1.06 |
| global | hyper_connection + norm | 4 | 270,356 | 0.00 |
| mtp | all roles | 4,705 | 10,862,838,300 | 10.86 |

Storage formats: `I8` 148.18 GB (packed FP4 expert payload), `F8_E8M0` 9.26 GB
(microscaling exponents), `F8_E4M3` 6.30 GB (dense projections), `BF16` 2.97 GB,
`F32` 0.15 GB, `I64` 0.02 GB.

## Expert geometry

Verified fact: every routed expert in the checkpoint is byte-identical in size.

| Tensor | dtype | Shape | Bytes |
|---|---|---|---:|
| `w1.weight` | `I8` | 2048 × 2048 | 4,194,304 |
| `w1.scale` | `F8_E8M0` | 2048 × 128 | 262,144 |
| `w2.weight` | `I8` | 4096 × 1024 | 4,194,304 |
| `w2.scale` | `F8_E8M0` | 4096 × 64 | 262,144 |
| `w3.weight` | `I8` | 2048 × 2048 | 4,194,304 |
| `w3.scale` | `F8_E8M0` | 2048 × 128 | 262,144 |
| **One expert** | | | **13,369,344 (12.75 MiB)** |

Inference: the logical FP4 matrices are 2048 × 4096 (`w1`, `w3`) and 4096 × 2048
(`w2`), stored two 4-bit values per `I8` byte along the input dimension, with one
`E8M0` exponent per 32 logical elements (4096 / 128 = 32, 2048 / 64 = 32). This is
the microscaling block-32 layout. The exact nibble order, sign convention, and
scale-application rule are **not** established here and must be read from
`inference/kernel.py` before any dequantization code is written.

Project decision: the 12.75 MiB expert unit is the engine's natural I/O and
cache granularity. Packing, prefetch, LRU accounting, and cache-hit telemetry
should all be denominated in whole expert units.

## Placement budgets

Derived from the verified inventory and the verified `config.json`
(43 layers, 256 routed experts, top-6, 1 shared expert).

| Quantity | Bytes | GB |
|---|---:|---:|
| One routed expert | 13,369,344 | 0.013 |
| Routed experts, one layer | 3,422,552,064 | 3.42 |
| Routed experts, all 43 layers | 147,169,738,752 | 147.17 |
| Non-expert layer tensors, all 43 layers | 6,727,565,512 | 6.73 |
| `embed` / `head` | 1,059,061,760 each | 1.06 each |
| MTP stack, non-expert portion | 595,182,108 | 0.60 |
| MTP stack, expert portion | 10,267,656,192 | 10.27 |
| **Decode traffic per token, cold cache** | **3,449,290,752** | **3.45** |
| **Prefill traffic per full-union chunk sweep** | **147,169,738,752** | **147.17** |

Decode traffic assumes no expert reuse between consecutive tokens. Prefill
traffic is the upper bound reached once a chunk's routing union covers every
expert, which any chunk of more than a few dozen tokens is expected to approach.

Inference, requiring measurement: chunk size is the dominant cost variable in
prefill. Eight chunks over one prompt cost eight sweeps, not one. The chunking
specification in Phase 5 must be written against this number.

## Constrained VRAM plan for the 11 GB target

`embed` is a per-token row gather and belongs in host memory. The MTP stack is
optional until speculative decoding is implemented. That leaves a candidate
permanently resident set:

| Item | Bytes |
|---|---:|
| Non-expert layer tensors, all 43 layers | 6,727,565,512 |
| `head` | 1,059,061,760 |
| Global norm and hyper-connection tensors | 270,356 |
| **Resident subtotal** | **7,786,897,628 (7.25 GiB)** |

Against a 9 GiB allocated budget this leaves 1.75 GiB, and against 10 GiB it
leaves 2.75 GiB, for the CUDA context and cuBLAS workspace, expert staging
buffers (two layers of six units is 153 MiB), context state, activations, and
allocator slack.

Project decision candidate, superseding "current-layer residency": on the 11 GB
target, keep **all** non-expert weights permanently resident and stream **only**
routed experts. This removes 6.73 GB per token from the transfer path and makes
per-token traffic exactly the 3.45 GB expert figure. Adding the MTP stack for
speculative decoding costs only 0.60 GB of residency, since its experts stream
like any other.

Verified consequence for the development GPU: the resident subtotal of 7.25 GiB
exceeds a 6 GB card, and the non-expert layer tensors alone (6.27 GiB) exceed it.
The RTX 3060 Laptop preset cannot run the full model under any placement policy
and remains a tiny-model and kernel-development target only.

## Numeric format findings for `sm_75`

Turing has no BF16 arithmetic, so every BF16 tensor must execute as FP16 or
FP32. FP32 would double `embed` and `head` to 4.24 GB and break the budget above,
so the question is whether FP16 is range-safe. Measured with
`tools/probe_weight_ranges.py`:

| Tensor | Coverage | max abs | Values above FP16 max | Values below FP16 min subnormal |
|---|---|---:|---:|---:|
| `embed.weight` | 16 MiB strided sample | 2.375 | 0 | 1 |
| `head.weight` | 16 MiB strided sample | 3.8125 | 0 | 1 |
| `layers.0.ffn.gate.weight` | complete | 0.498 | 0 | 1 |
| `layers.20.ffn.gate.weight` | complete | 0.260 | 0 | 2 |
| `layers.2.attn.compressor.wkv.weight` | complete | 0.170 | 0 | 6 |
| `layers.20.attn.indexer.weights_proj.weight` | complete | 7.281 | 0 | 0 |
| `layers.20.hc_attn_fn` (F32) | complete | 1.047 | 0 | 716 of 393,216 |
| `mtp.2.markov_head.markov_w1.weight` | 16 MiB strided sample | 17.125 | 0 | 1 |

Verified fact: no measured weight approaches the FP16 maximum of 65,504. The
largest magnitude found anywhere was 17.125.

Project decision: FP16 is admissible for BF16 weight storage on `sm_75`, and the
budget above stands. Two caveats carry forward. First, a handful of
denormal-magnitude entries flush to zero in FP16; parity tests must confirm this
is numerically irrelevant rather than assume it. Second, `hc_*_fn` is F32 with
8.5 percent of its values below the FP16 normal minimum, so the mHC tables
should stay F32 — they total 135 MB and the precision is nearly free.

Unresolved: activations and accumulation are a separate question from weight
storage. Router scores, `sqrtsoftplus`, Sinkhorn iterations, and attention
reductions still require declared accumulation precision, and the sampled
tensors above say nothing about them.

## Architecture facts feeding RAF-6

Verified facts from the container headers:

- 43 decoder layers, every one of them MoE. There are no dense prefix layers.
- All 43 layers carry 256 routed experts and one shared expert.
- Layers 0, 1, and 2 carry `ffn.gate.tid2eid`, an `I64` [129280, 6] table, and no
  `ffn.gate.bias`. Layers 3 through 42 carry `ffn.gate.bias` and no `tid2eid`.
- `attn.indexer.*` exists on exactly 21 layers: every even layer from 2 to 42.
- `attn.compressor.*` exists on 41 layers, 2 through 42, in two variants whose
  `ape` shapes are [4, 1024] on 21 layers and [128, 512] on 20 layers.
- Every layer carries `attn.attn_sink`, an F32 [64] tensor.
- Three complete MTP blocks exist: `mtp.0`, `mtp.1`, `mtp.2`. Each has its own
  attention, its own 256-expert MoE, and its own hyper-connection tensors.
  `mtp.0` alone has `main_proj` and `main_norm`; `mtp.2` alone has
  `markov_head`, `confidence_head`, `norm`, and `hc_head_*`.

Inference: the per-layer `ape` shapes correspond to the `compress_ratios` array
in `config.json`, where ratio 4 pairs with [4, 1024] and ratio 128 pairs with
[128, 512]. This mapping is strongly suggested by the counts but must be
confirmed against `inference/model.py` before it constrains implementation.

Resolution of a recorded S1.1 divergence: the open item
`next_token_prediction_layer_counts` recorded that `config.json` declares
`num_nextn_predict_layers=1` while `inference/config.json` declares
`n_mtp_layers=3`. The artifacts settle the count — three complete MTP blocks are
physically present, totaling 10.86 GB. The remaining question is semantic, not
structural: whether the Transformers-facing field describes how many blocks are
used per step rather than how many exist. The immutable S1.1 manifest is not
modified by this finding.

Engineering consequence, requiring measurement: expert selection on layers 0–2 is
a pure token-ID lookup, so those 18 expert units per token are known before any
compute begins. Every other layer's routing depends on the previous layer's
output. A streaming engine should exploit this asymmetry explicitly rather than
treat all 43 layers as equally unpredictable.

## Reproduction on Linux

Tool versions: Python `3.12.13` via `uv` `0.11.3`, locked environment.

```bash
uv sync --locked

uv run --locked python -B tools/build_tensor_inventory.py \
  --generated-at-utc 2026-08-05T18:00:00Z

cd manifests/derived && sha256sum -c tensor-inventory-0731.sha256
```

The builder refuses to overwrite a changed artifact without `--force`, aborts on
any tensor name outside its explicit role grammar, and aborts if the header
totals disagree with the immutable manifest.

Range measurement, which reads at most 16 MiB per tensor, never touches expert
payloads, and never writes weight bytes to disk:

```bash
uv run --locked python -B tools/probe_weight_ranges.py
```

Exit code 4 reports at least one tensor with values outside the FP16
representable range, which is the expected result while denormal entries exist.

## Boundary

This audit does not prove possession of any weight byte, does not establish the
FP4 nibble or scale-application semantics, does not define execution order, and
does not measure activation ranges. It constrains the memory plan and the
architecture inventory; it does not replace `inference/model.py` as the semantic
authority.
