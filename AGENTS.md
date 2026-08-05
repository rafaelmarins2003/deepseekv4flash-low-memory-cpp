# Project Agent Instructions

## Purpose

This repository is both a working inference engine and an educational artifact. Agents must not merely produce a correct result: they must make the architecture, decisions, evidence, and verification understandable to a developer learning from the work.

Solve the task and teach the reusable ideas behind it. Do not replace implementation with a tutorial, and do not deliver opaque changes without explaining their purpose.

## Communication language

- Match the user's language. Use Brazilian Portuguese by default when collaborating with Rafael.
- Keep source code, identifiers, schemas, commit messages, and durable technical documentation in English unless the surrounding project convention requires otherwise.
- Define unfamiliar terms on first use. Prefer plain language, then give the precise technical term.
- Explain conclusions and evidence clearly; do not expose private chain-of-thought or fill the conversation with low-value narration.

## Required didactic workflow

For every meaningful task:

1. **Orient:** identify the SDD issue, project phase, authoritative inputs, expected output, and prerequisite artifacts.
2. **Build the mental model:** briefly explain what the component does, why it exists, where it fits in the model or runtime, and its main failure modes.
3. **Separate knowledge types:** label important statements as verified fact, inference, project decision, or unresolved question when the distinction matters.
4. **Plan in inspectable checkpoints:** each checkpoint must produce an artifact or evidence the user can review.
5. **Implement minimally:** make the smallest clear change that satisfies the approved specification.
6. **Verify proportionally:** show the command, what it proves, and the observed result. Never imply that one test proves more than it actually does.
7. **Teach the handoff:** summarize what changed, why it is correct, how to reproduce it, what the user should inspect, and what remains unknown.

At useful milestones, explain the result before continuing. Do not pause for confirmation when the next action is already authorized, safe, and unambiguous.

## Spec-Driven Development contract

The project constitution is the combination of the Linear **Mission**, **Tech Stack**, and **Roadmap** documents. Feature issues are separate executable specifications.

Before changing code or durable artifacts:

- Read the relevant constitution sections and feature issue.
- Identify inputs, outputs, invariants, failure behavior, tests, and prohibited shortcuts.
- Trace each implementation change and test back to an approved requirement.
- If code, weights, or official evidence contradict the specification, record the discovery and update or escalate the specification before downstream work continues.
- Never guess an architecture detail to keep implementation moving. Preserve it as an explicit unresolved question with the evidence needed to answer it.
- Do not broaden a feature because a neighboring improvement appears convenient.

## Source authority and checkpoint lineage

The current model authority is:

- Model ID: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Normative model revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Informative documentation snapshot: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`

Rules:

- The normative revision defines weights, configuration, tokenizer, encoding protocol, reference code, and artifact membership.
- The documentation snapshot is separate evidence. Never mix its files into the normative model manifest.
- Never use mutable `main` as a source authority, download revision, test input, fixture identity, or runtime default.
- Distinguish upstream metadata verification from hashing local bytes. State the verification level explicitly.
- Keep the pristine source manifest immutable. Converted, packed, quantized, or fixture artifacts require a separate manifest containing `derived_from_manifest_sha256`.
- Never commit model weights to Git.
- Full-checkpoint download and local byte verification belong to RAF-12 and require a storage-capacity preflight.

## Correctness and reproducibility

- Correctness precedes performance.
- No silent fallback is allowed for missing metadata, unsupported formats, incompatible hardware, failed validation, or unresolved semantics.
- Tests must cover expected behavior and important failure modes.
- Numerical work must declare shapes, dtypes, layouts, accumulation precision, tolerances, and comparison method.
- Fixtures must record their source revision, manifest digest, environment, seed, and generation command.
- Commands should be reproducible on Linux and record relevant tool versions.
- CI must use small synthetic or tiny-model artifacts; it must not download the full checkpoint.
- A passing upstream hash or dry run does not prove that local checkpoint bytes were downloaded and verified.

## Memory and measurement discipline

- The primary hardware target is a physical 11 GB RTX 2080 Ti-class GPU.
- The initial safe allocated runtime target is approximately 9–10 GiB and must be finalized by measurement.
- Keep model-weight memory, context-state memory, compute workspace, staging buffers, and allocator/runtime overhead as separate budgets.
- Use `GB` for decimal units and `GiB` for binary units. Include raw byte counts where exactness matters.
- State whether a number is calculated, estimated, reported upstream, or measured locally.
- Refuse impossible configurations before allocation or download when they can be detected in advance.

## Coding minimalism

- First check whether the feature is needed.
- Reuse established project patterns, then standard-library or native features, then already-installed dependencies.
- Add a dependency only when its concrete benefit outweighs its reproducibility, security, build, and maintenance cost.
- Avoid speculative abstractions, generic frameworks, boilerplate, and premature optimization.
- Do not sacrifice readability, input validation, security, tests, numerical correctness, or academic reproducibility for fewer lines.
- Comments should explain non-obvious intent, invariants, numerical choices, or hardware constraints. Do not narrate obvious syntax.

## External code and prior art

- Treat official DeepSeek artifacts as normative only within their pinned revision and stated role.
- Treat papers and community runtimes as evidence or implementation references, not silent semantic authority.
- Before copying or adapting external code, record repository, exact revision, file or symbol, license, required attribution, modifications, and semantic-parity test.
- Preserve notices required by the source license and update `THIRD_PARTY_NOTICES.md` and the machine-readable source inventory.
- Clearly distinguish studying a technique, independently implementing it, and adapting source code.

## Review and completion standard

A task is complete only when:

- its required artifacts exist;
- acceptance criteria are checked with recorded evidence;
- negative and corruption paths required by the specification have been exercised;
- documentation describes actual behavior rather than intended behavior;
- no mutable source or unrecorded external dependency remains;
- unresolved questions and downstream consequences are explicit;
- the user receives a concise lesson covering the concept, the decision, the implementation, and the verification.

Preferred handoff structure:

1. Outcome
2. Concept learned
3. Decisions and tradeoffs
4. Files or specifications changed
5. Verification and evidence
6. Remaining questions and next checkpoint
