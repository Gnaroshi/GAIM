# Operational changes after the original experiment

The annotated tag `reference-20260921` preserves the execution source associated with the completed 2026-09-21 records. `main` adds the changes below so another researcher can use the public project. The historical run manifests keep their original code hashes; those hashes do not describe the later main branch.

## Changes

- GPU selection is shared across launchers, workers and preflight: explicit `GAIM_CUDA_DEVICES`, then existing `CUDA_VISIBLE_DEVICES`, then original default `4,5,6,7`. Four unique numeric physical IDs are required; order is preserved. Examples using another server's 0–3 are not authorization to use those devices on the original server.
- GPU-use checks and child processes follow the selected mapping. Saved evaluation/training/replay manifests are checked before continuation. A changed mapping fails before model loading. Replay now records its mapping; use a fresh replay directory when moving from the historical schema.
- Setup checks Ubuntu 22.04 x86_64, Python 3.10, required tools and selected GPU availability before installation. `--check` performs only these checks. Dependencies remain project-local; original observed package versions constrain resolution. The bootstrap script and optional Ubuntu Python-header package are checked against observed SHA-256 values before execution/extraction.
- `scripts/verify_reference.py` verifies public-record hashes, reconstructs original training rows, rescores actual answers and checks same-input replay summaries using the standard library. It performs no model inference, network calls or record writes.
- `CLAUDE.md` and `REPRODUCING_KO.md` distinguish CPU verification, fixed-input retraining, and complete regeneration. Historical source records and result reports remain immutable.

The model/dataset revisions, question selection, prompts, feedback, sampling, seed derivation, candidate validation, training selection, QLoRA settings and answer scoring were not changed by this publication work. A new launch records current code hashes rather than pretending to use the historical code.

The original execution selected GPUs 4–7. New manifests from another GPU allocation are expected to differ in operational metadata. Fixed seed and same inputs do not guarantee bitwise-identical training across environments. Report differences instead of editing source observations or tuning to match them.

## Validation boundary

Validation of the publication changes is recorded in `PUBLICATION_VALIDATION.md`. GPU preflight is a small NF4 forward/backward check, not a repeat of the medical experiment. The reference scores come from the completed historical runs; portability validation does not create new clinical or robustness results.
