# Historical execution metadata

These records describe the completed 2026-09-21 experiment, not a run on the machine that clones this repository.

- `test_run.json`: original fixed test protocol, question IDs, source dataset checksums, model revisions and execution-code hashes.
- `protocol_lock.json`: configuration, IDs and artifact hashes captured before test model calls.
- `training.json`: historical training settings, ten selected question IDs and selected perturbations.
- `replay.json`: historical same-input comparison and adapter hashes.
- `environment.txt`: observed full Python environment, including CUDA wheel versions. It is an environment reference, not a cross-platform compatibility guarantee.
- `publication_paths.json`: original and publication-copy file hashes for documents/metadata whose personal absolute paths were replaced by relative paths.

Path normalization changes serialized metadata hashes. Historical hashes inside records continue to identify original private execution artifacts; do not reinterpret them as hashes of redacted public copies. Numerical observations, scientific settings, source IDs and model revisions are unchanged.

The `reference-20260921` Git tag preserves the runtime source used by the experiment. Main-branch operational changes are documented separately. Fixed source records for reproducing input selection are under `repro/reference/`; full caches, model weights, virtual environments and process logs are not distributed.
