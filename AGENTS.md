# GAIM research workspace

This is the owner-authorized public research repository https://github.com/Gnaroshi/GAIM.
Run commands from this repository's root. Personal server paths are not prerequisites.
Read README.md and the relevant experiment/result documents before changes. Claude users should read CLAUDE.md when available.

- Only use CUDA physical devices 4,5,6,7. Do not stop other users' processes or change global environments.
- This frozen runtime assumes four GPUs at those indices. A later operational portability change must preserve the scientific protocol and identify the historical runtime separately.
- Keep experiment setup, generated data, observed results, and hypothetical examples distinct.
- Preserve original MedQA questions, choices, answer keys, source IDs and split provenance. Never claim a mirror or derived validation split is the official validation split.
- Separate syntactic validity from clinically confirmed answer preservation. Model or heuristic checks are provisional; human review is needed before confirmed attack-success claims.
- Compare clean-repeat, independent and adaptive conditions with the same target-call budget; do not stop after success. Start target context fresh for every attempt.
- Save actual generations and response errors. No fabricated results or silently substituted test data.
- Use code-only tests with explicit fixtures, never count fixtures as experiments.
- Implement core research operations with helpful Korean comments connecting inputs, outputs and control logic.
- Keep environments and downloads within the remote project/cache directories. No paid API calls by default.
- Keep published reference records immutable. Write new outputs under a new runs/ directory. Never tune on the observed test sample to force reference scores.
- Only publish intentionally selected code, documentation and attributed research records. Do not add credentials, personal paths, environments, model weights, caches, or unrelated attachments.
- Keep source/code-hash provenance and empirical observations distinct from later operational changes. A fixed seed is not a promise of bitwise-identical CUDA output.
