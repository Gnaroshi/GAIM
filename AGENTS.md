# GAIM research workspace

This is the owner-authorized public research repository https://github.com/Gnaroshi/GAIM.
Run commands from this repository's root. Personal server paths are not prerequisites.
Read README.md and the relevant experiment/result documents before changes. Claude users should read CLAUDE.md when available.

- Use only four GPUs allocated to this user. On the original server those are physical devices 4,5,6,7; do not use its devices 0–3. On another server explicitly select its allocated devices with GAIM_CUDA_DEVICES. Do not stop other users' processes or change global environments.
- GPU selection precedence is GAIM_CUDA_DEVICES, existing CUDA_VISIBLE_DEVICES, then the original default 4,5,6,7. Four unique numeric physical IDs are required. Never change the mapping while resuming a run.
- The reference-20260921 tag preserves the original execution code. Main includes documented operational changes; keep the scientific protocol unchanged for reproduction.
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
- When the user asks for tmux commands, prepare code/data and return one command per pane. Do not start training, GPU batch probes, polling, or long waits unless asked. Use gaim.train_arm for independent one-GPU panes; keep historical gaim.training behavior separate.
- Define each model by its public starting checkpoint and the data we additionally train it on. “Untrained” means no project-specific added training, never no prior training. Use clear data descriptions instead of unexplained clean/random/independent/adaptive labels.
- Keep checks proportional to the change. Preserve source gold, train/test separation, valid gradients and existing outputs; do not add duplicate gates or repeat completed checks without a concrete reason.
- Readability edits to historical Markdown are authorized; preserve observations and raw result JSON, with earlier wording in Git history.
