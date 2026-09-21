# Publication validation — 2026-09-21

These checks validate the public packaging and operational changes on main. They do not represent a new medical QA run or new training results.

| Check | Observed result |
|---|---|
| Local standard-library test suite | `python3 -m unittest discover -s tests -q`: 101 tests passed |
| Server test suite using the existing research environment | `.venv/bin/python -m pytest -q`: 101 passed in 3.91 seconds |
| CPU reference verifier, local and server | 39 file hashes passed; 228 train and 1,900 test responses rescored; four original training datasets reconstructed with matching hashes; 4,760 replay responses rescored with identical input/gold alignment and matching summaries |
| Shell syntax and working-tree whitespace | `bash -n scripts/setup.sh` and `git diff --check` passed |
| Supported-environment guard | `GAIM_CUDA_DEVICES=4,5,6,7 bash scripts/setup.sh --check` passed on Ubuntu 22.04 x86_64, Python 3.10.12 |
| Dependency constraints | `pip install --dry-run -r requirements.txt -c docs/results/reference/environment.txt` passed against the existing environment; no packages changed |
| Actual four-GPU NF4 preflight | Forward and backward passed on each RTX 3090 at physical 4,5,6,7, mapped to logical 0,1,2,3 |
| Official bootstrap availability | Current `get-pip.py` bytes matched the pinned SHA-256; pinned Ubuntu header URL and CUDA 12.4 Torch index returned HTTP 200 |

Server validation used an isolated publication-source directory with the existing environment, cache and local headers. It did not replace the historical experiment outputs. The GPU check used torch 2.6.0+cu124, CUDA runtime 12.4, transformers 4.57.1, peft 0.17.1 and bitsandbytes 0.48.1.

A clean machine installation from an empty environment and a complete post-portability medical experiment were **not** rerun for publication. The setup downloads and original observed dependency constraints were checked, but a fresh user's network, driver and compiler remain environment prerequisites. Read the supported-platform requirements and failure handling in [the reproduction guide](REPRODUCING_KO.md).

Claude Code was not used as a separate agent to run this validation. The repository supplies its project instructions and verified experiment commands; it does not claim a separately observed Claude-driven reproduction.
