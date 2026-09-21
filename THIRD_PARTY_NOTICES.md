# Third-party data and models

## MedQA

This experiment downloads the English four-option mirror `GBaker/MedQA-USMLE-4-options` at revision `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`.

The [pinned mirror's dataset card](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/blob/0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18/README.md) declares **CC BY 4.0**. See the [license](https://creativecommons.org/licenses/by/4.0/). Attribution: Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits, *What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams*, 2020, [paper](https://arxiv.org/abs/2009.13081), [original MedQA repository](https://github.com/jind11/MedQA).

Question excerpts, choices and answer keys included in the reference records retain their original wording. Stable source IDs, split provenance and hashes identify their source. Korean summaries/translations, generated perturbations and model responses are additions by this experiment; they are not original MedQA medical explanations. Text formatting and personal execution paths in publication copies are normalized as documented in the publication manifests. Referenced source images are not included.

The [original MedQA repository's MIT license](https://github.com/jind11/MedQA/blob/master/LICENSE) and the mirror's dataset-card declaration are distinct; this notice does not substitute the original repository license for the mirror's stated data license.

## Qwen

Model: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The model card declares Apache 2.0. Model weights and trained adapters are downloaded/generated locally, not included in this Git repository. Consult the [pinned model license](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/LICENSE).

## Software dependencies and research references

Python packages retain their respective licenses. Direct versions are recorded in `requirements.txt`; the observed complete environment is in `docs/results/reference/environment.txt`.

PAIR ([paper](https://arxiv.org/abs/2310.08419)) and MedFuzz ([paper](https://arxiv.org/abs/2406.06573)) motivate the comparison. This repository does not claim to reproduce their complete original implementations or benchmark results. No project-wide license for newly authored code is assigned by this third-party notice.
