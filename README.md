# DrugAgent

Multi-agent workflow for drug-target interaction (DTI) evidence. It combines ML scores (DeepPurpose), KG signals, and PubMed RAG evidence, then produces a reasoning tree and final label.

[![arXiv](https://img.shields.io/badge/arXiv-2408.13378-b31b1b.svg)](https://arxiv.org/abs/2408.13378)
[![CI](https://github.com/inoue0426/DrugAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/inoue0426/DrugAgent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Last Commit](https://img.shields.io/github/last-commit/inoue0426/DrugAgent)](https://github.com/inoue0426/DrugAgent/commits/main)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Import Sorting: isort](https://img.shields.io/badge/imports-isort-1674b9.svg)](https://pycqa.github.io/isort/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

## Setup (Repo Root)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv sync
source .venv/bin/activate
```

## Environment Variables

Create a `.env` file (see `.env.example`) or export these variables:

```bash
AZURE_OPENAI_API_LLM_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT_NAME=...
AZURE_OPENAI_API_VERSION=...
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=...
AZURE_OPENAI_EMBEDDING_API_VERSION=...
```

Embedding variables are required for PubMed RAG.

## Run

Full DrugAgent run:

```bash
uv run python -m drugagent.cli --drug Imatinib --gene KIT --enabled_agents ML,KG,RAG
```

Plausibility/Faithfulness evaluation (Claude Opus):

```bash
uv run python src/faithfulness_plausibility_eval.py \
  --input data/plausibility_faithfulness_demo.jsonl \
  --output data/plausibility_faithfulness_results.jsonl
```

Claude setup:

```bash
CLAUDE_DEPLOYMENT=claude-opus-4-6
```

Uses `AZURE_OPENAI_API_LLM_KEY` and `CLAUDE_ENDPOINT` from `.env` (see `src/drugagent/kinase/config_utils.py`).

## Outputs

- `output/trees/{config_id}/{drug}_{gene}.json` reasoning trees
- `output/summary_{ablation}.csv` (CLI)
- `output/summary.csv` (legacy summary output)
- `output/ml_dti_scores`
- `output/ml_lookup_cache`
- `output/rag_dti_cache.csv`
- `output/graph_dti_cache.csv`

## Data Assets

- ML: DeepPurpose model downloads automatically if not present.
- RAG: place files at `data/kinase_rag_index.faiss` and `data/kinase_rag_metadata.json`. If missing, the app tries to download via `DRUGAGENT_RAG_GDRIVE_URL` into `DRUGAGENT_RAG_DOWNLOAD_DIR`.
- KG: provide a local KG CSV and set `DRUGAGENT_KG_PATH` or place it at `data/KG+BDB.csv.gz`.

## Citation

```bibtex
@article{inoue2025drugagent,
  title={Drugagent: Multi-agent large language model-based reasoning for drug-target interaction prediction},
  author={Inoue, Yoshitaka and Song, Tianci and Wang, Xinling and Luna, Augustin and Fu, Tianfan},
  journal={ArXiv},
  pages={arXiv--2408},
  year={2025}
}
```
