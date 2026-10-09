# DrugAgent

**Multi-agent evidence integration and reasoning for drug–target interaction (DTI) prediction**

[![arXiv](https://img.shields.io/badge/arXiv-2408.13378-b31b1b.svg)](https://arxiv.org/abs/2408.13378)
[![CI](https://github.com/inoue0426/DrugAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/inoue0426/DrugAgent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

DrugAgent brings together **machine-learning predictions**, **knowledge-graph evidence**, and **biomedical literature** to assess candidate drug–target interactions. Instead of presenting a model score alone, the workflow combines evidence from specialized agents and produces a **reasoning tree** alongside a final prediction.

**[Paper (arXiv)](https://arxiv.org/abs/2408.13378)** · **[Quick start](#quick-start)** · **[Outputs](#outputs)**

## How it works

Given a drug and a target gene, DrugAgent can gather complementary evidence through three components:

| Component | Evidence source | Role |
| --- | --- | --- |
| **ML** | DeepPurpose | Provides model-based DTI scores. |
| **KG** | Local knowledge graph | Provides graph-based evidence about the interaction. |
| **RAG** | PubMed literature index | Retrieves relevant biomedical literature evidence. |

The workflow organizes the collected signals into a reasoning tree and a final interaction label.

```text
Drug + target gene
       |
       +---- ML agent  ---- DeepPurpose prediction
       |
       +---- KG agent  ---- Knowledge-graph evidence
       |
       +---- RAG agent ---- Retrieved literature evidence
       |
       v
Evidence integration / multi-agent reasoning
       |
       +---- Reasoning tree (JSON)
       +---- Prediction summary (CSV)
```

The agents are selected with the `--enabled_agents` option. The full example below enables all three.

## Quick start

### Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

### Install

Run from the repository root:

```bash
git clone https://github.com/inoue0426/DrugAgent.git
cd DrugAgent
uv sync
```

### Configure credentials

Copy the example environment file and supply the credentials for the services you intend to use:

```bash
cp .env.example .env
```

The example file documents settings for Azure OpenAI, Claude, and NCBI. In particular, the PubMed RAG workflow requires embedding configuration. **Do not commit API keys or a populated `.env` file.**

### Run a drug–target example

```bash
uv run python -m drugagent.cli --drug Imatinib --gene KIT --enabled_agents ML,KG,RAG
```

This is an **example invocation**, not a guaranteed zero-configuration demo: ML model files, KG data, RAG assets, and the relevant service credentials must be available as described below. The repository includes a local KG asset, while the literature index may need to be obtained separately.

## Data and model assets

| Component | Required resource | Configuration |
| --- | --- | --- |
| ML | DeepPurpose model | Downloaded automatically if missing. |
| KG | Knowledge-graph CSV | Set `DRUGAGENT_KG_PATH`, or use `data/KG+BDB.csv.gz`. |
| RAG | FAISS index and metadata | Provide `data/kinase_rag_index.faiss` and `data/kinase_rag_metadata.json`, or configure the paths in `.env.example`. |

If the RAG index is not present, the application can attempt to download it using `DRUGAGENT_RAG_GDRIVE_URL` and `DRUGAGENT_RAG_DOWNLOAD_DIR`. Check [`.env.example`](.env.example) for available overrides.

## Outputs

DrugAgent writes predictions, intermediate evidence, and evaluation artifacts under `output/`. Not every file is produced by every command.

| Path | Contents |
| --- | --- |
| `output/trees/{config_id}/{drug}_{gene}.json` | Reasoning tree for a drug–target pair. |
| `output/summary_{ablation}.csv` | CLI prediction summary for an agent configuration. |
| `output/summary.csv` | Legacy summary output. |
| `output/input_payloads/{drug}__{target}.json` | Saved evaluation input payload. |
| `output/ml_dti_scores/` | ML prediction artifacts. |
| `output/ml_lookup_cache/` | Cached ML lookups. |
| `output/rag_dti_cache.csv` | Cached literature evidence. |
| `output/graph_dti_cache.csv` | Cached graph evidence. |

For an actual example of a saved result, see the [sample summary CSV](output/summary_v0_ml_kg_rag_nomodel_none.csv). These files are artifacts of the workflow; they should not be interpreted as a claim of validated DTI performance without the evaluation protocol in the paper.

## Plausibility and faithfulness evaluation

The repository also provides a script for evaluating the plausibility and faithfulness of saved reasoning inputs.

**From a prediction summary CSV:**

```bash
uv run python src/faithfulness_plausibility_eval.py \
  --summary-csv output/summary_v0_ml_kg_rag_nomodel_none.csv \
  --output output/plausibility_faithfulness_from_summary.jsonl
```

The script reads the `input_payload` column (either JSON content or a path such as `output/input_payloads/*.json`).

**From JSONL inputs:**

```bash
uv run python src/faithfulness_plausibility_eval.py \
  --input data/plausibility_faithfulness_demo.jsonl \
  --output data/plausibility_faithfulness_results.jsonl
```

These evaluations use `AZURE_OPENAI_API_LLM_KEY` and `CLAUDE_ENDPOINT` from `.env`; see [`src/drugagent/kinase/config_utils.py`](src/drugagent/kinase/config_utils.py). They are separate from the basic drug–target inference command.

## Repository guide

| Location | Purpose |
| --- | --- |
| [`src/drugagent/cli.py`](src/drugagent/cli.py) | Command-line entry point used in the quick start. |
| [`src/drugagent/`](src/drugagent/) | Packaged agent orchestration and related modules. |
| [`src/`](src/) | Supporting utilities and evaluation scripts. |
| [`data/`](data/) | Bundled reference data and example evaluation inputs. |
| [`output/`](output/) | Stored predictions, evidence, and evaluation results. |
| [`.env.example`](.env.example) | Configuration variables and optional resource paths. |

The repository includes both packaged modules and older top-level scripts. For the documented inference workflow, begin with `python -m drugagent.cli`.

## Paper and citation

For the experimental methodology, datasets, baselines, and reported performance, refer to the [DrugAgent paper](https://arxiv.org/abs/2408.13378). The commands above demonstrate inference and reasoning evaluation; they are **not** a complete end-to-end reproduction script for every paper experiment.

```bibtex
@article{inoue2025drugagent,
  title={Drugagent: Multi-agent large language model-based reasoning for drug-target interaction prediction},
  author={Inoue, Yoshitaka and Song, Tianci and Wang, Xinling and Luna, Augustin and Fu, Tianfan},
  journal={ArXiv},
  pages={arXiv--2408},
  year={2025}
}
```

## License

Released under the [MIT License](LICENSE).
