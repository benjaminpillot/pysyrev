# pysyrev

**pysyrev** (PYthon SYstematic REView) is an automated, LLM-assisted PRISMA workflow for systematic literature reviews. It covers the full pipeline — from raw bibliographic records to screened, deduplicated, and thematically structured corpora — and can produce a PDF report at each stage.

![image](docs/espace-dev-ird.png)

---

## Features

- **Multi-source ingestion** — Web of Science (file or REST API), OpenAlex (file or REST API), Scopus, PubMed
- **Automatic deduplication** — fuzzy title matching across sources
- **LLM-based title/abstract screening** — multi-reviewer workflows with majority or mean voting, powered by any provider supported by LiteLLM (Anthropic, OpenAI, Ollama, LiteLLM proxy…)
- **Bibliographic network analysis** — bibliographic coupling and co-citation graphs exported as GraphML
- **Topic modelling** — BERTopic-based clustering with UMAP + HDBSCAN grid search, ranked by coherence scores
- **PDF report generation** — declarative, theme-aware PDF engine built on ReportLab

---

## Pipeline stages

| Stage | Key | Description |
|---|---|---|
| Bibliography | `bib` | Fetch, clean, filter, deduplicate, and optionally resolve references |
| LLM review | `review` | Screen documents against inclusion/exclusion criteria with one or more LLM reviewers |
| Bibliographic network | `bib-network` | Build coupling and co-citation networks from the included corpus |
| Topic modelling | `topic-model` | Cluster documents into topics using BERTopic; rank configurations by coherence |

Stages share data in memory when run together, and each stage auto-detects the most recent output of the previous one when run standalone.

---

## Installation

> **Prerequisite:** Python ≥ 3.10.

```bash
# 1. Clone the repository
git clone <repo-url>
cd pysyrev

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install berteley (local text pre-processing package)
pip install /path/to/berteley

# 4. Install pysyrev itself (editable mode recommended for development)
pip install -e .
```

To enable Plotly figure embedding in PDF reports, install the optional extras:

```bash
pip install -e ".[plotly]"
```

---

## Quick start

### CLI

```bash
# Run the full pipeline
python -m pysyrev config.yaml

# Run a single stage
python -m pysyrev config.yaml --stage bib
python -m pysyrev config.yaml --stage review
python -m pysyrev config.yaml --stage bib-network
python -m pysyrev config.yaml --stage topic-model
```

If installed via `setup.py`, the `pysyrev` command is also available directly:

```bash
pysyrev config.yaml --stage review
```

### Python API

```python
from pysyrev import Pipeline

# Full pipeline in one call
pipeline = Pipeline.from_config("config.yaml")
pipeline.run()

# Or stage by stage — results persist on the instance between calls
pipeline.run(stages=["bib"])
pipeline.run(stages=["review"])   # uses pipeline.bib.dataset automatically

# Access results
df_all    = pipeline.bib.dataset          # pd.DataFrame — all collected documents
df_kept   = pipeline.review.included_docs # pd.DataFrame — LLM-screened inclusions
network   = pipeline.network              # BibNetwork
topic     = pipeline.topic                # TopicModel
```

---

## Getting started

See the `tutorials/` folder for step-by-step Jupyter notebooks and annotated configuration examples covering each pipeline stage.

---

## Contributing

### Development and improvement

- Benjamin Pillot
- Théo Chamarande
- Kevin Chapuis

### Conceptualization and Coordination

- Benjamin Pillot
