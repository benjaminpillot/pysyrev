<div align="center">
  <img src="docs/logo.png" alt="pysyrev" width="350"/>
</div>

<div align="center">

[![tests](https://github.com/benjaminpillot/pysyrev/actions/workflows/tests.yml/badge.svg)](https://github.com/benjaminpillot/pysyrev/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/pysyrev/badge/?version=latest)](https://pysyrev.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/pysyrev)](https://pypi.org/project/pysyrev/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-≥3.10-blue)](https://www.python.org/)

</div>

**pysyrev** (PYthon SYstematic REView) is an automated, LLM-assisted PRISMA workflow for systematic literature reviews. It covers the full pipeline — from raw bibliographic records to screened, deduplicated, and thematically structured corpora — and produces a PDF report at the end.

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
| Report | `topic-report` | Generate a PDF report from the selected topic model run |

All sections are optional — only the stages declared in the config file are executed. Each stage auto-detects the most recent output of the previous one when run standalone.

---

## Installation

> **Prerequisite:** Python ≥ 3.10.

### From PyPI

```bash
pip install pysyrev
```

To enable Plotly figure embedding in PDF reports:

```bash
pip install "pysyrev[plotly]"
```

### From source

```bash
git clone <repo-url>
cd pysyrev
pip install -e .
```

## Documentation
Documentation is available from [here](https://pysyrev.readthedocs.io/en/latest/)

## Quick start

### CLI

```bash
# Check the installed version
pysyrev --version

# Run all configured stages (only stages present in the config are executed)
pysyrev config.yaml

# Run a single stage
pysyrev config.yaml --stage bib
pysyrev config.yaml --stage review
pysyrev config.yaml --stage bib-network
pysyrev config.yaml --stage topic-model
pysyrev config.yaml --stage topic-report
```

> If the `pysyrev` command is not available (e.g. editable install not yet registered), use `python -m pysyrev` as a drop-in replacement.

### Python API

```python
from pysyrev import Pipeline

# Full pipeline in one call — runs only the stages declared in the config
pipeline = Pipeline.from_config("config.yaml")
pipeline.run()

# Or stage by stage — results persist on the instance between calls
pipeline.run(stages=["bib"])
pipeline.run(stages=["review"])        # uses pipeline.bib.dataset automatically
pipeline.run(stages=["topic-report"])  # generates the PDF report

# Access results
df_all    = pipeline.bib.dataset          # pd.DataFrame — all collected documents
df_kept   = pipeline.review.included_docs # pd.DataFrame — LLM-screened inclusions
network   = pipeline.network              # BibNetwork
topic     = pipeline.topic                # TopicModel
report    = pipeline.report               # TopicReport
```

### Report-only run

A config containing only the `topic_report` (and optionally `report` and `llm`) sections is valid. This lets you generate or regenerate a report from a previous topic-model run without re-running the full pipeline:

```yaml
# report_only.yaml
topic_report:
  run_dir: /path/to/topic_modeling/run_2026-05-01T120000/  # or leave blank to auto-detect
  model_index: 0
  export_to: /path/to/output/report/
```

```bash
pysyrev report_only.yaml
```

---

## Configuration

A single YAML file controls all stages. Copy `pysyrev/config_examples/config_template.yaml` and fill in the sections you need. Sections not present in the file are simply skipped.

Key auto-detection rules (when fields are left blank):

| Blank field | Auto-detected from |
|---|---|
| `review.doc_dataset` | latest run in `bib.export.export_dir` |
| `bib_network.doc_dataset` | latest run in `review.export.export_dir` |
| `topic_model.doc_dataset` | latest run in `review.export.export_dir` |
| `topic_report.run_dir` | latest run in `topic_model.export.export_dir` |
| bib-network graphs in report | latest run in `bib_network.export.export_dir` |

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
- 
---

<div align="center">
  <img src="docs/espace-dev-ird.png" alt="organizations" width="600"/>
</div>

---