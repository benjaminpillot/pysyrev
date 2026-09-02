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
- **Automatic deduplication** — fuzzy title matching across sources, with a configurable source priority (OpenAlex wins by default, keeping its stable IDs; duplicates are aliased so references still resolve)
- **Canonical reference keys** — optionally remap every reference onto a shared DOI key space so bibliographic coupling holds across merged sources regardless of merge order: intra-corpus and DOI-bearing references map for free, extra-corpus OpenAlex ids are resolved to DOIs via the OpenAlex API and cached (skipped automatically when there is no other-source DOI to bridge to)
- **LLM-based title/abstract screening** — multi-reviewer workflows with majority or mean voting, powered by any provider supported by LiteLLM (Anthropic, OpenAI, Ollama, LiteLLM proxy…) or by [Albert](https://albert.api.etalab.gouv.fr), the French State's sovereign gateway (`provider: albert`, key in `ALBERT_API_KEY`) — useful when the corpus may not leave French public infrastructure
- **Bibliographic network panels** — bibliographic coupling and co-citation networks (Salton similarity + Leiden communities, readable backbone layout) cross-coloured by BERTopic topic, plus an inter-topic connectivity matrix (mean coupling between topics), all rendered in the report
- **Topic modelling** — BERTopic-based clustering with UMAP + HDBSCAN grid search, ranked by coherence scores
- **Reading-list selection** — rank the most relevant papers per topic by citations, network centrality, or a three-axis composite (coupling centrality + citation impact + thematic representativeness)
- **PDF report generation** — declarative, theme-aware PDF engine built on ReportLab

---

## Pipeline stages

| Stage | Key | Description |
|---|---|---|
| Bibliography | `bib` | Fetch, clean, filter, deduplicate, and optionally resolve references |
| LLM review | `review` | Screen documents against inclusion/exclusion criteria with one or more LLM reviewers |
| Topic modelling | `topic-model` | Cluster documents into topics using BERTopic; rank configurations by coherence |
| Report | `topic-report` | Generate a PDF report from the selected topic model run, including the bibliographic coupling, co-citation and topic-connectivity network panels (recomputed from the reviewed corpus' references) |

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
pysyrev config.yaml --stage topic-report

# Run multiple specific stages in one call (always executed in canonical order)
pysyrev config.yaml --stage topic-model topic-report

# Run from a given stage to the end (all configured stages from that point onwards)
pysyrev config.yaml --from topic-model
pysyrev config.yaml --from topic-report

# Download full-text PDFs for a list of candidates (Unpaywall → OpenAlex → Elsevier TDM)
pysyrev download candidates.csv output_folder/
pysyrev download candidates.csv output_folder/ --config download_config.yaml
```

Valid stage names: `bib` | `review` | `topic-model` | `topic-report`.
`--stage` and `--from` are mutually exclusive.

> If the `pysyrev` command is not available (e.g. editable install not yet registered), use `python -m pysyrev` as a drop-in replacement.

### Python API

```python
from pysyrev import Pipeline, ALL_STAGES

# Full pipeline in one call — runs only the stages declared in the config
pipeline = Pipeline.from_config("config.yaml")
pipeline.run()

# Run specific stages (always executed in canonical order)
pipeline.run(stages=["bib", "review"])
pipeline.run(stages=["topic-report"])  # generates the PDF report

# Run from a given stage to the end
start = ALL_STAGES.index("topic-model")
pipeline.run(stages=ALL_STAGES[start:])

# Access results
df_all    = pipeline.bib.dataset          # pd.DataFrame — all collected documents
df_kept   = pipeline.review.included_docs # pd.DataFrame — LLM-screened inclusions
topic     = pipeline.topic                # TopicModel
report    = pipeline.report               # TopicReport (incl. coupling + co-citation panels)
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
| `topic_model.doc_dataset` | latest run in `review.export.export_dir` (also the source the report's network panels are recomputed from) |
| `topic_report.run_dir` | latest run in `topic_model.export.export_dir` |

### LLM providers

Each reviewer (and the optional `llm:` labelling section) names a `provider`. Credentials never live in the YAML: every provider reads its own environment variable, resolved from the `env:` file the config points at.

| `provider` | Key | Default endpoint |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Anthropic API |
| `openai` | `OPENAI_API_KEY` | OpenAI API |
| `albert` | `ALBERT_API_KEY` | `https://albert.api.etalab.gouv.fr/v1` |
| `ollama` | — | `http://localhost:11434/v1` |
| `litellm` (default) | provider-dependent | LiteLLM routing |

`provider: albert` targets the French State's sovereign gateway (Etalab / DINUM), where access is granted to public-sector users rather than purchased — reported cost is therefore always 0. It fronts open-weight models served by vLLM, so structured output depends on the deployment: pysyrev asks for a strict `json_schema` first and, if the endpoint refuses, falls back to `json_object` and then to plain JSON parsing, keeping the level that worked for the rest of the run. List the model ids your key can reach with:

```bash
curl -H "Authorization: Bearer $ALBERT_API_KEY" https://albert.api.etalab.gouv.fr/v1/models
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
- 
---

<div align="center">
  <img src="docs/espace-dev-ird.png" alt="organizations" width="600"/>
</div>

---