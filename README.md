# Zepto Data & AI Platform — Capstone Project

One connected repository, three internally-linked modules, submitted as a
single coherent AI/ML engineering deliverable for Zepto's analytics guild.

| Module | Folder | Marks |
|---|---|---|
| Data Pipeline | [`/data_pipeline`](data_pipeline/) | 25 |
| Analytics Pipeline | [`/analytics`](analytics/) | 50 |
| Support Assistant (RAG) | [`/support_assistant`](support_assistant/) | 25 |

**Total: 100 marks.**

## Setup

Each module has its own `requirements.txt` (kept separate since the modules
have non-overlapping dependencies — pick this over one consolidated file to
keep each module independently runnable). From the repo root:

```bash
python -m venv venv && source venv/bin/activate    # or venv\Scripts\activate on Windows

pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```

## Running each module end to end

```bash
# Module 1 — Data Pipeline
cd data_pipeline
python scrape.py && python pipeline.py
cd ..

# Module 2 — Analytics Pipeline
cd analytics
python 01_eda.py && python 02_modeling.py
cd ..

# Module 3 — Support Assistant
cd support_assistant
python ingest.py && python example_calls.py
uvicorn main:app --host 0.0.0.0 --port 7860   # separate terminal, MOCK_LLM left at default
```

See each module's own README for full details, task-by-task design
decisions, and (for Module 3) the RAG architecture description.

## Design decisions summary

- **Module 1** scrapes books.toscrape.com (a public scraping-practice site)
  across 3+ categories into 60+ rows, cleans/types the fields, converts GBP
  to INR with the required fixed baseline (1 GBP = 105.50 INR, no API call),
  and loads a normalized two-table SQLite schema queried with 5+ SQL
  statements (including a JOIN), cross-checked against an equivalent
  `pandas.merge`.
- **Module 2** loads the Titanic dataset once via Seaborn, profiles and
  cleans it per a percentage-based missing-value threshold rule, tells a
  4-chart data story, then runs a full modeling pipeline (stratified split,
  leak-free `ColumnTransformer`/`Pipeline` preprocessing, three classifiers,
  full evaluation suite, three-way imbalance handling, `GridSearchCV`
  tuning with OOB score, and a regression side-task), ending in a saved,
  reloadable `joblib` pipeline.
- **Module 3** builds a small RAG service over 8 Zepto policy documents
  (local embeddings via `sentence-transformers`, ChromaDB for storage, a
  3-node LangGraph flow with a keyword-based mock-mode graded baseline that
  needs no LLM signup or network call), wrapped in a FastAPI `/ask`
  endpoint and a locally buildable/runnable Docker image.

## Git workflow

This repository's commit history includes at least one feature branch
created, committed to at least twice, and merged back into `main` (visible
via `git log --graph --all` or the repository's branch/merge history), as
required once across the whole repo (see `data_pipeline/README.md` for
where this is scored under the grading rubric).

## Academic integrity

All code, analysis, and written interpretations in this repository are
original work, produced with the aid of standard library/framework
documentation only.

## Project Status
Capstone project implementation completed.
