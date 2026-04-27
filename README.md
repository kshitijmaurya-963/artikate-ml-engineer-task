# Artikate AI / ML / LLM Engineer Assessment – Solution

This repository contains a complete solution for the Artikate Studio AI Engineer assessment. It is designed to run locally using free-tier and self-hosted models only.

## Quick Start

### 1. Clone and create environment

```bash
git clone <your-repo-url>.git
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

### 2. Configure environment variables

Create the .env file and fill in GROQ API key from https://console.groq.com/keys.

`.env` contains placeholders like:
```bash
LLM_API_BASE="https://api.groq.com/openai/v1"
LLM_API_KEY=""
LLM_MODEL_NAME="llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
```

### 3. Section 1 – Diagnosis only

Section 1 is written-only. See **Section 1** in `ANSWERS.md`.

### 4. Section 2 – RAG pipeline

Ingest the sample PDFs, build embeddings and index, then run a test query.

The ingestion and retrieval evaluation steps use the local SentenceTransformers model configured by `EMBEDDING_MODEL_NAME`, defaulting to `sentence-transformers/all-MiniLM-L6-v2`. The first run may download the model from Hugging Face.

```bash
python -m rag.ingest                  # ingests pdf's and creates indexing
python -m rag.eval_rag                # runs for precision@3 eval
python -m rag.pipeline --demo         # interactive demo from CLI
```

### 5. Section 3 – Ticket classifier

Create a classifier for five support categories:

- `billing`
- `technical_issue`
- `feature_request`
- `complaint`
- `other`

Training data is stored in:

```bash
data/tickets_train.jsonl
```
Evaluation data is stored in:
```bash
data/tickets_eval.jsonl
```
Train the classifier, then run evaluation and latency tests.

```bash
python -m classifier.train        # Trains the classifier
python -m classifier.evaluate     # Evaluates the classifier
python -m classifier.latency_test # Run the latency test; The latency test uses 20 raw ticket strings and asserts that every prediction is one of the five valid labels and that average inference latency is under 500ms per ticket.
```
> For approach and evaluation, visit **Section 3** in `ANSWERS.MD`

### 6. Section 4 – Systems design

Written answers are under **Section 4** in `ANSWERS.md`.

### 7. Loom recording 

```markdown
Loom walkthrough: https://www.loom.com/share/5467928882aa426da1e4464d07a91f31
```
