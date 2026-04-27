## Problem Context

Create a question-answering system over 500+ legal PDF contracts that:
- Answers precise queries (e.g., notice periods, liability limits).
- Returns exact document + page citations.
- Minimizes hallucinations; refusal is preferred when context is insufficient.

The solution must use free-tier or self-hosted models and be locally runnable.

## Chunking Strategy

- **Extractor:** Using `pdfplumber` to extract text with page numbers.
- **Granularity:** Creating overlapping, page-aware chunks of ~800–1,000 tokens with 150–200 token overlap.
This helps with:
  - Legal clauses are dense and often span multiple sentences; small chunks risk breaking clauses mid-way.
  - Overlap preserves cross-sentence context while avoiding excessive duplication.
- **Structure:** Each chunk record stores: `document_id`, `filename`, `page_number`, `chunk_index`, `text`.

## Embedding Model Choice

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Pros:**
  - Runs on CPU-only machines with low memory footprint.
  - Provides strong semantic retrieval performance for English legal text.
  - Free and easy to install using HuggingFace.

## Vector Store Choice

- **FAISS (flat index) via `faiss-cpu`.**
- **Pros:**
  - Purely local, free, and easy to serialize to disk.
  - Scales to hundreds of thousands of chunks on a single machine.
- **Alternative:** Chroma for persistent, queryable store with metadata filters.

For 500 contracts (~40 pages each), we would expect on the order of tens of thousands of chunks, which is well within FAISS CPU capabilities.

## Retrieval Strategy

- **Stage 1 – Dense retrieval:**
  - The implementation currently uses FAISS `IndexFlatL2` over dense embeddings and converts smaller L2 distances into a similarity-like confidence score. In a production version, I would L2-normalize embeddings and use inner product search to approximate cosine similarity more directly.
- **Future Improvements**:
- **Re-ranking:**
  - Use a small cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the top 20 chunks and keep the top 5–8 as LLM context.
- **Page grouping:** When multiple chunks from the same page are selected, merge them to avoid redundant context.

## Generation Model Choice

- **Model:** "llama-3.1-8b-instant"
- **Interface:** Abstracted through a simple `LLMClient` wrapper that:
  - Reads `LLM_API_BASE` and `LLM_API_KEY` from environment.
  - Accepts system + user messages.
  - Returns the assistant message text.

## Prompt Design

System prompt template:

```text
You are a legal contracts assistant.
You must answer ONLY using the provided context from company contracts.
If the context does not contain enough information to answer confidently,
respond with: "I do not have enough information in the documents to answer this question."

Always:
- Quote the relevant clause.
- Return the answer in concise prose.
- Never fabricate contract terms beyond the context.
```

User message template includes:

- The question.
- A serialized list of retrieved chunks with their document and page metadata.

## Hallucination Mitigation

Strategy implemented:

1. **Context sufficiency check:**
   - Compute a simple confidence measure based on average similarity of the top retrieved chunks.
   - If `max_sim` or `avg_topk_sim` falls below a threshold (i.e. 0.4), refuse to answer and return a low confidence score.

## Future Improvements
1. **Answer grounding check:**
   - After generation, scan the answer for contract-specific entities (numbers, dates, "₹" amounts, names) and verify they appear in at least one retrieved chunk.
   - If not, downgrade confidence and optionally replace the answer with a refusal template.

2. **Strict instructions:**
   - Prompt explicitly instructs the model never to make up terms not present in context.

## Evaluation Harness (Precision@3)

- **Dataset:**
  - `data/rag_qa_eval.jsonl` with 10 manually written QA pairs.
  - Each record: `question`, `expected_doc`, `expected_page`, `notes`.
- **Metric:**
  - For each question:
    - Retrieve top 3 chunks (before generation).
    - If any of the top 3 chunks originates from `expected_doc` and `expected_page`, count as a hit.
  - `precision_at_3 = hits / total_questions`.
- **Reporting:**
  - Script prints `precision_at_3` and a per-question table showing hits/misses.
  - Running:
    ```bash
    python -m rag.ingest
    python -m rag.eval_rag
    ```
  - Produced:
    Precision@3: 1.000


## Scaling to 50,000 Documents

If the corpus grows to 50,000 documents, the main bottlenecks and mitigations are:

1. **Ingestion & Embedding Time:**
   - Bottleneck: CPU-only embedding of millions of chunks.
   - Mitigation: Batch encoding, using GPU if available, parallelize over processes, or use an external free-tier embedding service if allowed.

2. **Index Build & Memory Footprint:**
   - Bottleneck: FAISS flat index size and query time.
   - Mitigation:
     - Switch to an IVF or HNSW index for approximate nearest neighbor search.
     - Shard the index by document type or time.

3. **Retrieval Latency:**
   - Bottleneck: Querying a very large index for each user question.
   - Mitigation:
     - Use ANN indices with tuned recall/latency trade-off.
     - Add lightweight keyword filtering before dense retrieval (hybrid retrieval).

4. **Storage & Persistence:**
   - Bottleneck: Storing embeddings and metadata.
   - Mitigation:
     - Move from local FAISS to a managed vector DB (e.g., Qdrant, Weaviate, pgvector) running on self-managed infrastructure.
