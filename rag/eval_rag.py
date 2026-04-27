import json
from pathlib import Path

from .embeddings import EmbeddingClient
from .vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "faiss_index"
EVAL_PATH = PROJECT_ROOT / "data" / "rag_qa_eval.jsonl"


def load_eval_questions(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main() -> None:
    embedding_client = EmbeddingClient.from_env()
    vector_store = VectorStore.from_disk(str(INDEX_DIR))
    # For retrieval-only evaluation, LLM is not strictly needed, but we construct the pipeline for API consistency.

    questions = list(load_eval_questions(str(EVAL_PATH)))

    hits = 0
    total = len(questions)

    for q in questions:
        question = q["question"]
        expected_doc = q["expected_doc"]
        expected_page = q["expected_page"]

        q_emb = embedding_client.embed_text(question)
        results = vector_store.search(q_emb, top_k=3)

        hit = any(
            r.document == expected_doc and r.page == expected_page for r in results
        )
        if hit:
            hits += 1

        print(f"Q: {question}")
        print(f"Expected: {expected_doc} (page {expected_page}) | Hit: {hit}")
        for r in results:
            print(f"  - {r.document} (page {r.page}) score={r.score:.3f}")
        print("-")

    precision_at_3 = hits / total if total else 0.0
    print(f"Precision@3: {precision_at_3:.3f}")


if __name__ == "__main__":
    main()
