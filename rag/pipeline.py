import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from .embeddings import EmbeddingClient
    from .generate import LLMClient
    from .vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "faiss_index"


@dataclass
class SourceChunk:
    document: str
    page: int
    chunk: str
    score: float


class RAGPipeline:
    def __init__(
        self,
        embedding_client: "EmbeddingClient",
        vector_store: "VectorStore",
        llm_client: "LLMClient",
        top_k: int = 8,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def query(self, question: str) -> Dict[str, Any]:
        # 1) Embed question
        q_emb = self.embedding_client.embed_text(question)

        # 2) Retrieve candidate chunks
        results = self.vector_store.search(q_emb, top_k=self.top_k)
        # results: List[SourceChunk]

        if not results:
            return {
                "answer": "I do not have enough information in the documents to answer this question.",
                "sources": [],
                "confidence": 0.0,
            }

        # 3) Compute a simple confidence score based on retrieval scores
        scores = [r.score for r in results]
        max_score = max(scores)
        avg_top3 = sum(scores[:3]) / min(3, len(scores))
        confidence = float((max_score + avg_top3) / 2.0)

        if max_score < 0.4:  # heuristic threshold
            return {
                "answer": "I do not have enough information in the documents to answer this question.",
                "sources": [
                    {
                        "document": r.document,
                        "page": r.page,
                        "chunk": r.chunk,
                    }
                    for r in results[:3]
                ],
                "confidence": confidence,
            }

        # 4) Build context string
        context_lines = []
        for idx, r in enumerate(results):
            context_lines.append(
                f"[DOC={r.document} | PAGE={r.page} | RANK={idx+1}]\n{r.chunk}\n"
            )
        context = "\n\n".join(context_lines)

        # 5) Call LLM
        system_prompt = (
            "You are a legal contracts assistant. "
            "Answer ONLY using the provided contract context. "
            "If the context does not contain enough information to answer confidently, "
            "respond with: 'I do not have enough information in the documents to answer this question.' "
            "Always quote the relevant clause and keep answers concise."
        )

        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )

        answer = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)

        # 6) Prepare sources
        sources = [
            {
                "document": r.document,
                "page": r.page,
                "chunk": r.chunk,
            }
            for r in results[:3]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }


def main() -> None:
    from .embeddings import EmbeddingClient
    from .generate import LLMClient
    from .vector_store import VectorStore

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    embedding_client = EmbeddingClient.from_env()
    vector_store = VectorStore.from_disk(str(INDEX_DIR))
    llm_client = LLMClient.from_env()

    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store,
        llm_client=llm_client,
    )

    if args.demo:
        while True:
            try:
                question = input("Question (or 'quit'): ")
            except EOFError:
                break
            if not question or question.lower() == "quit":
                break
            result = pipeline.query(question=question)
            print("\nAnswer:", result["answer"])
            print("Confidence:", result["confidence"])
            print("Sources:")
            for s in result["sources"]:
                print(f"- {s['document']} (page {s['page']})")
            print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
