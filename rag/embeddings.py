import os
from typing import List
from dotenv import load_dotenv


import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

class EmbeddingClient:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    @classmethod
    def from_env(cls) -> "EmbeddingClient":
        # For this assignment, prefer a local model to avoid paid APIs.
        # Example: all-MiniLM-L6-v2 (CPU-friendly).
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        return cls(model)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
