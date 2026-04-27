from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pdfplumber  # or pymupdf; keep it consistent in requirements.txt

from .embeddings import EmbeddingClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "faiss_index"


@dataclass
class RawChunk:
    document: str
    page: int
    chunk_index: int
    text: str


def extract_text_from_pdf(pdf_path: Path) -> List[RawChunk]:
    """
    Extract text from a single PDF into page-level RawChunk objects.
    We then re-chunk by token length below.
    """
    raw_chunks: List[RawChunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            raw_chunks.append(
                RawChunk(
                    document=pdf_path.name,
                    page=page_idx,
                    chunk_index=0,
                    text=text,
                )
            )
    return raw_chunks


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer. For production, replace with a
    tokenizer matching your embedding model's tokenization.
    """
    return text.split()


def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def make_overlapping_chunks(
    raw_chunks: List[RawChunk],
    target_tokens: int = 1000,
    overlap_tokens: int = 200,
) -> List[RawChunk]:
    """
    Turn page-level RawChunk objects into overlapping token-length chunks.
    """
    final_chunks: List[RawChunk] = []
    for rc in raw_chunks:
        tokens = tokenize(rc.text)
        if not tokens:
            continue

        start = 0
        chunk_idx = 0
        while start < len(tokens):
            end = start + target_tokens
            slice_tokens = tokens[start:end]
            if not slice_tokens:
                break
            text_chunk = detokenize(slice_tokens)
            final_chunks.append(
                RawChunk(
                    document=rc.document,
                    page=rc.page,
                    chunk_index=chunk_idx,
                    text=text_chunk,
                )
            )
            chunk_idx += 1
            # Overlap
            start += target_tokens - overlap_tokens

    return final_chunks


def build_index(chunks: List[RawChunk], embedding_client: EmbeddingClient, out_dir: Path) -> None:
    """
    Embed chunks, build a FAISS index, and save metadata to disk.
    """
    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedding_client.embed_batch(texts)  # shape: (N, D)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = out_dir / "index.faiss"
    meta_path = out_dir / "chunks_metadata.npy"

    print(f"Writing FAISS index to {faiss_path} ...")
    faiss.write_index(index, str(faiss_path))

    # store metadata as a numpy array of dicts
    meta = np.array(
        [
            {
                "document": c.document,
                "page": c.page,
                "chunk": c.text,
            }
            for c in chunks
        ],
        dtype=object,
    )
    print(f"Writing chunk metadata to {meta_path} ...")
    np.save(meta_path, meta, allow_pickle=True)

    print("Ingestion complete.")


def main() -> None:
    pdf_dir = PDF_DIR
    assert pdf_dir.exists(), f"{pdf_dir} does not exist. Add at least 3 sample PDFs."

    pdf_paths = sorted(p for p in pdf_dir.glob("*.pdf"))
    assert pdf_paths, "No PDFs found in data/pdfs. Add at least 3 sample legal PDFs."

    all_raw_chunks: List[RawChunk] = []
    for pdf_path in pdf_paths:
        print(f"Extracting from {pdf_path} ...")
        all_raw_chunks.extend(extract_text_from_pdf(pdf_path))

    print(f"Extracted {len(all_raw_chunks)} page-level chunks. Re-chunking...")
    all_chunks = make_overlapping_chunks(all_raw_chunks)

    print(f"Total final chunks: {len(all_chunks)}")

    embedding_client = EmbeddingClient.from_env()
    build_index(all_chunks, embedding_client, INDEX_DIR)


if __name__ == "__main__":
    main()
