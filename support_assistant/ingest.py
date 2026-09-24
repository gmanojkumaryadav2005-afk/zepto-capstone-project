"""
Module 3 - Support Assistant: Ingestion
Loads the 8 policy docs, chunks them, embeds with all-MiniLM-L6-v2
(sentence-transformers, local, free), and stores the embeddings in a
persistent ChromaDB collection.

Run:
    python ingest.py
"""
import glob
import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400  # characters; docs here are short so one chunk/doc is typical


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def build_index():
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, texts, metadatas = [], [], []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for i, chunk in enumerate(chunk_text(content)):
            ids.append(f"{doc_id}_chunk{i}")
            texts.append(chunk)
            metadatas.append({"doc_id": doc_id, "chunk_index": i})

    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {len(texts)} chunks from {len(set(m['doc_id'] for m in metadatas))} documents into '{COLLECTION_NAME}'")
    return collection


if __name__ == "__main__":
    build_index()
