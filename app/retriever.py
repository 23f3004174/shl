import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class Retriever:

    def __init__(self):
        self.model = None
        self.index = None
        self.catalog: List[Dict[str, Any]] = []

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.model = None

        try:
            import faiss
            self._faiss = faiss
        except Exception:
            self._faiss = None

        try:
            index_path = DATA_DIR / "faiss.index"
            if self._faiss is not None and index_path.exists():
                self.index = self._faiss.read_index(str(index_path))
        except Exception:
            self.index = None

        try:
            catalog_path = DATA_DIR / "catalog_processed.json"
            if catalog_path.exists():
                with catalog_path.open("r", encoding="utf-8") as f:
                    self.catalog = json.load(f)
        except Exception:
            self.catalog = []

    def search(self, query: str, top_k: int = 10):
        if self.model is None or self.index is None or not self.catalog:
            return []

        try:
            import numpy as np

            embedding = self.model.encode(
                [query],
                convert_to_numpy=True
            ).astype("float32")

            distances, indices = self.index.search(embedding, top_k)
        except Exception:
            return []

        results: List[Dict[str, Any]] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            if 0 <= idx < len(self.catalog):
                results.append(self.catalog[idx])

        return results


_retriever: Optional[Retriever] = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
