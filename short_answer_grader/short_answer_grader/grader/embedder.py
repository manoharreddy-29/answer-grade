"""Embedding back-ends.

* SentenceTransformerEmbedder - real semantic model (needs `sentence-transformers` + model download).
* LexicalEmbedder - dependency-free fallback (word + char n-gram hashing). Handles spelling variants
  but NOT true paraphrase; use it only for demos/tests.
"""
import hashlib
import math
from .preprocess import normalize


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class LexicalEmbedder:
    name = "lexical-fallback"
    semantic = False
    DIM = 2048

    def encode(self, text: str):
        v = [0.0] * self.DIM
        t = normalize(text)
        feats = [("w", w) for w in t.split()]
        padded = f" {t} "
        feats += [("c", padded[i:i + 4]) for i in range(len(padded) - 3)]
        for kind, f in feats:
            h = int(hashlib.md5(f"{kind}:{f}".encode()).hexdigest(), 16)
            v[h % self.DIM] += 2.0 if kind == "w" else 1.0
        return v


class SentenceTransformerEmbedder:
    semantic = True

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self._m = SentenceTransformer(model_name)

    def encode(self, text: str):
        return self._m.encode(text or "", normalize_embeddings=True).tolist()


def get_embedder(prefer_semantic=True):
    if prefer_semantic:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return LexicalEmbedder()
