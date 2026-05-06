from __future__ import annotations
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is available in the repo environment, but keep a fallback.
    np = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
except ImportError:
    TfidfVectorizer = None
    sklearn_cosine_similarity = None

class EvidenceClusterer:
    def __init__(self, threshold: float = 0.72):
        self.threshold = threshold
        self._model = None

    @property
    def model(self):
        if self._model is None and SentenceTransformer is not None:
            try:
                # Using a small, fast model for cluster discovery when available locally.
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self._model = None
        return self._model

    @property
    def has_embedding_model(self) -> bool:
        return self.model is not None and util is not None

    @staticmethod
    def _normalize(pattern: str) -> str:
        return " ".join(str(pattern or "").lower().split())

    def is_similar(self, pattern_a: str, pattern_b: str) -> float:
        a = self._normalize(pattern_a)
        b = self._normalize(pattern_b)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        if self.has_embedding_model:
            try:
                emb_a = self.model.encode(a, convert_to_tensor=True, show_progress_bar=False)
                emb_b = self.model.encode(b, convert_to_tensor=True, show_progress_bar=False)
                return float(util.cos_sim(emb_a, emb_b).item())
            except Exception:
                pass

        if TfidfVectorizer is not None and sklearn_cosine_similarity is not None:
            try:
                vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
                matrix = vectorizer.fit_transform([a, b])
                return float(sklearn_cosine_similarity(matrix[0], matrix[1])[0][0])
            except Exception:
                pass

        return self._jaccard_similarity(a, b)

    def get_similarity_score(self, pattern_a: str, pattern_b: str) -> float:
        return self.is_similar(pattern_a, pattern_b)

    def mean_similarity(self, patterns: list[str]) -> float:
        cleaned = [self._normalize(p) for p in patterns if self._normalize(p)]
        if len(cleaned) < 2:
            return 1.0 if cleaned else 0.0

        if self.has_embedding_model:
            try:
                embeddings = self.model.encode(cleaned, convert_to_tensor=True, show_progress_bar=False)
                matrix = util.cos_sim(embeddings, embeddings)
                return self._mean_upper_triangle(matrix.cpu().numpy() if hasattr(matrix, "cpu") else matrix)
            except Exception:
                pass

        if TfidfVectorizer is not None and sklearn_cosine_similarity is not None:
            try:
                vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
                matrix = vectorizer.fit_transform(cleaned)
                sims = sklearn_cosine_similarity(matrix)
                return self._mean_upper_triangle(sims)
            except Exception:
                pass

        scores = [self._jaccard_similarity(cleaned[i], cleaned[j]) for i in range(len(cleaned)) for j in range(i + 1, len(cleaned))]
        return sum(scores) / len(scores) if scores else 0.0

    def _mean_upper_triangle(self, matrix: Any) -> float:
        if np is None:
            values = []
            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix)):
                    values.append(float(matrix[i][j]))
            return sum(values) / len(values) if values else 0.0
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return 0.0
        values = arr[np.triu_indices(arr.shape[0], k=1)]
        return float(values.mean()) if values.size else 0.0

    @staticmethod
    def _jaccard_similarity(pattern_a: str, pattern_b: str) -> float:
        set_a = set(pattern_a.split())
        set_b = set(pattern_b.split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def find_best_cluster(self, pattern: str, existing_clusters: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not existing_clusters:
            return None
            
        best_score = -1.0
        best_cluster = None
        pattern_norm = self._normalize(pattern)
        
        for cluster in existing_clusters:
            cluster_patterns = [p for p in cluster.get("trigger_patterns", []) if self._normalize(p)]
            rep = cluster.get("representative_pattern") or (cluster_patterns[0] if cluster_patterns else "")
            candidates = cluster_patterns or ([rep] if rep else [])
            score = max((self.get_similarity_score(pattern_norm, candidate) for candidate in candidates), default=0.0)
            if score >= self.threshold and score > best_score:
                best_score = score
                best_cluster = cluster
                
        return best_cluster
