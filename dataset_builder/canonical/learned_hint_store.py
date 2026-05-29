from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

class LearnedHintStore:
    """
    Persistent store for learned aspect clusters.
    Replaces static domain-biased hint dictionaries.
    """
    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self.clusters: Dict[str, Dict] = {}
        self.load()

    def get_cluster(self, cluster_id: str) -> Optional[Dict]:
        return self.clusters.get(cluster_id)

    def add_cluster(self, cluster_id: str, data: Dict):
        self.clusters[cluster_id] = data
        self.save()

    def save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(self.clusters, indent=2), encoding="utf-8")

    def load(self):
        if not self.storage_path.exists():
            return
        try:
            self.clusters = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            self.clusters = {}
