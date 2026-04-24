from __future__ import annotations
import hashlib
import sqlite3
import threading
from pathlib import Path

class LLMDiskCache:
    def __init__(self, cache_file: str = ".llm_cache.db"):
        self.cache_file = Path(cache_file)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.cache_file, check_same_thread=False) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        model TEXT,
                        prompt TEXT,
                        response TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()

    def _get_key(self, prompt: str, model: str) -> str:
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, prompt: str, model: str) -> str | None:
        key = self._get_key(prompt, model)
        try:
            with sqlite3.connect(self.cache_file, check_same_thread=False) as conn:
                cursor = conn.execute("SELECT response FROM cache WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def set(self, prompt: str, model: str, response: str) -> None:
        key = self._get_key(prompt, model)
        try:
            with self._lock:
                with sqlite3.connect(self.cache_file, check_same_thread=False) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, model, prompt, response) VALUES (?, ?, ?, ?)",
                        (key, model, prompt, response)
                    )
                    conn.commit()
        except Exception:
            pass
