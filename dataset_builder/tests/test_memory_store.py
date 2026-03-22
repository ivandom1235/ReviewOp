from __future__ import annotations

import unittest
from pathlib import Path
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(ROOT))

from memory_store import AspectMemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_upsert_resolve_increment_and_promotion(self) -> None:
        tmp = Path(__file__).resolve().parents[1] / "_tmp_memory_store"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            store = AspectMemoryStore(tmp)
            store.upsert_term("battery", "battery_life", "electronics", confidence=0.84)
            store.increment_hit("battery", "electronics")
            store.increment_hit("battery", "electronics")
            resolved = store.resolve_term("battery", "electronics")
            self.assertEqual(resolved.canonical_aspect, "battery_life")
            self.assertGreaterEqual(resolved.hit_count, 2)
            candidates = store.list_candidates_for_promotion(min_hits=2, min_confidence=0.8)
            self.assertTrue(any(row["term"] == "battery" for row in candidates))
            promo = store.promote_term("battery", "battery_life", "manual review", approved_by="tester", domain="electronics")
            self.assertEqual(promo["new_label"], "battery_life")
            store.write_snapshot()
            store.write_promotions()
            self.assertTrue((tmp / "aspect_memory_terms.jsonl").exists())
            self.assertTrue((tmp / "aspect_memory_promotions.jsonl").exists())

            reloaded = AspectMemoryStore(tmp)
            resolved2 = reloaded.resolve_term("battery", "electronics")
            self.assertEqual(resolved2.canonical_aspect, "battery_life")
            self.assertGreaterEqual(resolved2.hit_count, 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_read_only_store_blocks_writes(self) -> None:
        tmp = Path(__file__).resolve().parents[1] / "_tmp_memory_store_ro"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            store = AspectMemoryStore(tmp, read_only=True)
            with self.assertRaises(RuntimeError):
                store.upsert_term("battery", "battery_life", "electronics")
            with self.assertRaises(RuntimeError):
                store.write_snapshot()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
