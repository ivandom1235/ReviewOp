from __future__ import annotations

import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.bootstrap import apply_schema_patches


class BootstrapSchemaPatchTests(unittest.TestCase):
    def test_apply_schema_patches_adds_reliability_columns(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        CREATE TABLE reviews (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            text TEXT NOT NULL,
                            domain VARCHAR(64) NULL,
                            product_id VARCHAR(128) NULL,
                            created_at DATETIME NOT NULL
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE TABLE predictions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            review_id INTEGER NOT NULL,
                            aspect_raw VARCHAR(255) NOT NULL,
                            aspect_cluster VARCHAR(255) NOT NULL,
                            sentiment VARCHAR(16) NOT NULL,
                            confidence FLOAT NOT NULL DEFAULT 0.5
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE TABLE abstained_predictions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            review_id INTEGER NOT NULL,
                            reason VARCHAR(128) NOT NULL,
                            confidence FLOAT NOT NULL DEFAULT 0.0,
                            ambiguity_score FLOAT NOT NULL DEFAULT 0.0,
                            created_at DATETIME NOT NULL
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE TABLE novel_candidates (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            review_id INTEGER NOT NULL,
                            aspect VARCHAR(255) NOT NULL,
                            novelty_score FLOAT NOT NULL DEFAULT 0.0,
                            confidence FLOAT NULL,
                            evidence TEXT NULL,
                            evidence_start INTEGER NULL,
                            evidence_end INTEGER NULL,
                            created_at DATETIME NOT NULL
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE TABLE admin_dismissed_alerts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            type VARCHAR(64) NOT NULL,
                            aspect VARCHAR(255) NOT NULL,
                            message VARCHAR(512) NOT NULL,
                            domain VARCHAR(64) NULL,
                            signature VARCHAR(64) NOT NULL UNIQUE,
                            dismissed_at DATETIME NOT NULL
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE TABLE rejected_aspect_candidates (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            review_id INTEGER NOT NULL,
                            raw_text VARCHAR(255) NOT NULL,
                            normalized_text VARCHAR(255) NOT NULL,
                            reason VARCHAR(128) NOT NULL,
                            quality_score FLOAT NOT NULL DEFAULT 0.0,
                            evidence_text TEXT NULL,
                            source_rule VARCHAR(64) NULL,
                            created_at DATETIME NOT NULL
                        )
                        """
                    )
                )

            apply_schema_patches(engine)

            inspector = inspect(engine)
            self.assertIn("contradiction_score", {col["name"] for col in inspector.get_columns("predictions")})
            self.assertIn("contradiction_types", {col["name"] for col in inspector.get_columns("predictions")})
            self.assertIn("quarantine_status", {col["name"] for col in inspector.get_columns("predictions")})
            self.assertIn("contradiction_score", {col["name"] for col in inspector.get_columns("abstained_predictions")})
            self.assertIn("contradiction_types", {col["name"] for col in inspector.get_columns("abstained_predictions")})
            self.assertIn("quarantine_status", {col["name"] for col in inspector.get_columns("abstained_predictions")})
            self.assertIn("contradiction_score", {col["name"] for col in inspector.get_columns("novel_candidates")})
            self.assertIn("contradiction_types", {col["name"] for col in inspector.get_columns("novel_candidates")})
            self.assertIn("quarantine_status", {col["name"] for col in inspector.get_columns("novel_candidates")})
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
