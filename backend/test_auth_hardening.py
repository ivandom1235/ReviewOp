from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

warnings.simplefilter("ignore", DeprecationWarning)

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.db import Base
from models.tables import User
from routes.user_portal import assert_no_default_passwords, seed_default_accounts
from services.auth import IdentityManager


class AuthHardeningTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_demo_users_are_not_seeded_without_explicit_flag(self) -> None:
        db = self.make_db()

        seed_default_accounts(db, app_env="dev", seed_demo_users=False)

        self.assertEqual(db.query(User).count(), 0)

    def test_demo_users_can_seed_when_explicitly_enabled(self) -> None:
        db = self.make_db()

        seed_default_accounts(db, app_env="dev", seed_demo_users=True)

        self.assertEqual(db.query(User).count(), 2)

    def test_production_rejects_default_demo_passwords(self) -> None:
        db = self.make_db()
        manager = IdentityManager()
        salt = "fixedsalt"
        db.add(User(username="admin", password_salt=salt, password_hash=manager.hash_password("12345", salt), role="admin"))
        db.commit()

        with self.assertRaisesRegex(RuntimeError, "default demo password"):
            assert_no_default_passwords(db, app_env="production")

    def test_password_hash_is_versioned_and_verifiable(self) -> None:
        manager = IdentityManager()
        salt = "fixedsalt"
        password_hash = manager.hash_password("correct horse", salt)

        self.assertTrue(password_hash.startswith("pbkdf2_sha256$"))
        self.assertTrue(manager.verify_password("correct horse", salt, password_hash))
        self.assertFalse(manager.verify_password("wrong", salt, password_hash))


if __name__ == "__main__":
    unittest.main()
