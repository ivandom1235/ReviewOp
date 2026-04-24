from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi import Response
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.db import Base
from models.schemas import AuthLoginIn
from routes.user_portal import get_current_user, login
from services.auth import IdentityManager


class CookieAuthTests(unittest.TestCase):
    def make_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        self.addCleanup(engine.dispose)
        self.addCleanup(db.close)
        return db

    def test_login_sets_httponly_session_cookie(self) -> None:
        db = self.make_db()
        IdentityManager().register_user(db, "alice", "password1")
        response = Response()

        payload = login(AuthLoginIn(username="alice", password="password1"), response, db)

        self.assertTrue(payload.token)
        cookie = response.headers.get("set-cookie", "")
        self.assertIn("reviewop_session=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite", cookie)

    def test_current_user_accepts_session_cookie_without_authorization_header(self) -> None:
        db = self.make_db()
        manager = IdentityManager()
        user = manager.register_user(db, "alice", "password1")
        token = manager.issue_session(db, user)

        current = get_current_user(db=db, authorization=None, session_cookie=token)

        self.assertEqual(current.username, "alice")


if __name__ == "__main__":
    unittest.main()
