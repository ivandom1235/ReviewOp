from __future__ import annotations
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Tuple, Optional

from sqlalchemy import func, and_
from sqlalchemy.orm import Session
from models.tables import User, UserSession

class IdentityManager:
    """
    Deep module encapsulating authentication, password hashing, 
    and session lifecycle management.
    """
    SESSION_HOURS = 24 * 7

    def hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256", 
            password.encode("utf-8"), 
            salt.encode("utf-8"), 
            120_000
        ).hex()

    def hash_session_token(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def register_user(self, db: Session, username: str, password: str, role: str = "user") -> User:
        username = username.strip()
        salt = secrets.token_hex(16)
        password_hash = self.hash_password(password, salt)
        
        user = User(
            username=username, 
            password_hash=password_hash, 
            password_salt=salt, 
            role=role
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def authenticate_user(self, db: Session, username: str, password: str) -> Tuple[User, str]:
        user = db.query(User).filter(func.lower(User.username) == username.lower().strip()).first()
        if not user:
            return None, None
            
        if self.hash_password(password, user.password_salt) != user.password_hash:
            return None, None
            
        token = self.issue_session(db, user)
        return user, token

    def issue_session(self, db: Session, user: User) -> str:
        token = secrets.token_urlsafe(48)
        expiry = datetime.utcnow() + timedelta(hours=self.SESSION_HOURS)
        
        # Cleanup expired sessions
        db.query(UserSession).filter(UserSession.expires_at <= datetime.utcnow()).delete(synchronize_session=False)
        
        db.add(UserSession(
            user_id=user.id, 
            token=self.hash_session_token(token), 
            expires_at=expiry
        ))
        db.commit()
        return token

    def verify_session(self, db: Session, token: str) -> Optional[User]:
        token_hash = self.hash_session_token(token)
        session = (
            db.query(UserSession)
            .filter(and_(
                UserSession.token == token_hash, 
                UserSession.expires_at > datetime.utcnow()
            ))
            .first()
        )
        if not session:
            return None
        return session.user
