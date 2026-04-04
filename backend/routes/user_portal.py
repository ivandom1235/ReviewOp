from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException, Query, BackgroundTasks
from sqlalchemy import and_, case, desc, func, or_, select
from sqlalchemy.orm import Session

from core.db import get_db
from models.schemas import (
    AuthLoginIn,
    AuthLoginOut,
    AuthRegisterIn,
    AuthUserOut,
    ProductCardOut,
    ProductDetailOut,
    ProductReviewOut,
    ProductSuggestionOut,
    StarDistributionOut,
    SubmitReviewIn,
    SubmitReviewOut,
    AspectSummaryOut,
)
from models.tables import (
    Prediction,
    ProductCatalog,
    Review,
    User,
    UserProductReview,
    UserSession,
)
from services.review_pipeline import (
    refresh_corpus_graph,
    _refresh_corpus_graph_task,
    run_single_review_pipeline,
)


router = APIRouter(prefix="/user", tags=["user_portal"])
logger = logging.getLogger(__name__)

SESSION_HOURS = 24 * 7


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()


def _hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def seed_default_accounts(db: Session, defaults: list[dict[str, str]] | None = None) -> None:
    defaults = defaults or [
        {"username": "admin", "password": "12345", "role": "admin"},
        {"username": "user", "password": "12345", "role": "user"},
    ]
    changed = False
    for item in defaults:
        existing = db.query(User).filter(func.lower(User.username) == item["username"]).first()
        if existing:
            continue
        salt = secrets.token_hex(16)
        db.add(
            User(
                username=item["username"],
                password_salt=salt,
                password_hash=_hash_password((item.get("password") or "").strip(), salt),
                role=item["role"],
            )
        )
        changed = True
    if changed:
        db.commit()


def _issue_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(48)
    expiry = datetime.utcnow() + timedelta(hours=SESSION_HOURS)
    db.query(UserSession).filter(UserSession.expires_at <= datetime.utcnow()).delete(synchronize_session=False)
    db.add(UserSession(user_id=user.id, token=_hash_session_token(token), expires_at=expiry))
    db.commit()
    return token


def _user_out(user: User) -> AuthUserOut:
    return AuthUserOut(id=user.id, username=user.username, role=user.role)  # type: ignore[arg-type]


def _get_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix):].strip()
    return authorization.strip()


def get_current_user(
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
) -> User:
    token = _get_token(authorization)
    token_hash = _hash_session_token(token)
    session = (
        db.query(UserSession)
        .filter(and_(UserSession.token == token_hash, UserSession.expires_at > datetime.utcnow()))
        .first()
    )
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return session.user


def require_user(current: User = Depends(get_current_user)) -> User:
    if current.role != "user":
        raise HTTPException(status_code=403, detail="User role required")
    return current


def _ensure_products_seeded(db: Session) -> None:
    if db.query(ProductCatalog.id).first():
        return
    rows = (
        db.query(Review.product_id)
        .filter(Review.product_id.isnot(None))
        .distinct()
        .limit(40)
        .all()
    )
    for row in rows:
        pid = (row[0] or "").strip()
        if not pid:
            continue
        db.add(ProductCatalog(product_id=pid, name=f"Product {pid}", category="General", summary=""))
    db.commit()


@router.post("/auth/register", response_model=AuthUserOut)
def register(payload: AuthRegisterIn, db: Session = Depends(get_db)):
    username = payload.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="username must be at least 3 chars")
    existing = db.query(User).filter(func.lower(User.username) == username.lower()).first()
    if existing:
        raise HTTPException(status_code=409, detail="username already exists")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(payload.password, salt)
    user = User(username=username, password_hash=password_hash, password_salt=salt, role="user")
    db.add(user)
    db.commit()
    db.refresh(user)
    return _user_out(user)


@router.post("/auth/login", response_model=AuthLoginOut)
def login(payload: AuthLoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(func.lower(User.username) == payload.username.lower().strip()).first()
    if not user:
        raise HTTPException(status_code=401, detail="invalid username or password")
    if _hash_password(payload.password, user.password_salt) != user.password_hash:
        raise HTTPException(status_code=401, detail="invalid username or password")
    token = _issue_session(db, user)
    return AuthLoginOut(token=token, user=_user_out(user))


@router.get("/auth/me", response_model=AuthUserOut)
def me(current: User = Depends(get_current_user)):
    return _user_out(current)


def _product_query_base(db: Session):
    return db.query(
        ProductCatalog.product_id.label("product_id"),
        ProductCatalog.name.label("name"),
        ProductCatalog.category.label("category"),
        ProductCatalog.summary.label("summary"),
        ProductCatalog.cached_average_rating.label("avg_rating"),
        ProductCatalog.cached_review_count.label("review_count"),
        ProductCatalog.cached_latest_review_at.label("latest_review_at"),
    )


def _row_to_card(row) -> ProductCardOut:
    return ProductCardOut(
        product_id=row.product_id,
        name=row.name,
        category=row.category,
        summary=row.summary,
        average_rating=float(row.avg_rating or 0.0),
        review_count=int(row.review_count or 0),
        latest_review_at=row.latest_review_at.isoformat() if row.latest_review_at else None,
    )


@router.get("/products/suggestions", response_model=ProductSuggestionOut)
def product_suggestions(
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    _ensure_products_seeded(db)

    recently_ids = [
        pid
        for (pid,) in (
            db.query(UserProductReview.product_id)
            .filter(UserProductReview.user_id == current.id)
            .order_by(desc(UserProductReview.created_at))
            .limit(8)
            .all()
        )
    ]

    recent_cards = []
    if recently_ids:
        rows = _product_query_base(db).filter(ProductCatalog.product_id.in_(recently_ids)).all()
        recent_cards = [_row_to_card(r) for r in rows]

    similar_rows = (
        _product_query_base(db)
        .order_by(desc("review_count"), desc("avg_rating"), ProductCatalog.name.asc())
        .limit(12)
        .all()
    )
    similar_cards = [_row_to_card(r) for r in similar_rows if r.product_id not in recently_ids][:8]

    return ProductSuggestionOut(recently_reviewed=recent_cards, similar_products=similar_cards)


@router.get("/products/search", response_model=list[ProductCardOut])
def search_products(
    q: str = Query(default=""),
    min_rating: int = Query(default=0, ge=0, le=5),
    sort: str = Query(default="most_recent"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    _ensure_products_seeded(db)

    ml_review_pids = (
        db.query(Review.product_id)
        .outerjoin(ProductCatalog, ProductCatalog.product_id == Review.product_id)
        .filter(
            Review.product_id.isnot(None),
            Review.product_id != "",
            ProductCatalog.id.is_(None),
        )
        .distinct()
        .limit(500)
        .all()
    )
    new_pids = [row[0].strip() for row in ml_review_pids if row[0] and row[0].strip()]
    if new_pids:
        for pid in new_pids:
            db.add(ProductCatalog(product_id=pid, name=f"Product {pid}", category="General", summary=""))
        db.commit()

    query = _product_query_base(db)
    needle = q.strip()
    if needle:
        escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{escaped}%"
        ml_product_ids = (
            select(Review.product_id)
            .where(
                and_(
                    Review.product_id.isnot(None),
                    Review.text.ilike(like, escape="\\"),
                )
            )
            .distinct()
        )
        user_review_product_ids = (
            select(UserProductReview.product_id)
            .where(
                or_(
                    UserProductReview.title.ilike(like, escape="\\"),
                    UserProductReview.review_text.ilike(like, escape="\\"),
                )
            )
            .distinct()
        )
        query = query.filter(
            or_(
                ProductCatalog.product_id.ilike(like, escape="\\"),
                ProductCatalog.name.ilike(like, escape="\\"),
                ProductCatalog.product_id.in_(user_review_product_ids),
                ProductCatalog.product_id.in_(ml_product_ids),
            )
        )

    if min_rating and min_rating > 1:
        query = query.filter(ProductCatalog.cached_average_rating >= min_rating)

    sort_key = sort.strip().lower()
    if sort_key == "highest_rated":
        query = query.order_by(desc(ProductCatalog.cached_average_rating), desc(ProductCatalog.cached_review_count))
    elif sort_key == "lowest_rated":
        query = query.order_by(ProductCatalog.cached_average_rating, desc(ProductCatalog.cached_review_count))
    elif sort_key == "most_helpful":
        query = query.order_by(desc(ProductCatalog.cached_helpful_count), desc(ProductCatalog.cached_review_count))
    else:
        query = query.order_by(desc(ProductCatalog.cached_latest_review_at), desc(ProductCatalog.cached_review_count))

    return [_row_to_card(row) for row in query.offset(offset).limit(limit).all()]


@router.get("/products/{product_id}", response_model=ProductDetailOut)
def product_detail(
    product_id: str,
    _: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    product = db.query(ProductCatalog).filter(ProductCatalog.product_id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="product not found")

    # Combined query for stats and counts per star
    stats = (
        db.query(
            func.avg(UserProductReview.rating).label("avg_rating"),
            func.count(UserProductReview.id).label("total"),
            # Distribution aggregation
            func.count(case((UserProductReview.rating == 5, 1))).label("s5"),
            func.count(case((UserProductReview.rating == 4, 1))).label("s4"),
            func.count(case((UserProductReview.rating == 3, 1))).label("s3"),
            func.count(case((UserProductReview.rating == 2, 1))).label("s2"),
            func.count(case((UserProductReview.rating == 1, 1))).label("s1"),
        )
        .filter(UserProductReview.product_id == product_id)
        .first()
    )

    distribution = [
        StarDistributionOut(stars=5, count=int(stats.s5 or 0)),
        StarDistributionOut(stars=4, count=int(stats.s4 or 0)),
        StarDistributionOut(stars=3, count=int(stats.s3 or 0)),
        StarDistributionOut(stars=2, count=int(stats.s2 or 0)),
        StarDistributionOut(stars=1, count=int(stats.s1 or 0)),
    ]

    return ProductDetailOut(
        product_id=product.product_id,
        name=product.name,
        category=product.category,
        summary=product.summary,
        average_rating=float(stats.avg_rating or 0.0),
        review_count=int(stats.total or 0),
        star_distribution=distribution,
    )


@router.get("/products/{product_id}/reviews", response_model=list[ProductReviewOut])
def product_reviews(
    product_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="most_recent"),
    min_rating: int = Query(default=1, ge=1, le=5),
    _: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    query = (
        db.query(UserProductReview)
        .filter(and_(UserProductReview.product_id == product_id, UserProductReview.rating >= min_rating))
    )

    sort_key = sort.strip().lower()
    if sort_key == "highest_rated":
        query = query.order_by(desc(UserProductReview.rating), desc(UserProductReview.created_at))
    elif sort_key == "lowest_rated":
        query = query.order_by(UserProductReview.rating, desc(UserProductReview.created_at))
    elif sort_key == "most_helpful":
        query = query.order_by(desc(UserProductReview.helpful_count), desc(UserProductReview.created_at))
    else:
        query = query.order_by(desc(UserProductReview.created_at))

    rows = query.offset((page - 1) * page_size).limit(page_size).all()
    
    linked_review_ids = [row.linked_review_id for row in rows if row.linked_review_id]
    predictions_by_review = {}
    if linked_review_ids:
        all_preds = (
            db.query(Prediction.review_id, Prediction.aspect_cluster, Prediction.sentiment)
            .filter(Prediction.review_id.in_(linked_review_ids))
            .order_by(Prediction.review_id, desc(Prediction.confidence))
            .all()
        )
        for r_id, aspect, sentiment in all_preds:
            if r_id not in predictions_by_review:
                predictions_by_review[r_id] = []
            predictions_by_review[r_id].append((aspect, sentiment))

    out: list[ProductReviewOut] = []
    for row in rows:
        aspects: list[AspectSummaryOut] = []
        if row.linked_review_id and row.linked_review_id in predictions_by_review:
            preds = predictions_by_review[row.linked_review_id][:6]
            seen = set()
            for aspect, sentiment in preds:
                key = f"{aspect}:{sentiment}"
                if key not in seen:
                    seen.add(key)
                    aspects.append(AspectSummaryOut(aspect=aspect, sentiment=sentiment))
        out.append(
            ProductReviewOut(
                review_id=row.id,
                product_id=row.product_id,
                reviewer_name=row.user.username,
                rating=row.rating,
                review_title=row.title,
                review_text=row.review_text,
                review_date=row.created_at.isoformat(),
                helpful_count=row.helpful_count,
                aspects=aspects,
            )
        )
    return out


@router.post("/reviews", response_model=SubmitReviewOut)
def submit_review(
    payload: SubmitReviewIn,
    background_tasks: BackgroundTasks,
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    clean_product_id = (payload.product_id or "").strip()
    clean_product_name = (payload.product_name or "").strip()
    if not clean_product_id:
        raise HTTPException(status_code=400, detail="product_id is required")

    product = db.query(ProductCatalog).filter(ProductCatalog.product_id == clean_product_id).first()
    if not product:
        product = ProductCatalog(
            product_id=clean_product_id,
            name=clean_product_name or f"Product {clean_product_id}",
            category="General",
            summary="",
        )
        db.add(product)
        db.flush()
    elif clean_product_name and product.name != clean_product_name:
        product.name = clean_product_name
        product.updated_at = datetime.utcnow()

    from app import app as fastapi_app

    review = run_single_review_pipeline(
        db,
        engine=fastapi_app.state.seq2seq_engine,
        text=payload.review_text,
        domain=product.category,
        product_id=clean_product_id,
    )
    db.flush()

    user_review = UserProductReview(
        user_id=current.id,
        product_id=clean_product_id,
        rating=payload.rating,
        title=(payload.review_title or "").strip() or None,
        review_text=payload.review_text.strip(),
        pros=(payload.pros or "").strip() or None,
        cons=(payload.cons or "").strip() or None,
        recommendation=payload.recommendation,
        linked_review_id=review.id,
    )
    db.add(user_review)
    
    product.cached_review_count += 1
    old_count = product.cached_review_count - 1
    product.cached_average_rating = ((product.cached_average_rating * old_count) + payload.rating) / product.cached_review_count
    product.cached_latest_review_at = datetime.utcnow()

    db.commit()
    background_tasks.add_task(_refresh_corpus_graph_task, product.category)
    db.refresh(user_review)
    return SubmitReviewOut(review_id=user_review.id, product_id=clean_product_id, linked_review_id=review.id)


@router.get("/reviews/me", response_model=list[ProductReviewOut])
def my_reviews(
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(UserProductReview)
        .filter(UserProductReview.user_id == current.id)
        .order_by(desc(UserProductReview.created_at))
        .limit(200)
        .all()
    )

    linked_ids = [row.linked_review_id for row in rows if row.linked_review_id]
    preds_by_id = {}
    if linked_ids:
        all_preds = (
            db.query(Prediction.review_id, Prediction.aspect_cluster, Prediction.sentiment)
            .filter(Prediction.review_id.in_(linked_ids))
            .all()
        )
        for r_id, aspect, sentiment in all_preds:
            if r_id not in preds_by_id:
                preds_by_id[r_id] = []
            preds_by_id[r_id].append(AspectSummaryOut(aspect=aspect, sentiment=sentiment))

    return [
        ProductReviewOut(
            review_id=row.id,
            product_id=row.product_id,
            reviewer_name=current.username,
            rating=row.rating,
            review_title=row.title,
            review_text=row.review_text,
            review_date=row.created_at.isoformat(),
            helpful_count=row.helpful_count,
            aspects=preds_by_id.get(row.linked_review_id, [])[:6],
        )
        for row in rows
    ]


@router.put("/reviews/{review_id}", response_model=SubmitReviewOut)
def edit_review(
    review_id: int,
    payload: SubmitReviewIn,
    background_tasks: BackgroundTasks,
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    row = (
        db.query(UserProductReview)
        .filter(and_(UserProductReview.id == review_id, UserProductReview.user_id == current.id))
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="review not found")
    if row.product_id != payload.product_id:
        raise HTTPException(status_code=400, detail="product_id cannot be changed")

    product = db.query(ProductCatalog).filter(ProductCatalog.product_id == row.product_id).first()
    if product and product.cached_review_count > 0:
        total_rating = (product.cached_average_rating * product.cached_review_count) - row.rating + payload.rating
        product.cached_average_rating = total_rating / product.cached_review_count
        product.updated_at = datetime.utcnow()

    clean_text = payload.review_text.strip()
    from app import app as fastapi_app

    linked_review = None
    if row.linked_review_id:
        linked_review = db.query(Review).filter(Review.id == row.linked_review_id).first()
    linked_review = run_single_review_pipeline(
        db,
        engine=fastapi_app.state.seq2seq_engine,
        text=clean_text,
        domain=product.category if product else None,
        product_id=row.product_id,
        review=linked_review,
        replace_existing=True,
    )

    row.linked_review_id = linked_review.id
    row.rating = payload.rating
    row.title = (payload.review_title or "").strip() or None
    row.review_text = clean_text
    row.pros = (payload.pros or "").strip() or None
    row.cons = (payload.cons or "").strip() or None
    row.recommendation = payload.recommendation
    row.updated_at = datetime.utcnow()
    db.commit()
    background_tasks.add_task(_refresh_corpus_graph_task, linked_review.domain)
    return SubmitReviewOut(review_id=row.id, product_id=row.product_id, linked_review_id=row.linked_review_id)


@router.delete("/reviews/{review_id}")
def delete_review(
    review_id: int,
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    row = (
        db.query(UserProductReview)
        .filter(and_(UserProductReview.id == review_id, UserProductReview.user_id == current.id))
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="review not found")
        
    product = db.query(ProductCatalog).filter(ProductCatalog.product_id == row.product_id).first()
    if product:
        if product.cached_review_count <= 1:
            product.cached_review_count = 0
            product.cached_average_rating = 0.0
            product.cached_helpful_count = 0
        else:
            total_rating = (product.cached_average_rating * product.cached_review_count) - row.rating
            product.cached_review_count -= 1
            product.cached_average_rating = total_rating / product.cached_review_count
            product.cached_helpful_count -= row.helpful_count

    linked_id = row.linked_review_id
    db.delete(row)
    if linked_id:
        linked_r = db.query(Review).filter(Review.id == linked_id).first()
        if linked_r:
            db.delete(linked_r)

    db.commit()
    return {"ok": True}
