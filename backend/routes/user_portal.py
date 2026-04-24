from __future__ import annotations

import secrets
from datetime import datetime
import os

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Query, BackgroundTasks, Request, Response
from sqlalchemy import and_, case, desc, func, or_, select
from sqlalchemy.orm import Session, aliased, selectinload

from core.db import get_db
from models.schemas import (
    AuthLoginIn,
    AuthLoginOut,
    AuthRegisterIn,
    AuthUserOut,
    JobStatusOut,
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
    Job,
    JobItem,
    User,
    UserProductReview,
)
from services.auth import IdentityManager
from services.hybrid_pipeline import run_single_review_hybrid_pipeline
from services.review_jobs import create_review_analysis_job, schedule_review_analysis_job
from services.review_pipeline import _refresh_corpus_graph_task


router = APIRouter(prefix="/user", tags=["user_portal"])

SESSION_HOURS = 24 * 7


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def assert_no_default_passwords(db: Session, app_env: str | None = None) -> None:
    env = str(app_env or os.getenv("REVIEWOP_ENV") or os.getenv("APP_ENV") or "dev").lower()
    if env in {"dev", "demo", "local"}:
        return
    manager = IdentityManager()
    for username in ("admin", "user"):
        existing = db.query(User).filter(func.lower(User.username) == username).first()
        if existing and manager.verify_password("12345", existing.password_salt, existing.password_hash):
            raise RuntimeError(f"default demo password is not allowed in {env}: {username}")


def seed_default_accounts(
    db: Session,
    defaults: list[dict[str, str]] | None = None,
    *,
    app_env: str | None = None,
    seed_demo_users: bool | None = None,
) -> None:
    env = str(app_env or os.getenv("REVIEWOP_ENV") or os.getenv("APP_ENV") or "dev").lower()
    if seed_demo_users is None:
        seed_demo_users = _env_flag("SEED_DEMO_USERS") or _env_flag("REVIEWOP_SEED_DEMO_USERS")
    assert_no_default_passwords(db, env)
    if not seed_demo_users:
        return
    if env not in {"dev", "demo", "local"}:
        raise RuntimeError("demo users can only be seeded in dev, demo, or local environments")
    defaults = defaults or [
        {"username": "admin", "password": "12345", "role": "admin"},
        {"username": "user", "password": "12345", "role": "user"},
    ]
    changed = False
    for item in defaults:
        username = (item.get("username") or "").strip()
        if not username:
            continue
        existing = db.query(User).filter(func.lower(User.username) == username.lower()).first()
        if existing:
            continue
        salt = secrets.token_hex(16)
        db.add(
            User(
                username=username,
                password_salt=salt,
                password_hash=IdentityManager().hash_password((item.get("password") or "").strip(), salt),
                role=item["role"],
            )
        )
        changed = True
    if changed:
        db.commit()


def _user_out(user: User) -> AuthUserOut:
    return AuthUserOut(id=user.id, username=user.username, role=user.role)  # type: ignore[arg-type]


SESSION_COOKIE_NAME = "reviewop_session"


def _get_token(authorization: str | None, session_cookie: str | None = None) -> str:
    if session_cookie:
        return session_cookie.strip()
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix):].strip()
    return authorization.strip()


def get_current_user(
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> User:
    token = _get_token(authorization, session_cookie)
    user = IdentityManager().verify_session(db, token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


def require_user(current: User = Depends(get_current_user)) -> User:
    if current.role != "user":
        raise HTTPException(status_code=403, detail="User role required")
    return current


def require_admin(current: User = Depends(get_current_user)) -> User:
    if current.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
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

    user = IdentityManager().register_user(db, username, payload.password)
    return _user_out(user)


@router.post("/auth/login", response_model=AuthLoginOut)
def login(payload: AuthLoginIn, response: Response, db: Session = Depends(get_db)):
    user, token = IdentityManager().authenticate_user(db, payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid username or password")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=IdentityManager.SESSION_HOURS * 3600,
    )
    return AuthLoginOut(token=token, user=_user_out(user))


@router.get("/auth/me", response_model=AuthUserOut)
def me(current: User = Depends(get_current_user)):
    return _user_out(current)


@router.get("/jobs/{job_id}", response_model=JobStatusOut)
def get_my_review_job(
    job_id: str,
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    item = db.query(JobItem).filter(JobItem.job_id == job.id, JobItem.row_index == 0).first()
    if not item or item.review_id is None:
        raise HTTPException(status_code=404, detail="Job not found")
    owned = (
        db.query(UserProductReview.id)
        .filter(
            UserProductReview.user_id == current.id,
            UserProductReview.linked_review_id == item.review_id,
            _active_review_filter(),
        )
        .first()
    )
    if not owned:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusOut(
        job_id=job.id,
        status=job.status,
        total=job.total,
        processed=job.processed,
        failed=job.failed,
        error=job.error,
    )


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


def _row_to_product_review_out(row, aspects: list[AspectSummaryOut], reply_title: str | None = None) -> ProductReviewOut:
    return ProductReviewOut(
        review_id=row.id,
        product_id=row.product_id,
        reviewer_name=row.user.username,
        rating=row.rating,
        review_title=row.title,
        review_text=row.review_text,
        review_date=row.created_at.isoformat(),
        helpful_count=row.helpful_count,
        aspects=aspects,
        reply_to_review_id=row.reply_to_review_id,
        reply_to_review_title=reply_title,
        is_reply=bool(row.reply_to_review_id),
    )


def _active_review_filter():
    return UserProductReview.deleted_at.is_(None)


def _apply_cached_product_delta(
    product: ProductCatalog,
    *,
    rating_delta: float = 0.0,
    review_delta: int = 0,
    helpful_delta: int = 0,
    latest_review_at: datetime | None = None,
) -> None:
    current_reviews = int(product.cached_review_count or 0)
    next_reviews = max(0, current_reviews + review_delta)

    if next_reviews <= 0:
        product.cached_review_count = 0
        product.cached_average_rating = 0.0
        product.cached_helpful_count = 0
        product.cached_latest_review_at = latest_review_at or product.cached_latest_review_at
        return

    current_total_rating = float(product.cached_average_rating or 0.0) * current_reviews
    product.cached_review_count = next_reviews
    product.cached_average_rating = (current_total_rating + rating_delta) / next_reviews
    product.cached_helpful_count = max(0, int(product.cached_helpful_count or 0) + helpful_delta)
    if latest_review_at is not None:
        product.cached_latest_review_at = latest_review_at


def _recompute_product_cache(db: Session, product: ProductCatalog) -> None:
    stats = (
        db.query(
            func.avg(UserProductReview.rating).label("avg_rating"),
            func.count(UserProductReview.id).label("total"),
            func.sum(UserProductReview.helpful_count).label("helpful"),
            func.max(UserProductReview.created_at).label("latest"),
        )
        .filter(UserProductReview.product_id == product.product_id, _active_review_filter())
        .first()
    )
    product.cached_average_rating = float(stats.avg_rating or 0.0)
    product.cached_review_count = int(stats.total or 0)
    product.cached_helpful_count = int(stats.helpful or 0)
    product.cached_latest_review_at = stats.latest
    product.updated_at = datetime.utcnow()


def _recompute_product_cache_for_review(db: Session, review: UserProductReview | None) -> None:
    if not review:
        return
    product = db.query(ProductCatalog).filter(ProductCatalog.product_id == review.product_id).first()
    if product:
        _recompute_product_cache(db, product)


def _lock_reply_context(
    clean_product_id: str,
    clean_product_name: str,
    parent_review: UserProductReview | None,
) -> tuple[str, str]:
    if parent_review is None:
        return clean_product_id, clean_product_name
    return parent_review.product_id, parent_review.title or parent_review.product_id


@router.get("/products/suggestions", response_model=ProductSuggestionOut)
def product_suggestions(
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    _ensure_products_seeded(db)

    recently_ids = [
        pid.strip()
        for (pid,) in (
            db.query(UserProductReview.product_id)
            .filter(UserProductReview.user_id == current.id, _active_review_filter())
            .order_by(desc(UserProductReview.created_at))
            .limit(8)
            .all()
        )
        if pid and pid.strip()
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
                _active_review_filter(),
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

    if min_rating and min_rating > 0:
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
    _recompute_product_cache(db, product)

    stats = (
        db.query(
            func.count(case((UserProductReview.rating == 5, 1))).label("s5"),
            func.count(case((UserProductReview.rating == 4, 1))).label("s4"),
            func.count(case((UserProductReview.rating == 3, 1))).label("s3"),
            func.count(case((UserProductReview.rating == 2, 1))).label("s2"),
            func.count(case((UserProductReview.rating == 1, 1))).label("s1"),
        )
        .filter(UserProductReview.product_id == product_id)
        .filter(_active_review_filter())
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
        average_rating=float(product.cached_average_rating or 0.0),
        review_count=int(product.cached_review_count or 0),
        star_distribution=distribution,
    )


@router.get("/products/{product_id}/reviews", response_model=list[ProductReviewOut])
def product_reviews(
    product_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="most_recent"),
    min_rating: int = Query(default=1, ge=1, le=5),
    aspect: str | None = Query(default=None),
    _: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    query = (
        db.query(UserProductReview)
        .options(selectinload(UserProductReview.user))
        .filter(and_(UserProductReview.product_id == product_id, UserProductReview.rating >= min_rating, _active_review_filter()))
    )
    clean_aspect = (aspect or "").strip()
    if clean_aspect:
        parent = aliased(UserProductReview)
        matching_review_ids = (
            select(Prediction.review_id)
            .where(
                or_(
                    Prediction.aspect_cluster == clean_aspect,
                    Prediction.aspect_raw == clean_aspect,
                )
            )
        )
        query = (
            query.outerjoin(parent, parent.id == UserProductReview.reply_to_review_id)
            .filter(
                or_(
                    UserProductReview.linked_review_id.in_(matching_review_ids),
                    parent.linked_review_id.in_(matching_review_ids),
                )
            )
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
    parent_user_review_ids = [row.reply_to_review_id for row in rows if row.reply_to_review_id]
    parent_linked_review_by_id: dict[int, int] = {}
    if parent_user_review_ids:
        parent_rows = (
            db.query(UserProductReview.id, UserProductReview.linked_review_id)
            .filter(UserProductReview.id.in_(parent_user_review_ids), _active_review_filter())
            .all()
        )
        parent_linked_review_by_id = {int(parent_id): int(linked_id) for parent_id, linked_id in parent_rows if linked_id}
    parent_review_ids = list(parent_linked_review_by_id.values())
    predictions_by_review = {}
    if linked_review_ids or parent_review_ids:
        all_ids = [*linked_review_ids, *parent_review_ids]
        all_preds = (
            db.query(Prediction.review_id, Prediction.aspect_cluster, Prediction.sentiment)
            .filter(Prediction.review_id.in_(all_ids))
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
        origin_review_id = parent_linked_review_by_id.get(row.reply_to_review_id) if row.reply_to_review_id else row.linked_review_id
        if origin_review_id and origin_review_id in predictions_by_review:
            preds = predictions_by_review[origin_review_id][:6]
            seen = set()
            for aspect, sentiment in preds:
                key = f"{aspect}:{sentiment}"
                if key not in seen:
                    seen.add(key)
                    aspects.append(AspectSummaryOut(aspect=aspect, sentiment=sentiment))
        reply_title = None
        if row.reply_to_review_id:
            parent = db.query(UserProductReview.title).filter(UserProductReview.id == row.reply_to_review_id, _active_review_filter()).first()
            reply_title = parent[0] if parent else None
        out.append(_row_to_product_review_out(row, aspects, reply_title=reply_title))
    return out


@router.post("/reviews", response_model=SubmitReviewOut)
def submit_review(
    payload: SubmitReviewIn,
    background_tasks: BackgroundTasks,
    request: Request,
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    clean_product_id = (payload.product_id or "").strip()
    clean_product_name = (payload.product_name or "").strip()
    reply_to_review_id = payload.reply_to_review_id
    if not clean_product_id:
        raise HTTPException(status_code=400, detail="product_id is required")

    parent_review = None
    if reply_to_review_id is not None:
        parent_review = db.query(UserProductReview).filter(UserProductReview.id == int(reply_to_review_id), _active_review_filter()).first()
        if not parent_review:
            raise HTTPException(status_code=404, detail="parent review not found")
        clean_product_id, clean_product_name = _lock_reply_context(clean_product_id, clean_product_name, parent_review)

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

    review = Review(text=payload.review_text.strip(), domain=product.category, product_id=clean_product_id)
    db.add(review)
    db.flush()
    job = create_review_analysis_job(db, review)
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
        reply_to_review_id=parent_review.id if parent_review else None,
    )
    db.add(user_review)
    _apply_cached_product_delta(
        product,
        rating_delta=float(payload.rating),
        review_delta=1,
        latest_review_at=datetime.utcnow(),
    )

    db.commit()
    background_tasks.add_task(
        schedule_review_analysis_job,
        job_id=job.id,
        explicit_engine=request.app.state.seq2seq_engine,
        implicit_client=request.app.state.implicit_client,
    )
    db.refresh(user_review)
    return SubmitReviewOut(
        review_id=user_review.id,
        product_id=clean_product_id,
        linked_review_id=review.id,
        reply_to_review_id=parent_review.id if parent_review else None,
        job_id=job.id,
        analysis_status=job.status,
    )


@router.get("/reviews/me", response_model=list[ProductReviewOut])
def my_reviews(
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(UserProductReview)
        .filter(UserProductReview.user_id == current.id, _active_review_filter())
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
            recommendation=row.recommendation,
            aspects=preds_by_id.get(row.linked_review_id, [])[:6],
            reply_to_review_id=row.reply_to_review_id,
            is_reply=bool(row.reply_to_review_id),
        )
        for row in rows
    ]


@router.get("/reviews/{review_id}", response_model=ProductReviewOut)
def get_review(
    review_id: int,
    _: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    row = (
        db.query(UserProductReview)
        .options(selectinload(UserProductReview.user))
        .filter(UserProductReview.id == review_id, _active_review_filter())
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="review not found")

    origin_review_id = row.linked_review_id
    if row.reply_to_review_id:
        parent = (
            db.query(UserProductReview.linked_review_id)
            .filter(UserProductReview.id == row.reply_to_review_id, _active_review_filter())
            .first()
        )
        origin_review_id = parent[0] if parent and parent[0] else row.linked_review_id
    aspects: list[AspectSummaryOut] = []
    if origin_review_id:
        preds = (
            db.query(Prediction.aspect_cluster, Prediction.sentiment)
            .filter(Prediction.review_id == origin_review_id)
            .order_by(desc(Prediction.confidence))
            .limit(6)
            .all()
        )
        aspects = [AspectSummaryOut(aspect=aspect, sentiment=sentiment) for aspect, sentiment in preds]

    reply_title = None
    if row.reply_to_review_id:
        parent = db.query(UserProductReview.title).filter(UserProductReview.id == row.reply_to_review_id, _active_review_filter()).first()
        reply_title = parent[0] if parent else None
    return _row_to_product_review_out(row, aspects, reply_title=reply_title)


@router.put("/reviews/{review_id}", response_model=SubmitReviewOut)
def edit_review(
    review_id: int,
    payload: SubmitReviewIn,
    background_tasks: BackgroundTasks,
    request: Request,
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    row = (
        db.query(UserProductReview)
        .filter(and_(UserProductReview.id == review_id, UserProductReview.user_id == current.id, _active_review_filter()))
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="review not found")
    if row.product_id != payload.product_id:
        raise HTTPException(status_code=400, detail="product_id cannot be changed")
    if payload.reply_to_review_id != row.reply_to_review_id:
        raise HTTPException(status_code=400, detail="reply_to_review_id cannot be changed")

    product = db.query(ProductCatalog).filter(ProductCatalog.product_id == row.product_id).first()

    clean_text = payload.review_text.strip()

    linked_review = None
    if row.linked_review_id:
        linked_review = db.query(Review).filter(Review.id == row.linked_review_id).first()
    linked_review, _, _, _ = run_single_review_hybrid_pipeline(
        db,
        explicit_engine=request.app.state.seq2seq_engine,
        implicit_client=request.app.state.implicit_client,
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
    if product:
        _recompute_product_cache(db, product)
    db.commit()
    background_tasks.add_task(_refresh_corpus_graph_task, linked_review.domain)
    return SubmitReviewOut(
        review_id=row.id,
        product_id=row.product_id,
        linked_review_id=row.linked_review_id,
        reply_to_review_id=row.reply_to_review_id,
    )


@router.delete("/reviews/{review_id}")
def delete_review(
    review_id: int,
    current: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    row = (
        db.query(UserProductReview)
        .filter(and_(UserProductReview.id == review_id, UserProductReview.user_id == current.id, _active_review_filter()))
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="review not found")
        
    product = db.query(ProductCatalog).filter(ProductCatalog.product_id == row.product_id).first()
    if product:
        _apply_cached_product_delta(
            product,
            rating_delta=-float(row.rating),
            review_delta=-1,
            helpful_delta=-int(row.helpful_count or 0),
        )

    row.deleted_at = datetime.utcnow()
    row.updated_at = row.deleted_at
    _recompute_product_cache_for_review(db, row)

    db.commit()
    return {"ok": True}
