"""SQLAlchemy ORM, prediction logging, statistics, and drift detection for the API."""

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

_logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

if DATABASE_URL.startswith("sqlite"):
    # Local dev: SQLite — disable pooling (avoids thread-safety issues with file locks)
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    # Production: PostgreSQL (Neon / Cloud SQL)
    # pool_pre_ping tests connections before use — critical for Cloud Run cold starts
    # pool_recycle drops connections older than 5 min to avoid "server closed connection" errors
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=10,
    )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class PredictionLog(Base):
    """Persisted record of a single /predict call. Text is never stored — only its hash."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_hash = Column(String, nullable=False, index=True)
    model_type = Column(String, nullable=False)
    label = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(engine)


def hash_text(text: str) -> str:
    """SHA-256 of normalised text — anonymises the input per project policy."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def log_prediction(text: str, model_type: str, label: int, probability: float, risk_level: str) -> None:
    """Persist one prediction entry. Intended to run as a FastAPI BackgroundTask."""
    try:
        with SessionLocal() as session:
            session.add(
                PredictionLog(
                    text_hash=hash_text(text),
                    model_type=model_type,
                    label=int(label),
                    probability=float(probability),
                    risk_level=risk_level,
                )
            )
            session.commit()
    except Exception as exc:
        _logger.error("log_prediction failed — model=%s label=%s prob=%s: %s", model_type, label, probability, exc)


_DRIFT_THRESHOLD = 0.05  # Flag if 7-day mean confidence deviates more than 5% from baseline


def get_drift() -> dict:
    """Compute confidence drift: 7-day rolling mean vs all-time baseline.

    Returns a dict with baseline_confidence, recent_confidence, delta,
    drift_detected flag, and per-model breakdown for the last 7 days.
    """
    with SessionLocal() as session:
        baseline_conf = float(session.scalar(select(func.avg(PredictionLog.probability))) or 0.0)
        baseline_distress_rate = 0.0
        total = session.scalar(select(func.count()).select_from(PredictionLog)) or 0
        if total > 0:
            distress = session.scalar(select(func.count()).select_from(PredictionLog).where(PredictionLog.label == 1)) or 0
            baseline_distress_rate = round(distress / total, 3)

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        recent_conf = float(session.scalar(select(func.avg(PredictionLog.probability)).where(PredictionLog.created_at >= cutoff)) or 0.0)
        recent_total = session.scalar(select(func.count()).select_from(PredictionLog).where(PredictionLog.created_at >= cutoff)) or 0
        recent_distress_rate = 0.0
        if recent_total > 0:
            recent_distress = (
                session.scalar(select(func.count()).select_from(PredictionLog).where(PredictionLog.created_at >= cutoff, PredictionLog.label == 1))
                or 0
            )
            recent_distress_rate = round(recent_distress / recent_total, 3)

        model_drift_rows = session.execute(
            select(PredictionLog.model_type, func.avg(PredictionLog.probability).label("avg_conf"))
            .where(PredictionLog.created_at >= cutoff)
            .group_by(PredictionLog.model_type)
        ).all()
        model_drift = {row.model_type: round(float(row.avg_conf), 3) for row in model_drift_rows}

    delta = round(recent_conf - baseline_conf, 3)
    return {
        "baseline_confidence": round(baseline_conf, 3),
        "recent_confidence": round(recent_conf, 3),
        "confidence_delta": delta,
        "drift_detected": abs(delta) > _DRIFT_THRESHOLD,
        "drift_threshold": _DRIFT_THRESHOLD,
        "baseline_distress_rate": baseline_distress_rate,
        "recent_distress_rate": recent_distress_rate,
        "recent_predictions_count": recent_total,
        "model_confidence_7d": model_drift,
    }


def get_stats() -> dict:
    """Return aggregated statistics used by GET /stats."""
    with SessionLocal() as session:
        total = session.scalar(select(func.count()).select_from(PredictionLog)) or 0
        distress = session.scalar(select(func.count()).select_from(PredictionLog).where(PredictionLog.label == 1)) or 0

        risk_rows = session.execute(select(PredictionLog.risk_level, func.count().label("cnt")).group_by(PredictionLog.risk_level)).all()
        risk_level_counts = {row.risk_level: row.cnt for row in risk_rows}

        model_rows = session.execute(select(PredictionLog.model_type, func.count().label("cnt")).group_by(PredictionLog.model_type)).all()
        model_usage = {row.model_type: row.cnt for row in model_rows}

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        day_rows = session.execute(
            select(
                func.date(PredictionLog.created_at).label("day"),
                func.count().label("cnt"),
            )
            .where(PredictionLog.created_at >= cutoff)
            .group_by(func.date(PredictionLog.created_at))
            .order_by(func.date(PredictionLog.created_at))
        ).all()
        predictions_by_day = [{"date": str(row.day), "count": row.cnt} for row in day_rows]

        avg_confidence = float(session.scalar(select(func.avg(PredictionLog.probability))) or 0.0)

        distress_model_rows = session.execute(
            select(PredictionLog.model_type, func.count().label("cnt")).where(PredictionLog.label == 1).group_by(PredictionLog.model_type)
        ).all()
        distress_by_model = {row.model_type: row.cnt for row in distress_model_rows}

    return {
        "total_predictions": total,
        "distress_count": distress,
        "no_distress_count": total - distress,
        "risk_level_counts": risk_level_counts,
        "model_usage": model_usage,
        "predictions_by_day": predictions_by_day,
        "avg_confidence": round(avg_confidence, 3),
        "distress_by_model": distress_by_model,
    }
