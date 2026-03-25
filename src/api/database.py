import hashlib
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

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
        print(f"[log_prediction] ERROR — model={model_type} label={label} prob={probability}: {exc}")


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
