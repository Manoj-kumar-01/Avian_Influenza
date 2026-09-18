"""
AvianGuard AI — Relational Database Layer
Handles persistent storage of all bioacoustic diagnostic records and disease surveillance logs.
Supports SQLite by default and PostgreSQL in production via the DATABASE_URL environment variable.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, desc
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# Resolve Database URL (SQLite default, PostgreSQL in cloud deployment)
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avian_guard.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Fix Heroku / Render postgres:// vs postgresql:// scheme
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()


class DiagnosticRecord(Base):
    """Stores full bioacoustic inference results and veterinary assessments."""
    __tablename__ = "diagnostic_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String(50), default="api", index=True)  # 'web_upload', 'telegram', 'url', 'monitor'
    user_identifier = Column(String(100), nullable=True)     # Telegram Chat ID, IP hash, or farm identifier
    filename = Column(String(255), nullable=True)
    prediction = Column(String(50), index=True)              # 'Healthy', 'Unhealthy', 'Noise', 'Rejected'
    confidence = Column(Float, default=0.0)
    prob_healthy = Column(Float, default=0.0)
    prob_unhealthy = Column(Float, default=0.0)
    prob_noise = Column(Float, default=0.0)
    is_alert = Column(Boolean, default=False, index=True)
    clinical_note = Column(Text, nullable=True)
    duration_sec = Column(Float, default=0.0)
    segments_analyzed = Column(Integer, default=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else None,
            "source": self.source,
            "user_identifier": self.user_identifier,
            "filename": self.filename,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4) if self.confidence else 0.0,
            "probabilities": {
                "Healthy": round(self.prob_healthy, 4) if self.prob_healthy else 0.0,
                "Unhealthy": round(self.prob_unhealthy, 4) if self.prob_unhealthy else 0.0,
                "Noise": round(self.prob_noise, 4) if self.prob_noise else 0.0,
            },
            "is_alert": self.is_alert,
            "clinical_note": self.clinical_note,
            "duration_sec": round(self.duration_sec, 2) if self.duration_sec else 0.0,
            "segments_analyzed": self.segments_analyzed
        }


def init_db():
    """Initializes tables in the target database."""
    Base.metadata.create_all(bind=engine)
    print(f"[DATABASE] Initialized database storage ({DATABASE_URL.split('://')[0]}).")


def save_diagnostic_record(
    prediction: str,
    confidence: float,
    prob_dict: Optional[Dict[str, float]] = None,
    source: str = "api",
    user_identifier: Optional[str] = None,
    filename: Optional[str] = None,
    clinical_note: Optional[str] = None,
    duration_sec: float = 0.0,
    segments_analyzed: int = 0,
    is_alert: bool = False
) -> DiagnosticRecord:
    """Inserts a new diagnostic inference record into the database."""
    session = SessionLocal()
    try:
        probs = prob_dict or {}
        record = DiagnosticRecord(
            source=source,
            user_identifier=str(user_identifier) if user_identifier else None,
            filename=filename,
            prediction=prediction,
            confidence=confidence,
            prob_healthy=probs.get("Healthy", 0.0),
            prob_unhealthy=probs.get("Unhealthy", 0.0),
            prob_noise=probs.get("Noise", 0.0),
            is_alert=is_alert or (prediction == "Unhealthy"),
            clinical_note=clinical_note,
            duration_sec=duration_sec,
            segments_analyzed=segments_analyzed,
            timestamp=datetime.utcnow()
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    except Exception as e:
        session.rollback()
        print(f"[DATABASE] Failed to save record: {e}")
        return None
    finally:
        session.close()


def get_recent_records(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Fetches recent diagnostic history."""
    session = SessionLocal()
    try:
        records = session.query(DiagnosticRecord).order_by(desc(DiagnosticRecord.timestamp)).offset(offset).limit(limit).all()
        return [r.to_dict() for r in records]
    finally:
        session.close()


def get_diagnostic_stats() -> Dict[str, Any]:
    """Computes high-level surveillance statistics for dashboards and reports."""
    session = SessionLocal()
    try:
        total = session.query(DiagnosticRecord).count()
        unhealthy = session.query(DiagnosticRecord).filter(DiagnosticRecord.prediction == "Unhealthy").count()
        healthy = session.query(DiagnosticRecord).filter(DiagnosticRecord.prediction == "Healthy").count()
        rejected = session.query(DiagnosticRecord).filter(DiagnosticRecord.prediction == "Rejected").count()
        noise = session.query(DiagnosticRecord).filter(DiagnosticRecord.prediction == "Noise").count()
        alerts = session.query(DiagnosticRecord).filter(DiagnosticRecord.is_alert == True).count()

        return {
            "total_diagnoses": total,
            "healthy_count": healthy,
            "unhealthy_count": unhealthy,
            "noise_count": noise,
            "rejected_non_hen_count": rejected,
            "total_alerts": alerts,
            "infection_rate_pct": round((unhealthy / total * 100), 2) if total > 0 else 0.0
        }
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    stats = get_diagnostic_stats()
    print("Database initial check stats:", stats)
