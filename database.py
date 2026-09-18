"""
AvianGuard AI — MongoDB Database Layer
Handles persistent document storage of all bioacoustic diagnostic records,
spectrogram results, and disease surveillance logs using MongoDB Atlas.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "avian_guard")
COLLECTION_NAME = "diagnostic_records"

_client = None
_db = None
_collection = None


def get_db():
    """Lazily initializes and returns the MongoDB database instance."""
    global _client, _db, _collection
    if _db is not None:
        return _db

    uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
    if not uri:
        # No MongoDB connection string provided yet
        return None

    try:
        from pymongo import MongoClient
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _db = _client[DB_NAME]
        _collection = _db[COLLECTION_NAME]
        return _db
    except Exception as e:
        print(f"[MONGODB] Connection initialization error: {e}")
        return None


def get_collection():
    """Returns the diagnostic_records collection."""
    global _collection
    if _collection is not None:
        return _collection
    get_db()
    return _collection


def init_db():
    """
    Initializes connection to MongoDB Atlas, validates ping,
    and sets up query performance indexes.
    """
    global _client
    uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
    if not uri:
        print("[MONGODB] Notice: MONGODB_URI not set in .env. Set your MongoDB Atlas connection string.")
        return

    try:
        from pymongo import MongoClient, DESCENDING
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Verify connection with ping command
        _client.admin.command("ping")
        db = _client[DB_NAME]
        col = db[COLLECTION_NAME]

        # Create optimized query indexes
        col.create_index([("timestamp", DESCENDING)])
        col.create_index([("prediction", 1)])
        col.create_index([("is_alert", 1)])
        col.create_index([("source", 1)])
        col.create_index([("user_identifier", 1)])

        print(f"[MONGODB] Connected successfully to MongoDB Atlas database '{DB_NAME}' with indexes.")
    except Exception as e:
        print(f"[MONGODB] Warning: Could not connect to MongoDB Atlas ({e}). Operations will retry when URI is configured.")


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
) -> Optional[Dict[str, Any]]:
    """Inserts a new diagnostic document into the MongoDB collection."""
    col = get_collection()
    if col is None:
        return None

    probs = prob_dict or {}
    record = {
        "timestamp": datetime.utcnow(),
        "source": source,
        "user_identifier": str(user_identifier) if user_identifier else None,
        "filename": filename,
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "probabilities": {
            "Healthy": round(float(probs.get("Healthy", 0.0)), 4),
            "Unhealthy": round(float(probs.get("Unhealthy", 0.0)), 4),
            "Noise": round(float(probs.get("Noise", 0.0)), 4),
        },
        "is_alert": bool(is_alert or (prediction == "Unhealthy")),
        "clinical_note": clinical_note,
        "duration_sec": round(float(duration_sec), 2),
        "segments_analyzed": int(segments_analyzed)
    }

    try:
        res = col.insert_one(record)
        record["_id"] = str(res.inserted_id)
        record["timestamp"] = record["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return record
    except Exception as e:
        print(f"[MONGODB] Failed to insert diagnostic document: {e}")
        return None


def get_recent_records(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Fetches recent diagnostic records sorted by newest first."""
    col = get_collection()
    if col is None:
        return []

    from pymongo import DESCENDING
    try:
        cursor = col.find().sort("timestamp", DESCENDING).skip(offset).limit(limit)
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            if isinstance(doc.get("timestamp"), datetime):
                doc["timestamp"] = doc["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            results.append(doc)
        return results
    except Exception as e:
        print(f"[MONGODB] Query error: {e}")
        return []


def get_diagnostic_stats() -> Dict[str, Any]:
    """Calculates aggregate surveillance metrics using MongoDB aggregation."""
    col = get_collection()
    if col is None:
        return {
            "total_diagnoses": 0,
            "healthy_count": 0,
            "unhealthy_count": 0,
            "noise_count": 0,
            "rejected_non_hen_count": 0,
            "total_alerts": 0,
            "infection_rate_pct": 0.0
        }

    try:
        total = col.count_documents({})
        healthy = col.count_documents({"prediction": "Healthy"})
        unhealthy = col.count_documents({"prediction": "Unhealthy"})
        noise = col.count_documents({"prediction": "Noise"})
        rejected = col.count_documents({"prediction": "Rejected"})
        alerts = col.count_documents({"is_alert": True})

        return {
            "total_diagnoses": total,
            "healthy_count": healthy,
            "unhealthy_count": unhealthy,
            "noise_count": noise,
            "rejected_non_hen_count": rejected,
            "total_alerts": alerts,
            "infection_rate_pct": round((unhealthy / total * 100), 2) if total > 0 else 0.0
        }
    except Exception as e:
        print(f"[MONGODB] Stats aggregation error: {e}")
        return {
            "total_diagnoses": 0,
            "healthy_count": 0,
            "unhealthy_count": 0,
            "noise_count": 0,
            "rejected_non_hen_count": 0,
            "total_alerts": 0,
            "infection_rate_pct": 0.0
        }


if __name__ == "__main__":
    init_db()
    stats = get_diagnostic_stats()
    print("MongoDB initial stats check:", stats)
