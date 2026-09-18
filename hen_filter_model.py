import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class HenVoiceFilter:
    """
    Stage 1 Gatekeeper: Accurately identifies whether an incoming audio recording
    is actually a hen's vocalization vs human speech, barking dogs, traffic, or noise.
    """
    def __init__(self, n_estimators: int = 150):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        )
        self.isolation_forest = IsolationForest(
            n_estimators=150,
            contamination=0.03,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, X_hen: np.ndarray, X_non_hen: np.ndarray):
        """
        Fits both the supervised discriminator and the one-class anomaly detector.
        """
        # Supervised binary dataset: 1 = Hen Vocalization, 0 = Non-Hen Audio
        y_hen = np.ones(len(X_hen), dtype=int)
        y_non_hen = np.zeros(len(X_non_hen), dtype=int)

        X_all = np.vstack([X_hen, X_non_hen])
        y_all = np.concatenate([y_hen, y_non_hen])

        X_scaled = self.scaler.fit_transform(X_all)

        print(f"Training Stage 1 Gatekeeper with {len(X_hen)} hen samples and {len(X_non_hen)} non-hen samples...")
        self.classifier.fit(X_scaled, y_all)

        # Fit Isolation Forest on hen samples only for boundary calibration
        X_hen_scaled = self.scaler.transform(X_hen)
        self.isolation_forest.fit(X_hen_scaled)

        self.is_fitted = True
        print("Stage 1 Gatekeeper training complete.")

    def predict(self, feature_vector: np.ndarray):
        """
        Evaluates a single 1D feature vector.
        Returns:
            dict with:
                is_hen: bool
                hen_probability: float (0.0 to 1.0)
                anomaly_score: float
                status: "APPROVED" | "REJECTED"
                reason: str
        """
        if not self.is_fitted:
            raise RuntimeError("HenVoiceFilter is not fitted yet.")

        feat_scaled = self.scaler.transform(feature_vector.reshape(1, -1))

        probs = self.classifier.predict_proba(feat_scaled)[0]
        # Class 1 is Hen, Class 0 is Non-Hen
        hen_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        
        # Decision score from isolation forest (> 0 means inlier, < 0 means anomaly)
        iso_score = float(self.isolation_forest.decision_function(feat_scaled)[0])

        # Dual condition: High supervised confidence + reasonable acoustic structure
        is_hen = (hen_prob >= 0.50) and (iso_score > -0.15)

        if is_hen:
            reason = "Acoustic signature matches domestic poultry vocalization profile."
            status = "APPROVED"
        else:
            if hen_prob < 0.35:
                reason = "Acoustic pattern strongly indicates non-hen audio (human, environmental noise, or other animal sound)."
            else:
                reason = "Audio signal diverges significantly from domestic chicken vocalization dynamics."
            status = "REJECTED"

        return {
            "is_hen": bool(is_hen),
            "hen_probability": round(hen_prob, 4),
            "anomaly_score": round(iso_score, 4),
            "status": status,
            "reason": reason
        }

    def save(self, filepath: str = "ood_detector.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "classifier": self.classifier,
                "isolation_forest": self.isolation_forest
            }, f)
        print(f"HenVoiceFilter successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "ood_detector.pkl"):
        instance = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            instance.scaler = data["scaler"]
            instance.classifier = data["classifier"]
            instance.isolation_forest = data["isolation_forest"]
            instance.is_fitted = True
        return instance
