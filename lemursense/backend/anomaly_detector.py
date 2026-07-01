"""
anomaly_detector.py
--------------------
Per-host unsupervised anomaly detection using Isolation Forest.

Each host gets its own model since "normal" for a database VM (high
disk I/O baseline) looks very different from "normal" for a cache VM.
This mirrors the biomimicry idea of "territory": each host has its own
learned baseline range, and the model flags departures from that
specific territory rather than a single global threshold.

Features used per row:
  - raw metric values (cpu_pct, mem_pct, disk_io_mbps, net_io_mbps)
  - rolling std (short window) per metric, as a volatility signal
  - hour-of-day encoded as sin/cos (circadian signal)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_METRICS = ["cpu_pct", "mem_pct", "disk_io_mbps", "net_io_mbps"]
ROLL_WINDOW = 6  # 6 * 5min = 30 minutes of context


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must be sorted by timestamp and belong to a single host.
    Returns a feature dataframe aligned to df's index.
    """
    feats = pd.DataFrame(index=df.index)
    for m in FEATURE_METRICS:
        feats[m] = df[m]
        feats[f"{m}_roll_std"] = (
            df[m].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0.0)
        )
    hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    feats["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    return feats


class HostAnomalyModel:
    def __init__(self, host: str, contamination: float = 0.03, random_state: int = 42):
        self.host = host
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        self.feature_columns = None

    def fit(self, df: pd.DataFrame):
        feats = build_features(df)
        self.feature_columns = feats.columns.tolist()
        self.model.fit(feats.values)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly scores normalized roughly to [0, 1], where
        higher = more anomalous. sklearn's decision_function gives
        higher = more normal, so we invert and min-max scale.
        """
        feats = build_features(df)[self.feature_columns]
        raw = -self.model.decision_function(feats.values)  # higher = more anomalous
        lo, hi = raw.min(), raw.max()
        if hi - lo < 1e-9:
            return np.zeros_like(raw)
        return (raw - lo) / (hi - lo)

    def predict_flags(self, df: pd.DataFrame) -> np.ndarray:
        feats = build_features(df)[self.feature_columns]
        preds = self.model.predict(feats.values)  # -1 = anomaly, 1 = normal
        return preds == -1


class AnomalyEngine:
    """Owns one HostAnomalyModel per host."""

    def __init__(self):
        self.models: dict[str, HostAnomalyModel] = {}

    def fit_all(self, df: pd.DataFrame):
        for host, group in df.groupby("host"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            self.models[host] = HostAnomalyModel(host).fit(group)
        return self

    def score_host(self, host: str, df_host: pd.DataFrame) -> np.ndarray:
        return self.models[host].score(df_host)

    def flags_host(self, host: str, df_host: pd.DataFrame) -> np.ndarray:
        return self.models[host].predict_flags(df_host)
