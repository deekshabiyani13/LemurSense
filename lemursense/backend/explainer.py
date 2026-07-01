"""
explainer.py
-------------
Explainable AI layer. Wraps a trained HostAnomalyModel with a SHAP
explainer so that every anomaly flag comes with a human-readable
"why" -- which features (metrics/volatility/time-of-day) pushed the
anomaly score up, and by how much.

We use shap.Explainer with a permutation algorithm over the model's
decision_function. This is model-agnostic (works regardless of the
underlying sklearn estimator) and robust across shap versions, which
matters since IsolationForest is not always fully supported by
shap's tree-specific fast path.
"""

import numpy as np
import pandas as pd
import shap

from anomaly_detector import build_features


class HostExplainer:
    def __init__(self, host_model, background_df: pd.DataFrame, background_size: int = 50):
        self.host_model = host_model
        bg_feats = build_features(background_df)[host_model.feature_columns]
        if len(bg_feats) > background_size:
            bg_feats = bg_feats.sample(background_size, random_state=42)
        self.background = bg_feats

        def score_fn(X):
            raw = -self.host_model.model.decision_function(X)
            return raw

        self.explainer = shap.Explainer(
            score_fn, self.background, algorithm="permutation"
        )

    def explain_row(self, df_window: pd.DataFrame, row_index: int, max_evals: int = 200):
        """
        df_window: dataframe (single host, sorted by time) that includes
        the row we want to explain plus enough history for rolling features.
        row_index: positional index (0-based) within df_window to explain.
        """
        feats = build_features(df_window)[self.host_model.feature_columns]
        target_row = feats.iloc[[row_index]]

        shap_values = self.explainer(target_row, max_evals=max_evals)

        contributions = list(zip(
            self.host_model.feature_columns,
            shap_values.values[0].tolist(),
            target_row.values[0].tolist(),
        ))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "base_value": float(np.array(shap_values.base_values[0]).item()),
            "score": float(shap_values.base_values[0] + shap_values.values[0].sum()),
            "contributions": [
                {"feature": f, "shap_value": round(v, 4), "raw_value": round(rv, 3)}
                for f, v, rv in contributions
            ],
        }
