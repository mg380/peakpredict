"""B6 — the peak-age predictor ladder.

Baseline-first: ``GroupMeanBaseline`` (predict the event+sex mean peak age) is
the floor every other model must beat. ``PooledRidge`` is the v1 working model —
a regularized regression over the engineered features with event/sex one-hot
(partial pooling via regularization). Both expose ``predict_one`` so the
dashboard can score an uploaded athlete. More expressive models (MixedLM,
gradient boosting, sequence models) are pluggable here but must beat these on
the temporal split (see evaluate.py) to be adopted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, PHYSICAL_NAMES

ENGINEERED = list(FEATURE_NAMES)       # always present, no missing
PHYSICAL = list(PHYSICAL_NAMES)        # height/weight, often missing -> imputed
NUMERIC = ENGINEERED + PHYSICAL
CATEGORICAL = ["event_id", "sex"]
TARGET = "peak_age"
Z80 = 1.2816  # ~80% normal interval


class GroupMeanBaseline:
    """Predict the mean peak age for the athlete's (event, sex) group."""

    def __init__(self) -> None:
        self.group_means: dict[tuple, float] = {}
        self.global_mean: float = 0.0
        self.resid_std: float = 1.0

    def fit(self, df: pd.DataFrame, calibrate: bool = True) -> GroupMeanBaseline:
        self.global_mean = float(df[TARGET].mean())
        self.group_means = {
            (e, int(s)): float(m)
            for (e, s), m in df.groupby(["event_id", "sex"])[TARGET].mean().items()
        }
        preds = np.array(
            [self._mean(e, int(s)) for e, s in zip(df["event_id"], df["sex"], strict=False)]
        )
        self.resid_std = float((df[TARGET].to_numpy() - preds).std(ddof=0)) or 1.0
        return self

    def _mean(self, event_id: str, sex: int) -> float:
        return self.group_means.get((event_id, int(sex)), self.global_mean)

    def predict_one(self, feats: dict, event_id: str, sex: int) -> tuple[float, float, float]:
        m = self._mean(event_id, int(sex))
        return m, m - Z80 * self.resid_std, m + Z80 * self.resid_std


class PooledRidge:
    """Ridge regression over engineered features + one-hot(event, sex)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.pipe = None
        self.resid_std: float = 1.0

    def _build(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        # physical features are often missing -> median-impute (per train fold) and
        # add a missing-indicator so the model can distinguish imputed from real
        physical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                         keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
        pre = ColumnTransformer(
            [
                ("eng", StandardScaler(), ENGINEERED),
                ("phys", physical, PHYSICAL),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ]
        )
        return Pipeline([("pre", pre), ("ridge", Ridge(alpha=self.alpha))])

    def fit(self, df: pd.DataFrame, calibrate: bool = True) -> PooledRidge:
        x = df[NUMERIC + CATEGORICAL]
        y = df[TARGET].to_numpy()
        self.pipe = self._build()
        # honest out-of-fold residual std for the prediction interval
        if calibrate and df["pid"].nunique() >= 2:
            from sklearn.model_selection import GroupKFold, cross_val_predict

            n_splits = min(5, df["pid"].nunique())
            oof = cross_val_predict(self.pipe, x, y, cv=GroupKFold(n_splits), groups=df["pid"])
            self.resid_std = float((y - oof).std(ddof=0)) or 1.0
        self.pipe.fit(x, y)
        if not calibrate or self.resid_std == 1.0:
            self.resid_std = float((y - self.pipe.predict(x)).std(ddof=0)) or 1.0
        return self

    def predict_one(self, feats: dict, event_id: str, sex: int) -> tuple[float, float, float]:
        # physical features may be absent at inference (None/missing) -> NaN, imputed
        row = {k: (np.nan if feats.get(k) is None else feats.get(k)) for k in NUMERIC}
        row["event_id"], row["sex"] = event_id, int(sex)
        p = float(self.pipe.predict(pd.DataFrame([row]))[0])
        return p, p - Z80 * self.resid_std, p + Z80 * self.resid_std


MODEL_FACTORIES: dict[str, type] = {"baseline": GroupMeanBaseline, "ridge": PooledRidge}
