"""B6b — the sequence-model rung of the predictor ladder.

A bidirectional RNN over the per-season ``(age, score)`` sequence plus static
inputs (event, sex, height, weight). It is the most expressive rung; ``publish``
adopts it only when it beats the tabular rungs on the temporal split — the same
rule every model on the ladder must pass.

Two contracts mirror ``model.py``:
  * ``fit(features, season_bests)`` — grouped-CV for an honest held-out report
    AND an out-of-fold residual std (the prediction interval), then a final fit
    on all data. Stores only picklable state (config, numpy weights, normalization
    stats) so the predictor serializes with ``joblib`` like the tabular models.
  * ``predict_series(series, event_id, sex, height, weight)`` — score one athlete
    from their observed ``(age, score)`` season series. The dashboard upload path
    already builds that series, so this parallels ``PooledRidge.predict_one``.

``torch`` is imported lazily inside the functions that need it, so importing this
module (e.g. to unpickle a bundle) never requires torch; training and inference do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Z80

EVENTS = ["40", "50", "70"]
MAX_LEN = 7                  # model is trained on the first <=7 seasons
N_SEQ = 2                    # per-timestep features: (age_norm, score)
N_STATIC = len(EVENTS) + 1 + 4  # event one-hot + sex + (height_z, weight_z, h_missing, w_missing)
RNN_MIN_ATHLETES = 50       # below this, training is too noisy to be worth a rung

# The bidirectional winner of the sweep/confirmation/optimizer battery: lowest
# seed variance at the floor (~1.39y test MAE), and the simplest such config.
RNN_CFG: dict = {
    "rnn_type": "lstm", "hidden": 24, "num_layers": 1, "bidirectional": True,
    "dropout": 0.3, "head_dim": 32, "lr": 1e-3, "weight_decay": 1e-4,
    "batch_size": 64, "epochs": 200, "patience": 20,
}

_RNN_CLS = None


def _rnn_cls():
    """Build (and cache) the ``PeakRNN`` nn.Module, importing torch lazily."""
    global _RNN_CLS
    if _RNN_CLS is not None:
        return _RNN_CLS
    import torch
    from torch import nn

    class PeakRNN(nn.Module):
        def __init__(self, n_seq: int, n_static: int, cfg: dict):
            super().__init__()
            rnn_cls = nn.GRU if cfg["rnn_type"] == "gru" else nn.LSTM
            self.is_lstm = cfg["rnn_type"] != "gru"
            self.bidir = bool(cfg["bidirectional"])
            self.rnn = rnn_cls(
                n_seq, cfg["hidden"], num_layers=cfg["num_layers"], batch_first=True,
                bidirectional=self.bidir,
                dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
            )
            out_h = cfg["hidden"] * (2 if self.bidir else 1)
            self.head = nn.Sequential(
                nn.Linear(out_h + n_static, cfg["head_dim"]), nn.ReLU(),
                nn.Dropout(cfg["dropout"]), nn.Linear(cfg["head_dim"], 1),
            )

        def forward(self, seq, lengths, static):
            packed = nn.utils.rnn.pack_padded_sequence(
                seq, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out = self.rnn(packed)
            h_n = out[1][0] if self.is_lstm else out[1]  # [num_layers*dirs, B, hidden]
            last = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidir else h_n[-1]
            return self.head(torch.cat([last, static], dim=1)).squeeze(1)

    _RNN_CLS = PeakRNN
    return _RNN_CLS


# -- sample construction --------------------------------------------------

def _seq(ages: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """One athlete's per-season sequence, normalized the same way in train + serve."""
    ages = ages[:MAX_LEN]
    scores = scores[:MAX_LEN]
    return np.stack([(ages - 24) / 6, scores], axis=1).astype(np.float32)


def _static_cat(event_id: str, sex: int) -> np.ndarray:
    return np.array(
        [*[1.0 if event_id == e else 0.0 for e in EVENTS], 1.0 if sex == 1 else 0.0],
        dtype=np.float32,
    )


def _build_samples(features: pd.DataFrame, season_bests: pd.DataFrame) -> list[dict]:
    """Leakage-safe sequences: the first ``cutoff_k`` seasons of each labelled career."""
    groups = {k: g.sort_values("age") for k, g in season_bests.groupby(["pid", "event_id", "sex"])}
    out: list[dict] = []
    for r in features.itertuples(index=False):
        g = groups.get((r.pid, r.event_id, r.sex))
        if g is None:
            continue
        obs = g.iloc[: r.cutoff_k]
        out.append({
            "pid": int(r.pid), "event_id": r.event_id, "sex": int(r.sex),
            "seq": _seq(obs["age"].to_numpy(float), obs["score"].to_numpy(float)),
            "height_cm": r.height_cm, "weight_kg": r.weight_kg,
            "static_cat": _static_cat(r.event_id, int(r.sex)),
            "peak_age": float(r.peak_age),
        })
    return out


def _stats(samples: list[dict]) -> dict:
    """Train-fold normalization constants for physical inputs and the target."""
    h = np.array([s["height_cm"] for s in samples], float)
    w = np.array([s["weight_kg"] for s in samples], float)
    t = np.array([s["peak_age"] for s in samples], float)
    return {
        "h_med": float(np.nanmedian(h)), "h_std": float(np.nanstd(h[~np.isnan(h)]) or 1.0),
        "w_med": float(np.nanmedian(w)), "w_std": float(np.nanstd(w[~np.isnan(w)]) or 1.0),
        "t_mean": float(t.mean()), "t_std": float(t.std() or 1.0),
    }


def _static_vec(static_cat: np.ndarray, height, weight, st: dict) -> np.ndarray:
    hm, wm = float(pd.isna(height)), float(pd.isna(weight))
    h = st["h_med"] if pd.isna(height) else float(height)
    w = st["w_med"] if pd.isna(weight) else float(weight)
    return np.concatenate([
        static_cat,
        [(h - st["h_med"]) / st["h_std"], (w - st["w_med"]) / st["w_std"], hm, wm],
    ]).astype(np.float32)


def _to_arrays(samples: list[dict], st: dict):
    """Pad sequences to ``MAX_LEN`` and assemble (seqs, lengths, statics, targets)."""
    seqs, lengths, statics, targets = [], [], [], []
    for s in samples:
        seq = s["seq"]
        length = len(seq)
        seqs.append(np.vstack([seq, np.zeros((MAX_LEN - length, N_SEQ), np.float32)]))
        lengths.append(length)
        statics.append(_static_vec(s["static_cat"], s["height_cm"], s["weight_kg"], st))
        targets.append((s["peak_age"] - st["t_mean"]) / st["t_std"])
    return (np.asarray(seqs, np.float32), np.asarray(lengths, np.int64),
            np.asarray(statics, np.float32), np.asarray(targets, np.float32))


def _split_pids(pids, seed: int, val_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    p = np.array(sorted(pids))
    rng.shuffle(p)
    n_val = int(val_frac * len(p))
    return set(p[n_val:].tolist()), set(p[:n_val].tolist())


# -- training -------------------------------------------------------------

def _train_weights(train: list[dict], val: list[dict], cfg: dict, seed: int):
    """Train one model with early stopping on ``val``; return (numpy weights, stats)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    st = _stats(train)
    Atr, Ava = _to_arrays(train, st), _to_arrays(val, st)
    tr = [torch.from_numpy(a) for a in Atr]
    va = [torch.from_numpy(a) for a in Ava]
    model = _rnn_cls()(N_SEQ, N_STATIC, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    bs, n = cfg["batch_size"], len(train)
    best_val, best_state, patience = float("inf"), None, 0
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for j in range(0, n, bs):
            idx = perm[j: j + bs]
            opt.zero_grad()
            loss_fn(model(tr[0][idx], tr[1][idx], tr[2][idx]), tr[3][idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.abs(model(va[0], va[1], va[2]).numpy() - va[3].numpy()).mean())
        if v < best_val - 1e-4:
            best_val, patience = v, 0
            best_state = {k: t.detach().cpu().numpy() for k, t in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
    weights = best_state or {k: t.detach().cpu().numpy() for k, t in model.state_dict().items()}
    return weights, st


def _predict(weights, st, cfg, samples: list[dict]) -> np.ndarray:
    """Peak-age predictions (years) for a list of samples from stored weights."""
    import torch

    A = _to_arrays(samples, st)
    model = _rnn_cls()(N_SEQ, N_STATIC, cfg)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(A[0]), torch.from_numpy(A[1]), torch.from_numpy(A[2])).numpy()
    return z * st["t_std"] + st["t_mean"]


class RNNPredictor:
    """Sequence rung; same role as ``PooledRidge`` but over the season sequence."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = {**RNN_CFG, **(cfg or {})}
        self.weights: dict | None = None
        self.st: dict | None = None
        self.resid_std: float = 1.0
        self.cv_report: dict | None = None

    def fit(
        self,
        features: pd.DataFrame,
        season_bests: pd.DataFrame,
        *,
        calibrate: bool = True,
        n_splits: int = 5,
        seed: int = 42,
    ) -> RNNPredictor:
        samples = _build_samples(features, season_bests)
        pids = sorted({s["pid"] for s in samples})
        if len(pids) < 2:
            raise ValueError("not enough athletes to train the RNN rung")

        if calibrate and len(pids) >= n_splits:
            self.cv_report, self.resid_std = self._cv(samples, pids, n_splits, seed)

        # final model on all data (internal val split for early stopping)
        train_p, val_p = _split_pids(pids, seed)
        tr = [s for s in samples if s["pid"] in train_p]
        va = [s for s in samples if s["pid"] in val_p]
        self.weights, self.st = _train_weights(tr, va, self.cfg, seed)
        if self.cv_report is None:  # tiny data: fall back to in-sample residuals
            pred = _predict(self.weights, self.st, self.cfg, samples)
            true = np.array([s["peak_age"] for s in samples])
            self.resid_std = float((true - pred).std(ddof=0)) or 1.0
        return self

    def _cv(self, samples, pids, n_splits, seed) -> tuple[dict, float]:
        from sklearn.model_selection import GroupKFold

        by_pid = np.array([s["pid"] for s in samples])
        order = np.arange(len(samples))
        gkf = GroupKFold(min(n_splits, len(pids)))
        true_all, pred_all = [], []
        for tr_idx, te_idx in gkf.split(order, groups=by_pid):
            tr_pids = {by_pid[i] for i in tr_idx}
            inner_tr, inner_va = _split_pids(sorted(tr_pids), seed)
            train = [samples[i] for i in tr_idx if by_pid[i] in inner_tr]
            val = [samples[i] for i in tr_idx if by_pid[i] in inner_va]
            if not val:  # degenerate fold
                val = train
            w, st = _train_weights(train, val, self.cfg, seed)
            test = [samples[i] for i in te_idx]
            pred_all.append(_predict(w, st, self.cfg, test))
            true_all.append(np.array([s["peak_age"] for s in test]))
        true = np.concatenate(true_all)
        pred = np.concatenate(pred_all)
        resid_std = float((true - pred).std(ddof=0)) or 1.0
        lo, hi = pred - Z80 * resid_std, pred + Z80 * resid_std
        report = {
            "mae": float(np.abs(true - pred).mean()),
            "rmse": float(np.sqrt(((true - pred) ** 2).mean())),
            "bias": float((pred - true).mean()),
            "interval_coverage": float(((true >= lo) & (true <= hi)).mean()),
            "n": int(len(true)),
        }
        return report, resid_std

    def _series_sample(self, series: pd.DataFrame, event_id: str, sex: int, height, weight) -> dict:
        obs = series.sort_values("age")
        return {
            "seq": _seq(obs["age"].to_numpy(float), obs["score"].to_numpy(float)),
            "height_cm": height, "weight_kg": weight,
            "static_cat": _static_cat(event_id, int(sex)), "peak_age": 0.0,
        }

    def predict_series(
        self, series: pd.DataFrame, event_id: str, sex: int, height_cm=None, weight_kg=None
    ) -> tuple[float, float, float]:
        """Peak age + 80% interval from an athlete's observed (age, score) series."""
        if self.weights is None or self.st is None:
            raise RuntimeError("RNNPredictor is not fitted")
        sample = self._series_sample(series, event_id, sex, height_cm, weight_kg)
        p = float(_predict(self.weights, self.st, self.cfg, [sample])[0])
        return p, p - Z80 * self.resid_std, p + Z80 * self.resid_std

    def predict_series_batch(self, items: list[tuple]) -> np.ndarray:
        """Peak ages for many athletes in one forward pass.

        ``items`` = list of ``(series, event_id, sex, height_cm, weight_kg)``.
        Returns an array of peak ages (years), aligned to ``items``.
        """
        if self.weights is None or self.st is None:
            raise RuntimeError("RNNPredictor is not fitted")
        if not items:
            return np.empty(0, dtype=float)
        samples = [self._series_sample(*it) for it in items]
        return _predict(self.weights, self.st, self.cfg, samples)
