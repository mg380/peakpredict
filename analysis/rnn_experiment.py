"""RNN experiment — does a sequence model beat the Ridge baseline?

Predicts peak age from the early-career (age, score) SEQUENCE plus static inputs
(event, sex, height, weight), and is judged on the SAME athlete-grouped held-out
split as the Ridge model. Per the ML spec, the RNN is exploratory: it is only
worth adopting if it actually beats Ridge here.

Run: python analysis/rnn_experiment.py  [--epochs 200] [--seed 42]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn

from peakpredict.common.io import read_parquet
from peakpredict.pipeline.model import NUMERIC, GroupMeanBaseline, PooledRidge

EVENTS = ["40", "50", "70"]
MAX_LEN = 7  # cutoffs are 3/5/7 -> sequence length <= 7


def build_samples(processed: str):
    feats = read_parquet(f"{processed}/features.parquet")
    sb = read_parquet(f"{processed}/season_bests.parquet")
    groups = {k: g.sort_values("age") for k, g in sb.groupby(["pid", "event_id", "sex"])}
    samples = []
    for i, r in enumerate(feats.itertuples(index=False)):
        g = groups.get((r.pid, r.event_id, r.sex))
        if g is None:
            continue
        obs = g.iloc[: r.cutoff_k]
        seq = np.stack([obs["age"].to_numpy(float), obs["score"].to_numpy(float)], axis=1)
        samples.append({
            "row": i, "pid": int(r.pid), "event_id": r.event_id, "sex": int(r.sex),
            "seq": seq, "height_cm": r.height_cm, "weight_kg": r.weight_kg,
            "peak_age": float(r.peak_age),
        })
    return feats, samples


def _stats(samples) -> dict:
    h = np.array([s["height_cm"] for s in samples], float)
    w = np.array([s["weight_kg"] for s in samples], float)
    t = np.array([s["peak_age"] for s in samples], float)
    return {
        "h_med": np.nanmedian(h), "h_std": np.nanstd(h[~np.isnan(h)]) or 1.0,
        "w_med": np.nanmedian(w), "w_std": np.nanstd(w[~np.isnan(w)]) or 1.0,
        "t_mean": t.mean(), "t_std": t.std() or 1.0,
    }


def make_tensors(samples, st: dict):
    seqs, lengths, statics, targets = [], [], [], []
    for s in samples:
        seq = s["seq"].astype(np.float32).copy()
        seq[:, 0] = (seq[:, 0] - 24.0) / 6.0  # scale age; score already standardized
        length = len(seq)
        seqs.append(np.vstack([seq, np.zeros((MAX_LEN - length, 2), np.float32)]))
        lengths.append(length)
        h, w = s["height_cm"], s["weight_kg"]
        hm, wm = float(pd.isna(h)), float(pd.isna(w))
        h = st["h_med"] if pd.isna(h) else h
        w = st["w_med"] if pd.isna(w) else w
        onehot = [1.0 if s["event_id"] == e else 0.0 for e in EVENTS]
        statics.append([*onehot, 1.0 if s["sex"] == 1 else 0.0,
                        (h - st["h_med"]) / st["h_std"], (w - st["w_med"]) / st["w_std"], hm, wm])
        targets.append((s["peak_age"] - st["t_mean"]) / st["t_std"])
    return (
        torch.tensor(np.array(seqs), dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.int64),
        torch.tensor(np.array(statics), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32),
    )


class PeakRNN(nn.Module):
    def __init__(self, n_static: int, hidden: int = 24, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, seq, lengths, static):
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)


def _split(pids, seed, test_frac=0.25, val_frac=0.2):
    rng = np.random.default_rng(seed)
    p = np.array(sorted(pids))
    rng.shuffle(p)
    n_test = int(test_frac * len(p))
    test = set(p[:n_test].tolist())
    rest = p[n_test:]
    n_val = int(val_frac * len(rest))
    return set(rest[n_val:].tolist()), set(rest[:n_val].tolist()), test


def _mae(model, tensors, st):
    model.eval()
    with torch.no_grad():
        pred = model(tensors[0], tensors[1], tensors[2]).numpy() * st["t_std"] + st["t_mean"]
    true = tensors[3].numpy() * st["t_std"] + st["t_mean"]
    return float(np.abs(pred - true).mean())


def run(processed: str = "data/processed", seed: int = 42, epochs: int = 200) -> None:
    torch.manual_seed(seed)
    feats, samples = build_samples(processed)
    pids = {s["pid"] for s in samples}
    train_p, val_p, test_p = _split(pids, seed)
    tr = [s for s in samples if s["pid"] in train_p]
    va = [s for s in samples if s["pid"] in val_p]
    te = [s for s in samples if s["pid"] in test_p]
    st = _stats(tr)
    T_tr, T_va, T_te = make_tensors(tr, st), make_tensors(va, st), make_tensors(te, st)

    model = PeakRNN(n_static=T_tr[2].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val, best_state, patience = float("inf"), None, 0
    n = len(tr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for j in range(0, n, 64):
            idx = perm[j : j + 64]
            opt.zero_grad()
            out = model(T_tr[0][idx], T_tr[1][idx], T_tr[2][idx])
            loss_fn(out, T_tr[3][idx]).backward()
            opt.step()
        v = _mae(model, T_va, st)
        if v < best_val - 1e-4:
            best_val, patience = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best_state)

    rnn_mae = _mae(model, T_te, st)

    # Ridge + baseline on the SAME test athletes, trained on train+val
    nontest = feats[~feats["pid"].isin(test_p)]
    test_rows = feats[feats["pid"].isin(test_p)]
    ridge = PooledRidge().fit(nontest)
    base = GroupMeanBaseline().fit(nontest)

    def tab_mae(model_):
        err = []
        for r in test_rows.itertuples(index=False):
            f = {k: getattr(r, k) for k in NUMERIC}
            err.append(model_.predict_one(f, r.event_id, int(r.sex))[0] - r.peak_age)
        return float(np.abs(err).mean())

    return {"base": tab_mae(base), "ridge": tab_mae(ridge), "rnn": rnn_mae,
            "n_test": len(te), "splits": (len(train_p), len(val_p), len(test_p))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RNN vs Ridge on the held-out split")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--repeats", type=int, default=1, help="run N seeds and aggregate")
    args = p.parse_args(argv)
    results = [run(args.processed, args.seed + i, args.epochs) for i in range(args.repeats)]
    if args.repeats == 1:
        r = results[0]
        print(f"\nRNN EXPERIMENT (seed={args.seed}; held-out split)")
        print(f"train/val/test athletes: {r['splits']} | test preds: {r['n_test']}\n")
        print(f"  B0 population mean   MAE {r['base']:.2f}y")
        print(f"  B2 ridge             MAE {r['ridge']:.2f}y")
        print(f"  B4 RNN (seq+static)  MAE {r['rnn']:.2f}y")
    else:
        ridge = np.array([r["ridge"] for r in results])
        rnn = np.array([r["rnn"] for r in results])
        wins = int((rnn < ridge).sum())
        print(f"\nRNN vs RIDGE over {args.repeats} seeds (held-out MAE, years):")
        print(f"  ridge  {ridge.mean():.2f} +/- {ridge.std():.2f}")
        print(f"  RNN    {rnn.mean():.2f} +/- {rnn.std():.2f}")
        delta = (ridge - rnn).mean()
        print(f"  RNN beat ridge in {wins}/{args.repeats} seeds; mean delta {delta:+.2f}y")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
