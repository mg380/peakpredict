"""RNN experiment — base vs feature-enriched sequence model, vs Ridge.

Builds two sequence representations and judges both on the SAME athlete-grouped
held-out split as Ridge:
  base      : per-season (age, score)            + static (event, sex, height, weight)
  enriched  : per-season (age, score, wind, percentile-vs-population-at-age,
              season volume, competition tier, finishing place, indoor, month)
              + static (event, sex, height, weight, country)

Run: python analysis/rnn_experiment.py [--epochs 200] [--seed 42] [--repeats 5]
"""

from __future__ import annotations

import argparse

import comet_ml  # noqa: F401 - must import before torch so Comet can instrument it
import numpy as np
import pandas as pd
import torch
from torch import nn

from peakpredict.common.io import RAW_DB_PATH, read_parquet
from peakpredict.pipeline.aggregates import build_population_aggregates
from peakpredict.pipeline.model import NUMERIC, GroupMeanBaseline, PooledRidge

EVENTS = ["40", "50", "70"]
MAX_LEN = 7
_PCT_LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
_GLOBAL = {"OG", "WC", "WI", "WCH", "WIC"}
_CONTINENTAL = {"EC", "CWG", "CC", "AC", "ASC", "PANAM", "ECP", "EJ", "WJ", "WY", "WYC", "WUG"}
_NATIONAL = {"NC", "NCAA", "OT", "NC-J", "SEC", "BIG"}


def competition_tier(comp) -> int:
    c = ("" if comp is None else str(comp)).strip().upper()
    if c in _GLOBAL:
        return 3
    if c in _CONTINENTAL:
        return 2
    if c in _NATIONAL or c.startswith("NC") or c.startswith("BIG"):
        return 1
    return 0


def _pct_lookup(season_bests: pd.DataFrame):
    agg = build_population_aggregates(season_bests)
    lut: dict = {}
    for r in agg.itertuples(index=False):
        vals = np.array([r.p10, r.p25, r.p50, r.p75, r.p90], dtype=float)
        if np.all(np.diff(vals) >= 0):
            lut[(r.event_id, int(r.sex), int(r.age_bin))] = vals

    def pct(event_id, sex, age, score) -> float:
        vals = lut.get((event_id, int(sex), int(round(age))))
        return 0.5 if vals is None else float(np.interp(score, vals, _PCT_LEVELS))

    return pct


def _country_onehot(country, top: list[str]) -> list[float]:
    vec = [0.0] * (len(top) + 1)
    if country in top:
        vec[top.index(country)] = 1.0
    else:
        vec[-1] = 1.0  # "other"
    return vec


def _make_sample(pid, event_id, sex, seq, height, weight, peak_age) -> dict:
    return {"pid": int(pid), "event_id": event_id, "sex": int(sex),
            "seq": np.asarray(seq, dtype=np.float32), "height_cm": height,
            "weight_kg": weight, "peak_age": float(peak_age),
            "static_cat": None}


def build_base_samples(processed: str) -> list[dict]:
    feats = read_parquet(f"{processed}/features.parquet")
    sb = read_parquet(f"{processed}/season_bests.parquet")
    groups = {k: g.sort_values("age") for k, g in sb.groupby(["pid", "event_id", "sex"])}
    out = []
    for r in feats.itertuples(index=False):
        g = groups.get((r.pid, r.event_id, r.sex))
        if g is None:
            continue
        obs = g.iloc[: r.cutoff_k]
        seq = np.stack([(obs["age"].to_numpy(float) - 24) / 6, obs["score"].to_numpy(float)], 1)
        s = _make_sample(r.pid, r.event_id, r.sex, seq, r.height_cm, r.weight_kg, r.peak_age)
        s["static_cat"] = np.array(
            [*[1.0 if r.event_id == e else 0.0 for e in EVENTS], 1.0 if r.sex == 1 else 0.0],
            dtype=np.float32,
        )
        out.append(s)
    return out


def build_raw_samples(processed: str) -> list[dict]:
    """Feed the RAW mark (no derived score) + event per-timestep, so the model
    must learn any event-relative structure itself."""
    feats = read_parquet(f"{processed}/features.parquet")
    sb = read_parquet(f"{processed}/season_bests.parquet")
    groups = {k: g.sort_values("age") for k, g in sb.groupby(["pid", "event_id", "sex"])}
    out = []
    for r in feats.itertuples(index=False):
        g = groups.get((r.pid, r.event_id, r.sex))
        if g is None:
            continue
        obs = g.iloc[: r.cutoff_k]
        ev = [1.0 if r.event_id == e else 0.0 for e in EVENTS]
        seq = np.column_stack([
            (obs["age"].to_numpy(float) - 24) / 6,
            (obs["mark"].to_numpy(float) - 25) / 15,  # raw seconds, fixed global affine scale
            np.tile(ev, (len(obs), 1)),
        ]).astype(np.float32)
        s = _make_sample(r.pid, r.event_id, r.sex, seq, r.height_cm, r.weight_kg, r.peak_age)
        s["static_cat"] = np.array([*ev, 1.0 if r.sex == 1 else 0.0], dtype=np.float32)
        out.append(s)
    return out


def build_enriched_samples(processed: str, db_path: str) -> list[dict]:
    import duckdb

    feats = read_parquet(f"{processed}/features.parquet")
    sb = read_parquet(f"{processed}/season_bests.parquet")
    pct = _pct_lookup(sb)

    con = duckdb.connect(str(db_path), read_only=True)
    perf = con.execute(
        "SELECT pid, event_id, indoor, perf_date, mark, wind, competition, round_pos "
        "FROM raw.performance WHERE event_id IN ('40','50','70')"
    ).df()
    countries = con.execute(
        "SELECT pid, country FROM raw.athlete WHERE country IS NOT NULL"
    ).df().set_index("pid")["country"].to_dict()
    top_countries = con.execute(
        "SELECT country FROM raw.athlete WHERE country IS NOT NULL "
        "GROUP BY country ORDER BY count(*) DESC LIMIT 10"
    ).df()["country"].tolist()
    con.close()

    perf = perf[perf["mark"].notna() & perf["perf_date"].notna()]
    perf = perf[perf["wind"].isna() | (perf["wind"] <= 2.0)]
    perf["perf_date"] = pd.to_datetime(perf["perf_date"])
    perf["year"] = perf["perf_date"].dt.year
    perf["tier"] = perf["competition"].map(competition_tier)
    perf["place"] = perf["round_pos"].astype(str).str.extract(r"^(\d+)").astype(float)
    gb = perf.groupby(["pid", "event_id", "year"])
    ctx = gb.agg(volume=("mark", "size"), tier=("tier", "max")).reset_index()
    best_cols = ["pid", "event_id", "year", "indoor", "perf_date", "place"]
    best = perf.loc[gb["mark"].idxmin(), best_cols]
    best["month"] = best["perf_date"].dt.month
    ctx = ctx.merge(best[["pid", "event_id", "year", "indoor", "place", "month"]],
                    on=["pid", "event_id", "year"])

    enr = sb.merge(ctx, left_on=["pid", "event_id", "season"],
                   right_on=["pid", "event_id", "year"], how="left")
    enr["pct"] = [pct(e, s, a, sc) for e, s, a, sc in
                  zip(enr["event_id"], enr["sex"], enr["age"], enr["score"], strict=False)]
    cols = ["age", "score", "wind", "pct", "volume", "tier", "place", "indoor", "month"]
    groups = {k: g.sort_values("age") for k, g in enr.groupby(["pid", "event_id", "sex"])}

    out = []
    for r in feats.itertuples(index=False):
        g = groups.get((r.pid, r.event_id, r.sex))
        if g is None:
            continue
        obs = g.iloc[: r.cutoff_k][cols].copy()
        seq = np.stack([
            (obs["age"].to_numpy(float) - 24) / 6,
            obs["score"].to_numpy(float),
            np.nan_to_num(obs["wind"].to_numpy(float)) / 2,
            obs["pct"].to_numpy(float) - 0.5,
            np.minimum(obs["volume"].fillna(1).to_numpy(float), 20) / 10,
            obs["tier"].fillna(0).to_numpy(float) / 3,
            1.0 / np.clip(obs["place"].fillna(20).to_numpy(float), 1, 20),
            obs["indoor"].fillna(False).astype(float).to_numpy(),
            (obs["month"].fillna(6).to_numpy(float) - 6) / 3,
        ], axis=1)
        s = _make_sample(r.pid, r.event_id, r.sex, seq, r.height_cm, r.weight_kg, r.peak_age)
        s["static_cat"] = np.array(
            [*[1.0 if r.event_id == e else 0.0 for e in EVENTS], 1.0 if r.sex == 1 else 0.0,
             *_country_onehot(countries.get(r.pid), top_countries)],
            dtype=np.float32,
        )
        out.append(s)
    return out


def _stats(samples) -> dict:
    h = np.array([s["height_cm"] for s in samples], float)
    w = np.array([s["weight_kg"] for s in samples], float)
    t = np.array([s["peak_age"] for s in samples], float)
    return {"h_med": np.nanmedian(h), "h_std": np.nanstd(h[~np.isnan(h)]) or 1.0,
            "w_med": np.nanmedian(w), "w_std": np.nanstd(w[~np.isnan(w)]) or 1.0,
            "t_mean": t.mean(), "t_std": t.std() or 1.0}


def make_tensors(samples, st: dict):
    n_seq = samples[0]["seq"].shape[1]
    seqs, lengths, statics, targets = [], [], [], []
    for s in samples:
        seq = s["seq"]
        length = len(seq)
        seqs.append(np.vstack([seq, np.zeros((MAX_LEN - length, n_seq), np.float32)]))
        lengths.append(length)
        h, w = s["height_cm"], s["weight_kg"]
        hm, wm = float(pd.isna(h)), float(pd.isna(w))
        h = st["h_med"] if pd.isna(h) else h
        w = st["w_med"] if pd.isna(w) else w
        statics.append(np.concatenate([
            s["static_cat"],
            [(h - st["h_med"]) / st["h_std"], (w - st["w_med"]) / st["w_std"], hm, wm],
        ]))
        targets.append((s["peak_age"] - st["t_mean"]) / st["t_std"])
    return (torch.tensor(np.array(seqs), dtype=torch.float32),
            torch.tensor(lengths, dtype=torch.int64),
            torch.tensor(np.array(statics), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


DEFAULT_CFG = {
    "rnn_type": "lstm", "hidden": 24, "num_layers": 1, "bidirectional": False,
    "dropout": 0.3, "head_dim": 32, "lr": 1e-3, "weight_decay": 1e-4,
    "batch_size": 64, "epochs": 200, "patience": 20,
}


def start_experiment(params: dict | None = None, tags: list[str] | None = None):
    """Open a Comet experiment if COMET_API_KEY is configured, else return None."""
    from peakpredict.common.config import get_secret

    key = get_secret("COMET_API_KEY", required=False)
    if not key:
        return None
    from comet_ml import Experiment

    exp = Experiment(
        api_key=key,
        project_name=get_secret("COMET_PROJECT", required=False) or "peakpredict-rnn",
        workspace=get_secret("COMET_WORKSPACE", required=False),
        display_summary_level=0, auto_metric_logging=False, auto_param_logging=False,
    )
    if params:
        exp.log_parameters(params)
    for t in tags or []:
        exp.add_tag(t)
    return exp


class PeakRNN(nn.Module):
    def __init__(self, n_seq: int, n_static: int, cfg: dict):
        super().__init__()
        rnn_cls = nn.GRU if cfg["rnn_type"] == "gru" else nn.LSTM
        self.is_lstm = cfg["rnn_type"] != "gru"
        self.bidir = bool(cfg["bidirectional"])
        self.rnn = rnn_cls(
            n_seq, cfg["hidden"], num_layers=cfg["num_layers"], batch_first=True,
            bidirectional=self.bidir, dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
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


def _split(pids, seed, test_frac=0.25, val_frac=0.2):
    rng = np.random.default_rng(seed)
    p = np.array(sorted(pids))
    rng.shuffle(p)
    n_test = int(test_frac * len(p))
    test = set(p[:n_test].tolist())
    rest = p[n_test:]
    n_val = int(val_frac * len(rest))
    return set(rest[n_val:].tolist()), set(rest[:n_val].tolist()), test


def _mae(model, tensors, st) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(tensors[0], tensors[1], tensors[2]).numpy() * st["t_std"] + st["t_mean"]
    true = tensors[3].numpy() * st["t_std"] + st["t_mean"]
    return float(np.abs(pred - true).mean())


def train_rnn(samples, train_p, val_p, test_p, seed, cfg=None, comet=None):
    """Train one config; returns (test_mae, best_val_mae). Logs to Comet if given."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    torch.manual_seed(seed)
    tr = [s for s in samples if s["pid"] in train_p]
    va = [s for s in samples if s["pid"] in val_p]
    te = [s for s in samples if s["pid"] in test_p]
    st = _stats(tr)
    T_tr, T_va, T_te = make_tensors(tr, st), make_tensors(va, st), make_tensors(te, st)
    model = PeakRNN(T_tr[0].shape[2], T_tr[2].shape[1], cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    bs = cfg["batch_size"]
    best_val, best_state, patience, n = float("inf"), None, 0, len(tr)
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for j in range(0, n, bs):
            idx = perm[j : j + bs]
            opt.zero_grad()
            loss_fn(model(T_tr[0][idx], T_tr[1][idx], T_tr[2][idx]), T_tr[3][idx]).backward()
            opt.step()
        v = _mae(model, T_va, st)
        if comet is not None:
            comet.log_metric("val_mae", v, epoch=epoch)
        if v < best_val - 1e-4:
            best_val, patience = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return _mae(model, T_te, st), best_val


def ridge_eval(processed, test_p):
    feats = read_parquet(f"{processed}/features.parquet")
    nontest = feats[~feats["pid"].isin(test_p)]
    test_rows = feats[feats["pid"].isin(test_p)]
    models = {"base": GroupMeanBaseline().fit(nontest), "ridge": PooledRidge().fit(nontest)}
    out = {}
    for name, m in models.items():
        err = [m.predict_one({k: getattr(r, k) for k in NUMERIC}, r.event_id, int(r.sex))[0]
               - r.peak_age for r in test_rows.itertuples(index=False)]
        out[name] = float(np.abs(err).mean())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RNN base vs enriched vs Ridge")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--db", default=str(RAW_DB_PATH))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--variant", choices=["raw", "enriched"], default="raw",
                   help="challenger to compare against the score RNN")
    args = p.parse_args(argv)

    builders = {"base": build_base_samples, "raw": build_raw_samples}
    if args.variant == "enriched":
        builders = {"base": build_base_samples,
                    "enriched": lambda pr: build_enriched_samples(pr, args.db)}
    sets = {name: fn(args.processed) for name, fn in builders.items()}
    challenger = next(k for k in sets if k != "base")
    pids = sorted({s["pid"] for s in sets["base"]})
    print(f"samples: {len(sets['base'])} | athletes: {len(pids)} | "
          f"seq dims base={sets['base'][0]['seq'].shape[1]} "
          f"{challenger}={sets[challenger][0]['seq'].shape[1]}")

    rows = []
    cfg = {"epochs": args.epochs}
    for i in range(args.repeats):
        sd = args.seed + i
        tr, va, te = _split(pids, sd)
        rid = ridge_eval(args.processed, te)
        rows.append({
            "ridge": rid["ridge"], "base": rid["base"],
            "rnn_score": train_rnn(sets["base"], tr, va, te, sd, cfg)[0],
            "rnn_alt": train_rnn(sets[challenger], tr, va, te, sd, cfg)[0],
        })
    df = pd.DataFrame(rows)
    print(f"\nHELD-OUT MAE over {args.repeats} seeds (years):")
    for col, label in [("base", "B0 population mean"), ("ridge", "B2 ridge"),
                       ("rnn_score", "RNN base (age, score)"),
                       ("rnn_alt", f"RNN {challenger}")]:
        print(f"  {label:26} {df[col].mean():.3f} +/- {df[col].std():.3f}")
    win = int((df["rnn_alt"] < df["rnn_score"]).sum())
    print(f"\n  {challenger} beat score-RNN in {win}/{args.repeats} seeds; "
          f"delta {(df['rnn_score'] - df['rnn_alt']).mean():+.3f}y")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
