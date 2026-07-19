"""Write a train-window feature-distribution baseline so ml/eval drift checks
can fire later. Without a stored baseline the drift monitor has nothing to
compare live serving features against."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


def write_baseline(df: pd.DataFrame, feature_cols: List[str], out_dir: Path) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if col.empty:
            stats[c] = {"mean": None, "std": None, "p10": None, "p50": None, "p90": None}
            continue
        stats[c] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "p10": float(col.quantile(0.10)), "p50": float(col.quantile(0.50)),
            "p90": float(col.quantile(0.90)),
        }
    payload = {"features": list(feature_cols), "n_rows": int(len(df)), "stats": stats}
    p = out_dir / "drift_baseline.json"
    p.write_text(json.dumps(payload, indent=2))
    return p
