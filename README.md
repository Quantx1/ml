# Quant X — ML

The `ml` Python package: feature engineering, labeling, purged-CV training (LambdaRank momentum/swing/positional, meta-conviction), regime detection, backtesting, eval.

Part of the [Quantx1](https://github.com/Quantx1) org: [landing](https://github.com/Quantx1/landing) · [frontend](https://github.com/Quantx1/frontend) · [backend](https://github.com/Quantx1/backend) · **ml**

## Layout note (important)

**This repo's root IS the `ml` package** — it is consumed as a git submodule mounted at `./ml` inside the [backend](https://github.com/Quantx1/backend) repo, so `from ml.features...` resolves identically there and here.

Standalone, that means: clone into a folder named `ml` (the default) and run Python from the **parent** directory:

```bash
git clone https://github.com/Quantx1/ml.git   # clones into ./ml
pip install -r ml/requirements-train.txt
pytest ml/tests -q                             # run from the parent dir
```

## Data

- `data/nse_tiers/` (tracked) — NSE universe tier lists; trainers fall back to these when no local OHLCV cache exists (bars are then fetched via the data plane).
- `data/cache/` (gitignored, optional) — local OHLCV CSVs for offline training.

## Backend interface

`ml/_vendor/backend/data/providers/` vendors the two provider files (`base.py`, `free_provider.py`) from the backend repo so this package imports cleanly without a backend checkout. If those files change in backend, re-copy them here.
