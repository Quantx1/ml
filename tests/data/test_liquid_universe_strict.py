"""
Tests for LiquidUniverseConfig strict mode (audit fix).

When strict=True, liquid_universe() must RAISE RuntimeError (matching
"liquid universe") rather than silently substitute NIFTY_200_FALLBACK
when the data source returns nothing.

The seam we monkeypatch is `yfinance.download` — the sole network call
inside liquid_universe().  Returning an empty DataFrame triggers the
"yfinance returned empty frame" branch, which is the total-data-loss
scenario the audit identified as a silent substitution risk.
"""

import importlib
import sys

import pandas as pd
import pytest

# Import the submodule directly (not via ml.data package, which shadows
# the submodule name with the re-exported function).
_lu_mod = importlib.import_module("ml.data.liquid_universe")

from ml.data.liquid_universe import LiquidUniverseConfig, liquid_universe  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_empty_download(*args, **kwargs) -> pd.DataFrame:
    """Stub for yfinance.download that returns an empty DataFrame."""
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_strict_raises_when_source_unavailable(monkeypatch):
    """strict=True + empty yfinance response → RuntimeError, never fallback."""
    # Clear cache so no stale hit masks the network call.
    _lu_mod.clear_cache()

    # Patch yfinance.download in the yfinance module namespace so that the
    # `import yfinance as yf` inside liquid_universe() sees the stub.
    import yfinance as _yf  # noqa: PLC0415
    monkeypatch.setattr(_yf, "download", _make_empty_download)

    with pytest.raises(RuntimeError, match="liquid universe"):
        liquid_universe(LiquidUniverseConfig(top_n=50, strict=True))


def test_non_strict_falls_back_quietly(monkeypatch):
    """strict=False (default) + empty yfinance response → static fallback list, no raise."""
    _lu_mod.clear_cache()

    import yfinance as _yf  # noqa: PLC0415
    monkeypatch.setattr(_yf, "download", _make_empty_download)

    syms = liquid_universe(LiquidUniverseConfig(top_n=10, strict=False))
    assert isinstance(syms, list), "expected a list"
    assert len(syms) > 0, "expected non-empty fallback list"
