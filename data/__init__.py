"""ml.data — data-layer utilities for trainers (PR 164 / PR 177 / PR 179)."""

from .bhavcopy_source import (
    BhavcopyError,
    bhavcopy_download,
    bhavcopy_download_with_fallback,
)
from .corporate_actions import (
    CORPORATE_ACTIONS,
    CorporateAction,
    actions_for,
    adjust_batch,
    adjust_volume_for_actions,
)
from .delisted_registry import (
    DELISTED_NSE,
    DelistingEvent,
    historical_universe_extras,
    was_listed_at,
)
from .fii_dii_history import (
    FlowFeatureConfig,
    backfill_from_jugaad,
    compute_flow_features,
    fii_dii_series,
    reindex_flow_features_to,
)
from .liquid_universe import (
    LiquidUniverseConfig,
    NIFTY_200_FALLBACK,
    NIFTY_50_FALLBACK,
    liquid_universe,
)
from .news_ingester import (
    backfill_sentiment_cache,
    fetch_headlines_for_symbol,
)
from .fundamentals_pit import (
    FUNDAMENTAL_FIELDS,
    FUNDAMENTALS_FEATURE_NAMES,
    FundamentalsFeatureConfig,
    FundamentalsRecord,
    compute_fundamentals_features,
    get_pit_fundamentals,
    ingest_yfinance_fundamentals,
    reindex_fundamentals_to,
    upsert_records,
)
from .quality_check import (
    DataQualityError,
    QualityCheckConfig,
    QualityReport,
    audit_feature_matrix,
    run_quality_checks,
)
from .sentiment_history import (
    SentimentFeatureConfig,
    reindex_sentiment_to,
    score_headlines_to_daily,
    sentiment_features_for,
)

__all__ = [
    "BhavcopyError",
    "CORPORATE_ACTIONS",
    "CorporateAction",
    "DELISTED_NSE",
    "DelistingEvent",
    "FlowFeatureConfig",
    "LiquidUniverseConfig",
    "NIFTY_200_FALLBACK",
    "NIFTY_50_FALLBACK",
    "DataQualityError",
    "FUNDAMENTAL_FIELDS",
    "FUNDAMENTALS_FEATURE_NAMES",
    "FundamentalsFeatureConfig",
    "FundamentalsRecord",
    "QualityCheckConfig",
    "QualityReport",
    "SentimentFeatureConfig",
    "actions_for",
    "adjust_batch",
    "adjust_volume_for_actions",
    "audit_feature_matrix",
    "backfill_from_jugaad",
    "backfill_sentiment_cache",
    "fetch_headlines_for_symbol",
    "bhavcopy_download",
    "bhavcopy_download_with_fallback",
    "compute_flow_features",
    "compute_fundamentals_features",
    "fii_dii_series",
    "get_pit_fundamentals",
    "historical_universe_extras",
    "ingest_yfinance_fundamentals",
    "liquid_universe",
    "reindex_flow_features_to",
    "reindex_fundamentals_to",
    "reindex_sentiment_to",
    "run_quality_checks",
    "score_headlines_to_daily",
    "sentiment_features_for",
    "upsert_records",
    "was_listed_at",
]
