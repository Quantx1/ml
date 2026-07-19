from backend.ai.signals.style_types import Style, StyleSignal, MomentumSignal


def test_momentum_signal_is_style_signal_and_serializes():
    sig = MomentumSignal(
        symbol="RELIANCE", style=Style.MOMENTUM, rank=1, percentile=1.0,
        confidence=100.0, direction="BUY", entry_price=100.0, stop_loss=85.0,
        target=130.0, risk_reward=2.0, reasons=["top of book"],
        expected_return=0.0369, top_decile_prob=1.0,
    )
    assert isinstance(sig, StyleSignal)
    assert sig.style == Style.MOMENTUM
    d = sig.to_dict()
    assert d["style"] == "momentum"
    assert d["symbol"] == "RELIANCE"
    assert d["expected_return"] == 0.0369
    assert d["top_decile_prob"] == 1.0
    assert d["risk_reward"] == 2.0
