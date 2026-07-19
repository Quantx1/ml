import pytest
from ml.training.base import PipelineTrainer, Trainer, TrainResult
from ml.training.specs import EngineSpec


def test_pipeline_trainer_is_a_trainer_and_requires_hooks():
    class Incomplete(PipelineTrainer):
        name = "incomplete"
        def engine_spec(self):
            return EngineSpec(name="incomplete")
    t = Incomplete()
    assert isinstance(t, Trainer)
    with pytest.raises(NotImplementedError):
        t.build_features(None)
    with pytest.raises(NotImplementedError):
        t.build_labels(None)
    with pytest.raises(NotImplementedError):
        t.make_model({})


def test_train_delegates_to_run_pipeline(monkeypatch, tmp_path):
    called = {}
    import ml.training.base as base_mod

    class T(PipelineTrainer):
        name = "t"
        def engine_spec(self):
            return EngineSpec(name="t")

    def fake_run_pipeline(trainer, out_dir):
        called["trainer"] = trainer.name
        called["out_dir"] = out_dir
        return TrainResult(artifacts=[], metrics={"ok": True})

    monkeypatch.setattr(base_mod, "_run_pipeline", fake_run_pipeline, raising=False)
    res = T().train(tmp_path)
    assert called["trainer"] == "t" and res.metrics["ok"] is True
