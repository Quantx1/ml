"""
PR 128 — trainer modules. Each ML feature in the v1 plan adds one file
here. ``ml/training/discovery.py`` auto-imports every module on runner
startup.

Naming convention: filename matches ``Trainer.name`` slug (e.g.
``regime_hmm.py`` → ``RegimeHMMTrainer.name = "regime_hmm"``).
"""
