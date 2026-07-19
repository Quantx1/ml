"""
PR 128 — trainer discovery.

Imports every module under ``ml/training/trainers/`` and collects every
``Trainer`` subclass defined in them. Trainers are auto-registered by
import side-effect so a new feature only needs to drop a file into
``trainers/``.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Dict, List

from .base import Trainer

logger = logging.getLogger(__name__)


def discover_trainers() -> List[Trainer]:
    """Import every trainer module and instantiate each Trainer subclass."""
    trainers_pkg = "ml.training.trainers"
    pkg = importlib.import_module(trainers_pkg)
    found: List[Trainer] = []
    seen_names: Dict[str, str] = {}

    pkg_path = Path(pkg.__file__).parent if pkg.__file__ else None
    if pkg_path is None:
        return found

    # 2026-05-23 — entry-side RL removed from v1 (failed the Sharpe gate);
    # the FinRL-X subsystem (ml/rl/ + finrl_x_ensemble trainer) was deleted
    # 2026-06-14. F4 AutoPilot ships on the supervised stack (Qlib+HMM+VIX).
    for mod_info in pkgutil.iter_modules([str(pkg_path)]):
        if mod_info.name.startswith("_"):
            continue
        full_name = f"{trainers_pkg}.{mod_info.name}"
        try:
            module = importlib.import_module(full_name)
        except Exception as exc:  # noqa: BLE001 — discovery must keep going
            logger.warning("trainer module %s failed to import: %s", full_name, exc)
            continue
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, Trainer)
                and obj is not Trainer
                and not getattr(obj, "__abstractmethods__", None)
                # Skip class-level _abstract_base=True (e.g. _SB3AlgoTrainer
                # shared parent of finrl_x_ppo/ddpg/a2c)
                and not getattr(obj, "_abstract_base", False)
            ):
                inst = obj()
                if not inst.name:
                    logger.warning("trainer %s has no .name; skipping", obj.__name__)
                    continue
                if inst.name in seen_names:
                    logger.warning(
                        "duplicate trainer name %r (in %s and %s); skipping latter",
                        inst.name, seen_names[inst.name], full_name,
                    )
                    continue
                seen_names[inst.name] = full_name
                found.append(inst)
    return found


def _topo_sort(trainers: List[Trainer]) -> List[Trainer]:
    """Order trainers so each runs after its declared dependencies."""
    by_name = {t.name: t for t in trainers}
    visited: Dict[str, bool] = {}  # False = visiting, True = done
    out: List[Trainer] = []

    def visit(t: Trainer) -> None:
        if visited.get(t.name) is True:
            return
        if visited.get(t.name) is False:
            raise RuntimeError(f"trainer dependency cycle at {t.name}")
        visited[t.name] = False
        for dep_name in t.depends_on or []:
            dep = by_name.get(dep_name)
            if dep is None:
                logger.warning(
                    "trainer %s depends on unknown trainer %r; ignoring",
                    t.name, dep_name,
                )
                continue
            visit(dep)
        visited[t.name] = True
        out.append(t)

    for t in trainers:
        visit(t)
    return out


def discover_sorted() -> List[Trainer]:
    """Discovery + topo-sort by ``depends_on``."""
    return _topo_sort(discover_trainers())
