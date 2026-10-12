"""Continuing a training run: learning-rate cycles and checkpoint retention.

A run is continued by repeating its command with ``resume`` (the default) and, to train longer, a
larger step budget. The latest checkpoint provides the full training state; this module decides
which learning-rate cycle the continued run is in and which checkpoints may be deleted.
Torch-free, so that it can be tested without the training environment.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def checkpoint_steps(ckpt_dir: str | Path) -> list[int]:
    """Sorted steps of the complete checkpoints (``<ckpt_dir>/<step>/``) of a run."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    return sorted(int(d.name) for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit())


def is_kept(step: int, keep_every: int) -> bool:
    """Whether ``step`` is a permanent checkpoint (a positive multiple of ``keep_every``; 0 = none)."""
    return keep_every > 0 and step > 0 and step % keep_every == 0


def checkpoints_to_remove(steps: Iterable[int], keep_last: int, keep_every: int = 0) -> list[int]:
    """Steps to delete: all but the ``keep_last`` most recent, except multiples of ``keep_every``."""
    steps = sorted(steps)
    older = steps[: max(0, len(steps) - keep_last)]
    return [s for s in older if not is_kept(s, keep_every)]


def lr_cycle_start(global_step: int, max_train_steps: int, saved_metadata: dict | None) -> int:
    """First step of the learning-rate cycle (warmup + decay) that the run is in.

    A run resumed with the step budget it was saved with (e.g. after a crash) stays in its cycle; a
    run resumed with a different budget (e.g. more epochs) starts a new cycle at the resumed step.

    Args:
        global_step: step the run continues from (0 for a new run).
        max_train_steps: step budget of this invocation.
        saved_metadata: ``metadata.pt`` of the resumed checkpoint; ``None`` for a new run.
    """
    if not saved_metadata:
        return 0
    saved_budget = (saved_metadata.get("config") or {}).get("max_train_steps")
    if saved_budget == max_train_steps:
        return int(saved_metadata.get("lr_cycle_start", 0))
    return global_step
