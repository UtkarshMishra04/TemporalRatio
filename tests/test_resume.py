"""Continuing training runs: checkpoint retention and learning-rate cycles."""

from temporal_ratio.training import resume


def test_checkpoints_to_remove():
    steps = [1000, 2000, 3000, 3633, 4000, 5000, 6000, 7000, 7266]
    assert resume.checkpoints_to_remove(steps, 5) == [1000, 2000, 3000, 3633]
    # checkpoints at multiples of keep_every (here: epochs of 3,633 steps) are never removed
    assert resume.checkpoints_to_remove(steps, 5, keep_every=3633) == [1000, 2000, 3000]
    assert resume.checkpoints_to_remove([1000, 2000], 5) == []


def test_lr_cycle_start():
    one_epoch = {"config": {"max_train_steps": 3633}}
    assert resume.lr_cycle_start(0, 3633, None) == 0  # new run
    assert resume.lr_cycle_start(2000, 3633, one_epoch) == 0  # crash recovery: same cycle
    assert resume.lr_cycle_start(3633, 10899, one_epoch) == 3633  # more epochs: new cycle
    continued = {"config": {"max_train_steps": 10899}, "lr_cycle_start": 3633}
    assert resume.lr_cycle_start(5000, 10899, continued) == 3633  # crash recovery in that cycle


def test_checkpoint_steps(tmp_path):
    for step in ("1000", "2000", "tmp_3000"):
        (tmp_path / step).mkdir()
    (tmp_path / "experiment.yaml").write_text("")
    assert resume.checkpoint_steps(tmp_path) == [1000, 2000]
    assert resume.checkpoint_steps(tmp_path / "missing") == []
