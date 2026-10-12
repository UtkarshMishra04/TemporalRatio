"""Action-head utilities; skipped unless the training environment (torch, einops, transformers) is installed."""

import pytest

action_head = pytest.importorskip("temporal_ratio.models.action_head")
torch = pytest.importorskip("torch")


def test_drop_first_frame_tokens():
    frames, tokens = 5, 6
    pad = torch.ones(256, frames * tokens, dtype=torch.bool)
    assert torch.equal(action_head.drop_first_frame_tokens(pad, frames, 0.0), pad)
    out = action_head.drop_first_frame_tokens(pad, frames, 1.0)
    assert not out[:, :tokens].any() and out[:, tokens:].all()

    torch.manual_seed(0)
    out = action_head.drop_first_frame_tokens(pad, frames, 0.2)
    first = out[:, :tokens]
    assert (first.all(1) | ~first.any(1)).all()  # a sample keeps or hides the whole frame
    assert out[:, tokens:].all()
    assert 0.1 < (~first.any(1)).float().mean() < 0.3
