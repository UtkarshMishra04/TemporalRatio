"""Server-side policy wrappers with a fake policy (no GPU, no checkpoint)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openpi_client")
from temporal_ratio import inference  # noqa: E402
from temporal_ratio.config import loader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakePolicy:
    """Returns a random 17-frame video when asked to generate one."""

    def __init__(self):
        self.calls: list[tuple[dict, dict]] = []

    def infer(self, obs: dict, **kwargs) -> dict:
        self.calls.append((obs, kwargs))
        video = None
        if kwargs.get("generate_video"):
            video = np.random.default_rng(0).integers(0, 256, size=(17, 16, 32, 3), dtype=np.uint8)
        return {"actions": np.zeros((17, 7)), "future_video": video, "features": np.zeros(3)}


@pytest.fixture
def served(monkeypatch):
    fake = FakePolicy()
    monkeypatch.setattr(inference, "load_policy", lambda exp: fake)
    monkeypatch.setattr(
        inference, "infer_kwargs", lambda exp, policy: {"generate_video": exp.generate_video, "tr_layers": None}
    )
    exp = loader.load_inference(ROOT / "configs/infer/libero/no_guidance.yaml", ["checkpoint=/c/1"])
    return fake, inference.ConfiguredPolicy(exp)


def test_generate_video_per_request(served):
    fake, policy = served
    out = policy.infer({"prompt": "x"})  # config default: generate_video false
    assert "future_video" not in out and "features" not in out
    assert fake.calls[-1][1]["generate_video"] is False
    out = policy.infer({"prompt": "x", "generate_video": True})
    assert out["future_video"].shape == (17, 16, 32, 3)
    obs, kwargs = fake.calls[-1]
    assert kwargs["generate_video"] is True and "generate_video" not in obs  # popped before the transforms


def test_guided_policy_toggles_decoding(monkeypatch):
    class Guided(FakePolicy):
        decode_video = False

        def infer(self, obs, **kwargs):
            return super().infer(obs, generate_video=self.decode_video)

    fake = Guided()
    monkeypatch.setattr(inference, "load_policy", lambda exp: fake)
    monkeypatch.setattr(inference, "infer_kwargs", lambda exp, policy: {})
    exp = loader.load_inference(ROOT / "configs/infer/libero/tr_adaptive.yaml", ["checkpoint=/c/1"])
    policy = inference.ConfiguredPolicy(exp)
    assert policy.infer({"generate_video": True})["future_video"] is not None
    assert "future_video" not in policy.infer({})


def test_save_video(served, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    imageio = pytest.importorskip("imageio.v2")
    _, policy = served
    saving = inference.VideoSavingPolicy(policy, tmp_path)
    saving.infer({"prompt": "x"})  # no video: nothing written
    out = saving.infer({"prompt": "x", "generate_video": True})
    assert "future_video" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["query_00001.mp4"]
    frames = imageio.mimread(tmp_path / "query_00001.mp4")
    assert len(frames) == 17 and frames[0].shape == (16, 32, 3)
    assert saving.metadata == {"policy_config": "videoVLA_libero_short", "checkpoint": "/c/1"}
