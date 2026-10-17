"""The action head must get the same video-timestep variable at inference as in training.

Training (``JointVideoActionModel.extract_features``) conditions the head on the sampler fraction
``u`` that the DiT's training scheduler maps to the features' noise level; inference maps the
sampling timestep back to that ``u``. Skipped unless the inference environment is installed.
"""

import pytest

torch = pytest.importorskip("torch")
video_policy = pytest.importorskip("temporal_ratio.policies.video_policy")
rf_mod = pytest.importorskip("cosmos_video._src.predict2.schedulers.rectified_flow")


@pytest.fixture
def rf():
    return rf_mod.RectifiedFlow(velocity_field=None, train_time_distribution="logitnormal", shift=5)


def test_inference_recovers_the_training_fraction(rf):
    grid = rf.noise_scheduler.timesteps
    tk = {"device": "cpu", "dtype": torch.float32}
    for u in torch.linspace(0, 0.999, 97):
        timestep = rf.get_discrete_timestamp(u.reshape(1, 1), tk)
        got = video_policy.head_timestep(timestep, grid)
        assert got.shape == (1,)
        assert float(got) == pytest.approx(int(float(u) * 1000) / 1000)


def test_same_noise_level_as_the_features(rf):
    # A sampling timestep between grid points maps to the training step of (nearly) the same sigma.
    grid = rf.noise_scheduler.timesteps
    tk = {"device": "cpu", "dtype": torch.float32}
    for t in (907.9, 600.3, 100.0):
        u = video_policy.head_timestep(torch.tensor(t), grid)
        sigma = rf.get_sigmas(rf.get_discrete_timestamp(u.reshape(1, 1), tk), tk)
        assert float(sigma) == pytest.approx(t / 1000, abs=2e-3)
    # e.g. sigma 0.908 (6 video steps, features after 2) is u = 0.336, not 0.908
    assert float(video_policy.head_timestep(torch.tensor(908.0), grid)) == pytest.approx(0.336, abs=1e-3)
