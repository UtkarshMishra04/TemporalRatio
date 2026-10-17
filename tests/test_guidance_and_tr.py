from __future__ import annotations

import numpy as np
import pytest

from temporal_ratio.config import loader
from temporal_ratio.guidance import scale as gm
from temporal_ratio.guidance import tr as trm

# ------------------------------------------------------------------ Temporal Ratio


def test_tr_is_future_over_current_mass():
    assert trm.temporal_ratio([0.5, 0.2, 0.2, 0.1]) == pytest.approx(1.0)
    assert trm.temporal_ratio([0.1, 0.3, 0.3, 0.3]) == pytest.approx(9.0)


def test_uniform_attention_gives_num_future_frames():
    # TR is not normalized by token count: uniform attention over T frames -> T - 1.
    assert trm.temporal_ratio(np.full(5, 0.2)) == pytest.approx(4.0)


def test_frame_aggregation_matches_manual_sum():
    rng = np.random.default_rng(0)
    tok = rng.random((3, 5 * 392))  # (actions, T * tokens_per_frame)
    per_frame = trm.frame_attention_from_tokens(tok, 392)
    assert per_frame.shape == (3, 5)
    np.testing.assert_allclose(per_frame.sum(-1), tok.sum(-1))
    with pytest.raises(ValueError):
        trm.frame_attention_from_tokens(tok[:, :-1], 392)


def test_tr_from_segments_ignores_non_video_segments():
    names = ["state", "video_t0", "video_t1", "video_t2"]
    seg = np.array([[[9.0, 1.0, 1.0, 1.0], [9.0, 1.0, 3.0, 3.0]]])  # (layers, actions, segments)
    # mean over actions: state 9, t0 1, t1 2, t2 2 -> TR = 4
    assert trm.temporal_ratio_from_segments(seg, names) == pytest.approx(4.0)
    assert trm.temporal_ratio_from_segments(seg[..., :2], names[:2]) == 1.0


def test_tr_rejects_single_frame():
    with pytest.raises(ValueError):
        trm.temporal_ratio([1.0])


# ------------------------------------------------------------------ adaptive scale


@pytest.mark.parametrize(
    ("tr_k", "tr_0", "s_max", "expected"),
    [(1.0, 1.0, 2.0, 0.0), (0.8, 1.0, 2.0, 0.0), (1.5, 1.0, 2.0, 0.5), (5.0, 1.0, 2.0, 2.0)],
)
def test_adaptive_scale(tr_k, tr_0, s_max, expected):
    raw, scale = gm.adaptive_scale(tr_k, tr_0, s_max)
    assert scale == pytest.approx(expected)
    assert raw >= scale - 1e-12


def test_zero_scale_means_no_guidance_by_default():
    d = gm.decide(0.0, 0.1, 0.1)
    assert (d.w_lang, d.w_plan, d.resample) == (0.0, 0.0, True)


def test_legacy_flat_floor_reproduces_original_runs():
    # Original code kept the base-weight sample when effective weights were ~0.
    d = gm.decide(0.0, 0.1, 0.1, legacy_flat_floor=True)
    assert d.resample is False
    # ...and behaved identically to the paper method whenever the scaled weights are non-trivial.
    for scale in (0.5, 1.7, 2.0):
        a = gm.decide(scale, 0.1, 0.1, legacy_flat_floor=True)
        b = gm.decide(scale, 0.1, 0.1)
        assert (a.w_lang, a.w_plan, a.resample) == (b.w_lang, b.w_plan, b.resample)


def test_unit_scale_reuses_base_sample():
    d = gm.decide(1.005, 0.3, 0.1)
    assert (d.w_lang, d.w_plan, d.resample) == (0.3, 0.1, False)


def test_guided_velocity_formulas():
    v_c, v_u, v_p = np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([2.0, 2.0])
    np.testing.assert_allclose(gm.guided_velocity(v_c, v_u, v_p, w_lang=0.0, w_plan=0.0), v_c)
    np.testing.assert_allclose(
        gm.guided_velocity(v_c, v_u, v_p, w_lang=1.0, w_plan=0.5), v_c + (v_c - v_u) + 0.5 * (v_p - v_c)
    )
    np.testing.assert_allclose(gm.guided_velocity(v_c, v_u, w_lang=1.0, formula="standard"), v_u + 2.0 * (v_c - v_u))
    with pytest.raises(ValueError):
        gm.guided_velocity(v_c, v_u, w_lang=1.0, formula="nope")


# ------------------------------------------------------------------ guidance presets


def test_guidance_presets():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs/infer/libero"
    assert loader.load_inference(root / "no_guidance.yaml").guidance.type == "none"
    flat = loader.load_inference(root / "flat.yaml").guidance
    assert (flat.type, flat.space, flat.w_lang, flat.w_plan, flat.plan_latent_frames) == ("flat", "velocity", 0.1, 0.1, 24)
    tr = loader.load_inference(root / "tr_adaptive.yaml").guidance
    assert (tr.type, tr.tr_baseline, tr.tr_layer, tr.s_max, tr.legacy_flat_floor) == ("tr_adaptive", "per_episode", 13, 2.0, False)
    assert tr.w_lang == tr.w_plan == 0.1  # inherited from flat.yaml
