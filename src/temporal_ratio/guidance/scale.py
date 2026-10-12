"""Pure math of TR-adaptive action guidance (no torch / jax imports; unit-tested on CPU).

The guided flow velocity at every action denoising step is

    v = v_c + w_lang * (v_c - v_u) + w_plan * (v_plan - v_c)          ("extrapolation")

with v_c conditioned on the task text, v_u on empty text and v_plan on features of a longer video
rollout. TR-adaptive guidance rescales both weights per replanning step:

    s_k = clip(TR_k / TR_baseline - 1, 0, s_max),    w_* = s_k * w_base_*

so guidance switches on only while the action head relies more on predicted future frames than
at the episode start (TR_baseline = TR_0, the first replan's Temporal Ratio).
"""

from __future__ import annotations

from dataclasses import dataclass

EPS = 1e-8
#: Scales within this distance of 1 reuse the sample drawn with the base weights.
SCALE_ONE_TOL = 0.01
#: Effective weights at or below this threshold are treated as "no guidance".
WEIGHT_TOL = 0.01


def adaptive_scale(tr_k: float, tr_baseline: float, s_max: float) -> tuple[float, float]:
    """Return ``(raw_scale, clipped_scale)`` for TR-adaptive guidance."""
    raw = max(0.0, tr_k / (tr_baseline + EPS) - 1.0)
    return raw, min(raw, s_max)


def damped_scale(tr_k: float, tr_0: float, s_max: float) -> tuple[float, float]:
    """Ablation: guidance shrinks as TR grows (``tr_baseline_mode='damped'``)."""
    raw = tr_0 / (tr_k + EPS)
    return raw, min(raw, s_max)


@dataclass(frozen=True)
class GuidanceDecision:
    w_lang: float
    w_plan: float
    #: Whether the actions must be re-sampled with (w_lang, w_plan) instead of reusing the sample
    #: drawn with the base weights.
    resample: bool


def decide(scale: float, w_base_lang: float, w_base_plan: float, *, legacy_flat_floor: bool = False) -> GuidanceDecision:
    """Effective weights for a scale, and whether the base-weight sample can be reused.

    The method samples once with the base weights (which also yields TR at Euler step 0), then
    re-samples with ``scale * w_base`` unless the scale is ~1.

    ``legacy_flat_floor=True`` reproduces the original research code, which skipped the re-sample
    whenever the *effective* weights were ~0 — so a zero scale silently kept the full base-weight
    guidance. ``False`` (default) follows the method: a zero scale means no guidance.
    """
    w_lang, w_plan = w_base_lang * scale, w_base_plan * scale
    if abs(scale - 1.0) <= SCALE_ONE_TOL:
        return GuidanceDecision(w_base_lang, w_base_plan, resample=False)
    if legacy_flat_floor and not (w_lang > WEIGHT_TOL or w_plan > WEIGHT_TOL):
        # Original behaviour: keep the base-weight sample (weights reported as the scaled ones).
        return GuidanceDecision(w_lang, w_plan, resample=False)
    return GuidanceDecision(w_lang, w_plan, resample=True)


def guided_velocity(v_c, v_u=None, v_plan=None, *, w_lang: float = 0.0, w_plan: float = 0.0, formula: str = "extrapolation"):
    """Combine branch velocities (works for floats, numpy arrays and torch tensors)."""
    if formula == "extrapolation":
        v = v_c
        if v_u is not None and w_lang:
            v = v + w_lang * (v_c - v_u)
    elif formula == "standard":
        v = v_c if v_u is None else v_u + (1.0 + w_lang) * (v_c - v_u)
    else:
        raise ValueError(f"unknown guidance formula '{formula}'")
    if v_plan is not None and w_plan:
        v = v + w_plan * (v_plan - v_c)
    return v
