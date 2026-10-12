"""Cross-attention rank collapse monitoring for DiT LoRA training.

Logs effective rank ratios and drift metrics to wandb at checkpoint intervals.
Detects when text-conditioning pathways collapse during robot finetuning.
"""
import logging
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch

_log = logging.getLogger(__name__)

MONITOR_BLOCKS = [20, 22, 25, 27]
KEEP_LAST_N = 5


def effective_rank(singular_values: torch.Tensor) -> float:
    S = singular_values[singular_values > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    return float(torch.exp(-(p * p.log()).sum()).item())


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item())


class RankCollapseMonitor:
    def __init__(self, net, device="cuda"):
        self._net = net
        self._device = device
        self._base_svs = {}
        self._base_flat = {}
        self._prev_self_drift = {}
        self._prev_step = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        _log.info("[RankMonitor] Caching base weight SVDs...")
        for block_id in MONITOR_BLOCKS:
            block = self._get_block(block_id)
            for attn_type in ("cross_attn", "self_attn"):
                projs = ["v_proj", "output_proj", "k_proj"] if attn_type == "cross_attn" else ["output_proj"]
                for proj_name in projs:
                    key = (block_id, attn_type, proj_name)
                    base_w = self._get_base_weight(block, attn_type, proj_name)
                    self._base_svs[key] = torch.linalg.svdvals(base_w.float().to(self._device)).cpu()
                    self._base_flat[key] = base_w.float().reshape(-1).cpu()
        self._initialized = True
        _log.info(f"[RankMonitor] Cached {len(self._base_svs)} base weight SVDs")

    def _get_block(self, block_id):
        blocks_owner = self._net.base_model.model if hasattr(self._net, "base_model") else self._net
        block = blocks_owner.blocks[block_id]
        if hasattr(block, '_checkpoint_wrapped_module'):
            block = block._checkpoint_wrapped_module
        return block

    def _get_base_weight(self, block, attn_type, proj_name):
        attn = getattr(block, attn_type)
        proj = getattr(attn, proj_name)
        if hasattr(proj, 'base_layer'):
            return proj.base_layer.weight.data
        return proj.weight.data

    def _get_merged_weight(self, block, attn_type, proj_name):
        attn = getattr(block, attn_type)
        proj = getattr(attn, proj_name)
        if hasattr(proj, 'base_layer'):
            base_w = proj.base_layer.weight.data.float()
            adapter_name = list(proj.lora_A.keys())[0]
            A = proj.lora_A[adapter_name].weight.data.float()
            B = proj.lora_B[adapter_name].weight.data.float()
            scaling = proj.scaling[adapter_name]
            return base_w + (B @ A) * scaling
        return proj.weight.data.float()

    @torch.no_grad()
    def compute_metrics(self, global_step):
        self._lazy_init()
        metrics = {}

        cross_drifts = []
        self_drifts = []

        for block_id in MONITOR_BLOCKS:
            block = self._get_block(block_id)

            # Cross-attn rank ratios
            for proj_name in ("v_proj", "output_proj"):
                key = (block_id, "cross_attn", proj_name)
                merged = self._get_merged_weight(block, "cross_attn", proj_name).to(self._device)
                current_svs = torch.linalg.svdvals(merged)
                current_rank = effective_rank(current_svs)
                base_rank = effective_rank(self._base_svs[key].to(self._device))
                ratio = current_rank / (base_rank + 1e-8)
                metrics[f"rank_monitor/b{block_id}_cross_{proj_name}_ratio"] = ratio

            # Cross-attn drift (output_proj)
            key = (block_id, "cross_attn", "output_proj")
            merged = self._get_merged_weight(block, "cross_attn", "output_proj").to(self._device)
            base = self._base_flat[key].to(self._device)
            cross_drift = 1.0 - _cos_sim(merged, base)
            metrics[f"rank_monitor/b{block_id}_cross_drift"] = cross_drift
            cross_drifts.append(cross_drift)

            # Self-attn drift (output_proj)
            key = (block_id, "self_attn", "output_proj")
            merged = self._get_merged_weight(block, "self_attn", "output_proj").to(self._device)
            base = self._base_flat[key].to(self._device)
            self_drift = 1.0 - _cos_sim(merged, base)
            metrics[f"rank_monitor/b{block_id}_self_drift"] = self_drift
            self_drifts.append(self_drift)

            # Cross/self drift ratio
            drift_ratio = cross_drift / (self_drift + 1e-8)
            metrics[f"rank_monitor/b{block_id}_drift_ratio"] = drift_ratio

            # Self-attn drift delta
            if block_id in self._prev_self_drift and self._prev_step is not None:
                step_diff = global_step - self._prev_step
                if step_diff > 0:
                    delta = (self_drift - self._prev_self_drift[block_id]) / (step_diff / 10000.0)
                    metrics[f"rank_monitor/b{block_id}_self_drift_delta_per_10k"] = delta
            self._prev_self_drift[block_id] = self_drift

            # Spectral gap of K_proj (block 27 only)
            if block_id == 27:
                merged_k = self._get_merged_weight(block, "cross_attn", "k_proj").to(self._device)
                svs_k = torch.linalg.svdvals(merged_k)
                if len(svs_k) >= 2:
                    metrics["rank_monitor/b27_k_proj_spectral_gap"] = float(svs_k[0] / svs_k[1])

            # Norm growth
            key_out = (block_id, "cross_attn", "output_proj")
            merged_out = self._get_merged_weight(block, "cross_attn", "output_proj").to(self._device)
            base_norm = self._base_flat[key_out].norm().item()
            current_norm = merged_out.norm().item()
            metrics[f"rank_monitor/b{block_id}_cross_norm_growth"] = current_norm / (base_norm + 1e-8)

        # Averages
        metrics["rank_monitor/avg_cross_drift"] = float(np.mean(cross_drifts))
        metrics["rank_monitor/avg_self_drift"] = float(np.mean(self_drifts))
        metrics["rank_monitor/avg_drift_ratio"] = float(np.mean(cross_drifts)) / (float(np.mean(self_drifts)) + 1e-8)

        self._prev_step = global_step
        return metrics

    def state_dict(self):
        return {
            "prev_self_drift": dict(self._prev_self_drift),
            "prev_step": self._prev_step,
        }

    def load_state_dict(self, d):
        self._prev_self_drift = d.get("prev_self_drift", {})
        self._prev_step = d.get("prev_step", None)


def cleanup_old_checkpoints(ckpt_dir: str, keep_n: int = KEEP_LAST_N, keep_every: int = 0):
    """Delete all but the ``keep_n`` most recent checkpoints; multiples of ``keep_every`` are kept."""
    from temporal_ratio.training.resume import checkpoint_steps, checkpoints_to_remove

    ckpt_dir = Path(ckpt_dir)
    import shutil
    for step in checkpoints_to_remove(checkpoint_steps(ckpt_dir), keep_n, keep_every):
        p = ckpt_dir / str(step)
        shutil.rmtree(p, ignore_errors=True)
        _log.info(f"[Cleanup] Removed old checkpoint: {p}")


def async_sync_checkpoint(local_dir: str, remote_uri: str):
    """Upload a checkpoint directory in the background with ``s5cmd sync`` (S3-compatible URIs)."""
    def _sync():
        try:
            cmd = ["s5cmd", "sync", f"{local_dir}/", f"{remote_uri}/"]
            subprocess.run(cmd, capture_output=True, timeout=600)
        except Exception as e:
            _log.warning(f"[CheckpointSync] Failed: {e}")

    t = threading.Thread(target=_sync, daemon=True)
    t.start()
