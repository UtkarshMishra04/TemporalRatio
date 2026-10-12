"""Check a running policy server (`trvam serve`) with recorded episode frames.

Sends observations of a LeRobot episode, built exactly like the open-loop evaluation (`trvam infer`)
builds them, and reports per frame the response, the latency and the action error against the
recorded actions: raw MSE and, in units of the dataset's per-dimension action standard deviation,
the RMSE over the first `--first-k` actions. Exits non-zero if an answer is malformed.

    trvam serve configs/infer/abc130k/fold_pen_bottle.yaml --set checkpoint=<ckpt> --port 8000 &
    python scripts/eval/check_server.py configs/infer/abc130k/fold_pen_bottle.yaml --port 8000 \\
        --dataset abc130k/open_the_pen_caps_val --episode 0 --frames 0 50 100 150

Use the inference config (and `--set policy_config=...`) the server was started with.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterator

import numpy as np

from temporal_ratio import offline
from temporal_ratio.config import loader, registry


def queries(policy_config: str, repo_id: str, episode: int, frames: list[int]) -> Iterator[tuple[int, dict, np.ndarray]]:
    """``(frame, observation, recorded action chunk)`` for the given frames of an episode."""
    train_config = registry.get_config(policy_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    ds, prompt_tf = offline._episode_dataset(train_config, data_config, repo_id, episode)
    for t in frames:
        sample = ds[t]
        if prompt_tf is not None:
            sample = prompt_tf(sample)
        sample = {k: (v[0] if k in data_config.obs_sequence_keys else v) for k, v in sample.items()}
        for tf in data_config.repack_transforms.inputs:
            sample = tf(sample)
        gt = np.asarray(sample.pop("actions"))
        yield t, {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in sample.items()}, gt


def main(argv: list[str] | None = None) -> int:
    from openpi_client import websocket_client_policy

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config", help="inference config the server was started with")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config overrides (as for trvam)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dataset", required=True, help="LeRobot dataset id, e.g. abc130k/open_the_pen_caps_val")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 50, 100, 150])
    p.add_argument("--first-k", type=int, default=25)
    a = p.parse_args(argv)

    exp = loader.load_inference(a.config, a.set)
    train_config = registry.get_config(exp.policy_config)
    key = train_config.data.create(train_config.assets_dirs, train_config.model).action_sequence_keys[0]
    std = offline._action_std(a.dataset, key)
    client = websocket_client_policy.WebsocketClientPolicy(host=a.host, port=a.port)
    print("server metadata:", client.get_server_metadata())
    ok = True
    for t, obs, gt in queries(exp.policy_config, a.dataset, a.episode, a.frames):
        start = time.perf_counter()
        out = client.infer(obs)
        latency = time.perf_counter() - start
        pred = np.asarray(out["actions"])[:, : gt.shape[-1]]
        valid = pred.ndim == 2 and len(pred) <= len(gt) and bool(np.isfinite(pred).all())
        ok &= valid
        line = f"frame {t}: actions {pred.shape} {'ok' if valid else 'INVALID'} | latency {latency:.2f}s"
        if valid:
            err = pred - gt[: len(pred)]
            line += f" | MSE {np.mean(err**2):.4f}"
            if std is not None:
                rmse = math.sqrt(offline.std_errors(pred, gt[: len(pred)], std, a.first_k)["first_k_mse_std"])
                line += f" | first-{a.first_k} RMSE {rmse:.3f} std"
        print(line + f" | keys {sorted(out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
