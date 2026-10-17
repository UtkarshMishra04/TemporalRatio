"""Check a running policy server (`trvam serve`).

Builds an observation from image files (one per camera, in the robot's camera order) or from random
images, sends `--queries` requests and reports per query the action chunk, the latency and the
response keys. Exits non-zero if an answer is malformed (actions not a finite 2-D chunk). The robot
(camera keys, state dimension) is taken from the server's `policy_config`; the server's own input
transforms resize the images.

    trvam serve configs/infer/abc130k/yam_pp24.yaml --set checkpoint=<ckpt> --port 8000 &
    python scripts/check_server.py --port 8000 --images top.png left_wrist.png right_wrist.png \\
        --prompt "fold the t-shirt" --video-dir results/check_server
    python scripts/check_server.py --port 8000 --random --queries 3

With `--video-dir` every request asks for the predicted future video (`generate_video`), which is
saved as `query_<n>.mp4` with a contact sheet `query_<n>.png` (frames 0, 4, 8, 12, 16 top to bottom).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

#: Observation layout per robot (``src/temporal_ratio/policies/*_policy.py``): camera keys in the
#: order of ``--images``, state dimension and the image size used for ``--random``.
ROBOTS = {
    "libero": {
        "cameras": ("observation/image", "observation/wrist_image"),
        "state_dim": 8,
        "image_size": (224, 224),
    },
    "yam": {
        "cameras": ("observation/left_image", "observation/left_wrist_left", "observation/right_wrist_left"),
        "state_dim": 14,
        "image_size": (240, 320),
    },
}

#: Frames of the predicted video shown in the contact sheet.
SHEET_FRAMES = (0, 4, 8, 12, 16)


def robot_of(policy_config: str) -> str:
    if policy_config.startswith("videoVLA_libero"):
        return "libero"
    if policy_config.startswith("videoVLA_abc130k"):
        return "yam"
    raise ValueError(f"unknown policy_config {policy_config!r}; pass --robot")


def observation(robot: str, images: list[str] | None, state: list[float] | None, prompt: str,
                rng: np.random.Generator) -> dict:
    spec = ROBOTS[robot]
    cameras = spec["cameras"]
    obs: dict = {}
    if images:
        from PIL import Image

        if len(images) != len(cameras):
            raise ValueError(f"{robot} needs {len(cameras)} images ({', '.join(cameras)}), got {len(images)}")
        for key, path in zip(cameras, images, strict=True):
            obs[key] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    else:
        h, w = spec["image_size"]
        for key in cameras:
            obs[key] = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    if state is not None and len(state) != spec["state_dim"]:
        raise ValueError(f"{robot} state has {spec['state_dim']} dimensions, got {len(state)}")
    obs["observation/state"] = np.asarray(state if state is not None else np.zeros(spec["state_dim"]), np.float32)
    obs["prompt"] = prompt
    return obs


def save_video(video: np.ndarray, out_dir: Path, index: int) -> Path:
    """Save the predicted video as mp4 and a PNG contact sheet of ``SHEET_FRAMES``."""
    import imageio.v2 as imageio
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"query_{index:05d}.mp4"
    imageio.mimwrite(path, video, fps=10)
    frames = [video[i] for i in SHEET_FRAMES if i < len(video)]
    Image.fromarray(np.concatenate(frames, axis=0)).save(path.with_suffix(".png"))
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--robot", choices=sorted(ROBOTS), help="default: from the server's policy_config")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", help="one image per camera, in the robot's camera order")
    src.add_argument("--random", action="store_true", help="send random images (smoke test)")
    p.add_argument("--state", type=float, nargs="+", help="robot state (default: zeros)")
    p.add_argument("--prompt", default="do something")
    p.add_argument("--queries", type=int, default=1)
    p.add_argument("--video-dir", type=Path, help="request the predicted video and save it here")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=a.host, port=a.port)
    metadata = client.get_server_metadata()
    print("server metadata:", metadata)
    robot = a.robot or robot_of(str(metadata.get("policy_config", "")))
    rng = np.random.default_rng(a.seed)
    ok = True
    for n in range(a.queries):
        obs = observation(robot, a.images, a.state, a.prompt, rng)
        if a.video_dir is not None:
            obs["generate_video"] = True
        start = time.perf_counter()
        out = client.infer(obs)
        latency = time.perf_counter() - start
        actions = np.asarray(out.get("actions"))
        valid = actions.ndim == 2 and actions.size > 0 and bool(np.isfinite(actions).all())
        ok &= valid
        line = f"query {n}: actions {actions.shape} {'ok' if valid else 'INVALID'} | latency {latency:.2f}s"
        video = out.get("future_video")
        if a.video_dir is not None:
            if video is None:
                ok = False
                line += " | no future_video"
            else:
                line += f" | video {np.shape(video)} -> {save_video(np.asarray(video), a.video_dir, n)}"
        print(line + f" | keys {sorted(out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
