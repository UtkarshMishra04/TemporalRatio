"""``trvam`` — command-line entry point.

    trvam show     <infer.yaml> [--set k=v ...]                                   resolved config
    trvam serve    <infer.yaml> [--set k=v ...] [--port 8000] [--save-video DIR]  websocket policy server
    trvam download <repo_id> [--subfolder S] [--out DIR]                          fetch a released checkpoint

Hugging Face credentials are read from the standard locations (``hf auth login`` / ``HF_TOKEN``)
and are never part of a config file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

from temporal_ratio.config import loader
from temporal_ratio.config.schema import ConfigError, as_dict

#: Runtime defaults for the in-process commands.
RUNTIME_ENV = {
    "TORCHDYNAMO_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
    "JAX_PLATFORMS": "cpu",  # JAX is only used by config utilities
}


# --------------------------------------------------------------------------------------------
def cmd_show(a: argparse.Namespace) -> int:
    cfg = loader.load_inference(a.config, a.set)
    print(yaml.safe_dump(as_dict(cfg), sort_keys=False))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from temporal_ratio.inference import ConfiguredPolicy, VideoSavingPolicy
    from temporal_ratio.serving.websocket_policy_server import WebsocketPolicyServer

    exp = loader.load_inference(a.config, a.set)
    policy = ConfiguredPolicy(exp)
    if a.save_video:
        policy = VideoSavingPolicy(policy, a.save_video)
    print(f"[trvam] serving {exp.checkpoint} ({exp.policy_config}) on 0.0.0.0:{a.port}", flush=True)
    WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=a.port, metadata=policy.metadata).serve_forever()
    return 0


def download(repo_id: str, subfolder: str | None, out: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/**"] if subfolder else None,
        local_dir=out,
    )
    return Path(path) / subfolder if subfolder else Path(path)


def cmd_download(a: argparse.Namespace) -> int:
    print(download(a.repo_id, a.subfolder, a.out))
    return 0


# --------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trvam", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def with_config(name: str, fn, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("config")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
        sp.set_defaults(fn=fn)
        return sp

    with_config("show", cmd_show, "print the resolved config")
    sp = with_config("serve", cmd_serve, "serve the policy over websockets (openpi-client protocol)")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--save-video", metavar="DIR", help="write each predicted future video to DIR/query_<n>.mp4")
    sp = sub.add_parser("download", help="download a released checkpoint")
    sp.add_argument("repo_id")
    sp.add_argument("--subfolder")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_download)
    return p


def main(argv: list[str] | None = None) -> int:
    for key, value in RUNTIME_ENV.items():
        os.environ.setdefault(key, value)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as e:
        print(f"[trvam] config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
