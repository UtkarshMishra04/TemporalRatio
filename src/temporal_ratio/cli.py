"""``trvam`` — command-line entry point.

    trvam show        <config.yaml> [--set k=v ...]            resolved config + generated commands
    trvam train       <train.yaml>  [--set k=v ...] [--gpus N] [--dry-run] [--only STAGE] [--from STAGE]
    trvam infer       <infer.yaml>  [--dataset ID ...] [--episodes 0 1] [--out DIR]   open-loop evaluation
    trvam serve       <infer.yaml>  [--port 8000]               websocket policy server
    trvam eval-libero <eval.yaml>   [--set k=v ...] [--dry-run]  closed-loop LIBERO evaluation
    trvam download    <repo_id> [--subfolder S] [--out DIR]     fetch a released checkpoint

Hugging Face and Weights & Biases credentials are read from the standard locations
(``hf auth login`` / ``HF_TOKEN``, ``wandb login`` / ``WANDB_API_KEY``) and are never part of a
config file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from temporal_ratio.config import loader, translate
from temporal_ratio.config.schema import ConfigError, as_dict

REPO_ROOT = Path(__file__).resolve().parents[2]


def _runtime_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT / "src"), env["PYTHONPATH"]) if p)
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("JAX_PLATFORMS", "cpu")  # JAX is only used by config utilities
    env.update(extra or {})
    return env


def _print_cmd(argv: list[str], env: dict[str, str] | None = None) -> None:
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
    print((prefix + " " if prefix else "") + shlex.join(argv))


def _load_any(path: str, overrides: list[str]):
    tree = loader.load_yaml_tree(path)
    if "stages" in tree:
        return loader.load_train(path, overrides)
    if "inference" in tree:
        return loader.load_libero_eval(path, overrides)
    return loader.load_inference(path, overrides)


# --------------------------------------------------------------------------------------------
def cmd_show(a: argparse.Namespace) -> int:
    cfg = _load_any(a.config, a.set)
    print(yaml.safe_dump(as_dict(cfg), sort_keys=False))
    for w in getattr(cfg, "warnings", []):
        print(f"# warning: {w}")
    if hasattr(cfg, "stages"):
        for cmd in translate.build_train_commands(cfg):
            print(f"\n# stage '{cmd.stage}' -> {cmd.checkpoint_dir}")
            _print_cmd(cmd.torchrun(cfg), cmd.env)
    elif hasattr(cfg, "eval"):
        for seed in cfg.eval.seeds:
            c = translate.build_libero_eval(cfg, seed)
            _print_cmd([sys.executable, c.script, *c.args], c.env)
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    if a.gpus:  # single node with N GPUs
        a.set = [*a.set, "distributed.nnodes=1", f"distributed.nproc_per_node={a.gpus}"]
    exp = loader.load_train(a.config, a.set)
    for w in exp.warnings:
        print(f"[trvam] warning: {w}", file=sys.stderr)
    commands = translate.build_train_commands(exp)
    names = [c.stage for c in commands]
    for opt in (a.only, a.start_from):
        if opt and opt not in names:
            raise ConfigError(f"unknown stage '{opt}'; stages are {names}")
    previous_latest: Path | None = None
    started = a.start_from is None
    for i, cmd in enumerate(commands):
        if cmd.stage == a.start_from:
            started = True
        selected = started and (a.only is None or a.only == cmd.stage)
        if cmd.pending:
            if previous_latest is None and i > 0:
                prev_dir = commands[i - 1].checkpoint_dir
                previous_latest = (
                    translate.latest_step_dir(prev_dir)
                    if not a.dry_run or prev_dir.exists()
                    else prev_dir / "<latest_step>"
                )
            if previous_latest is not None:
                cmd.resolve_previous_stage(previous_latest)
        if selected:
            argv = cmd.torchrun(exp)
            print(f"[trvam] stage '{cmd.stage}' ({i + 1}/{len(commands)}) -> {cmd.checkpoint_dir}")
            _print_cmd(argv, cmd.env)
            if not a.dry_run:
                cmd.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                (cmd.checkpoint_dir / "experiment.yaml").write_text(
                    yaml.safe_dump({"experiment": as_dict(exp), "stage": cmd.stage, "argv": argv}, sort_keys=False)
                )
                rc = subprocess.call(argv, cwd=REPO_ROOT, env=_runtime_env(_expand_paths(cmd.env)))
                if rc != 0:
                    print(f"[trvam] stage '{cmd.stage}' failed with exit code {rc}", file=sys.stderr)
                    return rc
        if not a.dry_run and cmd.checkpoint_dir.exists():
            try:
                previous_latest = translate.latest_step_dir(cmd.checkpoint_dir)
            except FileNotFoundError:
                previous_latest = None
        else:
            previous_latest = None
    return 0


def cmd_eval_libero(a: argparse.Namespace) -> int:
    exp = loader.load_libero_eval(a.config, a.set)
    if not a.dry_run:
        from temporal_ratio.inference import resolve_checkpoint

        exp.inference.checkpoint = str(resolve_checkpoint(exp.inference.checkpoint))
    for seed in exp.eval.seeds:
        c = translate.build_libero_eval(exp, seed)
        argv = [sys.executable, str(REPO_ROOT / c.script), *c.args]
        _print_cmd(argv, c.env)
        if not a.dry_run:
            rc = subprocess.call(argv, cwd=REPO_ROOT, env=_runtime_env(c.env))
            if rc != 0:
                return rc
    return 0


def cmd_infer(a: argparse.Namespace) -> int:
    exp = loader.load_inference(a.config, a.set)
    if exp.lerobot_home:  # read by LeRobot at import time
        os.environ["HF_LEROBOT_HOME"] = os.path.expanduser(exp.lerobot_home)
    from temporal_ratio import offline  # heavy imports (torch, models) only here

    report = offline.run(
        exp, out_dir=a.out, datasets=a.dataset, episodes=a.episodes, max_steps=a.max_steps, prompt=a.prompt
    )
    summary = {name: r["summary"] for name, r in report["datasets"].items()}
    out = a.out or exp.offline.output_dir
    print(json.dumps({"out": str(out), "summary": summary}, indent=2))
    return 0 if all(s["all_finite"] and s["num_queries"] for s in summary.values()) else 1


def _expand_paths(env: dict[str, str]) -> dict[str, str]:
    return {k: os.path.expanduser(v) for k, v in env.items()}


def cmd_serve(a: argparse.Namespace) -> int:
    from temporal_ratio.inference import ConfiguredPolicy
    from temporal_ratio.serving.websocket_policy_server import WebsocketPolicyServer

    exp = loader.load_inference(a.config, a.set)
    policy = ConfiguredPolicy(exp)
    print(f"[trvam] serving {exp.checkpoint} ({exp.policy_config}) on 0.0.0.0:{a.port}")
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

    with_config("show", cmd_show, "print the resolved config and the commands it generates")
    sp = with_config("train", cmd_train, "run a training experiment")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--only", help="run only this stage")
    sp.add_argument("--from", dest="start_from", help="resume the recipe from this stage")
    sp.add_argument("--gpus", type=int, help="run on one node with this many GPUs")
    sp = with_config("eval-libero", cmd_eval_libero, "closed-loop LIBERO evaluation")
    sp.add_argument("--dry-run", action="store_true")
    sp = with_config("infer", cmd_infer, "open-loop inference on recorded LeRobot episodes")
    sp.add_argument("--dataset", nargs="+", help="LeRobot dataset id(s) (default: offline.datasets of the config)")
    sp.add_argument("--episodes", type=int, nargs="+", help="episode indices (default: offline.episodes)")
    sp.add_argument("--out", help="output directory (default: offline.output_dir)")
    sp.add_argument("--max-steps", type=int, default=None, help="policy queries per episode (default: offline.max_steps)")
    sp.add_argument("--prompt", help="override the recorded task prompt")
    sp = with_config("serve", cmd_serve, "serve the policy over websockets (openpi-client protocol)")
    sp.add_argument("--port", type=int, default=8000)
    sp = sub.add_parser("download", help="download a released checkpoint")
    sp.add_argument("repo_id")
    sp.add_argument("--subfolder")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_download)
    return p


def main(argv: list[str] | None = None) -> int:
    # Runtime defaults for in-process commands (infer, serve); subprocesses get them via _runtime_env.
    for key, value in _runtime_env().items():
        if key in ("TORCHDYNAMO_DISABLE", "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM", "JAX_PLATFORMS"):
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
