"""Load experiment YAML files into validated dataclasses.

Features:
  * ``_base_: relative/or/absolute.yaml`` (string or list) — deep-merged, child wins.
  * Dotted overrides: ``["stages.0.steps=200", "adaptation.lora.rank=64"]`` (values parsed as YAML).
  * Strict construction: unknown keys and invalid ``Literal`` values raise :class:`ConfigError`.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
import types
import typing
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from temporal_ratio.config.schema import (
    ConfigError,
    InferenceExperiment,
    LiberoEvalExperiment,
    TrainExperiment,
)

T = TypeVar("T")


# --------------------------------------------------------------------------------------------
# YAML + inheritance + overrides
# --------------------------------------------------------------------------------------------


def _deep_merge(base: dict, child: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_string(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` and a leading ``~``."""

    def repl(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        if var in os.environ:
            return os.environ[var]
        if default is not None:
            return default
        raise ConfigError(f"environment variable {var} is not set (use ${{{var}:-default}})")

    out = _ENV.sub(repl, value)
    return os.path.expanduser(out) if out.startswith("~") else out


def _expand_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v) for v in node]
    if isinstance(node, str):
        return expand_string(node)
    return node


def _resolve_bases(data: dict, folder: Path, seen: tuple[Path, ...]) -> dict:
    """Resolve ``_base_`` at this level (relative to ``folder``) and in nested mappings."""
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = _deep_merge(merged, load_yaml_tree(folder / b, seen))
    out = _deep_merge(merged, data)
    for k, v in list(out.items()):
        if isinstance(v, dict):
            out[k] = _resolve_bases(v, folder, seen)
    return out


def load_yaml_tree(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Read a YAML file and resolve ``_base_`` chains (top-level and nested)."""
    path = Path(path).resolve()
    if path in _seen:
        raise ConfigError(f"circular _base_ include: {' -> '.join(map(str, (*_seen, path)))}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return _resolve_bases(data, path.parent, (*_seen, path))


def apply_overrides(tree: dict, overrides: list[str] | tuple[str, ...]) -> dict:
    """Apply ``a.b.0.c=value`` overrides (value parsed as YAML)."""
    tree = copy.deepcopy(tree)
    for ov in overrides:
        if "=" not in ov:
            raise ConfigError(f"override must look like key=value, got '{ov}'")
        key, raw = ov.split("=", 1)
        value = yaml.safe_load(raw)
        node: Any = tree
        parts = key.strip().split(".")
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            if isinstance(node, list):
                if not p.isdigit() or int(p) >= len(node):
                    raise ConfigError(f"override '{key}': index '{p}' out of range")
                if last:
                    node[int(p)] = value
                else:
                    node = node[int(p)]
            else:
                if last:
                    node[p] = value
                else:
                    node = node.setdefault(p, {})
    return tree


# --------------------------------------------------------------------------------------------
# dict -> dataclass
# --------------------------------------------------------------------------------------------


def _is_optional(tp: Any) -> tuple[bool, Any]:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(get_args(tp)) == 2:
            return True, args[0]
    return False, tp


def _build(tp: Any, value: Any, where: str) -> Any:
    optional, inner = _is_optional(tp)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{where}: value is required")
    tp = inner
    origin = get_origin(tp)

    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping for {tp.__name__}")
        return from_dict(tp, value, where)
    if origin is typing.Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{where}: '{value}' is not one of {list(allowed)}")
        return value
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        (item_tp,) = get_args(tp)[:1] or (Any,)
        return [_build(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    return value  # Any / unannotated


def from_dict(cls: type[T], data: dict, where: str = "") -> T:
    """Strictly build dataclass ``cls`` from ``data`` (unknown keys are errors)."""
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where or cls.__name__}: unknown keys {unknown}; valid keys: {sorted(fields)}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _build(hints[name], value, f"{where}.{name}" if where else name)
    missing = [
        n
        for n, f in fields.items()
        if n not in kwargs and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    if missing:
        raise ConfigError(f"{where or cls.__name__}: missing required keys {missing}")
    return cls(**kwargs)


# --------------------------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------------------------


def load_train(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> TrainExperiment:
    tree = _expand_tree(apply_overrides(load_yaml_tree(path), overrides))
    return from_dict(TrainExperiment, tree).validate()


def load_inference(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> InferenceExperiment:
    tree = _expand_tree(apply_overrides(load_yaml_tree(path), overrides))
    return from_dict(InferenceExperiment, tree).validate()


def load_libero_eval(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> LiberoEvalExperiment:
    tree = _expand_tree(apply_overrides(load_yaml_tree(path), overrides))
    return from_dict(LiberoEvalExperiment, tree).validate()
