"""Repository root, used for repo-relative defaults (``assets/``, ``checkpoints/``).

Found by walking up from this file to the directory containing ``pyproject.toml``;
``$TEMPORAL_RATIO_ROOT`` overrides it.
"""

import os
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("TEMPORAL_RATIO_ROOT")
    if env:
        return Path(env).resolve()
    d = Path(__file__).resolve().parent
    for _ in range(5):
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    return Path.cwd()


PROJECT_ROOT = _find_root()
