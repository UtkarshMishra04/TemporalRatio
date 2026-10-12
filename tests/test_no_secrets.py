"""The public repository must never contain credentials or internal infrastructure references.

This complements the gitleaks CI job with project-specific patterns (internal hosts, paths and
dataset namespaces that are not secrets per se but must not ship).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "huggingface token": r"hf_[A-Za-z0-9]{30,}",
    "wandb api key": r"(wandb_v1_|local-wandb)",
    "generic api key assignment": r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "aws access key": r"AKIA[0-9A-Z]{16}",
    "cluster file system path": r"/(k8s-nfs|nfs|storage/home|data/s3_downloads)/",
    "object store uri": r"\bs3://(?!bucket/)[a-z0-9]",  # upstream Cosmos uses s3://bucket/... as registry keys
    "self-hosted wandb": r"https?://[a-z0-9.-]+\.wandb\.io",
    "hard-coded wandb entity": r"entity\s*=\s*['\"][A-Za-z0-9_-]+['\"]",
    "private ip": r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "internal git host": r"gitlab-[a-z]+\.[a-z]+\.com",
    "corporate email": r"@(amazon|nvidia)\.com",
    "root home cache": r"/root/\.cache",
}
# Patterns not applied below these paths (vendored upstream code uses placeholder URIs in docs).
EXCLUDE = {"object store uri": ("src/cosmos_video/",)}
# Files allowed to *describe* the patterns (this test and the CI config).
ALLOW = {"tests/test_no_secrets.py", ".gitleaks.toml"}
# Individual lines kept verbatim from upstream sources: (file, substring).
ALLOW_LINES = {
    # Author credit in the public NVIDIA Cosmos-Predict2.5 source (vendored unchanged).
    ("src/cosmos_video/_src/predict2/tokenizers/wan2pt1_2d_plugins.py", "Original code author is Yan Wang"),
}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ""}


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / f for f in out.splitlines()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = [p for p in ROOT.rglob("*") if ".git" not in p.parts]
    return [f for f in files if f.is_file() and f.suffix in TEXT_SUFFIXES and f.stat().st_size < 5_000_000]


def test_no_secrets_or_internal_references():
    hits = []
    for f in _tracked_files():
        rel = str(f.relative_to(ROOT))
        if rel in ALLOW:
            continue
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if any(rel == path and marker in line for path, marker in ALLOW_LINES):
                continue
            for name, pat in FORBIDDEN.items():
                if rel.startswith(EXCLUDE.get(name, ())):
                    continue
                if re.search(pat, line):
                    hits.append(f"{rel}:{i}: [{name}] {line.strip()[:100]}")
    assert not hits, "forbidden content found:\n" + "\n".join(hits)
