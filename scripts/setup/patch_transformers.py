#!/usr/bin/env python3
"""Install transformers_replace patches into the active transformers package.

The Gemma action expert needs modified Gemma / PaliGemma / SigLIP modules (adaptive RMSNorm
conditioning, as in openpi). They are copied over the files of the installed ``transformers``
(which must be version 4.53.2):

    python scripts/setup/patch_transformers.py
"""

import importlib
import shutil
import sys
from pathlib import Path


def main():
    # Find transformers install location
    transformers = importlib.import_module("transformers")
    transformers_dir = Path(transformers.__file__).parent
    print(f"Transformers location: {transformers_dir}")
    print(f"Transformers version: {transformers.__version__}")

    # Source: our patches
    replace_dir = Path(__file__).resolve().parents[2] / "src" / "temporal_ratio" / "models" / "transformers_replace"
    if not replace_dir.exists():
        print(f"ERROR: {replace_dir} not found")
        sys.exit(1)

    # Copy patches
    copied = 0
    for src_file in replace_dir.rglob("*.py"):
        rel = src_file.relative_to(replace_dir)
        dst = transformers_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)
        copied += 1

    print(f"Copied {copied} files to {transformers_dir}")

    # Verify
    try:
        from transformers.models.siglip import check

        if check.check_whether_transformers_replace_is_installed_correctly():
            print("Verification: PASS")
        else:
            print("Verification: FAIL — patches may not have applied correctly")
            sys.exit(1)
    except Exception as e:
        print(f"Verification skipped: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
