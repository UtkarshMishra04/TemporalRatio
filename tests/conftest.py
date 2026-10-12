import sys
from pathlib import Path

# Tests import the light-weight modules straight from src/ (no package install needed in CI).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
