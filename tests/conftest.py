import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_DIR = ROOT / "tests" / "data"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    if not SAMPLE_DIR.exists():
        pytest.skip("tests/data with a few released jpg+json pairs is required")
    return SAMPLE_DIR
