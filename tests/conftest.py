"""Test configuration.

Puts the project root on ``sys.path`` so ``ACP_arden`` imports as a plain
module. Nothing outside this repository is added to the path, and no test in
this suite loads a real model binary or reads a real dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ACP_arden as acp  # noqa: E402  (import must follow the sys.path insert)

# A fixed, non-secret key so identifier-producing tests are deterministic
# without requiring the researcher's real FACE_ID_HMAC_KEY. It protects
# nothing and is never applied to real dataset identities.
TEST_ID_HMAC_KEY = "dGVzdC1vbmx5LWtleS1ub3QtYS1zZWNyZXQtMzJieXRlcy0wMDE"


@pytest.fixture(autouse=True)
def _configured_identifier_key():
    """Install the test key around every test, then restore. Tests that assert
    on a *missing* or *rejected* key use ``temporary_id_hmac_key`` themselves or
    clear the key explicitly inside the test body."""
    with acp.temporary_id_hmac_key(TEST_ID_HMAC_KEY):
        yield
