#!/usr/bin/env bash
# Install the optional Experiment 8 comparison environment deterministically.
#
# A plain "pip install -r requirements-comparison.txt" is unsafe: insightface
# depends on opencv-python, which shadows the pinned headless build and changes
# detection and embedding numerics without any version mismatch appearing in
# package metadata. This script installs insightface without dependencies and
# pins the rest explicitly, then verifies that exactly one OpenCV is effective.
set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: activate a virtual environment first; refusing to install globally." >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Project requirements"
python -m pip install -q -r requirements.txt

echo "==> insightface (no dependencies)"
python -m pip install -q --no-deps insightface==1.0.1

echo "==> Pinned comparison dependencies"
python -m pip install -q -r requirements-comparison-deps.txt

echo "==> Removing any conflicting OpenCV distribution"
if python -m pip show opencv-python >/dev/null 2>&1; then
    python -m pip uninstall -y opencv-python
    # Reinstall headless: uninstalling the conflicting wheel can remove the
    # shared cv2 directory that both distributions write into.
    python -m pip install -q --force-reinstall --no-deps opencv-python-headless==4.13.0.92
fi

echo "==> Verifying the dependency contract"
python - <<'PYCHECK'
import importlib.metadata as meta
import sys

def version(name):
    try:
        return meta.version(name)
    except meta.PackageNotFoundError:
        return None

failures = []

if version("opencv-python") is not None:
    failures.append("opencv-python is installed and will shadow the headless build")

headless = version("opencv-python-headless")
if headless != "4.13.0.92":
    failures.append(f"opencv-python-headless is {headless}, expected 4.13.0.92")

import cv2
print(f"cv2.__version__ = {cv2.__version__}")
print(f"cv2.__file__    = {cv2.__file__}")
if not str(cv2.__version__).startswith("4.13.0"):
    failures.append(f"imported cv2 is {cv2.__version__}, expected 4.13.0")

try:
    from insightface.model_zoo import get_model  # noqa: F401
    print("insightface.model_zoo         OK")
except Exception as exc:                          # pragma: no cover - install check
    failures.append(f"insightface.model_zoo failed to import: {exc}")

try:
    from insightface.utils import face_align  # noqa: F401
    print("insightface.utils.face_align  OK")
except Exception as exc:                          # pragma: no cover - install check
    failures.append(f"insightface.utils.face_align failed to import: {exc}")

# Every directly pinned version is verified, not merely printed. A silently
# resolved-away pin is exactly the failure that changed the published numbers
# once already.
import re
from pathlib import Path

pins = {}
for source in ("requirements-comparison.txt", "requirements-comparison-deps.txt"):
    for line in Path(source).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9._-]+)==([A-Za-z0-9._+!-]+)", line)
        if match:
            pins[match.group(1)] = match.group(2)

print("\nVerifying every direct pin:")
for package in sorted(pins):
    expected, installed = pins[package], version(package)
    status = "OK" if installed == expected else "MISMATCH"
    print(f"  {package:26} {str(installed):14} expected {expected:14} {status}")
    if installed != expected:
        failures.append(
            f"{package} is {installed}, expected {expected} (pin was not honoured)"
        )

if not pins:
    failures.append("no pinned versions were found; the requirement files are unreadable")

if failures:
    print("\nDEPENDENCY CONTRACT VIOLATED:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    raise SystemExit(1)

print("\nComparison environment verified.")
PYCHECK
