#!/usr/bin/env python3
"""COM7014 Advanced Computing Project — face verification coursework.

The whole project lives in this one file. It measures a face-verification
pipeline assembled from two pretrained OpenCV models: YuNet, which finds the
face in a photograph, and SFace, which turns that face into a vector of
numbers that can be compared. Neither model is modified.

Two questions are put to it. The first is the standard one-to-one problem:
shown two unconstrained photographs, does the pipeline judge correctly whether
they depict the same person? The second is harder, and is what the project is
really about. When a single photograph is searched against a gallery of many
enrolled profiles, can the same similarity score be trusted to flag a profile
that may already be registered?

The research question is whether a framework built by combining several
existing models performs better than any of those models used on its own. No
face detector or face-recognition network is trained or fine-tuned; each is
used exactly as published. What this project supplies is the arrangement
around them — enrolment, threshold calibration and a small review classifier —
together with the measurements that show what each addition is actually worth.

Five arrangements are compared on the same BFW protocol. They show the
development of the system; adjacent differences can include calibration and
coverage effects and are not automatically isolated component effects:

    1  one enrolled photograph, threshold borrowed from 1:1 verification
    2  three photographs averaged into a template, same borrowed threshold
    3  the same template, but a threshold calibrated for gallery search
    4  a logistic-regression classifier placed on top of the search
    5  SCRFD and ArcFace substituted for YuNet and SFace, calibrated afresh

Layer 5 earns its place for a particular reason. Swapping in stronger models
shows how much of any gain belongs to the framework and how much simply
belongs to better components. The classifier in layer 4 is the only model
trained here, and it is fitted on BFW development identities that appear
nowhere in the held-out test set.

Nothing in this project decides anything about a person. Every result is a
signal that opens a case for a human reviewer.

Which direction counts as suspicious depends on the question being asked, and
the two are easily confused:

    duplicate-profile screening   a *high* similarity to another enrolled
                                  gallery identity opens a duplicate-profile
                                  review
    profile-photo consistency     a *low* similarity to the profile's own
                                  enrolled template opens an inconsistency
                                  review
    extraction failure            no match or mismatch decision is made

Keeping development data apart from evaluation data is enforced in code rather
than merely promised in prose:

    pairsDevTrain.txt -> candidate thresholds only
    pairsDevTest.txt  -> deterministic selection, then freezing
    pairs.txt         -> official ten-fold CV, threshold fitted on nine folds
    pairs_CPLFW.txt   -> raw CPLFW under the separate frozen development threshold

Run it with the VS Code play button, or:

    python ACP_arden.py                    # interactive menu
    python ACP_arden.py --mode self-test   # deterministic synthetic tests

No image dataset and no ONNX weight file is kept in the repository. Their
locations are read from a local ``.env`` that Git ignores.
"""

# =============================================================================
# 1. Imports and programme metadata
# =============================================================================

# Allow type annotations to refer to classes defined later in this single-file
# research artefact.
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import heapq
import importlib.util
import hmac
import json
import math
import os
import platform
import random
import re
import sqlite3
import statistics
import string
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np

PROGRAMME_NAME = "COM7014 Advanced Computing Project"
PROGRAMME_TITLE = "COM7014 Advanced Computing Project — Face Verification"
PROGRAMME_VERSION = "1.0.0"

# Every path in this programme is derived from the file's own location, so the
# working directory never changes which results are read or written.
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
AGGREGATE_ROOT = RESULTS_ROOT / "aggregate"
RAW_ROOT = RESULTS_ROOT / "raw"
DEFAULT_REVIEW_DB = RAW_ROOT / "review.sqlite"
DEFAULT_GALLERY_MANIFEST = RAW_ROOT / "gallery_manifest.json"

# Attribution. Every line of this file is original project code: the OpenCV,
# NumPy, Pillow and Streamlit calls below use those libraries' documented
# public APIs, which is use rather than adaptation of their source. Wherever an
# external model, dataset, API or published method enters the pipeline, a
# four-field attribution header records its origin immediately above the code
# concerned. The complete register, with licences and digests, is REFERENCES.md.
# Neither benchmark dataset nor either model file is redistributed here.
#
# Reading order. The thirty numbered sections run from configuration through
# to the launcher, so the file may be read from top to bottom in roughly the
# order the pipeline executes. Comments that explain a language construct are
# addressed to a reader following the method rather than the syntax, and appear
# at the first use of each construct.


# =============================================================================
# 2. Configuration and pinned model hashes
# =============================================================================
#
# Changing any value in this section changes the evaluation partition. A
# threshold calibrated under one contract must never be applied under another,
# so these are constants rather than command-line options.

EMBEDDING_DIMENSIONS = 128
MODEL_VERSION = "opencv-sface-2021dec-yunet-2023mar"
PREPROCESSING_REVISION = "opencv-yunet-sface-exif-bgr-l2-v1"

# The detection and embedding stages are external pretrained artefacts, not
# code adapted into this file. Both are consumed through OpenCV's documented
# public API; the two blocks below record where each originates.
##############
# Title: YuNet: A Tiny Millisecond-level Face Detector
# Author: Wu, W., Peng, H. and Yu, S., Machine Intelligence Research, 20(5), pp. 656-665
# Date: 2023
# Availability: https://doi.org/10.1007/s11633-023-1423-y
# GitHub: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
##############
##############
# Title: SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition
# Author: Zhong, Y., Deng, W., Hu, J., Zhao, D., Li, X. and Wen, D.
# Date: 2021
# Availability: https://doi.org/10.1109/TIP.2020.3048632
# GitHub: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
##############
##############
# Title: OpenCV Zoo, distributor of the two pinned ONNX weight files
# Author: OpenCV team; YuNet weights MIT, SFace weights Apache-2.0
# Date: 2023 (YuNet 2023mar release), 2021 (SFace 2021dec release)
# Availability: https://github.com/opencv/opencv_zoo
##############
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"

# Digests of the official OpenCV Zoo release. A file that does not match is
# refused rather than loaded (see verify_model_file in section 4).
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"

DETECTOR_SCORE_THRESHOLD = 0.9
DETECTOR_NMS_THRESHOLD = 0.3
DETECTOR_TOP_K = 5000

# Dependencies whose exact installed version forms part of the evaluation
# partition, because they can move floating-point results at the margins.
EXPECTED_DEPENDENCY_VERSIONS = {
    "numpy": "2.5.1",
    "opencv-python-headless": "4.13.0.92",
    "Pillow": "12.3.0",
    # Fitting the review classifier is version-sensitive at the margins.
    "scikit-learn": "1.9.0",
}

DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
HARD_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 12_000_000
HARD_MAX_IMAGE_PIXELS = 40_000_000

DEFAULT_RANDOM_SEED = 20260727

# Archive checksums recorded for this project's own dataset acquisition. They
# describe the copies evaluated here; the CPLFW authors do not publish an
# official archive checksum of their own.
# The benchmark datasets are external inputs used under their authors'
# published terms. Neither is redistributed here; only these checksums, which
# identify the exact copies this project evaluated, are recorded.
##############
# Title: Labeled Faces in the Wild (LFW) benchmark database
# Author: Huang, G.B., Ramesh, M., Berg, T. and Learned-Miller, E. (UMass Amherst TR 07-49)
# Date: October 2007
# Availability: http://vis-www.cs.umass.edu/lfw/lfw.pdf
##############
# The alignment procedure that produced the funnelled image set evaluated here
# is a separate publication, and a distinct image set from the unaligned
# distribution, so it is cited in its own right.
##############
# Title: Unsupervised Joint Alignment of Complex Images
# Author: Huang, G.B., Jain, V. and Learned-Miller, E. (ICCV)
# Date: 2007
# Availability: http://vis-www.cs.umass.edu/lfw/
##############
LFW_ARCHIVE_FILENAME = "lfwfunneled.tgz"
LFW_ARCHIVE_MD5 = "1b42dfed7d15c9b2dd63d5e5840c86ad"
##############
# Title: Cross-Pose LFW (CPLFW) benchmark database
# Author: Zheng, T. and Deng, W. (Beijing Univ. of Posts and Telecommunications, TR 18-01)
# Date: February 2018
# Availability: http://www.whdeng.cn/cplfw/
##############
CPLFW_ARCHIVE_FILENAME = "CPLFW.zip"
CPLFW_ARCHIVE_SHA256 = "9a09dd1ebe1a000c52f69f365f5d564cd529f1fcf4f0479510231856f358f416"

# CPLFW ships two non-interchangeable image sets inside the same archive: the
# authors' raw, unconstrained images and a separately pre-cropped and aligned
# copy. A result must never be ambiguous about which one it scored, so the
# variant is an explicit, recorded field rather than an assumption.
CPLFW_RAW_ARCHIVE_FILENAME = "images.rar"
CPLFW_RAW_ARCHIVE_SHA256 = "7baca61dda21341eaa642f229eedfbba1d0aaa2d22447d79e158920106831165"
CPLFW_ALIGNED_ARCHIVE_FILENAME = "cp-aligned.zip"
CPLFW_ALIGNED_ARCHIVE_SHA256 = "420adcc13f1ab9510d8f99af04dbfb1695645ff73942c2a1010c5c01fd8367e2"

CPLFW_IMAGE_VARIANTS = ("raw", "aligned")

LFW_CALIBRATION_PROTOCOL = "pairsDevTrain.txt"
LFW_DEVELOPMENT_PROTOCOL = "pairsDevTest.txt"
LFW_FINAL_PROTOCOL = "pairs.txt"
CPLFW_PROTOCOL = "pairs_CPLFW.txt"

REQUIRED_LFW_PROTOCOLS = (
    LFW_CALIBRATION_PROTOCOL,
    LFW_DEVELOPMENT_PROTOCOL,
    LFW_FINAL_PROTOCOL,
)

CPLFW_EXPECTED_PAIRS = 6000
CPLFW_EXPECTED_PER_CLASS = 3000

SCHEMA_VERSION = 1

# Identifies gallery accounting that retains failed enrolments in the intended
# population. Conditional metrics count scored probes; end-to-end metrics
# count every intended probe, including those without an enrolled reference.
GALLERY_METHODOLOGY_REVISION = "open-set-gallery-accounting-v2"

POLICY_NOTE = (
    "Duplicate-profile screening: a high similarity to another enrolled gallery "
    "identity opens a case for human review only. It is not evidence of scam activity "
    "and does not ban, reject or accuse any identity. This polarity applies to gallery "
    "screening; profile-photo consistency refers a *low* similarity instead, and an "
    "extraction failure makes no decision at all."
)


def cplfw_provenance_fields(image_variant: str) -> Dict[str, str]:
    """Result fields that make a CPLFW evaluation's image variant explicit and
    impossible to omit. Anything other than 'raw' or 'aligned' is refused."""
    if image_variant == "raw":
        archive_filename = CPLFW_RAW_ARCHIVE_FILENAME
        archive_sha256 = CPLFW_RAW_ARCHIVE_SHA256
        image_source = "authors-distributed images.rar"
    elif image_variant == "aligned":
        archive_filename = CPLFW_ALIGNED_ARCHIVE_FILENAME
        archive_sha256 = CPLFW_ALIGNED_ARCHIVE_SHA256
        image_source = "authors-distributed cp-aligned.zip"
    else:
        raise ValueError(
            f"Unknown CPLFW image variant {image_variant!r}; expected one of {CPLFW_IMAGE_VARIANTS}"
        )
    return {
        "dataset_image_variant": image_variant,
        "dataset_image_source": image_source,
        "dataset_archive_filename": archive_filename,
        "dataset_archive_sha256": archive_sha256,
        "dataset_root_description": "private, gitignored local research storage; path omitted",
    }


# =============================================================================
# 3. Environment loading and path validation
# =============================================================================
#
# No dataset, protocol or model location is ever hard-coded. Each one is read
# from a git-ignored ``.env`` beside this file, or from the process
# environment, so the artefact carries no researcher-specific path.

ENV_FILENAME = ".env"

REQUIRED_ENVIRONMENT_VARIABLES = (
    "FACE_DATA_ROOT",
    "FACE_PROTOCOL_ROOT",
    "FACE_MODEL_ROOT",
    "FACE_CPLFW_RAW_ROOT",
)
OPTIONAL_ENVIRONMENT_VARIABLES = (
    "FACE_CACHE_ROOT",
    # Optional external benchmarks. Listed here so the privacy scanner adds
    # their configured roots to the forbidden-substring set as well.
    "FACE_BFW_ROOT",
    "FACE_BFW_METADATA_ROOT",
    "FACE_ARCFACE_MODEL_ROOT",
)


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def parse_env_text(text: str) -> Dict[str, str]:
    """Minimal ``.env`` reader: ``KEY=value``, single- or double-quoted values,
    blank lines and ``#`` comments. Deliberately not a dependency — quoted
    values matter here only because research storage paths contain spaces."""
    values: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Tolerate the shell "export" prefix, so the same file can be sourced
        # by a terminal as well as read here.
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        # Split at the first "=" only: a research storage path may itself
        # contain one, and it belongs to the value rather than the key.
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip a matched surrounding quote, which is how a path containing
        # spaces is written in this file.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def load_env_file(path: Path = PROJECT_ROOT / ENV_FILENAME) -> Dict[str, str]:
    """Read the local ``.env`` if present. A missing file is not an error: the
    caller may have exported the variables directly."""
    path = Path(path)
    if not path.is_file():
        return {}
    return parse_env_text(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class EnvironmentConfig:
    """Resolved research-storage roots. Every field is either explicitly
    supplied or ``None``; no default location is ever assumed."""

    data_root: Optional[Path]
    protocol_root: Optional[Path]
    model_root: Optional[Path]
    cplfw_raw_root: Optional[Path]
    cache_root: Optional[Path]
    # Optional external benchmarks. Absent configuration is not an error: the
    # five baseline experiments and every existing mode run without them.
    bfw_root: Optional[Path] = None
    bfw_metadata_root: Optional[Path] = None
    arcface_model_root: Optional[Path] = None

    # Build the configuration directly from the environment. Optional paths let
    # the baseline experiments run without BFW or the ArcFace comparison models.
    @classmethod
    def load(cls, env: Optional[Mapping[str, str]] = None) -> "EnvironmentConfig":
        """Process environment first, then the local ``.env`` as a fallback, so
        an explicit export always wins over the file."""
        source: Dict[str, str] = dict(load_env_file())
        source.update({k: v for k, v in (os.environ if env is None else env).items() if v})

        # Installed as a side effect of loading configuration, so every code
        # path that resolves storage roots also has identifiers available. The
        # key itself is never stored on the returned object, which is written
        # into provenance records.
        identifier_key = source.get(ID_HMAC_KEY_VARIABLE)
        if identifier_key:
            configure_id_hmac_key(identifier_key)

        def optional(name: str) -> Optional[Path]:
            value = source.get(name)
            return Path(value).expanduser() if value else None

        return cls(
            data_root=optional("FACE_DATA_ROOT"),
            protocol_root=optional("FACE_PROTOCOL_ROOT"),
            model_root=optional("FACE_MODEL_ROOT"),
            cplfw_raw_root=optional("FACE_CPLFW_RAW_ROOT"),
            cache_root=optional("FACE_CACHE_ROOT"),
            bfw_root=optional(BFW_ROOT_VARIABLE),
            bfw_metadata_root=optional(BFW_METADATA_ROOT_VARIABLE),
            arcface_model_root=optional(ARCFACE_MODEL_ROOT_VARIABLE),
        )

    def require_bfw_roots(self) -> Tuple[Path, Path]:
        """Resolve the BFW image root and metadata table, or stop on the exact
        blocker. The metadata root defaults to the image root, which is where
        the official archive places the datatable."""
        if self.bfw_root is None:
            raise BfwDatasetError(
                f"{BFW_ROOT_VARIABLE} is not set. The BFW open-set experiment needs the "
                f"official Balanced Faces in the Wild data, obtained from "
                f"https://github.com/visionjo/facerec-bias-bfw under its own terms. It is "
                f"never downloaded automatically and no mirror is used."
            )
        metadata_root = self.bfw_metadata_root or self.bfw_root
        matches = sorted(Path(metadata_root).glob("bfw*datatable*.csv"))
        if not matches:
            raise BfwDatasetError(
                f"No BFW datatable (bfw*datatable*.csv) found under the configured "
                f"{BFW_METADATA_ROOT_VARIABLE or BFW_ROOT_VARIABLE}. Expected the official "
                f"metadata table shipped with the dataset."
            )
        # The release ships the metadata table twice, once with a version in the
        # name and once without. Prefer the versioned copy so the recorded
        # provenance identifies which BFW release was evaluated.
        versioned = next((m for m in matches if re.search(r"v\d", m.name)), None)
        return Path(self.bfw_root), versioned or matches[0]

    def require_data_root(self) -> Path:
        return _require(self.data_root, "FACE_DATA_ROOT")

    def require_protocol_root(self) -> Path:
        return _require(self.protocol_root, "FACE_PROTOCOL_ROOT")

    def require_model_root(self) -> Path:
        return _require(self.model_root, "FACE_MODEL_ROOT")

    def require_lfw_root(self) -> Path:
        return self.require_data_root() / "lfw_funneled"

    def require_cplfw_raw_root(self) -> Path:
        if self.cplfw_raw_root is not None:
            return self.cplfw_raw_root
        return self.require_data_root() / "cplfw"

    def missing_variables(self) -> List[str]:
        present = {
            "FACE_DATA_ROOT": self.data_root,
            "FACE_PROTOCOL_ROOT": self.protocol_root,
            "FACE_MODEL_ROOT": self.model_root,
            "FACE_CPLFW_RAW_ROOT": self.cplfw_raw_root,
        }
        return [name for name, value in present.items() if value is None]

    def private_roots(self) -> List[Path]:
        return [
            root
            for root in (
                self.data_root,
                self.protocol_root,
                self.model_root,
                self.cplfw_raw_root,
                self.cache_root,
            )
            if root is not None
        ]


def _require(value: Optional[Path], name: str) -> Path:
    if value is None:
        raise ConfigurationError(
            f"{name} is not set. Copy .env.example to .env and fill it in, or export "
            f"{name} directly. This project never assumes a default path for real "
            f"dataset or model files."
        )
    return value


def project_relative(path: Path) -> str:
    """POSIX path relative to this file's directory, or the bare filename when
    the target lives outside the project. Artifacts record this rather than an
    absolute path, so a published result never carries a private location."""
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def redact_private_paths(text: str, config: Optional[EnvironmentConfig] = None) -> str:
    """Replace any configured private root, and any remaining home-directory
    prefix, with a placeholder. Applied to everything this programme prints, so
    a failure message can never disclose a storage location."""
    config = config if config is not None else EnvironmentConfig.load()
    replacements: List[Tuple[str, str]] = []
    for name, root in (
        ("FACE_DATA_ROOT", config.data_root),
        ("FACE_PROTOCOL_ROOT", config.protocol_root),
        ("FACE_MODEL_ROOT", config.model_root),
        ("FACE_CPLFW_RAW_ROOT", config.cplfw_raw_root),
        ("FACE_CACHE_ROOT", config.cache_root),
    ):
        if root is not None:
            replacements.append((str(root), f"<{name}>"))
    replacements.append((str(Path.home()), "<HOME>"))
    # Longest first, so a nested root is not partially rewritten by its parent.
    for needle, placeholder in sorted(replacements, key=lambda item: -len(item[0])):
        if needle:
            text = text.replace(needle, placeholder)
    return text


def announce(message: str) -> None:
    """Print a line with private locations redacted."""
    print(redact_private_paths(message))


# =============================================================================
# 4. Hashing, provenance and privacy helpers
# =============================================================================


class ModelUnavailableError(RuntimeError):
    """Raised when a model file is missing or fails hash verification."""


class DependencyContractError(RuntimeError):
    """Raised when an installed dependency does not match the pinned contract."""


# Opaque identifiers are keyed, not merely salted. A published fixed salt makes
# the mapping from a public identifier back to a dataset identity a dictionary
# attack over a known, enumerable name list; a secret key removes that.
OPAQUE_ID_VERSION = "hmac-sha256-v1"
OPAQUE_ID_HEX_LENGTH = 32
ID_HMAC_KEY_VARIABLE = "FACE_ID_HMAC_KEY"
MINIMUM_ID_HMAC_KEY_BYTES = 32

# Rejected outright so a template value can never reach a published artefact.
_PLACEHOLDER_KEY_MARKERS = (
    "changeme",
    "change-me",
    "replace",
    "example",
    "placeholder",
    "your-key",
    "yourkey",
    "todo",
    "secret",
    "xxxx",
)


class OpaqueIdentifierKeyError(RuntimeError):
    """Raised when the identifier HMAC key is missing, weak or malformed."""


# Held in memory only. Never written to a result, never logged, never hashed
# into a fingerprint that could confirm a guessed key.
_ID_HMAC_KEY: Optional[bytes] = None


def sha256_of_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        # Digest the file a megabyte at a time, so a large model binary is never
        # held in memory in full while its pinned hash is verified.
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_evaluated_image_set(paths: Iterable[Path], dataset_root: Path) -> str:
    """Fingerprint of *which* images an evaluation touched, without re-hashing
    gigabytes of pixels: SHA-256 of the sorted, newline-joined list of paths
    relative to the dataset root. Two runs sharing this value scored exactly
    the same image set."""
    dataset_root = Path(dataset_root).resolve()
    relative = sorted({str(Path(p).resolve().relative_to(dataset_root)) for p in paths})
    return hashlib.sha256("\n".join(relative).encode("utf-8")).hexdigest()


def verify_model_file(path: Path, expected_sha256: str) -> str:
    """Return the file's SHA-256 if it matches the pinned digest, else refuse."""
    path = Path(path)
    if not path.is_file():
        raise ModelUnavailableError(f"Model file not found: {path}")
    # The digest fingerprints the exact weight file. Comparing it against the
    # value pinned in source proves the evaluation used the intended model and
    # not a different release that happened to carry the same filename.
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        raise ModelUnavailableError(
            f"Model hash mismatch for {path}: expected {expected_sha256}, got {actual}. "
            f"Do not proceed — re-download the exact pinned OpenCV Zoo release."
        )
    return actual


def check_dependency_contract(*, strict: bool = True) -> Dict[str, Dict[str, Optional[str]]]:
    report: Dict[str, Dict[str, Optional[str]]] = {}
    mismatched: List[str] = []
    for package, expected in EXPECTED_DEPENDENCY_VERSIONS.items():
        try:
            installed: Optional[str] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            installed = None
        report[package] = {"expected": expected, "installed": installed}
        if installed != expected:
            mismatched.append(package)
    # Metadata alone is not sufficient. Installing opencv-python alongside
    # opencv-python-headless leaves both recorded while only one is imported,
    # so the runtime version of the library actually loaded is checked too.
    try:
        import cv2

        loaded = str(getattr(cv2, "__version__", ""))
    except Exception:  # pragma: no cover - cv2 is a pinned dependency
        loaded = ""
    expected_cv2 = EXPECTED_DEPENDENCY_VERSIONS.get("opencv-python-headless", "")
    if loaded and expected_cv2 and not expected_cv2.startswith(loaded):
        report["cv2 (imported)"] = {"expected": expected_cv2, "installed": loaded}
        mismatched.append(
            f"cv2 (imported {loaded}, expected {expected_cv2}; another OpenCV "
            f"distribution is shadowing opencv-python-headless)"
        )

    if mismatched and strict:
        raise DependencyContractError(
            "Dependency version mismatch for: "
            + ", ".join(mismatched)
            + ". Reinstall with the pinned versions in requirements.txt before running any "
            "evaluation that will be reported as evidence."
        )
    return report


def software_environment_report() -> Dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "dependencies": check_dependency_contract(strict=False),
    }


def decode_id_hmac_key(raw: str) -> bytes:
    """Decode a candidate key from URL-safe base64 or hexadecimal and refuse
    anything that would not carry at least 32 bytes of entropy. The value is
    never echoed back in an error message."""
    candidate = raw.strip()
    if not candidate:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} is empty. Generate one with: "
            f"python -c \"import secrets;print(secrets.token_urlsafe(32))\""
        )
    lowered = candidate.lower()
    for marker in _PLACEHOLDER_KEY_MARKERS:
        if marker in lowered:
            raise OpaqueIdentifierKeyError(
                f"{ID_HMAC_KEY_VARIABLE} looks like a placeholder value. Generate a real "
                f"key with: python -c \"import secrets;print(secrets.token_urlsafe(32))\""
            )

    decoded: Optional[bytes] = None
    # Hex first: a hex string is also valid base64 alphabet, and interpreting it
    # as base64 would understate its length.
    stripped = candidate.removeprefix("0x").removeprefix("0X")
    if len(stripped) % 2 == 0 and all(c in string.hexdigits for c in stripped):
        try:
            decoded = bytes.fromhex(stripped)
        except ValueError:
            decoded = None
    if decoded is None:
        padded = candidate + "=" * (-len(candidate) % 4)
        try:
            decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        except (ValueError, binascii.Error) as exc:
            raise OpaqueIdentifierKeyError(
                f"{ID_HMAC_KEY_VARIABLE} is not valid URL-safe base64 or hexadecimal."
            ) from exc

    if len(decoded) < MINIMUM_ID_HMAC_KEY_BYTES:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} decodes to {len(decoded)} bytes; at least "
            f"{MINIMUM_ID_HMAC_KEY_BYTES} are required. Generate one with: "
            f"python -c \"import secrets;print(secrets.token_urlsafe(32))\""
        )
    if len(set(decoded)) == 1:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} is a single repeated byte and carries no entropy."
        )
    return decoded


def configure_id_hmac_key(raw: Optional[str]) -> None:
    """Install the process-wide identifier key. Called once at start-up."""
    global _ID_HMAC_KEY
    if raw is None:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} is not set. It is required so that public identifiers "
            f"cannot be reversed by hashing a candidate name list. Add it to .env; generate "
            f"one with: python -c \"import secrets;print(secrets.token_urlsafe(32))\""
        )
    _ID_HMAC_KEY = decode_id_hmac_key(raw)


def id_hmac_key_is_configured() -> bool:
    return _ID_HMAC_KEY is not None


@contextmanager
def temporary_id_hmac_key(raw: str):
    """Install a key for the duration of a block. Used by the synthetic
    self-tests so they run without a configured environment, and by unit tests
    that need two different keys in one process."""
    global _ID_HMAC_KEY
    previous = _ID_HMAC_KEY
    try:
        _ID_HMAC_KEY = decode_id_hmac_key(raw)
        yield
    finally:
        _ID_HMAC_KEY = previous


# Fixed key used only by the in-process synthetic self-tests. It protects
# nothing: it exists so the self-tests are runnable on a machine with no
# research configuration, and it never touches real dataset identities.
SELF_TEST_ID_HMAC_KEY = "0" * 63 + "1"


# Keyed HMAC rather than a plain digest. An identifier derived from an unkeyed
# hash of a public benchmark name can be reversed by hashing a short list of
# candidate names, so a secret key is required and is never published, not even
# as a fingerprint.
def opaque_id(value: str) -> str:
    """Deterministic, one-way identifier standing in for a real identity or
    sample name. Deterministic under a fixed key so a re-run reproduces the same
    identifiers without ever storing the reversible mapping, and keyed so that
    nobody without the key can rebuild that mapping by hashing candidate names."""
    if _ID_HMAC_KEY is None:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} has not been configured; refusing to emit an "
            f"identifier that would be reversible by dictionary attack."
        )
    # Keyed hashing, not plain hashing. Without the secret key an attacker
    # could hash a list of candidate dataset names and recover which person
    # each published identifier refers to.
    digest = hmac.new(_ID_HMAC_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:OPAQUE_ID_HEX_LENGTH]


# Constant-time comparison, so the time taken to reject a value cannot reveal
# how much of it matched.
def opaque_ids_match(left: str, right: str) -> bool:
    """Constant-time comparison for values derived from the secret key."""
    return hmac.compare_digest(left, right)


def scrub_filename(path: Path) -> str:
    """Only the filename component, never an absolute or project-relative path."""
    return Path(path).name


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# 5. Image loading
# =============================================================================
#
# Bounded before decoding and after decoding, EXIF orientation normalised,
# animated images refused, and every failure raised as one catchable exception
# rather than a silent skip that would quietly shrink the denominator.


class ImageLoadError(RuntimeError):
    """Raised for any image that cannot be safely and strictly loaded."""


# Keep a loaded image immutable, so no stage of the pipeline can alter the
# pixels another stage has already measured.
@dataclass(frozen=True)
class LoadedImage:
    bgr: np.ndarray  # HxWx3 uint8, OpenCV's BGR channel order
    width: int
    height: int
    source_path: Path


# The size limits must be named at the call site, because two same-typed bounds
# passed positionally could be transposed without any error being raised.
def load_image_bgr(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> LoadedImage:
    from PIL import Image, ImageOps

    if max_bytes > HARD_MAX_IMAGE_BYTES:
        raise ImageLoadError(f"max_bytes {max_bytes} exceeds hard ceiling {HARD_MAX_IMAGE_BYTES}")
    if max_pixels > HARD_MAX_IMAGE_PIXELS:
        raise ImageLoadError(f"max_pixels {max_pixels} exceeds hard ceiling {HARD_MAX_IMAGE_PIXELS}")

    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Image file does not exist: {path}")

    size = path.stat().st_size
    if size == 0:
        raise ImageLoadError(f"Image file is empty: {path}")
    if size > max_bytes:
        raise ImageLoadError(f"Image file {path} is {size} bytes, exceeds max_bytes={max_bytes}")

    try:
        with Image.open(path) as raw:
            raw.load()
            if getattr(raw, "is_animated", False):
                raise ImageLoadError(f"Animated/multi-frame images are not supported: {path}")
            oriented = ImageOps.exif_transpose(raw)
            if oriented is None:
                raise ImageLoadError(f"Failed to normalise EXIF orientation for: {path}")
            width, height = oriented.size
            if width * height > max_pixels:
                raise ImageLoadError(
                    f"Image {path} has {width * height} pixels, exceeds max_pixels={max_pixels}"
                )
            rgb = oriented.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except ImageLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalise every decode failure
        # Keep the underlying decode error alongside this project's own failure
        # category, so an unreadable image is still counted as an extraction
        # failure rather than crashing the run.
        raise ImageLoadError(f"Could not decode image {path}: {exc}") from exc

    # Convert Pillow's RGB channel order to the BGR order OpenCV expects, and
    # copy into a contiguous buffer the detector can read. Every experiment
    # loads images this way, so the channel order is part of the pinned
    # preprocessing revision.
    bgr = np.ascontiguousarray(array[:, :, ::-1])
    return LoadedImage(bgr=bgr, width=width, height=height, source_path=path)


# =============================================================================
# 6. YuNet face detection
# =============================================================================
#
# Exactly one detectable face is required, matching the research question
# ("does this photo show one identifiable face"). Zero and multiple detections
# are counted as explicit outcomes in section 12, never silently dropped.


# Accept the real OpenCV wrapper, the InsightFace comparison wrapper and the
# deterministic synthetic stand-ins used by the self-tests through one
# evaluation interface, so no experiment depends on a particular class.
@runtime_checkable
class FaceDetector(Protocol):
    """Returns one detected face's row, or raises ``FaceCountError``."""

    def detect_single_face(self, bgr: np.ndarray) -> np.ndarray: ...


class FaceCountError(RuntimeError):
    """Raised when an image does not contain exactly one detectable face."""

    def __init__(self, face_count: int):
        super().__init__(f"Expected exactly one face, found {face_count}")
        self.face_count = face_count


@dataclass(frozen=True)
class DetectorSettings:
    score_threshold: float = DETECTOR_SCORE_THRESHOLD
    nms_threshold: float = DETECTOR_NMS_THRESHOLD
    top_k: int = DETECTOR_TOP_K


##############
# Title: The OpenCV Library (cv2.FaceDetectorYN and cv2.FaceRecognizerSF APIs)
# Author: Bradski, G., Dr. Dobb's Journal of Software Tools, 25(11), pp. 120-125
# Date: 2000 (library); APIs used as documented in the OpenCV 4.x release series
# Availability: https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html
##############
# The wrapper below is original code. It calls OpenCV's documented public API,
# which is use of the library rather than adaptation of its source; the header
# records the API's origin, not a borrowed implementation.
class YuNetDetector:
    def __init__(
        self,
        model_path: Path,
        expected_sha256: str,
        settings: DetectorSettings = DetectorSettings(),
    ):
        import cv2

        self.model_sha256 = verify_model_file(Path(model_path), expected_sha256)
        self.settings = settings
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            settings.score_threshold,
            settings.nms_threshold,
            settings.top_k,
        )

    def detect_single_face(self, bgr: np.ndarray) -> np.ndarray:
        """Return the single detection's row from YuNet's output matrix
        (bounding box, five landmarks, confidence), or raise FaceCountError."""
        height, width = bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(bgr)
        # YuNet returns None rather than an empty matrix when it finds nothing,
        # so the None case and the wrong-count case are tested together. Testing
        # them in one expression is also what lets a type checker prove the
        # subscript below is safe.
        if faces is None or len(faces) != 1:
            raise FaceCountError(0 if faces is None else len(faces))
        return faces[0]


# =============================================================================
# 7. SFace embedding
# =============================================================================
#
# Produces a raw, not yet L2-normalised 128-value feature vector aligned from a
# YuNet detection. Normalisation is a separate step (section 8) so the live
# pipeline and any offline re-analysis share one normalisation code path.


@runtime_checkable
class FaceEmbedder(Protocol):
    """Returns a raw, not yet L2-normalised embedding for a detected face."""

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray: ...


class EmbeddingShapeError(RuntimeError):
    """Raised when SFace returns a vector of unexpected shape."""


class SFaceEmbedder:
    def __init__(self, model_path: Path, expected_sha256: str):
        import cv2

        self.model_sha256 = verify_model_file(Path(model_path), expected_sha256)
        self._recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        # alignCrop reads the landmark columns as 32-bit floats and does not
        # check the array's type, so a 64-bit row is misread rather than
        # refused and the crop that follows bears no relation to the face.
        # The conversion costs nothing for YuNet's own rows, which are already
        # 32-bit, and makes the embedder safe to pair with any detector.
        aligned = self._recognizer.alignCrop(bgr, np.asarray(face_row, dtype=np.float32))
        feature = self._recognizer.feature(aligned)
        embedding = np.asarray(feature, dtype=np.float64).reshape(-1)
        if embedding.shape[0] != EMBEDDING_DIMENSIONS:
            raise EmbeddingShapeError(
                f"Unexpected embedding dimensionality {embedding.shape[0]}, "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        return embedding


def configure_deterministic_opencv() -> Dict[str, Any]:
    """Reduce OpenCV's execution variability as far as the platform allows.

    YuNet's detection score is not bit-stable, so an image scoring near the 0.9
    acceptance threshold can be detected on one run and missed on the next.
    Disabling OpenCL removes the largest source, and ``setNumThreads(0)``
    forces single-threaded execution -- note that under Apple's GCD parallel
    framework ``setNumThreads(1)`` is silently a no-op while ``0`` is honoured.

    This does not achieve bit-level reproducibility across processes: roughly
    one image in a thousand still flips between fresh interpreters, which
    appears to be floating-point variation inside OpenCV's DNN backend rather
    than anything this project controls. The canonical run cache, not this
    function, is what guarantees that every derived artefact reports the same
    extraction outcomes."""
    import cv2

    # setNumThreads(0) forces single-threaded under GCD; 1 is a no-op there.
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    return {
        "opencv_threads_reported": int(cv2.getNumThreads()),
        "opencv_opencl_enabled": bool(cv2.ocl.useOpenCL()),
        "opencv_parallel_framework": "GCD" if "GCD" in cv2.getBuildInformation() else "other",
        "bitwise_reproducible_across_processes": False,
        "reproducibility_mechanism": (
            "Canonical run cache. Detection is not bit-stable across fresh processes on "
            "this platform, so every derived artefact reads one cached extraction rather "
            "than re-deriving it."
        ),
    }


def load_models(model_root: Path) -> Tuple[YuNetDetector, SFaceEmbedder]:
    """Hash-verified detector and embedder pair. Any digest mismatch stops the
    run here rather than producing a result under an unknown model."""
    configure_deterministic_opencv()
    detector = YuNetDetector(Path(model_root) / YUNET_FILENAME, YUNET_SHA256)
    embedder = SFaceEmbedder(Path(model_root) / SFACE_FILENAME, SFACE_SHA256)
    return detector, embedder


# =============================================================================
# 8. Similarity and normalisation
# =============================================================================


class SimilarityError(ValueError):
    """Raised for malformed embeddings (wrong shape, non-finite, zero norm)."""


def l2_normalize(vector: np.ndarray, *, tolerance: float = 1e-7) -> np.ndarray:
    # Flatten to a single row of numbers, whatever shape the model returned.
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.shape[0] == 0:
        raise SimilarityError("Vector must have at least one dimension.")
    if not np.all(np.isfinite(vector)):
        raise SimilarityError("Vector must contain only finite numbers before normalisation.")
    # The norm is the vector's length. Dividing by it rescales the face
    # representation to unit length, so later comparisons measure direction
    # only and are unaffected by how strong the raw signal happened to be.
    norm = math.sqrt(float(np.dot(vector, vector)))
    if norm <= 1e-12:
        raise SimilarityError("Vector norm is too close to zero to normalise safely.")
    normalized = vector / norm
    # Confirm the result really is unit length. A silent failure here would
    # distort every similarity computed from this face.
    result_norm = math.sqrt(float(np.dot(normalized, normalized)))
    if abs(result_norm - 1.0) > tolerance:
        raise SimilarityError("Normalisation self-check failed.")
    return normalized


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    # Two faces can only be compared when described by the same number of
    # measurements. A mismatch means embeddings from two different models.
    if left.shape[0] == 0 or left.shape[0] != right.shape[0]:
        raise SimilarityError("Embeddings must have the same non-zero number of dimensions.")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise SimilarityError("Embeddings must contain only finite numbers.")
    left_norm = math.sqrt(float(np.dot(left, left)))
    right_norm = math.sqrt(float(np.dot(right, right)))
    if left_norm == 0.0 or right_norm == 0.0:
        raise SimilarityError("Embeddings must have a non-zero norm.")
    # The similarity score used throughout the project: 1.0 means the two
    # representations point the same way, 0.0 means they are unrelated.
    return float(np.dot(left, right) / (left_norm * right_norm))


# =============================================================================
# 9. LFW and CPLFW protocol parsing
# =============================================================================
#
# The two datasets' authors distribute genuinely different formats, confirmed
# against the real distributed files rather than assumed from secondary
# sources:
#
# - LFW (pairs.txt, pairsDevTrain.txt, pairsDevTest.txt): a header line, then
#   rows of either "identity image_a image_b" (matched) or
#   "identity_a image_a identity_b image_b" (mismatched), with images stored
#   per identity at identity/identity_%04d.jpg.
# - CPLFW (pairs_CPLFW.txt): no header; each pair is exactly two consecutive
#   "filename label" lines sharing the same label, with images stored flat
#   under the dataset root.


# The two file *formats* parsed below are defined by the respective dataset
# authors and are documented in the references recorded in section 2 (LFW:
# Huang et al., 2007; CPLFW: Zheng and Deng, 2018). The parsers themselves are
# original code, written against the authors' distributed files rather than
# adapted from any other implementation.
class ProtocolError(RuntimeError):
    """Raised for any malformed, inconsistent, or unsafe protocol file."""


@dataclass(frozen=True)
class Pair:
    left_path: Path
    right_path: Path
    same_identity: bool
    left_identity: str
    right_identity: str


def _image_filename(identity: str, image_number: str) -> str:
    try:
        number = int(image_number)
    except ValueError as exc:
        raise ProtocolError(f"Image number {image_number!r} is not an integer") from exc
    return f"{identity}_{number:04d}.jpg"


def _resolve_image_path(dataset_root: Path, identity: str, image_number: str) -> Path:
    filename = _image_filename(identity, image_number)
    candidate = (dataset_root / identity / filename).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise ProtocolError(f"Resolved image path escapes dataset root: {candidate}") from exc
    if not candidate.is_file():
        raise ProtocolError(f"Missing image referenced by protocol: {candidate}")
    return candidate


def _validate_header(
    header: Sequence[str], same_count: int, diff_count: int, protocol_path: Path
) -> None:
    # The published LFW protocols use two header forms. A single number states
    # one count for each class; two numbers state folds and pairs per fold.
    # Both are checked against what was actually parsed, so a truncated or
    # altered protocol file is refused rather than evaluated in part.
    if len(header) == 1:
        expected = int(header[0])
        if same_count != expected or diff_count != expected:
            raise ProtocolError(
                f"{protocol_path}: header declares {expected} matched and {expected} "
                f"mismatched pairs, found {same_count} matched and {diff_count} mismatched"
            )
    elif len(header) == 2:
        folds, per_fold = int(header[0]), int(header[1])
        expected = folds * per_fold
        if same_count != expected or diff_count != expected:
            raise ProtocolError(
                f"{protocol_path}: header declares {folds} folds x {per_fold} pairs per "
                f"class, expected {expected} matched and {expected} mismatched, found "
                f"{same_count} matched and {diff_count} mismatched"
            )
    else:
        raise ProtocolError(f"{protocol_path}: unrecognised header format: {list(header)}")


def parse_lfw_pairs(protocol_path: Path, dataset_root: Path) -> List[Pair]:
    protocol_path = Path(protocol_path)
    dataset_root = Path(dataset_root).resolve()

    if not protocol_path.is_file():
        raise ProtocolError(f"Protocol file does not exist: {protocol_path}")

    raw_lines = protocol_path.read_text(encoding="utf-8").strip("\n").split("\n")
    if not raw_lines or not raw_lines[0].strip():
        raise ProtocolError(f"Empty protocol file: {protocol_path}")

    # The first line states how many pairs the file should contain; the rest
    # are the pairs themselves.
    header = raw_lines[0].split()
    data_lines = raw_lines[1:]

    pairs: List[Pair] = []
    # Tracks pairs already seen, so a repeated pair cannot be scored twice and
    # quietly weight one comparison more heavily than the others.
    seen: Set[Tuple[str, str]] = set()
    same_count = 0
    diff_count = 0

    # Numbering starts at 2 because line 1 was the header, so any error message
    # points at the line a reader would actually find in the file.
    for line_number, raw_line in enumerate(data_lines, start=2):
        line = raw_line.strip()
        if not line:
            continue
        columns = line.split("\t") if "\t" in line else line.split()

        # The official format encodes the answer in the column count. Three
        # columns name one person twice, so the pair is a genuine match; four
        # columns name two different people, so it is an impostor pair.
        if len(columns) == 3:
            identity, image_a, image_b = columns
            left = _resolve_image_path(dataset_root, identity, image_a)
            right = _resolve_image_path(dataset_root, identity, image_b)
            same_identity = True
            left_identity = right_identity = identity
        elif len(columns) == 4:
            identity_a, image_a, identity_b, image_b = columns
            left = _resolve_image_path(dataset_root, identity_a, image_a)
            right = _resolve_image_path(dataset_root, identity_b, image_b)
            same_identity = False
            left_identity, right_identity = identity_a, identity_b
        else:
            raise ProtocolError(
                f"{protocol_path}:{line_number}: expected 3 or 4 columns, got {len(columns)}"
            )

        # Reversing a pair does not create an independent comparison.
        first, second = sorted((str(left), str(right)))
        key = (first, second)
        if key in seen:
            raise ProtocolError(f"{protocol_path}: duplicate pair detected: {key}")
        seen.add(key)

        pairs.append(Pair(left, right, same_identity, left_identity, right_identity))
        if same_identity:
            same_count += 1
        else:
            diff_count += 1

    if same_count == 0 or diff_count == 0:
        raise ProtocolError(
            f"{protocol_path} must contain both matched and mismatched pairs; "
            f"found {same_count} matched, {diff_count} mismatched"
        )

    _validate_header(header, same_count, diff_count, protocol_path)
    return pairs


def lfw_official_fold_indices(protocol_path: Path, pairs: Sequence[Pair]) -> List[List[int]]:
    """Preserve the ten folds supplied by LFW, including failed extractions.

    A fold contains 300 genuine pairs followed by 300 impostor pairs. Splitting
    only successfully scored pairs would move the published fold boundaries.
    This is pair-level cross-validation; it does not assert identity separation.
    """
    header = Path(protocol_path).read_text(encoding="utf-8").splitlines()[0].split()
    if header != ["10", "300"] or len(pairs) != 6000:
        raise ProtocolError("Final LFW evaluation requires the official 10 x 600-pair protocol.")
    folds = [list(range(start, start + 600)) for start in range(0, 6000, 600)]
    for number, indices in enumerate(folds, start=1):
        labels = [pairs[index].same_identity for index in indices]
        if labels != [True] * 300 + [False] * 300:
            raise ProtocolError(
                f"LFW fold {number} must contain 300 genuine then 300 impostor pairs."
            )
    return folds


def _cplfw_identity_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    # CPLFW filenames end in an image number, so the identity is everything
    # before the final underscore. A name without that suffix is returned
    # unchanged rather than truncated.
    prefix, _, suffix = stem.rpartition("_")
    return prefix if prefix and suffix.isdigit() else stem


def _resolve_flat_image_path(dataset_root: Path, filename: str) -> Path:
    candidate = (dataset_root / filename).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise ProtocolError(f"Resolved image path escapes dataset root: {candidate}") from exc
    if not candidate.is_file():
        raise ProtocolError(f"Missing image referenced by protocol: {candidate}")
    return candidate


def parse_cplfw_pairs(protocol_path: Path, dataset_root: Path) -> List[Pair]:
    protocol_path = Path(protocol_path)
    dataset_root = Path(dataset_root).resolve()

    if not protocol_path.is_file():
        raise ProtocolError(f"Protocol file does not exist: {protocol_path}")

    lines = [
        line for line in protocol_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise ProtocolError(f"Empty protocol file: {protocol_path}")
    if len(lines) % 2 != 0:
        raise ProtocolError(
            f"{protocol_path}: expected an even number of lines (two per pair), got {len(lines)}"
        )

    pairs: List[Pair] = []
    seen: Set[Tuple[str, str]] = set()
    same_count = 0
    diff_count = 0

    # CPLFW writes one photograph per line, so a pair occupies two consecutive
    # lines and the loop advances two at a time.
    for index in range(0, len(lines), 2):
        line_number = index + 1
        first_columns = lines[index].split()
        second_columns = lines[index + 1].split()

        if len(first_columns) != 2 or len(second_columns) != 2:
            raise ProtocolError(
                f"{protocol_path}:{line_number}: expected 'filename label' on each of a pair's two lines"
            )

        first_filename, first_label = first_columns
        second_filename, second_label = second_columns

        # The label states whether the two photographs show the same person.
        # Both lines of one pair must agree; disagreement means the file has
        # been misaligned and the pairs no longer describe what they claim.
        if first_label not in {"0", "1"} or second_label not in {"0", "1"}:
            raise ProtocolError(f"{protocol_path}:{line_number}: label must be 0 or 1")
        if first_label != second_label:
            raise ProtocolError(
                f"{protocol_path}:{line_number}: a pair's two lines must share the same label, "
                f"got {first_label!r} and {second_label!r}"
            )

        left = _resolve_flat_image_path(dataset_root, first_filename)
        right = _resolve_flat_image_path(dataset_root, second_filename)

        key = (str(left), str(right))
        if key in seen:
            raise ProtocolError(f"{protocol_path}: duplicate pair detected: {key}")
        seen.add(key)

        # Label 1 marks a genuine pair, 0 an impostor pair.
        same_identity = first_label == "1"
        pairs.append(
            Pair(
                left,
                right,
                same_identity,
                _cplfw_identity_from_filename(first_filename),
                _cplfw_identity_from_filename(second_filename),
            )
        )
        if same_identity:
            same_count += 1
        else:
            diff_count += 1

    if same_count == 0 or diff_count == 0:
        raise ProtocolError(
            f"{protocol_path} must contain both matched and mismatched pairs; "
            f"found {same_count} matched, {diff_count} mismatched"
        )

    return pairs


# =============================================================================
# 10. Verification metrics
# =============================================================================
#
# Convention: label 1 is the same identity (a match), label 0 a different
# identity. Higher similarity is more match-like, and the decision rule is
# "score >= threshold implies a predicted match". Implemented in plain NumPy so
# the dependency contract stays small and fully pinned.

# Every metric entry point accepts either a plain sequence or a NumPy array, so
# callers need not convert scores back and forth between the two.
ScoreInput = Union[Sequence[float], np.ndarray]
LabelInput = Union[Sequence[int], np.ndarray]


class MetricsError(ValueError):
    """Raised for empty, ragged, non-finite or single-class score/label input."""


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile, matching NumPy's default method."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return float(ordered[0])
    # The position the requested percentile falls at. It is rarely a whole
    # number, so it usually lies between two of the sorted values.
    index = (pct / 100.0) * (len(ordered) - 1)
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    # Interpolate between the two neighbours rather than rounding, which is
    # what makes a 95th-percentile latency comparable between runs.
    fraction = index - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return (
            self.true_positive + self.false_positive + self.true_negative + self.false_negative
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
        }


# Reject empty, ragged, non-finite or single-class input before any metric is
# calculated, so a malformed evaluation stops rather than reporting a figure
# derived from unusable scores.
def _validate_inputs(scores: ScoreInput, labels: LabelInput) -> Tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if scores_arr.shape[0] != labels_arr.shape[0]:
        raise MetricsError("scores and labels must have the same length")
    if scores_arr.shape[0] == 0:
        raise MetricsError("scores/labels must not be empty")
    # A missing or infinite score would silently distort every rate computed
    # from it, so the evaluation stops rather than reporting a corrupted figure.
    if not np.all(np.isfinite(scores_arr)):
        raise MetricsError("scores must contain only finite numbers")
    unique_labels = set(np.unique(labels_arr).tolist())
    if not unique_labels.issubset({0, 1}):
        raise MetricsError(f"labels must be 0 or 1, found {sorted(unique_labels)}")
    # Both classes must be present. With genuine pairs only there is nothing to
    # falsely match, and with impostor pairs only nothing to correctly match,
    # so the resulting rates would be meaningless rather than merely extreme.
    if unique_labels != {0, 1}:
        raise MetricsError(
            f"labels must contain both classes (0 and 1); found only {sorted(unique_labels)}"
        )
    return scores_arr, labels_arr


def confusion_matrix(scores: ScoreInput, labels: LabelInput, threshold: float) -> ConfusionMatrix:
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    predicted_match = scores_arr >= threshold
    actual_match = labels_arr == 1

    return ConfusionMatrix(
        true_positive=int(np.sum(predicted_match & actual_match)),
        false_positive=int(np.sum(predicted_match & ~actual_match)),
        true_negative=int(np.sum(~predicted_match & ~actual_match)),
        false_negative=int(np.sum(~predicted_match & actual_match)),
    )


def rates_from_confusion(matrix: ConfusionMatrix) -> Dict[str, float]:
    positives = matrix.true_positive + matrix.false_negative
    negatives = matrix.true_negative + matrix.false_positive
    total = matrix.total

    accuracy = (matrix.true_positive + matrix.true_negative) / total if total else float("nan")
    precision = (
        matrix.true_positive / (matrix.true_positive + matrix.false_positive)
        if (matrix.true_positive + matrix.false_positive) > 0
        else float("nan")
    )
    recall = matrix.true_positive / positives if positives > 0 else float("nan")
    # A self-comparison is the NaN test here: an undefined rate must propagate
    # as NaN rather than silently contribute a zero to the derived metric.
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )
    false_match_rate = matrix.false_positive / negatives if negatives > 0 else float("nan")
    false_non_match_rate = matrix.false_negative / positives if positives > 0 else float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_match_rate": false_match_rate,
        "false_non_match_rate": false_non_match_rate,
        "true_match_rate": recall,
    }


def roc_points(scores: ScoreInput, labels: LabelInput) -> List[Dict[str, float]]:
    """ROC curve as {threshold, false_match_rate, true_match_rate} points, one
    per distinct score plus high/low sentinels, ordered by descending
    threshold."""
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    # The curve only changes shape at an observed score, so one point per
    # distinct score describes it exactly without interpolation.
    thresholds = np.unique(scores_arr)[::-1]
    # Two sentinels beyond the observed range anchor the curve at its extremes:
    # a threshold above every score matches nothing, one below every score
    # matches everything, giving the (0,0) and (1,1) endpoints.
    sentinel_high = float(thresholds[0]) + 1.0 if thresholds.size else 1.0
    sentinel_low = float(thresholds[-1]) - 1.0 if thresholds.size else -1.0
    all_thresholds = np.concatenate(([sentinel_high], thresholds, [sentinel_low]))

    # Each point is the error pair a deployment would actually experience at
    # that threshold, recomputed from the confusion matrix rather than
    # accumulated, so a rounding error cannot propagate along the curve.
    points: List[Dict[str, float]] = []
    for threshold in all_thresholds:
        matrix = confusion_matrix(scores_arr, labels_arr, float(threshold))
        rates = rates_from_confusion(matrix)
        points.append(
            {
                "threshold": float(threshold),
                "false_match_rate": rates["false_match_rate"],
                "true_match_rate": rates["true_match_rate"],
            }
        )
    return points


##############
# Title: The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve
# Author: Hanley, J.A. and McNeil, B.J., Radiology, 143(1), pp. 29-36
# Date: 1982
# Availability: https://doi.org/10.1148/radiology.143.1.7063747
##############
# The equivalence this function relies on -- that the area under the ROC curve
# equals the probability that a randomly chosen positive outranks a randomly
# chosen negative, the quantity estimated by the Wilcoxon/Mann-Whitney rank
# statistic -- is the standard result established in the reference above. The
# implementation below is original and deliberately avoids a machine-learning
# dependency for one statistic.
def roc_auc(scores: ScoreInput, labels: LabelInput) -> float:
    """Rank-based ROC-AUC (the Mann-Whitney U identity), ties resolved with
    average ranks. Equivalent to the trapezoidal-rule area, without pulling in
    a machine-learning framework for one statistic."""
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    # A stable sort keeps equal scores in their original order, so the tie
    # handling below gives the same ROC-AUC on every run.
    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    ranks = np.empty(len(scores_arr), dtype=np.float64)

    n = len(sorted_scores)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        average_rank = (i + 1 + j) / 2.0  # 1-indexed rank, averaged across the tie block
        ranks[order[i:j]] = average_rank
        i = j

    positive_mask = labels_arr == 1
    n_pos = int(np.sum(positive_mask))
    n_neg = int(len(labels_arr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise MetricsError("roc_auc requires at least one example of each class")

    sum_ranks_positive = float(np.sum(ranks[positive_mask]))
    return float((sum_ranks_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def equal_error_rate(scores: ScoreInput, labels: LabelInput) -> Dict[str, float]:
    """EER by linear interpolation between the ROC points that bracket
    false_match_rate == false_non_match_rate."""
    points = roc_points(scores, labels)

    best_gap: Optional[float] = None
    best_eer: Optional[float] = None
    best_threshold: Optional[float] = None
    previous: Optional[Tuple[float, float, float, float]] = None

    for point in points:
        fmr = point["false_match_rate"]
        fnmr = 1.0 - point["true_match_rate"]
        gap = fmr - fnmr
        if previous is not None:
            prev_fmr, _prev_fnmr, prev_gap, prev_threshold = previous
            crosses = (prev_gap <= 0 <= gap) or (prev_gap >= 0 >= gap)
            if crosses:
                if gap == prev_gap:
                    eer, threshold = fmr, point["threshold"]
                else:
                    ratio = prev_gap / (prev_gap - gap)
                    eer = prev_fmr + ratio * (fmr - prev_fmr)
                    threshold = prev_threshold + ratio * (point["threshold"] - prev_threshold)
                if best_gap is None or abs(gap) < best_gap:
                    best_gap, best_eer, best_threshold = abs(gap), eer, threshold
        previous = (fmr, fnmr, gap, point["threshold"])

    # Both are assigned together above, so either both are set or neither is;
    # testing both is what makes that invariant explicit.
    if best_eer is None or best_threshold is None:
        closest = min(
            points, key=lambda p: abs(p["false_match_rate"] - (1.0 - p["true_match_rate"]))
        )
        best_eer = closest["false_match_rate"]
        best_threshold = closest["threshold"]

    return {"equal_error_rate": float(best_eer), "threshold": float(best_threshold)}


@dataclass(frozen=True)
class ThresholdCandidate:
    threshold: float
    strategy: str
    metrics: Dict[str, float]


def select_threshold(
    scores: ScoreInput,
    labels: LabelInput,
    *,
    strategy: str,
    target_false_match_rate: Optional[float] = None,
) -> ThresholdCandidate:
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    # Only an observed score can change a decision, so the search is exhaustive
    # over the distinct scores rather than over an arbitrary grid.
    candidate_thresholds = np.unique(scores_arr)

    if strategy == "eer":
        eer_result = equal_error_rate(scores_arr, labels_arr)
        threshold = eer_result["threshold"]
        metrics = rates_from_confusion(confusion_matrix(scores_arr, labels_arr, threshold))
        metrics["equal_error_rate"] = eer_result["equal_error_rate"]
        return ThresholdCandidate(threshold, strategy, metrics)

    if strategy == "target_fmr":
        if target_false_match_rate is None:
            raise MetricsError("target_fmr strategy requires target_false_match_rate")
        best_threshold: Optional[float] = None
        best_metrics: Optional[Dict[str, float]] = None
        # Walk downwards from the strictest threshold. Lowering it admits more
        # matches, so the last threshold still within the target bound is the
        # most permissive one that satisfies it, and therefore the one that
        # catches the most genuine matches at that error budget.
        for threshold in sorted(candidate_thresholds, reverse=True):
            metrics = rates_from_confusion(
                confusion_matrix(scores_arr, labels_arr, float(threshold))
            )
            fmr = metrics["false_match_rate"]
            if fmr == fmr and fmr <= target_false_match_rate:
                best_threshold, best_metrics = float(threshold), metrics
            elif best_threshold is not None:
                # The bound has been exceeded and cannot be regained by
                # lowering the threshold further, so the search stops here.
                break
        if best_threshold is None or best_metrics is None:
            raise MetricsError(
                f"No threshold achieves false_match_rate <= {target_false_match_rate}"
            )
        return ThresholdCandidate(best_threshold, strategy, best_metrics)

    if strategy not in {"balanced_accuracy", "f1"}:
        raise MetricsError(f"Unknown threshold-selection strategy: {strategy}")

    best_threshold = None
    best_score = float("-inf")
    best_metrics = None
    for threshold in candidate_thresholds:
        metrics = rates_from_confusion(confusion_matrix(scores_arr, labels_arr, float(threshold)))
        if strategy == "balanced_accuracy":
            tmr = metrics["true_match_rate"]
            fmr = metrics["false_match_rate"]
            tnr = 1.0 - fmr if fmr == fmr else float("nan")
            score = (tmr + tnr) / 2.0 if tmr == tmr and tnr == tnr else float("-inf")
        else:
            score = metrics["f1"] if metrics["f1"] == metrics["f1"] else float("-inf")
        # Ties are broken towards the stricter threshold, so two equally
        # scoring candidates always resolve the same way and the selection
        # stays reproducible.
        prefer_higher_on_tie = (
            score == best_score and best_threshold is not None and threshold > best_threshold
        )
        if score > best_score or prefer_higher_on_tie:
            best_score, best_threshold, best_metrics = score, float(threshold), metrics

    if best_threshold is None or best_metrics is None:
        raise MetricsError(f"Could not select a threshold using strategy={strategy}")
    return ThresholdCandidate(best_threshold, strategy, best_metrics)


# =============================================================================
# 11. Threshold calibration, selection and freezing
# =============================================================================
#
# The validation/held-out boundary is the single most important methodological
# guarantee in this artefact, so it is enforced in code across three stages:
#
# 1. calibrate() runs on pairsDevTrain.txt only and produces a table of
#    *candidate* thresholds. It never picks a winner; its status is
#    "candidates".
# 2. select_final_threshold() runs on pairsDevTest.txt, scores every candidate
#    and selects exactly one by a fixed, fully deterministic rule. Only this
#    step's output is marked "frozen".
# 3. require_frozen_threshold() protects CPLFW/gallery transfer evaluations.
#    Final LFW instead fits thresholds within official ten-fold cross-validation.

VALIDATION_SPLIT = "validation"
CANDIDATES_STATUS = "candidates"
FROZEN_STATUS = "frozen"

DEFAULT_TARGET_FALSE_MATCH_RATES: Sequence[float] = (0.001, 0.01, 0.05)

SELECTION_RULE = (
    "Maximum balanced accuracy on the development split (pairsDevTest.txt); "
    "ties broken by lower development-split false match rate, then by "
    "candidate name, for full determinism."
)


class CalibrationError(RuntimeError):
    """Raised when a calibration stage is run out of order or on the wrong split."""


@dataclass(frozen=True)
class CalibrationResult:
    split: str
    status: str
    candidates: Dict[str, ThresholdCandidate]


def calibrate(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    split: str,
    target_false_match_rates: Sequence[float] = DEFAULT_TARGET_FALSE_MATCH_RATES,
) -> CalibrationResult:
    """Stage 1: generate candidate thresholds from the validation split only."""
    if split != VALIDATION_SPLIT:
        raise CalibrationError(
            f"Calibration must only run on the '{VALIDATION_SPLIT}' split; got '{split}'. "
            f"This is enforced in code to prevent held-out/test-set leakage into threshold "
            f"selection, regardless of what the caller intended."
        )

    # Three strategies optimise a different balance between the two error types:
    # balanced accuracy weights both equally, F1 favours correct matches, and the
    # equal-error rate is the point at which the two rates coincide.
    candidates: Dict[str, ThresholdCandidate] = {
        "balanced_accuracy": select_threshold(scores, labels, strategy="balanced_accuracy"),
        "f1": select_threshold(scores, labels, strategy="f1"),
        "eer": select_threshold(scores, labels, strategy="eer"),
    }
    # One further candidate per target false match rate, for a deployment that
    # must hold false matches below a stated bound whatever it costs in misses.
    for target in target_false_match_rates:
        candidates[f"target_fmr_{target}"] = select_threshold(
            scores, labels, strategy="target_fmr", target_false_match_rate=target
        )

    # Returned with the "candidates" status, never a chosen threshold. Selection
    # is a separate stage on a different split, so this result cannot be
    # mistaken for a frozen operating point.
    return CalibrationResult(split=split, status=CANDIDATES_STATUS, candidates=candidates)


def require_candidates(payload: Dict[str, Any], *, context: str = "") -> Dict[str, Dict[str, Any]]:
    """Stage 2's guard: read the candidate table out of a loaded artifact,
    refusing anything already frozen or never calibrated."""
    if payload.get("status") != CANDIDATES_STATUS:
        raise CalibrationError(
            f"{context}: threshold artifact status is '{payload.get('status')}', not "
            f"'{CANDIDATES_STATUS}'. Expected a calibration artifact that has not yet been "
            f"frozen by a development-split selection step."
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise CalibrationError(f"{context}: threshold artifact has no candidates to select from")
    return candidates


def select_final_threshold(
    candidates: Dict[str, Dict[str, Any]],
    dev_scores: Sequence[float],
    dev_labels: Sequence[int],
) -> Dict[str, Any]:
    """Stage 2: score every stage-1 candidate against the development split and
    select exactly one by SELECTION_RULE. Returns the outcome plus every
    candidate's development metrics, so the choice is auditable."""
    if not candidates:
        raise CalibrationError("No candidate thresholds to select from")

    per_candidate_dev_metrics: Dict[str, Dict[str, float]] = {}
    for name, candidate in candidates.items():
        threshold = float(candidate["threshold"])
        matrix = confusion_matrix(dev_scores, dev_labels, threshold)
        rates = rates_from_confusion(matrix)
        tmr, fmr = rates["true_match_rate"], rates["false_match_rate"]
        # Self-comparison is the NaN test; an unscorable candidate sorts last
        # rather than winning the selection by accident.
        balanced_accuracy = (
            (tmr + (1.0 - fmr)) / 2.0 if tmr == tmr and fmr == fmr else float("-inf")
        )
        per_candidate_dev_metrics[name] = {
            **rates,
            "threshold": threshold,
            "balanced_accuracy": balanced_accuracy,
        }

    # The published selection rule in one key: highest balanced accuracy first,
    # then lowest false match rate, then the candidate name. Including the name
    # guarantees a total order, so the chosen threshold is deterministic and no
    # tie is left for chance to break.
    def sort_key(name: str) -> Tuple[float, float, str]:
        metrics = per_candidate_dev_metrics[name]
        return (-metrics["balanced_accuracy"], metrics["false_match_rate"], name)

    selected_name = min(per_candidate_dev_metrics, key=sort_key)
    selected = per_candidate_dev_metrics[selected_name]

    return {
        "selected_candidate": selected_name,
        "selected_threshold": selected["threshold"],
        "selection_rule": SELECTION_RULE,
        "all_candidates_dev_metrics": per_candidate_dev_metrics,
    }


# A reportable evaluation must refuse to run against a threshold that was not
# frozen beforehand. Enforced in code rather than left to procedure, so a
# recalibrated threshold cannot reach a published result unnoticed.
def require_frozen_threshold(payload: Dict[str, Any], *, context: str = "") -> float:
    """Stage 3's guard: read a threshold from a loaded artifact, refusing
    anything not explicitly marked frozen by select_final_threshold."""
    if payload.get("status") != FROZEN_STATUS:
        raise CalibrationError(
            f"{context}: threshold artifact status is '{payload.get('status')}', not "
            f"'{FROZEN_STATUS}'. Refusing to use a non-frozen threshold for a held-out or "
            f"final evaluation."
        )
    threshold = payload.get("threshold")
    if not isinstance(threshold, (int, float)):
        raise CalibrationError(
            f"{context}: threshold artifact is missing a numeric 'threshold' field"
        )
    return float(threshold)


# =============================================================================
# 12. Pair evaluation and failure accounting
# =============================================================================
#
# Face-extraction failures (zero or multiple detections, unreadable images) are
# recorded as their own outcome category and reported alongside accuracy, never
# dropped from the pair count. Per-image latency is timed so throughput can be
# reported rather than inferred from wall-clock runtime.


@dataclass(frozen=True)
class PairScore:
    pair: Pair
    similarity: Optional[float]
    failure_code: Optional[str]


@dataclass
class EvaluationResult:
    total_pairs: int
    scored_pairs: List[PairScore]
    failures: Dict[str, int] = field(default_factory=dict)
    embedding_times_seconds: List[float] = field(default_factory=list)

    # Each figure below is derived from the stored pairs on every access, so a
    # reported count can never drift out of step with the pairs it describes.
    @property
    def valid_scores(self) -> List[float]:
        return [s.similarity for s in self.scored_pairs if s.similarity is not None]

    @property
    def valid_labels(self) -> List[int]:
        return [
            1 if s.pair.same_identity else 0
            for s in self.scored_pairs
            if s.similarity is not None
        ]

    @property
    def scored_pair_count(self) -> int:
        # Only pairs yielding one valid face on both sides carry a similarity score.
        return len(self.valid_scores)

    @property
    def failed_pairs(self) -> int:
        # Failed pairs stay within the protocol total; they simply have no score.
        return self.total_pairs - self.scored_pair_count

    @property
    def failure_rate(self) -> float:
        # Stored as a fraction of the full protocol; reports render the percentage.
        return self.failed_pairs / self.total_pairs if self.total_pairs else float("nan")

    def validate_accounting(self) -> None:
        """Confirm every protocol pair is accounted for exactly once. The
        reported denominator is only meaningful if nothing was silently
        discarded and the breakdown describes precisely the failed pairs."""
        if self.scored_pair_count + self.failed_pairs != self.total_pairs:
            raise ValueError(
                f"Scored ({self.scored_pair_count}) and failed ({self.failed_pairs}) pairs must "
                f"sum to the protocol total ({self.total_pairs})."
            )
        # evaluate_pairs records exactly one terminal category per failed pair,
        # so the breakdown partitions the failures rather than tallying images.
        categorised = sum(self.failures.values())
        if categorised != self.failed_pairs:
            raise ValueError(
                f"Failure breakdown totals {categorised} but {self.failed_pairs} pairs failed; "
                f"every failed pair must carry exactly one extraction-failure category."
            )


def _embed_image(
    path: Path,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    cache: Dict[Path, np.ndarray],
    embedding_times: List[float],
) -> np.ndarray:
    # An image often appears in several protocol pairs. Reusing the stored
    # result keeps the evaluation honest as well as faster: the same photograph
    # must always yield the same numbers.
    if path in cache:
        return cache[path]
    start = time.perf_counter()
    # The three stages every photograph passes through: read the file, locate
    # exactly one face, then turn that face into a list of numbers.
    loaded = load_image_bgr(path)
    face_row = detector.detect_single_face(loaded.bgr)
    raw_embedding = embedder.embed(loaded.bgr, face_row)
    normalized = l2_normalize(raw_embedding)
    embedding_times.append(time.perf_counter() - start)
    cache[path] = normalized
    return normalized


def evaluate_pairs(
    pairs: List[Pair], *, detector: FaceDetector, embedder: FaceEmbedder
) -> EvaluationResult:
    scored: List[PairScore] = []
    failures: Dict[str, int] = {}
    cache: Dict[Path, np.ndarray] = {}
    embedding_times: List[float] = []

    def record(code: str) -> None:
        failures[code] = failures.get(code, 0) + 1

    # Returns the embedding and an empty failure code, or no embedding and the
    # code that terminated this side. The empty string carries the same meaning
    # as the absent embedding, so a caller can branch on either one.
    def embed_side(path: Path, side: str) -> Tuple[Optional[np.ndarray], str]:
        try:
            return _embed_image(path, detector, embedder, cache, embedding_times), ""
        except FaceCountError as exc:
            code = f"zero_faces_{side}" if exc.face_count == 0 else f"multiple_faces_{side}"
            return None, code
        except (ImageLoadError, SimilarityError) as exc:
            return None, f"image_error_{side}:{exc}"

    for pair in pairs:
        # Sides are attempted left first and the pair is abandoned on the first
        # terminal failure, so each failed pair records exactly one category.
        # A "_right" category therefore means the right image failed *after*
        # the left had already yielded one valid face: where both sides would
        # fail, only the left is ever counted. The four categories partition
        # the failed pairs; they are not per-image failure tallies.
        left_embedding, left_failure = embed_side(pair.left_path, "left")
        if left_embedding is None:
            record(left_failure.split(":")[0])
            scored.append(PairScore(pair, None, left_failure))
            continue

        right_embedding, right_failure = embed_side(pair.right_path, "right")
        if right_embedding is None:
            record(right_failure.split(":")[0])
            scored.append(PairScore(pair, None, right_failure))
            continue

        similarity = cosine_similarity(left_embedding, right_embedding)
        scored.append(PairScore(pair, similarity, None))

    return EvaluationResult(
        total_pairs=len(pairs),
        scored_pairs=scored,
        failures=failures,
        embedding_times_seconds=embedding_times,
    )


def summarize_metrics(result: EvaluationResult, threshold: float) -> Dict[str, Any]:
    scores = result.valid_scores
    labels = result.valid_labels
    if not scores:
        raise ValueError("No pairs were successfully scored; cannot compute metrics.")

    # Refuse to publish a failure rate whose denominator does not reconcile.
    result.validate_accounting()

    matrix = confusion_matrix(scores, labels, threshold)
    rates = rates_from_confusion(matrix)
    auc = roc_auc(scores, labels)
    eer = equal_error_rate(scores, labels)
    curve = roc_points(scores, labels)

    times_ms = [t * 1000.0 for t in result.embedding_times_seconds]

    return {
        "threshold": threshold,
        "total_pairs": result.total_pairs,
        # Score-based metrics below are conditional on these scored pairs alone.
        "scored_pairs": result.scored_pair_count,
        "failed_pairs": result.failed_pairs,
        "failure_rate": result.failure_rate,
        "failure_breakdown": dict(result.failures),
        "confusion_matrix": matrix.as_dict(),
        "accuracy": rates["accuracy"],
        "precision": rates["precision"],
        "recall": rates["recall"],
        "f1": rates["f1"],
        "false_match_rate": rates["false_match_rate"],
        "false_non_match_rate": rates["false_non_match_rate"],
        "roc_auc": auc,
        "equal_error_rate": eer["equal_error_rate"],
        "roc_points": curve,
        "embedding_time_mean_ms": statistics.fmean(times_ms) if times_ms else float("nan"),
        "embedding_time_median_ms": statistics.median(times_ms) if times_ms else float("nan"),
        "embedding_time_p95_ms": percentile(times_ms, 95) if times_ms else float("nan"),
        "unique_images_embedded": len(times_ms),
    }


def image_content_digest(paths: Iterable[Path]) -> str:
    """Content fingerprint in caller-defined order; no private path is published."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(bytes.fromhex(sha256_of_file(path)))
    return digest.hexdigest()


def select_lfw_training_threshold(scores: ScoreInput, labels: LabelInput) -> float:
    """Maximum training accuracy, breaking ties towards the stricter threshold."""
    values, targets = _validate_inputs(scores, labels)
    order = np.argsort(values, kind="stable")
    values, targets = values[order], targets[order]
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    positive_prefix = np.r_[0, np.cumsum(targets)]
    # Before each threshold, negatives become correct and positives become misses.
    correct = starts - positive_prefix[starts] + positive_prefix[-1] - positive_prefix[starts]
    thresholds = np.r_[values[starts], np.nextafter(values[-1], np.inf)]
    correct = np.r_[correct, len(targets) - positive_prefix[-1]]
    return float(thresholds[np.flatnonzero(correct == correct.max())[-1]])


def summarize_lfw_cross_validation(
    result: EvaluationResult, folds: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Fit a threshold on all other folds; score each pair exactly once out of fold."""
    indices = [i for fold in folds for i in fold]
    if len(folds) < 2 or any(not fold for fold in folds) or sorted(indices) != list(range(result.total_pairs)):
        raise ProtocolError("LFW folds must partition every protocol pair exactly once.")
    result.validate_accounting()
    fold_results: List[Dict[str, Any]] = []
    pooled: Dict[str, int] = dict.fromkeys(("true_positive", "false_positive", "true_negative", "false_negative"), 0)
    for number, fold in enumerate(folds, start=1):
        excluded = set(fold)
        training = [row for i, row in enumerate(result.scored_pairs)
                    if i not in excluded and row.similarity is not None]
        testing = [result.scored_pairs[i] for i in fold
                   if result.scored_pairs[i].similarity is not None]
        if {row.pair.same_identity for row in training} != {False, True} or not testing:
            raise ProtocolError(f"LFW fold {number} has insufficient scored training or test pairs.")
        threshold = select_lfw_training_threshold(
            [cast(float, row.similarity) for row in training],
            [int(row.pair.same_identity) for row in training],
        )
        matrix = confusion_matrix(
            [cast(float, row.similarity) for row in testing],
            [int(row.pair.same_identity) for row in testing], threshold,
        )
        for key, value in matrix.as_dict().items():
            pooled[key] += value
        fold_results.append({
            "fold": number, "threshold": threshold,
            "training_pairs": result.total_pairs - len(fold), "training_scored_pairs": len(training),
            "test_pairs": len(fold), "test_scored_pairs": len(testing),
            "test_failed_pairs": len(fold) - len(testing),
            "confusion_matrix": matrix.as_dict(), **rates_from_confusion(matrix),
        })
    # AUC/EER are descriptive pooled score summaries, not tuned operating points.
    summary = summarize_metrics(result, 0.0)
    summary.pop("threshold")
    matrix = ConfusionMatrix(**pooled)
    summary.update(rates_from_confusion(matrix))
    accuracies = [fold["accuracy"] for fold in fold_results]
    summary.update({
        "confusion_matrix": pooled,
        "evaluation_protocol": "lfw_official_10_fold_cross_validation",
        "threshold_status": "per_fold_training_only",
        "threshold_selection": "Maximum accuracy on the other nine folds; stricter threshold on ties",
        "fold_count": len(folds), "fold_results": fold_results,
        "mean_fold_accuracy": statistics.fmean(accuracies),
        "fold_accuracy_std": statistics.stdev(accuracies),
        "fold_accuracy_standard_error": statistics.stdev(accuracies) / math.sqrt(len(folds)),
        "pooled_accuracy": summary["accuracy"],
        "end_to_end_correct_decision_rate": (matrix.true_positive + matrix.true_negative) / result.total_pairs,
        "metric_note": "Accuracy/rates/confusion are pooled out-of-fold decisions; mean_fold_accuracy weights folds equally. Rates exclude extraction failures; coverage is reported separately. Pooled ROC/EER are descriptive only.",
        "transfer_threshold_note": "The separate LFW development threshold is used only for CPLFW/gallery transfer, never for final LFW decisions.",
    })
    return summary


def experiment_lfw_cross_validation(
    config: EnvironmentConfig, detector: FaceDetector, embedder: FaceEmbedder,
    *, output_path: Path, description: Optional[PipelineDescription] = None,
) -> Dict[str, Any]:
    protocol_path = config.require_protocol_root() / LFW_FINAL_PROTOCOL
    lfw_root = config.require_lfw_root()
    pairs = parse_lfw_pairs(protocol_path, lfw_root)
    folds = lfw_official_fold_indices(protocol_path, pairs)
    pipeline = pipeline_identity_fields(detector, embedder, description)
    images = sorted({p.left_path for p in pairs} | {p.right_path for p in pairs})
    context = {"revision": "lfw-pair-scores-v1", "protocol_sha256": sha256_of_file(protocol_path),
               "image_content_sha256": image_content_digest(images),
               "pipeline": pipeline, "software_environment": software_environment_report()}
    key = context_digest(context)
    cache_path = RAW_ROOT / "lfw_pair_scores" / f"{key}.json"
    payload = None
    if cache_path.is_file():
        try:
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if candidate.get("context") == context and candidate.get("scores_sha256") == context_digest(candidate["data"]):
                payload = candidate["data"]
        except (ValueError, KeyError, TypeError):
            pass
    if payload is None:
        announce("Scoring LFW images once; thresholds will be fitted independently within each fold.")
        result = evaluate_pairs(pairs, detector=detector, embedder=embedder)
        payload = {"rows": [[r.similarity, r.failure_code.split(":")[0] if r.failure_code else None]
                            for r in result.scored_pairs],
                   "failures": result.failures, "timings": result.embedding_times_seconds}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _restrict_permissions(cache_path.parent, 0o700)
        _atomic_private_write(cache_path, json.dumps({"context": context, "data": payload,
                              "scores_sha256": context_digest(payload)}))
    else:
        announce("Reusing verified LFW scores; fitting ten independent training-fold thresholds.")
        if len(payload["rows"]) != len(pairs):
            raise ProtocolError("Cached LFW pair count does not match the protocol.")
        result = EvaluationResult(len(pairs),
            [PairScore(pair, row[0], row[1]) for pair, row in zip(pairs, payload["rows"])],
            payload["failures"], payload["timings"])
    summary = summarize_lfw_cross_validation(result, folds)
    # Publish aggregate overlap counts to explain why View 1 is not a final holdout.
    final_keys = {tuple(sorted((p.left_path, p.right_path))) for p in pairs}
    overlaps: Dict[str, int] = {}
    for filename in (LFW_CALIBRATION_PROTOCOL, LFW_DEVELOPMENT_PROTOCOL):
        development = parse_lfw_pairs(config.require_protocol_root() / filename, lfw_root)
        overlaps[filename] = len(final_keys & {tuple(sorted((p.left_path, p.right_path))) for p in development})
    write_json_artifact(output_path, {
        "artifact_type": "lfw_verification_metrics", "split": "final",
        "protocol_file": LFW_FINAL_PROTOCOL, "protocol_sha256": context["protocol_sha256"],
        "evaluated_image_set_sha256": sha256_of_evaluated_image_set(images, lfw_root),
        "image_content_sha256": context["image_content_sha256"],
        "dataset_archive_md5": LFW_ARCHIVE_MD5, "development_protocol_pair_overlap": overlaps,
        **summary, **pipeline, "software_environment": software_environment_report(),
    })
    announce(f"LFW ten-fold mean accuracy: {summary['mean_fold_accuracy']:.4f}; scored {result.scored_pair_count}/{result.total_pairs}.")
    return summary


# =============================================================================
# 13. Duplicate-profile gallery evaluation
# =============================================================================
#
# Simulates registered profiles (the gallery) plus two kinds of probe: a second
# image of a gallery identity (a duplicate-registration attempt) and an image
# of an identity absent from the gallery (a legitimate new registration). Every
# manifest entry uses an opaque, one-way identifier rather than a real name,
# and each image holds exactly one role. Nothing here labels a result a scam:
# exceeding the threshold only opens a human-review case.


class GalleryError(RuntimeError):
    """Raised when a gallery cannot be built or embedded at all."""


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    identity_hash: str
    image_path: Path
    role: str  # "gallery" | "duplicate_probe" | "unknown_probe"


@dataclass(frozen=True)
class GalleryManifest:
    entries: List[ManifestEntry]
    seed: int


def build_manifest(
    identity_to_images: Dict[str, List[Path]],
    *,
    excluded_images: Iterable[Path] = (),
    seed: int = DEFAULT_RANDOM_SEED,
    max_unknown_identities: Optional[int] = None,
) -> GalleryManifest:
    excluded = {Path(p) for p in excluded_images}
    rng = random.Random(seed)

    # Drop any photograph already used elsewhere in the evaluation, and order
    # what remains by filename so the selection never depends on the order the
    # filesystem happened to list the directory.
    eligible = {
        identity: sorted(
            (Path(p) for p in images if Path(p) not in excluded), key=lambda p: p.name
        )
        for identity, images in identity_to_images.items()
    }
    eligible = {identity: images for identity, images in eligible.items() if images}

    # The number of photographs decides the role. Two or more allows one to
    # enrol the profile and another to search with, which is a known duplicate
    # case. A single photograph cannot do both, so that person stands in for a
    # new registration who is not in the gallery at all.
    gallery_identities = sorted(
        identity for identity, images in eligible.items() if len(images) >= 2
    )
    unknown_identities = sorted(
        identity for identity, images in eligible.items() if len(images) == 1
    )

    if max_unknown_identities is not None:
        rng.shuffle(unknown_identities)
        unknown_identities = sorted(unknown_identities[:max_unknown_identities])

    if not gallery_identities:
        raise GalleryError("No identity has at least two usable images; cannot build a gallery.")
    if not unknown_identities:
        raise GalleryError(
            "No identity with exactly one usable image is available as an unknown probe."
        )

    entries: List[ManifestEntry] = []

    for identity in gallery_identities:
        images = eligible[identity]
        gallery_image, duplicate_image = images[0], images[1]
        identity_hash = opaque_id(identity)
        entries.append(
            ManifestEntry(
                opaque_id(f"{identity}:{gallery_image.name}"),
                identity_hash,
                gallery_image,
                "gallery",
            )
        )
        entries.append(
            ManifestEntry(
                opaque_id(f"{identity}:{duplicate_image.name}"),
                identity_hash,
                duplicate_image,
                "duplicate_probe",
            )
        )

    for identity in unknown_identities:
        image = eligible[identity][0]
        identity_hash = opaque_id(identity)
        entries.append(
            ManifestEntry(
                opaque_id(f"{identity}:{image.name}"), identity_hash, image, "unknown_probe"
            )
        )

    seen_paths: Set[Path] = set()
    for entry in entries:
        if entry.image_path in seen_paths:
            raise GalleryError(f"Image assigned to more than one manifest role: {entry.image_path}")
        seen_paths.add(entry.image_path)

    return GalleryManifest(entries=entries, seed=seed)


@dataclass(frozen=True)
class ProbeResult:
    sample_id: str
    role: str
    identity_hash: str
    top_candidate_identity_hash: Optional[str]
    top_similarity: Optional[float]
    rank1_correct: Optional[bool]
    exceeds_duplicate_threshold: Optional[bool]
    failure_code: Optional[str]
    # Rank of the probe's own identity in the descending-similarity candidate
    # list, 1-based; None when the probe is not mated or was never scored.
    correct_identity_rank: Optional[int] = None
    rank5_correct: Optional[bool] = None


@dataclass(frozen=True)
class GalleryEntryResult:
    """Outcome of attempting to enrol one gallery reference image. Retained for
    every intended entry, including the ones that could not be embedded — the
    defect this replaces dropped those silently, which shrank the gallery
    without recording that it had shrunk."""

    sample_id: str
    identity_hash: str
    embedded: bool
    failure_code: Optional[str]


# Gallery enrolment failures are reported under their own suffixed codes so a
# reader can never confuse a gallery-side failure with a probe-side one. These
# three partition every enrolment failure.
GALLERY_FAILURE_CODES = (
    "zero_faces_gallery",
    "multiple_faces_gallery",
    "image_error_gallery",
)

# A mated probe whose own gallery reference never enrolled cannot be found by
# any threshold. It is a coverage failure, not a similarity miss, and is kept
# distinct from both so neither rate is quietly flattered.
GALLERY_REFERENCE_UNAVAILABLE = "gallery_reference_unavailable"


@dataclass(frozen=True)
class GalleryEvaluationResult:
    gallery_size: int
    probe_results: List[ProbeResult]
    search_times_seconds: List[float] = field(default_factory=list)
    # Enrolment accounting is optional when a caller has only scored-gallery
    # information. Detailed runs retain the outcome of every intended entry.
    intended_gallery_size: Optional[int] = None
    gallery_entry_results: List[GalleryEntryResult] = field(default_factory=list)

    @property
    def embedded_gallery_size(self) -> int:
        return self.gallery_size

    @property
    def resolved_intended_gallery_size(self) -> int:
        """Intended entries, falling back to the embedded count when this result
        came from a caller that predates enrolment accounting."""
        return self.gallery_size if self.intended_gallery_size is None else self.intended_gallery_size

    @property
    def gallery_failure_breakdown(self) -> Dict[str, int]:
        breakdown = {code: 0 for code in GALLERY_FAILURE_CODES}
        for entry in self.gallery_entry_results:
            if entry.embedded:
                continue
            code = entry.failure_code or "image_error_gallery"
            breakdown[code] = breakdown.get(code, 0) + 1
        return breakdown

    @property
    def gallery_entry_failure_count(self) -> int:
        return self.resolved_intended_gallery_size - self.embedded_gallery_size

    @property
    def gallery_entry_failure_rate(self) -> float:
        intended = self.resolved_intended_gallery_size
        return self.gallery_entry_failure_count / intended if intended else float("nan")


def _embed_entry(
    entry: ManifestEntry, detector: FaceDetector, embedder: FaceEmbedder
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        loaded = load_image_bgr(entry.image_path)
        face_row = detector.detect_single_face(loaded.bgr)
        raw = embedder.embed(loaded.bgr, face_row)
        return l2_normalize(raw), None
    except FaceCountError as exc:
        return None, "zero_faces" if exc.face_count == 0 else "multiple_faces"
    except (ImageLoadError, SimilarityError) as exc:
        return None, f"image_error:{exc}"


def _gallery_failure_code(probe_failure_code: Optional[str]) -> str:
    """Map a generic extraction failure onto its gallery-side category."""
    if probe_failure_code == "zero_faces":
        return "zero_faces_gallery"
    if probe_failure_code == "multiple_faces":
        return "multiple_faces_gallery"
    return "image_error_gallery"


def evaluate_gallery(
    manifest: GalleryManifest,
    *,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    duplicate_review_threshold: float,
) -> GalleryEvaluationResult:
    gallery_entries = [e for e in manifest.entries if e.role == "gallery"]
    probe_entries = [e for e in manifest.entries if e.role in ("duplicate_probe", "unknown_probe")]

    gallery_embeddings: List[Tuple[ManifestEntry, np.ndarray]] = []
    gallery_entry_results: List[GalleryEntryResult] = []
    for entry in gallery_entries:
        embedding, failure = _embed_entry(entry, detector, embedder)
        if embedding is not None:
            gallery_embeddings.append((entry, embedding))
            gallery_entry_results.append(
                GalleryEntryResult(entry.sample_id, entry.identity_hash, True, None)
            )
        else:
            # Retained rather than dropped: an identity that never enrolled is
            # unfindable, and the mated probes pointing at it must be accounted
            # for instead of quietly leaving the denominator.
            gallery_entry_results.append(
                GalleryEntryResult(
                    entry.sample_id, entry.identity_hash, False, _gallery_failure_code(failure)
                )
            )
    if not gallery_embeddings:
        raise GalleryError("No gallery entry could be embedded; cannot run the experiment.")

    enrolled_identity_hashes = {entry.identity_hash for entry, _ in gallery_embeddings}

    results: List[ProbeResult] = []
    search_times: List[float] = []
    for probe in probe_entries:
        # Checked before embedding: when the probe's own reference is missing
        # from the gallery, no similarity score could change the outcome, so
        # the protocol-level blocker is the honest attribution.
        if probe.role == "duplicate_probe" and probe.identity_hash not in enrolled_identity_hashes:
            results.append(
                ProbeResult(
                    probe.sample_id,
                    probe.role,
                    probe.identity_hash,
                    None,
                    None,
                    None,
                    None,
                    GALLERY_REFERENCE_UNAVAILABLE,
                )
            )
            continue

        probe_embedding, failure = _embed_entry(probe, detector, embedder)
        if probe_embedding is None:
            results.append(
                ProbeResult(
                    probe.sample_id, probe.role, probe.identity_hash, None, None, None, None, failure
                )
            )
            continue

        search_start = time.perf_counter()
        similarities = sorted(
            (
                (candidate_entry, cosine_similarity(probe_embedding, candidate_embedding))
                for candidate_entry, candidate_embedding in gallery_embeddings
            ),
            # Same lexicographic device as the threshold selection: rank by
            # descending similarity, and break an exact tie on the opaque
            # sample identifier so the winner is deterministic.
            key=lambda item: (-item[1], item[0].sample_id),
        )
        search_times.append(time.perf_counter() - search_start)
        top_entry, top_similarity = similarities[0]
        rank1_correct = (
            top_entry.identity_hash == probe.identity_hash
            if probe.role == "duplicate_probe"
            else None
        )

        correct_rank: Optional[int] = None
        rank5_correct: Optional[bool] = None
        if probe.role == "duplicate_probe":
            for position, (candidate_entry, _score) in enumerate(similarities, start=1):
                if candidate_entry.identity_hash == probe.identity_hash:
                    correct_rank = position
                    break
            rank5_correct = correct_rank is not None and correct_rank <= 5

        results.append(
            ProbeResult(
                probe.sample_id,
                probe.role,
                probe.identity_hash,
                top_entry.identity_hash,
                top_similarity,
                rank1_correct,
                top_similarity >= duplicate_review_threshold,
                None,
                correct_rank,
                rank5_correct,
            )
        )

    return GalleryEvaluationResult(
        gallery_size=len(gallery_embeddings),
        probe_results=results,
        search_times_seconds=search_times,
        intended_gallery_size=len(gallery_entries),
        gallery_entry_results=gallery_entry_results,
    )


def summarize_gallery_metrics(result: GalleryEvaluationResult) -> Dict[str, Any]:
    duplicate_probes = [r for r in result.probe_results if r.role == "duplicate_probe"]
    unknown_probes = [r for r in result.probe_results if r.role == "unknown_probe"]
    scored_duplicates = [r for r in duplicate_probes if r.failure_code is None]
    scored_unknowns = [r for r in unknown_probes if r.failure_code is None]

    # Conditional denominator: mated probes that were scored at all, which by
    # construction means both the probe and its gallery reference embedded.
    # End-to-end denominator: every mated probe the protocol intended, so an
    # extraction failure or an unavailable reference counts as a miss rather
    # than disappearing. The two answer different questions and the reporting
    # layer is required to show them together.
    detected_duplicates = sum(1 for r in scored_duplicates if r.exceeds_duplicate_threshold)
    intended_duplicates = len(duplicate_probes)
    unavailable_reference_count = sum(
        1 for r in duplicate_probes if r.failure_code == GALLERY_REFERENCE_UNAVAILABLE
    )

    duplicate_detection_rate = (
        detected_duplicates / len(scored_duplicates) if scored_duplicates else float("nan")
    )
    end_to_end_duplicate_detection_rate = (
        detected_duplicates / intended_duplicates if intended_duplicates else float("nan")
    )
    false_duplicate_review_rate = (
        sum(1 for r in scored_unknowns if r.exceeds_duplicate_threshold) / len(scored_unknowns)
        if scored_unknowns
        else float("nan")
    )
    rank1_identification_rate = (
        sum(1 for r in scored_duplicates if r.rank1_correct) / len(scored_duplicates)
        if scored_duplicates
        else float("nan")
    )
    end_to_end_rank1_identification_rate = (
        sum(1 for r in scored_duplicates if r.rank1_correct) / intended_duplicates
        if intended_duplicates
        else float("nan")
    )
    rank5_identification_rate = (
        sum(1 for r in scored_duplicates if r.rank5_correct) / len(scored_duplicates)
        if scored_duplicates
        else float("nan")
    )
    end_to_end_rank5_identification_rate = (
        sum(1 for r in scored_duplicates if r.rank5_correct) / intended_duplicates
        if intended_duplicates
        else float("nan")
    )
    # Self-comparison is the NaN test: with no scored duplicate probes the miss
    # rate stays undefined instead of being reported as a perfect 1.0.
    true_duplicate_miss_rate = (
        1.0 - duplicate_detection_rate
        if duplicate_detection_rate == duplicate_detection_rate
        else float("nan")
    )

    search_times_ms = [t * 1000.0 for t in result.search_times_seconds]

    return {
        "methodology_revision": GALLERY_METHODOLOGY_REVISION,
        "gallery_size": result.gallery_size,
        "intended_gallery_size": result.resolved_intended_gallery_size,
        "embedded_gallery_size": result.embedded_gallery_size,
        "gallery_entry_failure_count": result.gallery_entry_failure_count,
        "gallery_entry_failure_rate": result.gallery_entry_failure_rate,
        "gallery_failure_breakdown": result.gallery_failure_breakdown,
        "duplicate_probe_count": len(duplicate_probes),
        "unknown_probe_count": len(unknown_probes),
        "duplicate_probe_failures": len(duplicate_probes) - len(scored_duplicates),
        "unknown_probe_failures": len(unknown_probes) - len(scored_unknowns),
        "gallery_reference_unavailable_count": unavailable_reference_count,
        # Named explicitly so a reader cannot mistake the conditional figure for
        # the headline result; the legacy key keeps older artefacts readable.
        "conditional_duplicate_detection_rate": duplicate_detection_rate,
        "end_to_end_duplicate_detection_rate": end_to_end_duplicate_detection_rate,
        "duplicate_detection_rate": duplicate_detection_rate,
        "false_duplicate_review_rate": false_duplicate_review_rate,
        "conditional_rank1_identification_rate": rank1_identification_rate,
        "end_to_end_rank1_identification_rate": end_to_end_rank1_identification_rate,
        "conditional_rank5_identification_rate": rank5_identification_rate,
        "end_to_end_rank5_identification_rate": end_to_end_rank5_identification_rate,
        "rank1_identification_rate": rank1_identification_rate,
        "true_duplicate_miss_rate": true_duplicate_miss_rate,
        "gallery_search_time_mean_ms": (
            statistics.fmean(search_times_ms) if search_times_ms else float("nan")
        ),
        "gallery_search_time_p95_ms": (
            percentile(search_times_ms, 95) if search_times_ms else float("nan")
        ),
    }


def discover_identity_images(dataset_root: Path) -> Dict[str, List[Path]]:
    """Map each per-identity directory to its images, following LFW's
    identity/identity_NNNN.jpg layout."""
    identities: Dict[str, List[Path]] = {}
    for identity_dir in sorted(p for p in Path(dataset_root).iterdir() if p.is_dir()):
        images = sorted(identity_dir.glob(f"{identity_dir.name}_*.jpg"))
        if images:
            identities[identity_dir.name] = images
    return identities


def write_gallery_manifest(manifest: GalleryManifest, path: Path) -> None:
    """Write the manifest to a private location. It contains real image paths
    and is therefore git-ignored, never a published artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": manifest.seed,
        "entries": [
            {
                "sample_id": entry.sample_id,
                "identity_hash": entry.identity_hash,
                "role": entry.role,
                "image_path": str(entry.image_path),
            }
            for entry in manifest.entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_gallery_manifest(path: Path) -> GalleryManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [
        ManifestEntry(
            entry["sample_id"], entry["identity_hash"], Path(entry["image_path"]), entry["role"]
        )
        for entry in payload["entries"]
    ]
    return GalleryManifest(entries=entries, seed=payload["seed"])


# =============================================================================
# 14. Aggregate output generation
# =============================================================================
#
# Every artefact is self-describing enough that a reader never has to trust an
# unlabelled number: schema version, creation timestamp, software and model
# provenance, and dataset digests. Writes are atomic, so a crash mid-write
# never leaves a half-written result file.


class ArtifactError(RuntimeError):
    """Raised when a required artifact is missing or unreadable."""


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _atomic_write(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        # Replace the artefact in one step, so an interrupted run cannot leave a
        # half-written result that a later stage would read as complete.
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    body: Dict[str, Any] = dict(payload)
    body.setdefault("schema_version", SCHEMA_VERSION)
    body.setdefault("created_at", utc_now_iso())
    text = json.dumps(body, indent=2, sort_keys=True, default=_json_default) + "\n"
    _atomic_write(Path(path), text)
    return sha256_of_text(text)


def write_csv_artifact(
    path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]
) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    text = buffer.getvalue()
    _atomic_write(Path(path), text)
    return sha256_of_text(text)


def write_markdown_artifact(path: Path, text: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    _atomic_write(Path(path), text)
    return sha256_of_text(text)


def read_json_artifact(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


VERIFICATION_EXPERIMENTS = ("lfw_development", "lfw_final", "cplfw")

# Category keys carry the side on which extraction terminated; see
# evaluate_pairs for the left-first short-circuit rule.
FAILURE_CATEGORY_PROSE = {
    "zero_faces_left": "zero-face detections on the left image",
    "zero_faces_right": "zero-face detections on the right image",
    "multiple_faces_left": "multiple-face detections on the left image",
    "multiple_faces_right": "multiple-face detections on the right image",
}


def format_percentage(value: Any) -> str:
    try:
        number = float(value)
        return f"{number * 100:.2f}%" if math.isfinite(number) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def format_number(value: Any, digits: int = 4) -> str:
    try:
        # The caller chooses the precision, so a rate and a threshold can be
        # rendered to different numbers of decimal places.
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def format_count(value: Any) -> str:
    # Thousands separators for pair counts quoted in written work.
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def render_failure_breakdown(payload: Dict[str, Any]) -> str:
    """Describe the failure categories in prose, reading the counts from the
    result rather than restating them by hand."""
    breakdown = payload.get("failure_breakdown") or {}
    if not breakdown:
        return "No extraction-failure categories were recorded."

    described = [
        f"{format_count(breakdown[key])} {prose}"
        for key, prose in FAILURE_CATEGORY_PROSE.items()
        if key in breakdown
    ]
    # Any category outside the expected four is surfaced, never dropped.
    described += [
        f"{format_count(count)} {key}"
        for key, count in breakdown.items()
        if key not in FAILURE_CATEGORY_PROSE
    ]

    joined = ", ".join(described[:-1]) + (
        f" and {described[-1]}" if len(described) > 1 else described[0]
    )
    return (
        f"The {format_count(payload.get('failed_pairs'))} failures comprised {joined}. Each failed "
        f"pair carries exactly one category: sides are attempted left first and the pair is "
        f"abandoned at the first terminal failure, so a right-side category means the left image "
        f"had already yielded one valid face."
    )


def render_final_report(
    payloads: Dict[str, Dict[str, Any]], gallery_payload: Dict[str, Any]
) -> str:
    dev, final, cplfw = payloads["lfw_development"], payloads["lfw_final"], payloads["cplfw"]
    lines = [
        "# Final evaluation report",
        "",
        f"Generated from saved evaluation results on {utc_now_iso()}. Every number below is "
        "read directly from the corresponding `results/aggregate/*.json` file, each of which "
        "embeds its own software, model and dataset provenance (`software_environment`, "
        "`model_sha256`, `protocol_sha256`, `evaluated_image_set_sha256`, "
        "`dataset_archive_md5`/`dataset_archive_sha256`).",
        "",
        "## Experiments 1–2 — threshold calibration and selection",
        "",
        f"Experiment 1 (`pairsDevTrain.txt`) generated candidate thresholds and wrote them with "
        f"status `\"candidates\"`; it never selects a winner. Experiment 2 (`pairsDevTest.txt`) "
        f"evaluated every candidate and only then selected and froze "
        f"**{final.get('operating_strategy') or dev.get('selected_candidate', 'n/a')}** "
        f"at threshold **{format_number(dev.get('threshold'), 6)}**, by the rule: "
        f"\"{dev.get('selection_rule', 'n/a')}\"",
        "",
        "## Experiment 2 — LFW development validation (`pairsDevTest.txt`)",
        "",
        f"Scored {dev.get('scored_pairs', 'n/a')} / {dev.get('total_pairs', 'n/a')} pairs "
        f"(failure rate {format_percentage(dev.get('failure_rate'))}). "
        f"Accuracy **{format_percentage(dev.get('accuracy'))}**, F1 {format_number(dev.get('f1'))}, "
        f"ROC-AUC {format_number(dev.get('roc_auc'))}, "
        f"EER {format_percentage(dev.get('equal_error_rate'))}.",
        "",
        "## Experiment 3 — official LFW ten-fold evaluation (`pairs.txt`)",
        "",
        f"Scored {final.get('scored_pairs', 'n/a')} / {final.get('total_pairs', 'n/a')} pairs "
        f"(failure rate {format_percentage(final.get('failure_rate'))}). "
        f"**Mean fold accuracy {format_percentage(final.get('mean_fold_accuracy', final.get('accuracy')))}** "
        f"(standard error {format_percentage(final.get('fold_accuracy_standard_error'))}); "
        f"pooled out-of-fold accuracy {format_percentage(final.get('accuracy'))}, "
        f"precision {format_percentage(final.get('precision'))}, "
        f"recall {format_percentage(final.get('recall'))}, "
        f"false match rate {format_percentage(final.get('false_match_rate'))}, "
        f"false non-match rate {format_percentage(final.get('false_non_match_rate'))}, "
        f"ROC-AUC {format_number(final.get('roc_auc'))}, "
        f"EER {format_percentage(final.get('equal_error_rate'))}. "
        f"Confusion matrix: {final.get('confusion_matrix')}. "
        f"Mean embedding time {format_number(final.get('embedding_time_mean_ms'), 2)} ms "
        f"(p95 {format_number(final.get('embedding_time_p95_ms'), 2)} ms) over "
        f"{final.get('unique_images_embedded', 'n/a')} unique images.",
        "",
        "## Experiment 4 — CPLFW cross-pose generalisation (same frozen threshold, no recalibration)",
        "",
        # Counts and the decimal rate come from summarize_metrics(); the
        # percentage is derived here so no figure is ever transcribed by hand.
        f"Of the {format_count(cplfw.get('total_pairs'))} raw CPLFW protocol pairs, "
        f"{format_count(cplfw.get('scored_pairs'))} produced valid similarity scores and "
        f"{format_count(cplfw.get('failed_pairs'))} failed during face extraction. The "
        f"extraction-failure rate was therefore **{format_percentage(cplfw.get('failure_rate'))}** "
        f"({format_count(cplfw.get('failed_pairs'))} ÷ {format_count(cplfw.get('total_pairs'))}). "
        f"These failed pairs were retained in the protocol total and reported separately rather "
        f"than being silently discarded.",
        "",
        render_failure_breakdown(cplfw),
        "",
        f"Accuracy, precision, recall, F1-score, ROC-AUC and EER are conditional on the "
        f"{format_count(cplfw.get('scored_pairs'))} pairs for which both images produced exactly "
        f"one valid face: accuracy {format_percentage(cplfw.get('accuracy'))}, "
        f"F1 {format_number(cplfw.get('f1'))}, "
        f"false match rate {format_percentage(cplfw.get('false_match_rate'))}, "
        f"false non-match rate {format_percentage(cplfw.get('false_non_match_rate'))}, "
        f"ROC-AUC {format_number(cplfw.get('roc_auc'))}, "
        f"EER {format_percentage(cplfw.get('equal_error_rate'))}. An extraction failure is not a "
        f"verification error: the pipeline never produced a similarity score for those pairs, so "
        f"they can be neither correct nor incorrect.",
        "",
        "## Experiment 5 — real 1:N duplicate-profile gallery (LFW, seed "
        f"{gallery_payload.get('seed', 'n/a')})",
        "",
        f"Gallery size {gallery_payload.get('gallery_size', 'n/a')}; "
        f"{gallery_payload.get('duplicate_probe_count', 'n/a')} duplicate probes "
        f"({gallery_payload.get('duplicate_probe_failures', 'n/a')} extraction failures); "
        f"{gallery_payload.get('unknown_probe_count', 'n/a')} unknown probes "
        f"({gallery_payload.get('unknown_probe_failures', 'n/a')} extraction failures).",
        "",
        f"- Gallery enrolment: {gallery_payload.get('embedded_gallery_size', 'n/a')} enrolled of "
        f"{gallery_payload.get('intended_gallery_size', 'n/a')} intended "
        f"(failure rate {format_percentage(gallery_payload.get('gallery_entry_failure_rate'))}); "
        f"{gallery_payload.get('gallery_reference_unavailable_count', 'n/a')} mated probes had no "
        f"enrolled reference.",
        f"- Duplicate detection rate (conditional, scored mated probes only): "
        f"**{format_percentage(gallery_payload.get('duplicate_detection_rate'))}**",
        f"- Duplicate detection rate (end-to-end, all intended mated probes): "
        f"**{format_percentage(gallery_payload.get('end_to_end_duplicate_detection_rate'))}**",
        f"- Rank-1 identification rate: "
        f"{format_percentage(gallery_payload.get('rank1_identification_rate'))}",
        f"- Rank-5 identification rate (conditional): "
        f"{format_percentage(gallery_payload.get('conditional_rank5_identification_rate'))}",
        f"- True duplicate miss rate: "
        f"{format_percentage(gallery_payload.get('true_duplicate_miss_rate'))}",
        f"- **False duplicate-review rate: "
        f"{format_percentage(gallery_payload.get('false_duplicate_review_rate'))}**",
        "",
        gallery_payload.get("policy_note", POLICY_NOTE),
        "",
        "The gallery uses the development-frozen 1:1 threshold as a transfer control. "
        "Final LFW uses ten independently fitted fold thresholds; its FMR is not the "
        "FMR of this transferred policy. The observed gallery false-review rate directly "
        "measures transfer failure at this gallery size without assuming independent "
        "pairwise comparisons.",
        "",
        "## Limitations",
        "",
        "These figures describe LFW and CPLFW's own demographic composition, not any real user "
        "base; the gallery experiment is research-scale, not production-scale; and \"duplicate "
        "profile\" here means \"same face detected in the gallery\", not a legal or investigative "
        "finding.",
        "",
    ]
    return "\n".join(lines)


def write_aggregate_reports(
    output_root: Path, gallery_manifest_path: Path, cplfw_image_variant: str
) -> None:
    """Cross-reference the five per-experiment metrics files into a manifest,
    three CSVs and a written report, then refuse to finish if any of them
    contains a personal or absolute filesystem path."""
    # Reports are assembled by reading the artefacts back off disk, never from
    # values held in memory. A table therefore cannot disagree with the JSON
    # file it claims to summarise.
    threshold_path = output_root / "calibrated_threshold.json"
    payloads: Dict[str, Dict[str, Any]] = {
        "lfw_development": read_json_artifact(output_root / "lfw_development_metrics.json"),
        "lfw_final": read_json_artifact(output_root / "lfw_final_metrics.json"),
        "cplfw": read_json_artifact(output_root / "cplfw_metrics.json"),
    }
    # Prefer complete enrolment accounting; the fallback accepts summaries
    # that contain only the scored-gallery measurements.
    gallery_v2_path = output_root / "duplicate_gallery_metrics_v2.json"
    gallery_path = (
        gallery_v2_path if gallery_v2_path.is_file() else output_root / "duplicate_gallery_metrics.json"
    )
    gallery_payload = read_json_artifact(gallery_path)
    threshold_payload = read_json_artifact(threshold_path)

    output_files = {
        "calibrated_threshold.json": threshold_path,
        "lfw_development_metrics.json": output_root / "lfw_development_metrics.json",
        "lfw_final_metrics.json": output_root / "lfw_final_metrics.json",
        "cplfw_metrics.json": output_root / "cplfw_metrics.json",
        gallery_path.name: gallery_path,
    }

    write_json_artifact(
        output_root / "run_manifest.json",
        {
            "artifact_type": "run_manifest",
            "opaque_id_version": OPAQUE_ID_VERSION,
            "produced_by": "ACP_arden.py --mode full",
            "dataset_storage": (
                "private, gitignored local research storage; path omitted from public artifacts"
            ),
            "dataset_root_variable": "FACE_DATA_ROOT",
            "protocol_root_variable": "FACE_PROTOCOL_ROOT",
            "model_root_variable": "FACE_MODEL_ROOT",
            "cplfw_image_variant": cplfw_image_variant,
            "output_root": project_relative(output_root),
            "gallery_manifest": project_relative(gallery_manifest_path),
            "lfw_evaluation_protocol": payloads["lfw_final"].get("evaluation_protocol", "legacy_transferred_threshold"),
            "frozen_threshold_scope": "CPLFW and gallery transfer only; final LFW has per-fold training thresholds",
            "frozen_threshold": threshold_payload.get("threshold"),
            "frozen_threshold_candidate": threshold_payload.get("operating_strategy"),
            "output_file_sha256": {
                name: sha256_of_file(path) for name, path in output_files.items()
            },
            "software_environment": software_environment_report(),
        },
    )

    summary_field_order = [
        "experiment", "protocol_file", "total_pairs", "scored_pairs", "failure_rate", "threshold",
        "evaluation_protocol", "mean_fold_accuracy", "fold_accuracy_standard_error",
        "accuracy", "precision", "recall", "f1", "false_match_rate", "false_non_match_rate",
        "roc_auc", "equal_error_rate", "embedding_time_mean_ms", "embedding_time_median_ms",
        "embedding_time_p95_ms", "gallery_size", "duplicate_probe_count", "unknown_probe_count",
        "duplicate_detection_rate", "false_duplicate_review_rate", "rank1_identification_rate",
        "true_duplicate_miss_rate", "gallery_search_time_mean_ms", "gallery_search_time_p95_ms",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for name in VERIFICATION_EXPERIMENTS:
        payload = payloads[name]
        summary_rows.append(
            {"experiment": name, **{f: payload.get(f, "") for f in summary_field_order[1:]}}
        )
    summary_rows.append(
        {
            "experiment": "duplicate_gallery",
            **{f: gallery_payload.get(f, "") for f in summary_field_order[1:]},
        }
    )
    write_csv_artifact(
        output_root / "metrics_summary.csv", summary_rows, fieldnames=summary_field_order
    )

    confusion_fields = [
        "experiment", "true_positive", "false_positive", "true_negative", "false_negative",
    ]
    confusion_rows: List[Dict[str, Any]] = []
    for name in VERIFICATION_EXPERIMENTS:
        matrix = payloads[name].get("confusion_matrix") or {}
        confusion_rows.append(
            {"experiment": name, **{f: matrix.get(f, "") for f in confusion_fields[1:]}}
        )
    write_csv_artifact(
        output_root / "confusion_matrices.csv", confusion_rows, fieldnames=confusion_fields
    )

    roc_fields = ["experiment", "threshold", "false_match_rate", "true_match_rate"]
    roc_rows: List[Dict[str, Any]] = []
    for name in VERIFICATION_EXPERIMENTS:
        for point in payloads[name].get("roc_points", []):
            roc_rows.append({"experiment": name, **{f: point[f] for f in roc_fields[1:]}})
    write_csv_artifact(output_root / "roc_points.csv", roc_rows, fieldnames=roc_fields)

    write_markdown_artifact(
        output_root / "FINAL_EVALUATION_REPORT.md", render_final_report(payloads, gallery_payload)
    )

    leaks = find_path_leaks(output_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise SystemExit(
            "Refusing to finish: public aggregate output(s) contain a personal/absolute path:\n"
            + "\n".join(f"  {redact_private_paths(leak)}" for leak in leaks)
        )
    assert_no_identifier_key_leak(output_root)

    announce(
        f"Wrote run_manifest.json, metrics_summary.csv ({len(summary_rows)} rows), "
        f"confusion_matrices.csv ({len(confusion_rows)} rows), roc_points.csv "
        f"({len(roc_rows)} rows) and FINAL_EVALUATION_REPORT.md to "
        f"{project_relative(output_root)}"
    )


def _render_gallery_coverage_lines(gallery: Dict[str, Any]) -> List[str]:
    """Gallery enrolment coverage. Emitted unconditionally so a detection rate
    can never be read without knowing how much of the gallery actually
    enrolled; an artefact predating the corrected accounting says so plainly
    instead of implying full coverage."""
    if "gallery_entry_failure_rate" not in gallery:
        return [
            "  Gallery enrolment coverage: not recorded (artefact predates "
            f"{GALLERY_METHODOLOGY_REVISION}; re-run to obtain it)",
        ]
    return [
        f"  Intended gallery entries: {format_count(gallery.get('intended_gallery_size'))}, "
        f"enrolled {format_count(gallery.get('embedded_gallery_size'))}",
        f"  Gallery enrolment-failure rate: "
        f"{format_percentage(gallery.get('gallery_entry_failure_rate'))} "
        f"({format_count(gallery.get('gallery_entry_failure_count'))} references never enrolled)",
    ]


def _render_end_to_end_gallery_lines(gallery: Dict[str, Any]) -> List[str]:
    """The end-to-end detection rate, which shares the conditional figure's
    numerator but counts every intended mated probe. Printing it here is what
    stops the flattering conditional number standing on its own."""
    if "end_to_end_duplicate_detection_rate" not in gallery:
        return [
            "  Duplicate detection rate (end-to-end): not recorded (artefact "
            f"predates {GALLERY_METHODOLOGY_REVISION}; the conditional figure "
            "above therefore excludes unenrolled references and failed probes)",
        ]
    return [
        f"  Duplicate detection rate (end-to-end): "
        f"{format_percentage(gallery.get('end_to_end_duplicate_detection_rate'))}",
        f"  Mated probes with no enrolled reference: "
        f"{format_count(gallery.get('gallery_reference_unavailable_count'))}",
        "  LIMITATION: the conditional rate above is measured only over mated probes that "
        "were scored. The end-to-end rate counts every intended mated probe, so extraction "
        "failures and unenrolled references reduce it. Quote the two together.",
    ]


def render_results_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    """Headline figures for the terminal, read from the aggregate artifacts
    rather than hard-coded. Each conditional figure is printed together with
    the limitation that makes it interpretable: the CPLFW accuracy never
    appears without its extraction-failure rate, and the gallery detection rate
    never appears without its false-review rate."""
    final = read_json_artifact(output_root / "lfw_final_metrics.json")
    cplfw = read_json_artifact(output_root / "cplfw_metrics.json")
    # Complete enrolment accounting supplies both intended and scored counts.
    # The fallback supports summaries with scored-gallery measurements only.
    gallery_v2_path = output_root / "duplicate_gallery_metrics_v2.json"
    gallery = read_json_artifact(
        gallery_v2_path if gallery_v2_path.is_file() else output_root / "duplicate_gallery_metrics.json"
    )
    threshold = read_json_artifact(output_root / "calibrated_threshold.json")

    lines = [
        f"{PROGRAMME_TITLE} — results summary",
        "",
        f"Frozen transfer threshold (CPLFW/gallery): {format_number(threshold.get('threshold'), 6)} "
        f"(candidate: {threshold.get('operating_strategy')}, status: {threshold.get('status')})",
        f"Selection rule: {threshold.get('selection_rule')}",
        "",
        "Experiment 3 — final LFW (ten folds; thresholds trained on other folds)",
        f"  Final LFW mean fold accuracy: {format_percentage(final.get('mean_fold_accuracy', final.get('accuracy')))}",
        f"  Final LFW false-match rate: {format_percentage(final.get('false_match_rate'))}",
        f"  Final LFW false-non-match rate: {format_percentage(final.get('false_non_match_rate'))}",
        f"  Final LFW EER: {format_percentage(final.get('equal_error_rate'))}",
        f"  Final LFW extraction-failure rate: {format_percentage(final.get('failure_rate'))}",
        f"  Scored pairs: {format_count(final.get('scored_pairs'))} / "
        f"{format_count(final.get('total_pairs'))}",
        "",
        "Experiment 4 — raw CPLFW (same frozen threshold, cross-pose generalisation)",
        f"  Raw CPLFW conditional accuracy: {format_percentage(cplfw.get('accuracy'))}",
        f"  Raw CPLFW scored pairs: {format_count(cplfw.get('scored_pairs'))} / "
        f"{format_count(cplfw.get('total_pairs'))}",
        f"  Raw CPLFW failed pairs: {format_count(cplfw.get('failed_pairs'))}",
        f"  Raw CPLFW extraction-failure rate: {format_percentage(cplfw.get('failure_rate'))}",
        f"  Raw CPLFW false-match rate: {format_percentage(cplfw.get('false_match_rate'))}",
        f"  Raw CPLFW false-non-match rate: {format_percentage(cplfw.get('false_non_match_rate'))}",
        f"  Raw CPLFW EER: {format_percentage(cplfw.get('equal_error_rate'))}",
        f"  LIMITATION: the CPLFW accuracy above is conditional on the "
        f"{format_count(cplfw.get('scored_pairs'))} pairs that yielded one valid face on both "
        f"sides. {format_percentage(cplfw.get('failure_rate'))} of the protocol "
        f"({format_count(cplfw.get('failed_pairs'))} pairs) never reached the comparison stage, "
        f"and must be quoted alongside it.",
        "",
        "Experiment 5 — 1:N duplicate-profile gallery (real LFW images, "
        f"seed {gallery.get('seed')})",
        f"  Gallery size: {format_count(gallery.get('gallery_size'))}",
        *_render_gallery_coverage_lines(gallery),
        f"  Duplicate detection rate (conditional): "
        f"{format_percentage(gallery.get('duplicate_detection_rate'))}",
        *_render_end_to_end_gallery_lines(gallery),
        f"  Rank-1 identification rate: "
        f"{format_percentage(gallery.get('rank1_identification_rate'))}",
        f"  False duplicate-review rate: "
        f"{format_percentage(gallery.get('false_duplicate_review_rate'))}",
        f"  LIMITATION: the detection rate above is inseparable from the "
        f"{format_percentage(gallery.get('false_duplicate_review_rate'))} false duplicate-review "
        f"rate. Reusing a 1:1-calibrated threshold for 1:N search flags that share of genuinely "
        f"new identities for review, so the two figures must never be quoted apart.",
        "",
        "Policy: " + str(gallery.get("policy_note", POLICY_NOTE)),
        "",
        "Full write-up: results/aggregate/FINAL_EVALUATION_REPORT.md",
    ]
    return "\n".join(lines)


# =============================================================================
# 15. Privacy validation
# =============================================================================
#
# Guards that keep published outputs free of names, absolute paths and raw
# embedding vectors.

_FORBIDDEN_KEY_SUBSTRINGS = ("embedding", "path", "identity", "name")
_ALLOWED_KEY_EXCEPTIONS = {
    "strategy",
    "identity_count",
    "identity_hash",
    "candidate_identity_hash",
}

_TEXT_ARTIFACT_SUFFIXES = (".json", ".csv", ".md", ".txt")
_IMAGE_ARTIFACT_SUFFIXES = (".png",)

# Cloud-sync location markers, which identify a private storage layout even
# when they appear without a leading absolute-path prefix — for instance inside
# a rendered image's embedded metadata, or a relative-looking string. These are
# generic provider names; the researcher's own roots are added at scan time
# from the configured environment variables.
_PRIVATE_LOCATION_MARKERS = (
    "Library/CloudStorage",
    "OneDrive",
    "Dropbox",
    "Google Drive",
    "iCloud Drive",
)


class PrivacyLeakError(ValueError):
    """Raised when a record or artifact would disclose private data."""


def assert_no_leakage(record: Mapping[str, Any], *, context: str = "") -> None:
    """Raise PrivacyLeakError if any key or value in the record looks like a
    real name, an absolute filesystem path, or a raw embedding vector."""
    for key, value in record.items():
        label = f"{context}.{key}" if context else str(key)
        lowered_key = key.lower() if isinstance(key, str) else str(key)

        if key not in _ALLOWED_KEY_EXCEPTIONS:
            for banned in _FORBIDDEN_KEY_SUBSTRINGS:
                if banned in lowered_key:
                    raise PrivacyLeakError(f"{label}: key name may leak private data")

        if isinstance(value, str) and (value.startswith("/") or value.startswith("~")):
            raise PrivacyLeakError(f"{label}: value looks like an absolute path: {value}")

        # A long, purely numeric sequence has the shape of a raw face embedding,
        # which must never reach a published artefact.
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 32
            and all(isinstance(item, (int, float)) for item in value)
        ):
            raise PrivacyLeakError(f"{label}: value looks like a raw embedding vector")

        if isinstance(value, Mapping):
            assert_no_leakage(value, context=label)


def assert_no_identifier_key_leak(root: Path) -> None:
    """Confirm the identifier key appears in no published artefact.

    The key is compared against file contents but is never written into the
    error message, a log line or a return value — reporting *where* it leaked is
    useful, reporting *what* leaked would be a second leak. Nothing is raised
    when no key is configured: there is then nothing to disclose."""
    if _ID_HMAC_KEY is None:
        return
    # Both textual forms the key could plausibly take if it were ever written
    # out: the base64 form used in the environment file and the hexadecimal
    # form a debugging statement would produce.
    encoded = base64.urlsafe_b64encode(_ID_HMAC_KEY).decode("ascii").rstrip("=")
    needles = [encoded, _ID_HMAC_KEY.hex()]
    for path in sorted(Path(root).rglob("*")):
        # Only text artefacts are searched; a figure cannot contain the key in
        # a readable form, and reading one as text would waste the scan.
        if not path.is_file() or path.suffix.lower() not in _TEXT_ARTIFACT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for needle in needles:
            if needle and needle in text:
                raise PrivacyLeakError(
                    f"{path.name} contains the identifier HMAC key. The key must never be "
                    f"published; regenerate it and rebuild the affected artefacts."
                )


def default_forbidden_path_substrings(*, env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Substrings that must never appear in a published result: the user's home
    directory, common absolute-path prefixes, known private-storage location
    names, and the expanded value of every storage environment variable."""
    source = os.environ if env is None else env
    from_file = load_env_file()
    # General prefixes that would reveal a home directory on any platform.
    substrings = {"/Users/", "\\Users\\", "/home/", str(Path.home())}
    # Cloud-sync folder names, which disclose a private storage arrangement
    # even when the path itself looks unremarkable.
    substrings.update(_PRIVATE_LOCATION_MARKERS)
    # The researcher's own configured locations, taken from both the process
    # environment and the local file, since either may be in force.
    for variable in (*REQUIRED_ENVIRONMENT_VARIABLES, *OPTIONAL_ENVIRONMENT_VARIABLES):
        for value in (source.get(variable), from_file.get(variable)):
            if value:
                substrings.add(value)
    return sorted(s for s in substrings if s)


def _png_text_metadata(path: Path) -> str:
    """Every tEXt/iTXt/zTXt chunk in a PNG, concatenated. A renderer may write
    its own metadata, so an image can leak a path that never appears in the
    visible pixels.

    A corrupt or truncated PNG yields "" — there is nothing to read, and the
    file is not publishable evidence anyway. A missing Pillow is different: it
    would make every PNG silently pass, turning this check into a no-op that
    still reports "clean", so its absence is raised rather than swallowed."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a pinned dependency
        raise PrivacyLeakError(
            "Pillow is not installed, so PNG metadata cannot be scanned. Refusing to report a "
            "clean result from a check that did not run."
        ) from exc

    try:
        with Image.open(path) as image:
            return "\n".join(
                f"{key}: {value}"
                for key, value in (image.info or {}).items()
                if isinstance(value, str)
            )
    except OSError:
        return ""


# Published artefacts must carry no local storage path. A research-storage
# location identifies the person running the evaluation and can disclose where
# the benchmark images are held, so every published file is scanned before a run
# is allowed to report success.
def find_path_leaks(root: Path, *, forbidden_substrings: Sequence[str]) -> List[str]:
    """Recursively scan every JSON, CSV, Markdown or text file under the root —
    and the embedded text metadata of every PNG — for a forbidden substring.
    Returns "file:line: forbidden substring" findings; an empty list means
    clean. Unreadable or binary files are skipped rather than raising."""
    findings: List[str] = []
    root = Path(root)
    if not root.exists():
        return findings
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Two kinds of published file can carry a private path. A text artefact
        # can contain one directly; an image can hide one in the metadata a
        # plotting library writes when it saves the file.
        suffix = path.suffix.lower()
        if suffix in _TEXT_ARTIFACT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            location = "{path}:{line}"
        elif suffix in _IMAGE_ARTIFACT_SUFFIXES:
            text = _png_text_metadata(path)
            location = "{path} (embedded PNG metadata, entry {line})"
        else:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for needle in forbidden_substrings:
                if needle and needle in line:
                    where = location.format(path=path, line=line_number)
                    findings.append(f"{where}: contains forbidden substring {needle!r}")
    return findings


def check_public_outputs(paths: Sequence[Path]) -> bool:
    """Scan published output directories and report the outcome. Returns True
    when every scanned location is clean."""
    forbidden = default_forbidden_path_substrings()
    all_leaks: List[str] = []
    scanned: List[str] = []

    # Every location is scanned before anything is reported, so one clean
    # directory cannot mask a leak in a later one. A location that does not
    # exist yet is skipped rather than treated as a failure.
    for path in paths:
        if not path.exists():
            continue
        all_leaks.extend(find_path_leaks(path, forbidden_substrings=forbidden))
        scanned.append(project_relative(path))

    # Findings are themselves redacted before printing: naming the file is
    # useful, reprinting the private path would repeat the leak on screen.
    if all_leaks:
        print("FAIL personal/absolute path(s) found in published outputs:", file=sys.stderr)
        for leak in all_leaks:
            print(f"  {redact_private_paths(leak)}", file=sys.stderr)
        return False

    print(f"OK   no personal/absolute paths found under: {', '.join(scanned) or 'nothing to scan'}")
    return True


# =============================================================================
# 16. Optional human-review interface
# =============================================================================
#
# A Streamlit moderation page for manually reviewing anonymised
# duplicate-profile cases. It never displays a real name, real file path or raw
# embedding — only opaque identifiers, a similarity score, and the threshold
# that opened the case. It applies no sanction of any kind.

REVIEW_STATUSES = ["open", "confirmed_duplicate", "false_match", "dismissed"]

# Display wording is separate from stored status values, so the moderation
# page can display descriptive labels while database records retain their schema.
REVIEW_STATUS_WORDING = {
    "open": "awaiting review",
    "confirmed_duplicate": "marked for further review",
    "false_match": "closed as no action",
    "dismissed": "dismissed",
}
ALLOWED_REVIEW_STATUSES = set(REVIEW_STATUSES)

REVIEW_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_cases (
    case_id TEXT PRIMARY KEY,
    probe_sample_id TEXT NOT NULL,
    candidate_identity_hash TEXT NOT NULL,
    similarity REAL NOT NULL,
    threshold REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    -- Which identifier scheme produced probe_sample_id and
    -- candidate_identity_hash. Rows from two schemes are not comparable: the
    -- same person yields different identifiers under different keys.
    opaque_id_version TEXT NOT NULL DEFAULT 'hmac-sha256-v1'
);
"""


class ReviewDatabaseVersionError(RuntimeError):
    """Raised when a review database holds rows from another identifier scheme."""


def _migrate_review_schema(connection: sqlite3.Connection) -> None:
    """Add the version column to a database created before it existed. Rows
    already present came from the old fixed-salt scheme, so they are marked as
    such rather than silently relabelled."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(review_cases)")}
    if "opaque_id_version" not in columns:
        connection.execute(
            "ALTER TABLE review_cases ADD COLUMN opaque_id_version TEXT NOT NULL "
            "DEFAULT 'legacy-salted-sha256'"
        )


def assert_review_database_version(connection: sqlite3.Connection) -> None:
    """Refuse to mix identifier schemes in one database. The moderation
    database is private and disposable, so the remedy is to delete it."""
    # Every identifier scheme the stored cases were written under.
    versions = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT opaque_id_version FROM review_cases WHERE opaque_id_version IS NOT NULL"
        )
    }
    # Anything this build does not emit. Identifiers from a different scheme,
    # or from the same scheme under a different secret key, do not refer to the
    # same people, so rows from the two could never be compared or merged.
    foreign = versions - {OPAQUE_ID_VERSION}
    if foreign:
        raise ReviewDatabaseVersionError(
            f"The moderation database holds cases written under identifier scheme(s) "
            f"{sorted(foreign)!r}, but this build emits {OPAQUE_ID_VERSION!r}. Those "
            f"identifiers are not comparable, so the rows cannot be merged. The review "
            f"database is private, disposable local state: delete the file and re-run the "
            f"evaluation to regenerate it."
        )


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    probe_sample_id: str
    candidate_identity_hash: str
    similarity: float
    threshold: float
    status: str
    created_at: str
    decided_at: Optional[str]


# Close the review database even when the caller raises, so an interrupted
# review session cannot leave the local case file locked.
@contextmanager
def review_database(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(REVIEW_SCHEMA)
        _migrate_review_schema(connection)
        assert_review_database_version(connection)
        connection.row_factory = sqlite3.Row
        yield connection
        connection.commit()
    finally:
        connection.close()


def upsert_review_case(
    connection: sqlite3.Connection,
    *,
    case_id: str,
    probe_sample_id: str,
    candidate_identity_hash: str,
    similarity: float,
    threshold: float,
) -> None:
    # Insert the case, or refresh it if a previous run already raised it. The
    # stored status and decision are deliberately not overwritten, so re-running
    # the evaluation never discards a decision a reviewer has already recorded.
    connection.execute(
        """
        INSERT INTO review_cases
            (case_id, probe_sample_id, candidate_identity_hash, similarity, threshold,
             status, created_at, opaque_id_version)
        VALUES (?, ?, ?, ?, ?, 'open', ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
            probe_sample_id=excluded.probe_sample_id,
            candidate_identity_hash=excluded.candidate_identity_hash,
            similarity=excluded.similarity,
            threshold=excluded.threshold,
            opaque_id_version=excluded.opaque_id_version
        """,
        (
            case_id,
            probe_sample_id,
            candidate_identity_hash,
            similarity,
            threshold,
            utc_now_iso(),
            OPAQUE_ID_VERSION,
        ),
    )


def count_review_cases(connection: sqlite3.Connection, *, status: Optional[str] = None) -> int:
    if status is not None and status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unknown status filter: {status}")
    if status:
        row = connection.execute(
            "SELECT COUNT(*) FROM review_cases WHERE status = ?", (status,)
        ).fetchone()
    else:
        row = connection.execute("SELECT COUNT(*) FROM review_cases").fetchone()
    return int(row[0])


def list_review_cases(
    connection: sqlite3.Connection, *, status: Optional[str] = None, limit: Optional[int] = None
) -> List[ReviewCase]:
    """Cases ordered by descending similarity, so the strongest candidates for
    human attention come first. ``limit`` bounds how many are returned: a real
    queue can hold thousands, and rendering all of them at once would make the
    page unusable rather than more informative."""
    # The status filter is checked against the permitted set before it reaches
    # the query, so an unrecognised value fails here rather than silently
    # returning an empty queue that a reviewer would read as "no cases".
    if status is not None and status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unknown status filter: {status}")
    if limit is not None and limit < 0:
        raise ValueError("limit must not be negative")

    # The query is assembled from fixed fragments and every caller-supplied
    # value is bound as a parameter, so no input is ever interpolated into SQL.
    clauses = "SELECT * FROM review_cases"
    parameters: List[Any] = []
    if status:
        clauses += " WHERE status = ?"
        parameters.append(status)
    # Strongest candidates first: a reviewer works down the queue, so ordering
    # by descending similarity puts the most demanding cases at the top.
    clauses += " ORDER BY similarity DESC"
    if limit is not None:
        clauses += " LIMIT ?"
        parameters.append(limit)

    rows = connection.execute(clauses, tuple(parameters))
    return [_row_to_review_case(row) for row in rows]


def set_review_status(connection: sqlite3.Connection, *, case_id: str, status: str) -> None:
    if status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unknown status: {status}; must be one of {sorted(REVIEW_STATUSES)}")
    connection.execute(
        "UPDATE review_cases SET status = ?, decided_at = ? WHERE case_id = ?",
        (status, utc_now_iso(), case_id),
    )


def _row_to_review_case(row: sqlite3.Row) -> ReviewCase:
    return ReviewCase(
        case_id=row["case_id"],
        probe_sample_id=row["probe_sample_id"],
        candidate_identity_hash=row["candidate_identity_hash"],
        similarity=row["similarity"],
        threshold=row["threshold"],
        status=row["status"],
        created_at=row["created_at"],
        decided_at=row["decided_at"],
    )


def populate_review_database(
    db_path: Path, result: GalleryEvaluationResult, threshold: float
) -> int:
    """Record every probe that exceeded the review threshold as an open case.
    Only opaque identifiers and scores are stored — never a name or a path."""
    flagged = [p for p in result.probe_results if p.exceeds_duplicate_threshold]
    with review_database(db_path) as connection:
        for probe in flagged:
            # A probe only exceeds the threshold once it has been scored
            # against a candidate, so both fields are populated here.
            if probe.top_candidate_identity_hash is None or probe.top_similarity is None:
                continue
            upsert_review_case(
                connection,
                case_id=f"{probe.sample_id}:{probe.top_candidate_identity_hash}",
                probe_sample_id=probe.sample_id,
                candidate_identity_hash=probe.top_candidate_identity_hash,
                similarity=probe.top_similarity,
                threshold=threshold,
            )
    return len(flagged)


def running_under_streamlit() -> bool:
    """True when this file is already executing inside a Streamlit script run,
    so the review mode renders the page instead of launching another server."""
    if os.environ.get("ACP_ARDEN_REVIEW_CHILD") == "1":
        return True
    # Only Streamlit itself imports Streamlit before this file runs. Checking
    # that first keeps a plain interpreter run from importing the package at
    # all, which would emit a "missing ScriptRunContext" warning over the menu.
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001 - defensive across Streamlit versions
        return False


def render_review_page(db_path: Path) -> None:
    """The Streamlit page itself. Rendered only inside a Streamlit script run."""
    import streamlit as st

    st.set_page_config(page_title="Human review moderation", layout="wide")

    st.title("Human review moderation")
    st.error(
        "**These cases are not confirmed duplicate profiles.** They were created by "
        "Experiment 5 using the LFW gallery method. That threshold produced a "
        "high false-review rate and is included as a research baseline. This page "
        "demonstrates the review workflow, not a production moderation decision."
    )
    st.warning(
        "A high similarity to an enrolled profile is a review signal, not proof that two "
        "profiles belong to the same person, that a photograph was stolen or that fraud "
        "occurred. No account is banned, suspended or accused by this page. Case, probe "
        "and candidate identifiers are opaque one-way hashes; no real name, file path or "
        "face embedding is ever shown here."
    )

    # Aggregate counts only. No identifier, path or score is summarised here.
    with review_database(db_path) as summary_connection:
        totals = {
            "Cases currently stored": count_review_cases(summary_connection),
            **{
                f"Cases {REVIEW_STATUS_WORDING.get(state, state)}":
                    count_review_cases(summary_connection, status=state)
                for state in REVIEW_STATUSES
            },
        }
    count_columns = st.columns(len(totals))
    for column, (label, value) in zip(count_columns, totals.items()):
        column.metric(label, f"{value:,}")

    filter_column, page_size_column = st.columns([2, 1])
    status_filter = filter_column.selectbox("Filter by status", ["all", *REVIEW_STATUSES])
    page_size = page_size_column.number_input(
        "Cases to show", min_value=5, max_value=200, value=25, step=5
    )

    with review_database(db_path) as connection:
        status = None if status_filter == "all" else status_filter
        total = count_review_cases(connection, status=status)
        # A whole queue is thousands of cases; a reviewer works the strongest
        # candidates first, so the page shows a bounded, ordered slice of them.
        cases = list_review_cases(connection, status=status, limit=int(page_size))

        if not cases:
            st.info(
                "No cases match this filter. Run the complete evaluation "
                "(`python ACP_arden.py --mode full`) to populate the moderation database."
            )
            return

        st.caption(
            f"Showing the {len(cases)} highest-similarity case(s) of {total} matching this "
            f"filter, ordered by descending similarity. Each one is a prompt for a human "
            f"decision, not a finding."
        )

        for case in cases:
            with st.container(border=True):
                st.write(f"**Case:** `{case.case_id}`")
                col1, col2, col3 = st.columns(3)
                col1.metric("Similarity", f"{case.similarity:.4f}")
                col2.metric("Threshold", f"{case.threshold:.4f}")
                col3.metric("Status", case.status)
                st.caption(
                    f"Probe: `{case.probe_sample_id}` — "
                    f"Candidate identity: `{case.candidate_identity_hash}`"
                )
                st.caption(
                    f"Opened: {case.created_at}"
                    + (f" — Decided: {case.decided_at}" if case.decided_at else "")
                )

                chosen = st.radio(
                    "Decision",
                    REVIEW_STATUSES,
                    index=REVIEW_STATUSES.index(case.status),
                    key=f"decision_{case.case_id}",
                    horizontal=True,
                )
                if chosen != case.status:
                    set_review_status(connection, case_id=case.case_id, status=chosen)
                    st.rerun()


def launch_review_interface(db_path: Path) -> int:
    """Re-run this same file under Streamlit. The child process is marked with
    an environment flag so it renders the page rather than recursing."""
    if not db_path.exists():
        announce(
            f"No review database at {project_relative(db_path)} yet. The page will open empty; "
            f"run option 4 (complete evaluation) first to populate it."
        )
    # Stated before the browser opens, because the queue shown here was built
    # by the deliberately high-false-review Experiment 5 baseline.
    print("")
    print(render_experiment_preview("review"))
    print("")
    child_environment = dict(os.environ)
    child_environment["ACP_ARDEN_REVIEW_CHILD"] = "1"
    print("Starting human review moderation (Ctrl+C in this terminal to stop it).")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.address=127.0.0.1",
            "--",
            "--mode",
            "review",
            "--review-db",
            str(db_path),
        ],
        check=False,
        env=child_environment,
    )
    if completed.returncode != 0:
        print(
            "The review interface exited non-zero. Streamlit is an optional dependency: "
            "install it with `python -m pip install -r requirements.txt`.",
            file=sys.stderr,
        )
    return completed.returncode


# =============================================================================
# 17. BFW dataset adapter
# =============================================================================
#
# Balanced Faces in the Wild is an external real benchmark. It is never
# downloaded automatically, never vendored into this repository and never read
# from an unofficial mirror: the researcher obtains it from the official project
# under its own terms and points FACE_BFW_ROOT at the extracted tree.
#
# The metadata schema is pinned rather than sniffed. A file that does not match
# stops the run with a message naming what was expected and what was found,
# because silently mis-parsing an identity or a subgroup column would corrupt
# every downstream figure while still producing plausible-looking numbers.
##############
# Title: Face Recognition: Too Bias, or Not Too Bias?
# Author: Robinson, J.P., Livitz, G., Henon, Y., Qin, C., Fu, Y. and Timoner, S.
# Date: 2020
# Availability: https://doi.org/10.1109/CVPRW50498.2020.00008
##############
##############
# Title: Balanced Faces in the Wild (BFW), source data and metadata table
# Author: Robinson, J.P. and contributors (visionjo/facerec-bias-bfw)
# Date: 2020
# Availability: https://github.com/visionjo/facerec-bias-bfw
##############


class BfwDatasetError(RuntimeError):
    """Raised when BFW is absent, misconfigured or fails schema validation."""


# The eight official demographic subgroups. Used for stratification and for
# aggregate fairness reporting only; never joined to an identifiable name in
# any published artefact.
BFW_SUBGROUPS = (
    "asian_females",
    "asian_males",
    "black_females",
    "black_males",
    "indian_females",
    "indian_males",
    "white_females",
    "white_males",
)

# Columns of the official BFW datatable that this adapter relies on. Pinned so
# that a differently-shaped release is refused rather than misread.
BFW_REQUIRED_COLUMNS = ("fold", "p1", "p2", "label", "att1", "att2")

# This project's own open-set protocol built on top of the official data. BFW
# was published for verification and bias analysis and ships no official
# open-set identification protocol, so the version below identifies *our*
# construction, not an upstream standard.
BFW_PROTOCOL_VERSION = "acp-arden-bfw-open-set-v1"

BFW_ROOT_VARIABLE = "FACE_BFW_ROOT"
BFW_METADATA_ROOT_VARIABLE = "FACE_BFW_METADATA_ROOT"


@dataclass(frozen=True)
class BfwImage:
    """One BFW image. ``image_path`` is private and never published; the two
    opaque identifiers are what appear in public artefacts."""

    image_path: Path
    identity: str  # private, e.g. "asian_females/n000009"
    subgroup: str
    sample_id: str
    identity_hash: str


@dataclass(frozen=True)
class BfwDataset:
    images: List[BfwImage]
    metadata_sha256: str
    metadata_filename: str

    def by_identity(self) -> Dict[str, List[BfwImage]]:
        grouped: Dict[str, List[BfwImage]] = {}
        for image in self.images:
            grouped.setdefault(image.identity, []).append(image)
        # Deterministic ordering by filename, so role assignment never depends
        # on directory-iteration order.
        return {
            identity: sorted(items, key=lambda i: i.image_path.name)
            for identity, items in sorted(grouped.items())
        }

    def subgroup_of_identity(self) -> Dict[str, str]:
        return {image.identity: image.subgroup for image in self.images}


def _bfw_identity_and_subgroup(relative: str) -> Tuple[str, str]:
    """Split an official BFW relative path into (identity, subgroup).

    The published layout is ``<subgroup>/<identity>/<image>.jpg``; anything
    else is a schema violation rather than something to guess at."""
    # Backslashes are normalised first, so a table written on Windows parses
    # identically to one written on a POSIX system.
    parts = PurePosixPath(relative.strip().replace("\\", "/")).parts
    # Exactly three components are required. A shorter or longer path means the
    # dataset is not the official release, and guessing at the identity would
    # silently change which photographs belong to which person.
    if len(parts) != 3:
        raise BfwDatasetError(
            f"BFW image path {relative!r} does not match the official "
            f"'<subgroup>/<identity>/<image>' layout (found {len(parts)} component(s))."
        )
    subgroup, identity_folder, _filename = parts
    # The subgroup is pinned to the eight official categories, so an unexpected
    # folder cannot quietly create a ninth subgroup in the reported results.
    if subgroup not in BFW_SUBGROUPS:
        raise BfwDatasetError(
            f"BFW image path {relative!r} names subgroup {subgroup!r}, which is not one of "
            f"the eight official subgroups {list(BFW_SUBGROUPS)}."
        )
    return f"{subgroup}/{identity_folder}", subgroup


def load_bfw_dataset(
    image_root: Path,
    metadata_path: Path,
    *,
    require_files_exist: bool = True,
) -> BfwDataset:
    """Read the official BFW datatable and return the deduplicated image set.

    Every failure is explicit. Nothing is skipped, inferred or defaulted."""
    image_root = Path(image_root).resolve()
    metadata_path = Path(metadata_path)
    if not image_root.is_dir():
        raise BfwDatasetError(f"{BFW_ROOT_VARIABLE} does not point at a directory: {image_root}")
    if not metadata_path.is_file():
        raise BfwDatasetError(
            f"BFW metadata table not found: {metadata_path}. Obtain it from the official "
            f"project (visionjo/facerec-bias-bfw) and set {BFW_METADATA_ROOT_VARIABLE}."
        )

    text = metadata_path.read_text(encoding="utf-8")
    reader = csv.DictReader(StringIO(text))
    columns = tuple(reader.fieldnames or ())
    missing = [c for c in BFW_REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise BfwDatasetError(
            f"BFW metadata {metadata_path.name} is missing required column(s) {missing}. "
            f"Expected at least {list(BFW_REQUIRED_COLUMNS)}; found {list(columns)}. This "
            f"adapter pins the official schema and refuses to guess at a variant."
        )

    # relative path -> subgroup asserted by the table, so a row that contradicts
    # an earlier row is caught rather than silently overwriting it.
    asserted: Dict[str, str] = {}
    for line_number, row in enumerate(reader, start=2):
        for path_column, attribute_column in (("p1", "att1"), ("p2", "att2")):
            relative = (row.get(path_column) or "").strip()
            attribute = (row.get(attribute_column) or "").strip()
            if not relative:
                raise BfwDatasetError(
                    f"BFW metadata {metadata_path.name} line {line_number}: column "
                    f"{path_column!r} is empty."
                )
            _identity, subgroup = _bfw_identity_and_subgroup(relative)
            if attribute and attribute != subgroup:
                raise BfwDatasetError(
                    f"BFW metadata {metadata_path.name} line {line_number}: column "
                    f"{attribute_column!r} says {attribute!r} but the path implies "
                    f"{subgroup!r}. Refusing to choose between them."
                )
            previous = asserted.get(relative)
            if previous is not None and previous != subgroup:
                raise BfwDatasetError(
                    f"BFW metadata {metadata_path.name}: image {relative!r} is assigned to "
                    f"both {previous!r} and {subgroup!r}."
                )
            asserted[relative] = subgroup

    if not asserted:
        raise BfwDatasetError(f"BFW metadata {metadata_path.name} contained no image rows.")

    images: List[BfwImage] = []
    identity_subgroup: Dict[str, str] = {}
    for relative in sorted(asserted):
        identity, subgroup = _bfw_identity_and_subgroup(relative)
        absolute = (image_root / relative).resolve()
        # Containment check before any filesystem access, so a crafted relative
        # path cannot reach outside the configured root.
        if not absolute.is_relative_to(image_root):
            raise BfwDatasetError(
                f"BFW image {relative!r} resolves outside {BFW_ROOT_VARIABLE}."
            )
        if require_files_exist and not absolute.is_file():
            raise BfwDatasetError(f"BFW image listed in the metadata is missing: {relative}")
        established = identity_subgroup.setdefault(identity, subgroup)
        if established != subgroup:
            raise BfwDatasetError(
                f"BFW identity {identity!r} appears under two subgroups: "
                f"{established!r} and {subgroup!r}."
            )
        images.append(
            BfwImage(
                image_path=absolute,
                identity=identity,
                subgroup=subgroup,
                sample_id=opaque_id(f"bfw:{relative}"),
                identity_hash=opaque_id(f"bfw-identity:{identity}"),
            )
        )

    seen_sample_ids: Set[str] = set()
    for image in images:
        if image.sample_id in seen_sample_ids:
            raise BfwDatasetError(f"Duplicate BFW sample identifier for {image.image_path.name}.")
        seen_sample_ids.add(image.sample_id)

    return BfwDataset(
        images=images,
        metadata_sha256=sha256_of_text(text),
        metadata_filename=metadata_path.name,
    )


def bfw_dataset_provenance(dataset: BfwDataset) -> Dict[str, Any]:
    """Public provenance for BFW. Contains no path, no filename that embeds an
    identity and no demographic record joined to a name."""
    # Aggregate shape only: how many photographs each person has, and how many
    # people fall in each subgroup. No name, path or per-person record is
    # included, because this block is published.
    grouped = dataset.by_identity()
    per_identity_counts = sorted(len(v) for v in grouped.values())
    subgroup_of = dataset.subgroup_of_identity()
    subgroup_counts = {
        subgroup: sum(1 for s in subgroup_of.values() if s == subgroup)
        for subgroup in BFW_SUBGROUPS
    }
    return {
        "dataset_name": "BFW",
        "protocol_version": BFW_PROTOCOL_VERSION,
        "protocol_provenance": (
            "Project-defined open-set protocol constructed from the official BFW data. "
            "BFW publishes verification and bias-analysis protocols; it does not publish "
            "an open-set identification protocol, and none is implied here."
        ),
        "metadata_filename": dataset.metadata_filename,
        "metadata_sha256": dataset.metadata_sha256,
        "evaluated_image_set_sha256": sha256_of_text(
            "\n".join(sorted(i.sample_id for i in dataset.images))
        ),
        "total_identities": len(grouped),
        "total_images": len(dataset.images),
        "images_per_identity_min": per_identity_counts[0] if per_identity_counts else 0,
        "images_per_identity_max": per_identity_counts[-1] if per_identity_counts else 0,
        "images_per_identity_median": (
            statistics.median(per_identity_counts) if per_identity_counts else float("nan")
        ),
        "subgroup_identity_counts": subgroup_counts,
    }


# =============================================================================
# 18. Identity-disjoint open-set protocol
# =============================================================================
#
# Development and test identities are completely disjoint, so a threshold
# selected on one cannot have seen the other. Within each partition, identities
# split again into mated (enrolled in the gallery, with held-back probes) and
# non-mated (never enrolled, representing a legitimate new registration).
#
# Every division is stratified by the eight subgroups and driven only by the
# fixed research seed and the dataset itself — never by the secret identifier
# key, so the partition stays reproducible by someone who does not hold it.

OPEN_SET_ROLES = ("gallery_enrolment", "mated_probe", "non_mated_probe")

# A test evaluation is refused unless the policy carries exactly this status.
OPEN_SET_STATUS_DEVELOPMENT = "open_set_development"
OPEN_SET_STATUS_FROZEN = "open_set_frozen"
OPEN_SET_STATUS_TESTED = "open_set_tested"
OPEN_SET_STATUSES = (
    OPEN_SET_STATUS_DEVELOPMENT,
    OPEN_SET_STATUS_FROZEN,
    OPEN_SET_STATUS_TESTED,
)

# Enrolment sizing for the two methods compared in Experiment 6.
SINGLE_IMAGE_ENROLMENT = 1
MULTI_IMAGE_ENROLMENT = 3
MULTI_IMAGE_MINIMUM_ENROLMENT = 2

# Several probes are held back per identity rather than one. With one probe each,
# a partition of 200 non-mated identities yields 200 searches, so the smallest
# measurable FPIR is 1/200 = 0.5% and the 0.3% primary target could not be
# resolved at all. Drawing more probes per identity buys the resolution; the
# resulting within-identity correlation is precisely what the cluster bootstrap
# in section 22 accounts for by resampling identities rather than images.
MATED_PROBES_PER_IDENTITY = 5
NON_MATED_PROBES_PER_IDENTITY = 15

# An identity needs enough images to enrol the maximum template and still hold
# back its probes, otherwise the two methods would not see the same identities.
# Derived inside build_open_set_protocol from its arguments rather than fixed
# here, so that overriding the probe counts cannot leave a stale threshold
# behind.


class OpenSetProtocolError(RuntimeError):
    """Raised when the open-set partition cannot be built or is inconsistent."""


@dataclass(frozen=True)
class OpenSetEntry:
    sample_id: str
    identity_hash: str
    identity: str  # private
    subgroup: str
    image_path: Path  # private
    role: str
    partition: str  # "development" | "test"


@dataclass(frozen=True)
class OpenSetProtocol:
    entries: List[OpenSetEntry]
    seed: int
    provenance: Dict[str, Any]

    def partition(self, name: str) -> List[OpenSetEntry]:
        return [e for e in self.entries if e.partition == name]

    def identities(self, partition: str, role: str) -> Set[str]:
        return {e.identity for e in self.entries if e.partition == partition and e.role == role}


def _split_stratified_identities(
    identities_by_subgroup: Dict[str, List[str]], *, seed: int
) -> Tuple[List[str], List[str]]:
    """Halve each subgroup's identities into development and test.

    Deterministic rule for an odd count: the extra identity goes to
    development, so the held-out test partition is never the larger of the two
    and a tie can never be resolved by looking at test data."""
    development: List[str] = []
    test: List[str] = []
    for subgroup in sorted(identities_by_subgroup):
        members = sorted(identities_by_subgroup[subgroup])
        # Seeded per subgroup so adding a subgroup cannot reshuffle the others.
        rng = random.Random(f"{seed}:{subgroup}")
        rng.shuffle(members)
        midpoint = (len(members) + 1) // 2  # odd -> development takes the extra
        development.extend(members[:midpoint])
        test.extend(members[midpoint:])
    return sorted(development), sorted(test)


def _split_mated_and_non_mated(
    identities: Sequence[str],
    subgroup_of: Mapping[str, str],
    *,
    seed: int,
    partition: str,
) -> Tuple[List[str], List[str]]:
    """Divide a partition's identities into mated and non-mated, stratified by
    subgroup, with the extra identity of an odd subgroup going to mated."""
    # Group the identities by subgroup first, so the split below is stratified
    # and each subgroup contributes proportionally to both roles.
    by_subgroup: Dict[str, List[str]] = {}
    for identity in identities:
        by_subgroup.setdefault(subgroup_of[identity], []).append(identity)

    mated: List[str] = []
    non_mated: List[str] = []
    # Subgroups are visited in sorted order, and their members sorted before
    # shuffling, so the split depends only on the seed and never on the order
    # the filesystem happened to return.
    for subgroup in sorted(by_subgroup):
        members = sorted(by_subgroup[subgroup])
        # Seeded per subgroup, so adding or removing one subgroup cannot
        # reshuffle the identities assigned within any other.
        rng = random.Random(f"{seed}:{partition}:{subgroup}")
        rng.shuffle(members)
        # An odd count sends the extra identity to the mated side, a fixed rule
        # that keeps the split reproducible rather than resolved by chance.
        midpoint = (len(members) + 1) // 2
        mated.extend(members[:midpoint])
        non_mated.extend(members[midpoint:])
    return sorted(mated), sorted(non_mated)


def build_open_set_protocol(
    dataset: BfwDataset,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    mated_probes_per_identity: int = MATED_PROBES_PER_IDENTITY,
    non_mated_probes_per_identity: int = NON_MATED_PROBES_PER_IDENTITY,
) -> OpenSetProtocol:
    """Construct the deterministic, identity-disjoint open-set partition."""
    grouped = dataset.by_identity()
    subgroup_of = dataset.subgroup_of_identity()

    # Only identities with enough images can be mated under both methods; the
    # rest remain eligible as non-mated probes, which need a single image.
    minimum_images = MULTI_IMAGE_ENROLMENT + mated_probes_per_identity
    identities_by_subgroup: Dict[str, List[str]] = {}
    for identity in grouped:
        identities_by_subgroup.setdefault(subgroup_of[identity], []).append(identity)

    development_ids, test_ids = _split_stratified_identities(identities_by_subgroup, seed=seed)
    if not development_ids or not test_ids:
        raise OpenSetProtocolError(
            "BFW yielded too few identities to form disjoint development and test partitions."
        )

    entries: List[OpenSetEntry] = []
    for partition, partition_ids in (("development", development_ids), ("test", test_ids)):
        eligible_mated = [i for i in partition_ids if len(grouped[i]) >= minimum_images]
        remainder = [i for i in partition_ids if i not in set(eligible_mated)]
        mated, spare = _split_mated_and_non_mated(
            eligible_mated, subgroup_of, seed=seed, partition=partition
        )
        non_mated = sorted(set(spare) | set(remainder))
        if not mated or not non_mated:
            raise OpenSetProtocolError(
                f"The {partition} partition has no {'mated' if not mated else 'non-mated'} "
                f"identities; the open-set protocol needs both."
            )

        for identity in mated:
            images = grouped[identity]
            # Deterministic ordering already applied by by_identity(); the first
            # N enrol and the next is held back as the mated probe. Enrolment
            # images are chosen before any scoring, never by similarity outcome.
            for image in images[:MULTI_IMAGE_ENROLMENT]:
                entries.append(
                    OpenSetEntry(
                        image.sample_id, image.identity_hash, identity, image.subgroup,
                        image.image_path, "gallery_enrolment", partition,
                    )
                )
            # Probes are taken from the images immediately after the enrolment
            # block, so no image can hold both roles.
            probe_slice = images[
                MULTI_IMAGE_ENROLMENT : MULTI_IMAGE_ENROLMENT + mated_probes_per_identity
            ]
            for probe in probe_slice:
                entries.append(
                    OpenSetEntry(
                        probe.sample_id, probe.identity_hash, identity, probe.subgroup,
                        probe.image_path, "mated_probe", partition,
                    )
                )

        for identity in non_mated:
            # A non-mated identity enrols nothing, so every one of its images is
            # available as a probe representing a legitimate new registration.
            for probe in grouped[identity][:non_mated_probes_per_identity]:
                entries.append(
                    OpenSetEntry(
                        probe.sample_id, probe.identity_hash, identity, probe.subgroup,
                        probe.image_path, "non_mated_probe", partition,
                    )
                )

    _assert_protocol_invariants(entries)

    provenance = {
        "seed": seed,
        "protocol_version": BFW_PROTOCOL_VERSION,
        "odd_count_rule": (
            "When a subgroup holds an odd number of identities the extra identity is "
            "assigned to the development partition, and within a partition to the mated "
            "group, so the held-out test set is never enlarged by a tie."
        ),
        "development_identities": len(development_ids),
        "test_identities": len(test_ids),
    }
    return OpenSetProtocol(entries=entries, seed=seed, provenance=provenance)


def _assert_protocol_invariants(entries: Sequence[OpenSetEntry]) -> None:
    """The three properties that make the experiment meaningful, checked rather
    than assumed: no image holds two roles, no identity crosses the
    development/test boundary, and no identity is both enrolled and non-mated."""
    # First property: one photograph, one role. A photograph used both to
    # enrol a profile and to search against it would be compared with itself.
    seen_paths: Set[Path] = set()
    for entry in entries:
        if entry.image_path in seen_paths:
            raise OpenSetProtocolError(
                f"Image assigned to more than one open-set role: {entry.image_path.name}"
            )
        seen_paths.add(entry.image_path)

    # Collect which partitions and which roles each person appears under.
    partitions_of: Dict[str, Set[str]] = {}
    roles_of: Dict[str, Set[str]] = {}
    for entry in entries:
        partitions_of.setdefault(entry.identity, set()).add(entry.partition)
        roles_of.setdefault(entry.identity, set()).add(entry.role)

    # Second property: nobody appears in both partitions. A person seen while
    # the threshold was chosen would make the held-out test no longer unseen.
    crossing = sorted(i for i, p in partitions_of.items() if len(p) > 1)
    if crossing:
        raise OpenSetProtocolError(
            f"{len(crossing)} identity/identities appear in both the development and test "
            f"partitions; the open-set protocol requires them to be disjoint."
        )

    # Third property: a person is either enrolled or a stranger, never both.
    # Otherwise a "new profile" would be counted as falsely referred to a
    # gallery that legitimately contains them.
    for identity, roles in roles_of.items():
        if "non_mated_probe" in roles and roles & {"gallery_enrolment", "mated_probe"}:
            raise OpenSetProtocolError(
                "An identity is both enrolled in the gallery and used as a non-mated probe; "
                "a non-mated search must have no gallery reference."
            )


def open_set_protocol_summary(
    protocol: OpenSetProtocol,
    *,
    dataset: Optional[BfwDataset] = None,
    detector: Any = None,
    embedder: Any = None,
    description: Optional[PipelineDescription] = None,
) -> Dict[str, Any]:
    """Public manifest summary: opaque identifiers and counts only.

    Software, model and dataset provenance are embedded alongside them so that
    this artefact carries the same record as every other published JSON."""

    # Counts only, never identities or paths. This summary is published, so it
    # describes the shape of the protocol without disclosing who is in it.
    def counts(partition: str) -> Dict[str, Any]:
        rows = protocol.partition(partition)
        by_subgroup: Dict[str, Dict[str, int]] = {
            subgroup: {role: 0 for role in OPEN_SET_ROLES} for subgroup in BFW_SUBGROUPS
        }
        # Tally how many photographs each subgroup contributes to each role,
        # which is what shows the protocol is balanced across subgroups.
        for entry in rows:
            by_subgroup[entry.subgroup][entry.role] += 1
        return {
            # Counted over unique identity hashes, because one person supplies
            # several photographs and would otherwise be counted repeatedly.
            "identities": len({e.identity_hash for e in rows}),
            "images": len(rows),
            "roles": {role: sum(1 for e in rows if e.role == role) for role in OPEN_SET_ROLES},
            "mated_identities": len({e.identity_hash for e in rows if e.role == "mated_probe"}),
            "non_mated_identities": len(
                {e.identity_hash for e in rows if e.role == "non_mated_probe"}
            ),
            "by_subgroup": by_subgroup,
        }

    _identity = pipeline_identity_fields(detector, embedder, description)
    return {
        "artifact_type": "bfw_open_set_protocol_summary",
        "schema_version": SCHEMA_VERSION,
        "opaque_id_version": OPAQUE_ID_VERSION,
        "seed": protocol.seed,
        "development": counts("development"),
        "test": counts("test"),
        **protocol.provenance,
        "public_manifest_sha256": sha256_of_text(
            "\n".join(sorted(f"{e.partition}:{e.role}:{e.sample_id}" for e in protocol.entries))
        ),
        "pipeline_name": _identity["model_version"],
        **_identity,
        "software_environment": software_environment_report(),
        "dataset_provenance": (
            bfw_dataset_provenance(dataset) if dataset is not None else None
        ),
        "policy_note": POLICY_NOTE,
    }


def write_open_set_private_manifest(protocol: OpenSetProtocol, path: Path) -> None:
    """Private manifest with real paths. Written under results/raw, which is
    git-ignored, and never published."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # This is the one record that carries real image paths, so that a run can be
    # traced back to the photographs it used. The published manifest alongside
    # it carries the opaque identifiers only. The seed is stored with the
    # entries, because the split cannot be reproduced without it.
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": protocol.seed,
        "entries": [
            {
                "sample_id": e.sample_id,
                "identity_hash": e.identity_hash,
                "subgroup": e.subgroup,
                "role": e.role,
                "partition": e.partition,
                "image_path": str(e.image_path),
            }
            for e in protocol.entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# =============================================================================
# 19. Open-set enrolment methods
# =============================================================================
#
# Two methods evaluated over exactly the same identity partitions, so any
# difference between them is attributable to the enrolment representation and
# the threshold, not to a different sample of people.
#
#   Method A (control)  single_image_pairwise_threshold
#       One gallery image per identity, searched with the LFW-frozen 1:1
#       threshold. This is the threshold-transfer control: it measures what
#       happens when a threshold calibrated for comparing exactly two images is
#       reused for 1:N search.
#
#   Method B (proposed) three_image_open_set_calibrated
#       Up to three enrolment images averaged into one L2-normalised template,
#       searched with a threshold calibrated on the BFW development partition
#       at a target false-positive identification rate.

METHOD_A = "single_image_pairwise_threshold"
METHOD_B = "three_image_open_set_calibrated"
OPEN_SET_METHODS = (METHOD_A, METHOD_B)

INSUFFICIENT_GALLERY_IMAGES = "insufficient_gallery_images"


@dataclass(frozen=True)
class EnrolmentOutcome:
    """Outcome of enrolling one identity, retained whether or not it succeeded."""

    identity_hash: str
    subgroup: str
    enrolled: bool
    attempted_images: int
    embedded_images: int
    failure_code: Optional[str]


@dataclass(frozen=True)
class EnrolledIdentity:
    identity_hash: str
    subgroup: str
    template: np.ndarray


@dataclass(frozen=True)
class OpenSetSearchResult:
    sample_id: str
    identity_hash: str
    subgroup: str
    role: str
    failure_code: Optional[str] = None
    top_similarity: Optional[float] = None
    top_identity_hash: Optional[str] = None
    top2_similarity: Optional[float] = None
    correct_rank: Optional[int] = None
    correct_similarity: Optional[float] = None
    highest_impostor_similarity: Optional[float] = None
    top1_time_seconds: Optional[float] = None
    top5_time_seconds: Optional[float] = None
    # Inputs to the review classifier of section 26. Every one is computable at
    # inference time from the search alone; none reveals whether the probe's
    # identity is actually enrolled, which would leak the label.
    top5_similarity_mean: Optional[float] = None
    top5_similarity_stdev: Optional[float] = None
    top1_gallery_image_count: Optional[int] = None
    gallery_size: Optional[int] = None
    probe_detection_confidence: Optional[float] = None
    probe_face_area_ratio: Optional[float] = None
    # Supplementary direct profile-consistency control (section 27). One
    # deterministically assigned wrong enrolled template, compared with this
    # photograph alone. Recorded for non-mated probes only, and deliberately
    # excluded from the review classifier's feature list: it is an evaluation
    # measurement, not something a deployment would compute at decision time.
    assigned_wrong_identity_hash: Optional[str] = None
    assigned_wrong_template_similarity: Optional[float] = None


@dataclass(frozen=True)
class OpenSetRunResult:
    method: str
    partition: str
    enrolment_outcomes: List[EnrolmentOutcome]
    search_results: List[OpenSetSearchResult]
    gallery_size: int
    comparisons_per_probe: int
    # Per-image stage timings; model loading is excluded and reported separately.
    stage_times_seconds: Dict[str, List[float]] = field(default_factory=dict)


# One unit-length template per identity, formed by averaging and renormalising.
# This changes score distributions; it does not necessarily increase similarity
# to every other identity. Its benefit must be measured at calibrated thresholds.
def build_identity_template(
    embeddings: Sequence[np.ndarray],
) -> np.ndarray:
    """Average already-L2-normalised embeddings and re-normalise the mean.

    Re-normalisation matters: the arithmetic mean of unit vectors is not itself
    a unit vector, and cosine similarity against an un-normalised template would
    silently rescale every score for that identity."""
    if not embeddings:
        raise SimilarityError("Cannot build a template from zero embeddings.")
    # Stack the profile's photographs into rows, average them into a single
    # representation of that person, then rescale the average to unit length.
    stacked = np.vstack([l2_normalize(np.asarray(e, dtype=np.float64)) for e in embeddings])
    return l2_normalize(stacked.mean(axis=0))


def _embed_open_set_entry(
    entry: OpenSetEntry, detector: FaceDetector, embedder: FaceEmbedder
) -> Tuple[Optional[np.ndarray], Optional[str], Dict[str, float]]:
    """Embed one image and return the detector-derived quality signals with it.

    YuNet emits a fifteen-element row: bounding box in the first four columns
    and the detection score in the last. Both are captured here so the review
    classifier can use image quality without a second detection pass."""
    # Each stage is timed separately so a pipeline comparison can attribute
    # cost to loading, detection or embedding rather than to the whole call.
    started = time.perf_counter()
    try:
        loaded = load_image_bgr(entry.image_path)
        load_done = time.perf_counter()
        face_row = detector.detect_single_face(loaded.bgr)
        detect_done = time.perf_counter()
        height, width = loaded.bgr.shape[:2]
        metadata: Dict[str, float] = {
            "image_load_seconds": load_done - started,
            "detection_seconds": detect_done - load_done,
        }
        if face_row is not None and len(face_row) >= 15 and height and width:
            metadata["probe_detection_confidence"] = float(face_row[14])
            metadata["probe_face_area_ratio"] = (
                float(face_row[2] * face_row[3]) / float(height * width)
            )
        embedding = l2_normalize(embedder.embed(loaded.bgr, face_row))
        embed_done = time.perf_counter()
        metadata["embedding_seconds"] = embed_done - detect_done
        metadata["complete_pipeline_seconds"] = embed_done - started
        return embedding, None, metadata
    except FaceCountError as exc:
        return None, ("zero_faces" if exc.face_count == 0 else "multiple_faces"), {}
    except (ImageLoadError, SimilarityError) as exc:
        return None, f"image_error:{exc}", {}


def _call_embed(embed: Callable[..., Any], entry, detector, embedder):
    """Normalise an embedding callable to (embedding, failure, metadata).

    Test stubs supply the older two-value form, so both are accepted rather
    than forcing every caller to carry metadata it does not produce."""
    result = embed(entry, detector, embedder)
    if len(result) == 3:
        return result
    embedding, failure = result
    return embedding, failure, {}


# --- Probe search and the supplementary consistency control ------------------


def assigned_wrong_template(
    sample_id: str, enrolled: Sequence[EnrolledIdentity], *, seed: int
) -> Optional[EnrolledIdentity]:
    """Pick one wrong enrolled template for a control photograph.

    Deterministic in the protocol seed and the opaque sample identifier, so the
    same photograph always draws the same wrong profile and the control can be
    reproduced without storing the assignment. The gallery is ordered by its
    opaque identity hashes, never by a private name."""
    if not enrolled:
        return None
    # Order by opaque hash so the choice never depends on enrolment order.
    ordered = sorted(enrolled, key=lambda identity: identity.identity_hash)
    # Turn the seed and the sample identifier into one large number, then use
    # the remainder to pick a position in the list. The same photograph always
    # lands on the same profile, so the control repeats exactly.
    offset = int(sha256_of_text(f"{seed}:wrong-template:{sample_id}"), 16)
    return ordered[offset % len(ordered)]


# --- Gallery enrolment, then probe search against the enrolled gallery --------


def run_open_set_method(
    protocol: OpenSetProtocol,
    *,
    partition: str,
    method: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    embed_fn: Optional[Callable[..., Tuple[Optional[np.ndarray], Optional[str]]]] = None,
) -> OpenSetRunResult:
    """Enrol the gallery and search every probe. No threshold is applied here:
    raw scores and ranks are retained so one search can be scored at several
    operating points without re-running the pipeline."""
    if method not in OPEN_SET_METHODS:
        raise OpenSetProtocolError(f"Unknown open-set method {method!r}.")
    embed = embed_fn or _embed_open_set_entry

    rows = protocol.partition(partition)
    enrolment_images: Dict[str, List[OpenSetEntry]] = {}
    subgroup_of_hash: Dict[str, str] = {}
    for entry in rows:
        subgroup_of_hash[entry.identity_hash] = entry.subgroup
        if entry.role == "gallery_enrolment":
            enrolment_images.setdefault(entry.identity_hash, []).append(entry)

    # Method A enrols one image per identity and is the control; method B enrols
    # three and is the proposed change. The minimum is what an identity must
    # reach to be enrolled at all, so an identity short of it becomes a recorded
    # coverage failure rather than a silently smaller template.
    take = SINGLE_IMAGE_ENROLMENT if method == METHOD_A else MULTI_IMAGE_ENROLMENT
    minimum = SINGLE_IMAGE_ENROLMENT if method == METHOD_A else MULTI_IMAGE_MINIMUM_ENROLMENT

    outcomes: List[EnrolmentOutcome] = []
    enrolled: List[EnrolledIdentity] = []
    template_image_counts: Dict[str, int] = {}
    stage_times: Dict[str, List[float]] = {
        "image_load_seconds": [], "detection_seconds": [],
        "embedding_seconds": [], "complete_pipeline_seconds": [],
    }

    def _record_stages(meta: Mapping[str, float]) -> None:
        for stage in stage_times:
            value = meta.get(stage)
            if isinstance(value, (int, float)):
                stage_times[stage].append(float(value))
    for identity_hash in sorted(enrolment_images):
        # Ordered by filename, never by sample_id. sample_id is an HMAC under the
        # secret key, so ordering by it would make the choice of enrolment image
        # key-dependent and the published metrics irreproducible by anyone
        # holding a different key.
        candidates = sorted(enrolment_images[identity_hash], key=lambda e: e.image_path.name)[:take]
        embeddings: List[np.ndarray] = []
        for entry in candidates:
            embedding, _failure, meta = _call_embed(embed, entry, detector, embedder)
            _record_stages(meta)
            if embedding is not None:
                embeddings.append(embedding)
        subgroup = subgroup_of_hash[identity_hash]
        if len(embeddings) < minimum:
            # Recorded, never silently removed: an identity that cannot meet the
            # minimum is a coverage failure of the method under test.
            outcomes.append(
                EnrolmentOutcome(
                    identity_hash, subgroup, False, len(candidates), len(embeddings),
                    INSUFFICIENT_GALLERY_IMAGES,
                )
            )
            continue
        outcomes.append(
            EnrolmentOutcome(identity_hash, subgroup, True, len(candidates), len(embeddings), None)
        )
        enrolled.append(
            EnrolledIdentity(identity_hash, subgroup, build_identity_template(embeddings))
        )
        template_image_counts[identity_hash] = len(embeddings)

    if not enrolled:
        raise OpenSetProtocolError(
            f"No identity could be enrolled for method {method!r} in the {partition} partition."
        )
    enrolled_hashes = {e.identity_hash for e in enrolled}

    results: List[OpenSetSearchResult] = []
    for entry in rows:
        if entry.role not in ("mated_probe", "non_mated_probe"):
            continue
        # A mated probe whose identity failed enrolment has nothing to match
        # against. That is an unresolved protocol outcome, not a miss, so it is
        # recorded separately and kept out of the conditional denominators.
        if entry.role == "mated_probe" and entry.identity_hash not in enrolled_hashes:
            results.append(
                OpenSetSearchResult(
                    entry.sample_id, entry.identity_hash, entry.subgroup, entry.role,
                    failure_code=GALLERY_REFERENCE_UNAVAILABLE,
                )
            )
            continue

        probe_embedding, failure, probe_meta = _call_embed(embed, entry, detector, embedder)
        _record_stages(probe_meta)
        if probe_embedding is None:
            results.append(
                OpenSetSearchResult(
                    entry.sample_id, entry.identity_hash, entry.subgroup, entry.role,
                    failure_code=failure,
                )
            )
            continue

        # Two separate measurements of what a deployment would actually pay.
        # Retrieving the single best candidate costs a running maximum;
        # retrieving five costs a partial selection over the gallery, which is
        # the real cost of a five-candidate review queue.
        start = time.perf_counter()
        similarities = [(c, cosine_similarity(probe_embedding, c.template)) for c in enrolled]
        _best = max(similarities, key=lambda item: item[1])
        top1_elapsed = time.perf_counter() - start

        start_top5 = time.perf_counter()
        _five = heapq.nlargest(5, similarities, key=lambda item: item[1])
        top5_elapsed = top1_elapsed + (time.perf_counter() - start_top5)

        # The full ordering is an evaluation artefact needed to recover the
        # mate's rank; it is deliberately timed out of both figures above,
        # because no deployment would sort the entire gallery per probe.
        scored = sorted(
            similarities,
            key=lambda item: (-item[1], item[0].identity_hash),
        )

        top_candidate, top_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else None

        # The mate's rank is needed for TPIR and CMC, and the highest impostor
        # score for the margin feature. Both are meaningful only for a mated
        # probe: a non-mated probe has no correct identity in the gallery.
        correct_rank: Optional[int] = None
        correct_score: Optional[float] = None
        highest_impostor: Optional[float] = None
        if entry.role == "mated_probe":
            for position, (candidate, score) in enumerate(scored, start=1):
                if candidate.identity_hash == entry.identity_hash:
                    correct_rank, correct_score = position, score
                    break
            impostors = [s for c, s in scored if c.identity_hash != entry.identity_hash]
            highest_impostor = max(impostors) if impostors else None

        # Supplementary direct control: compare this photograph with exactly one
        # deterministically assigned wrong enrolled template. A non-mated probe
        # is enrolled nowhere, so every gallery identity is a wrong one and no
        # exclusion is needed. Assignment is by the protocol seed and the opaque
        # identifier alone, so it is reproducible and independent of scoring.
        wrong_hash: Optional[str] = None
        wrong_score: Optional[float] = None
        if entry.role == "non_mated_probe":
            wrong = assigned_wrong_template(entry.sample_id, enrolled, seed=protocol.seed)
            if wrong is not None:
                wrong_hash = wrong.identity_hash
                wrong_score = next(
                    score for candidate, score in similarities
                    if candidate.identity_hash == wrong_hash
                )

        top5_scores = [score for _candidate, score in scored[:5]]
        results.append(
            OpenSetSearchResult(
                entry.sample_id, entry.identity_hash, entry.subgroup, entry.role,
                None, top_score, top_candidate.identity_hash, second_score,
                correct_rank, correct_score, highest_impostor, top1_elapsed, top5_elapsed,
                statistics.fmean(top5_scores) if top5_scores else None,
                statistics.stdev(top5_scores) if len(top5_scores) > 1 else 0.0,
                template_image_counts.get(top_candidate.identity_hash),
                len(enrolled),
                probe_meta.get("probe_detection_confidence"),
                probe_meta.get("probe_face_area_ratio"),
                wrong_hash, wrong_score,
            )
        )

    return OpenSetRunResult(
        method=method,
        partition=partition,
        enrolment_outcomes=outcomes,
        search_results=results,
        gallery_size=len(enrolled),
        comparisons_per_probe=len(enrolled),
        stage_times_seconds=stage_times,
    )


# =============================================================================
# 20. Open-set metrics (FPIR, FNIR, TPIR)
# =============================================================================
#
# These are identification metrics and are not interchangeable with the
# pairwise rates in section 10:
#
#   FPIR  a *search* against the whole gallery returns at least one candidate
#         above threshold when the person is not enrolled. FMR is a single
#         comparison between two images. One non-mated search performs as many
#         comparisons as there are enrolled identities, which is exactly why a
#         1:1 threshold transfers so badly.
#
#   FNIR  a mated search fails to place the correct identity within rank k
#         above threshold. FNMR is a single genuine comparison falling below
#         threshold, with no competing candidates and no ranking involved.
#
# TPIR (also written DIR) is 1 - FNIR at the same rank.
##############
# Title: Face Recognition Technology Evaluation (FRTE) 1:N Identification
# Author: National Institute of Standards and Technology (NIST)
# Date: 2025
# Availability: https://pages.nist.gov/frvt/html/frvt1N.html
##############
##############
# Title: ISO/IEC 19795-1:2021 Biometric performance testing and reporting,
#        Part 1: Principles and framework
# Author: International Organization for Standardization
# Date: 2021
# Availability: https://www.iso.org/standard/73515.html
##############

# NIST commonly reports FNIR at an FPIR of 0.003; that is taken as the primary
# operating point here, with a stricter and a looser point either side.
FPIR_TARGETS = (0.001, 0.003, 0.01)
PRIMARY_FPIR_TARGET = 0.003


def _scored(results: Sequence[OpenSetSearchResult], role: str) -> List[OpenSetSearchResult]:
    return [r for r in results if r.role == role and r.failure_code is None]


# Conditional rates: every denominator here counts probes that were actually
# scored. Extraction failures are excluded and reported separately as coverage,
# because a failure is neither a correct nor an incorrect decision. The
# end-to-end rates elsewhere divide by all intended probes instead.
def open_set_rates_at_threshold(
    results: Sequence[OpenSetSearchResult], threshold: float
) -> Dict[str, float]:
    """FPIR, FNIR and TPIR at one operating threshold."""
    # Two groups of test photographs. Non-mated probes belong to people who are
    # not enrolled, so any match is a false referral. Mated probes belong to
    # people who are enrolled, so the correct profile ought to be found.
    non_mated = _scored(results, "non_mated_probe")
    mated = _scored(results, "mated_probe")

    # FPIR: a person absent from the gallery whose best match still passed the
    # threshold. Each one is a new profile sent for review unnecessarily.
    false_positives = sum(
        1 for r in non_mated if r.top_similarity is not None and r.top_similarity >= threshold
    )
    fpir = false_positives / len(non_mated) if non_mated else float("nan")

    # TPIR: the correct profile must appear within the first "rank" candidates
    # *and* score above the threshold. Rank alone would count a case the
    # reviewer never sees; the threshold alone would ignore who was matched.
    def found_within(rank: int) -> int:
        return sum(
            1
            for r in mated
            if r.correct_rank is not None
            and r.correct_rank <= rank
            and r.correct_similarity is not None
            and r.correct_similarity >= threshold
        )

    tpir1 = found_within(1) / len(mated) if mated else float("nan")
    tpir5 = found_within(5) / len(mated) if mated else float("nan")
    # CMC ignores the threshold and asks only about ranking. Comparing it with
    # TPIR shows how much detection is lost to the threshold rather than to the
    # model failing to rank the right person highly.
    cmc1 = (
        sum(1 for r in mated if r.correct_rank == 1) / len(mated) if mated else float("nan")
    )
    cmc5 = (
        sum(1 for r in mated if r.correct_rank is not None and r.correct_rank <= 5) / len(mated)
        if mated
        else float("nan")
    )
    return {
        "threshold": threshold,
        "fpir": fpir,
        # FNIR is simply the share not found, so it always complements TPIR.
        "fnir_rank1": 1.0 - tpir1 if tpir1 == tpir1 else float("nan"),
        "fnir_rank5": 1.0 - tpir5 if tpir5 == tpir5 else float("nan"),
        "tpir_rank1": tpir1,
        "tpir_rank5": tpir5,
        "cmc_rank1": cmc1,
        "cmc_rank5": cmc5,
        # The same FPIR expressed as a workload: how many unnecessary reviews a
        # moderator would face per thousand new profiles.
        "false_reviews_per_1000_non_mated": fpir * 1000.0 if fpir == fpir else float("nan"),
        "scored_non_mated_probes": len(non_mated),
        "scored_mated_probes": len(mated),
    }


# One canonical primary-pipeline run underpins Experiments 6, 7 and 8, the
# implementation layers and every subgroup breakdown, so the same method can
# never report two different scored counts.
#
# The cache holds derived face-comparison scores and decisions. Those are
# privacy-sensitive: they describe how a face model responded to identifiable
# people. It holds no raw photograph, face embedding or enrolled template, and
# it stays local, access-restricted and excluded from Git.
###############################################################################
# Canonical run cache
###############################################################################
#
# Experiments 6, 7 and 8 must report the same figures for the same method. Each
# scoring the primary pipeline separately would let them drift, because OpenCV
# detection is not bit-reproducible across processes. One run per partition is
# therefore scored, cached and reused. Original project logic; no external
# implementation is adapted.

CANONICAL_RUN_CACHE = RAW_ROOT / "canonical_primary_run.json"
CANONICAL_CACHE_SCHEMA_VERSION = 3

# Any change to extraction, template construction, search, ranking or failure
# accounting changes what a cached run means, without necessarily changing a
# model digest or a library version. Incrementing this revision retires every
# cache produced by the previous logic.
CANONICAL_PIPELINE_REVISION = "opencv-yunet-sface-open-set-v2"

CANONICAL_CACHE_PRIVACY_NOTE = (
    "The cache contains privacy-sensitive derived face-comparison scores and decisions, "
    "but no raw photographs, face embeddings or enrolled templates. It remains local, "
    "access-restricted and excluded from Git."
)

# Owner-only, because the cache holds derived scores about identifiable people.
CANONICAL_CACHE_DIR_MODE = 0o700
CANONICAL_CACHE_FILE_MODE = 0o600

# Every non-timing field of a search result contributes to the outcome digest.
# Timing is excluded because it varies between runs by design.
_DIGEST_EXCLUDED_FIELDS = frozenset({
    "top1_time_seconds", "top5_time_seconds",
})

# A cache is only trusted when every one of these is present with the right
# type. A missing or retyped field means the file is not what this programme
# wrote, so it is rebuilt rather than parsed defensively.
_CACHE_REQUIRED_FIELDS: Dict[str, Any] = {
    "cache_schema_version": int,
    "canonical_run_digest": str,
    "canonical_context": dict,
    "canonical_context_sha256": str,
    "method": str,
    "partition": str,
    "gallery_size": int,
    "comparisons_per_probe": int,
    "enrolment_outcomes": list,
    "search_results": list,
}


def canonical_cache_path(partition: str, base: Path = CANONICAL_RUN_CACHE) -> Path:
    """One cache per partition. The development partition matters as much as the
    held-out one: the classifier is fitted and calibrated on it, so recomputing
    it would move the frozen threshold between runs."""
    base = Path(base)
    return base.with_name(f"{base.stem}_{partition}{base.suffix}")


def canonical_run_context(
    protocol: OpenSetProtocol,
    *,
    partition: str,
    dataset: Optional[BfwDataset] = None,
    detector: Any = None,
    embedder: Any = None,
) -> Dict[str, Any]:
    """Every input capable of changing a cached run's result.

    Matching probe identifiers is not sufficient: the same images scored with a
    different model, detector setting or OpenCV build produce different
    outcomes, so the whole context is compared before a cache is trusted."""
    import cv2

    rows = protocol.partition(partition)
    provenance = bfw_dataset_provenance(dataset) if dataset is not None else {}
    # Prefer the detector's own settings over the module constants, so a
    # detector constructed with non-default settings cannot reuse a cache built
    # under the defaults.
    settings = getattr(detector, "settings", None) or DetectorSettings()
    return {
        "cache_schema_version": CANONICAL_CACHE_SCHEMA_VERSION,
        "pipeline_revision": CANONICAL_PIPELINE_REVISION,
        "partition": partition,
        "dataset_metadata_sha256": provenance.get("metadata_sha256"),
        "evaluated_image_set_sha256": provenance.get("evaluated_image_set_sha256"),
        "protocol_version": BFW_PROTOCOL_VERSION,
        # The published manifest digest covers sample, role and partition only.
        # Kept unchanged because artefacts already reference it.
        "public_manifest_digest": public_manifest_digest(protocol),
        # Cache invalidation needs more: moving a sample to another identity or
        # subgroup changes what a cached score means while leaving the sample
        # identifier, role and partition untouched.
        "private_cache_protocol_context_digest": private_protocol_context_digest(protocol),
        "model_filenames": [YUNET_FILENAME, SFACE_FILENAME],
        "model_sha256": {
            "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
            "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
        },
        "preprocessing_revision": PREPROCESSING_REVISION,
        # YuNet is re-sized to each image's own dimensions before detection, so
        # the constructor size is only the initial value, never the size the
        # detector actually runs at.
        "detector_initial_input_size": [320, 320],
        "detector_input_strategy": "native_image_dimensions_per_image",
        "detector_score_threshold": settings.score_threshold,
        "detector_nms_threshold": settings.nms_threshold,
        "detector_top_k": settings.top_k,
        "exactly_one_face_required": True,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "gallery_images_per_identity": MULTI_IMAGE_ENROLMENT,
        "minimum_valid_gallery_images": MULTI_IMAGE_MINIMUM_ENROLMENT,
        "seed": protocol.seed,
        "python_version": sys.version.split()[0],
        "opencv_version": str(getattr(cv2, "__version__", "")),
        "numpy_version": _package_version("numpy"),
        "pillow_version": _package_version("Pillow"),
        "platform": platform.platform(),
        # Complete role assignments, not just probe identifiers.
        "gallery_enrolment_samples": sorted(
            e.sample_id for e in rows if e.role == "gallery_enrolment"
        ),
        "mated_probe_samples": sorted(e.sample_id for e in rows if e.role == "mated_probe"),
        "non_mated_probe_samples": sorted(
            e.sample_id for e in rows if e.role == "non_mated_probe"
        ),
        "intended_gallery_identities": len(
            {e.identity_hash for e in rows if e.role == "gallery_enrolment"}
        ),
        "intended_mated_probes": sum(1 for e in rows if e.role == "mated_probe"),
        "intended_non_mated_probes": sum(1 for e in rows if e.role == "non_mated_probe"),
    }


def public_manifest_digest(protocol: OpenSetProtocol) -> str:
    """Digest over the published protocol manifest: sample, role and partition."""
    return sha256_of_text(
        "\n".join(sorted(f"{e.partition}:{e.role}:{e.sample_id}" for e in protocol.entries))
    )


def private_protocol_context_digest(protocol: OpenSetProtocol) -> str:
    """Digest over the complete opaque assignment of every protocol entry.

    Covers the identity and subgroup as well as the sample, role and partition,
    together with the grouping of enrolment images under each opaque identity.
    Reassigning a sample to a different identity or subgroup therefore
    invalidates a cache even though the published manifest is unchanged.

    Only opaque identifiers enter the digest. No private identity name,
    absolute path or raw filename is hashed or stored."""
    entries = sorted(
        f"{e.partition}:{e.role}:{e.sample_id}:{e.identity_hash}:{e.subgroup}"
        for e in protocol.entries
    )
    # The enrolment grouping is part of the context in its own right: the same
    # images distributed differently across identities build different
    # templates and so produce different scores.
    grouping: Dict[str, List[str]] = {}
    for entry in protocol.entries:
        if entry.role == "gallery_enrolment":
            grouping.setdefault(
                f"{entry.partition}:{entry.identity_hash}", []
            ).append(entry.sample_id)
    grouped = sorted(
        f"{key}=[{','.join(sorted(samples))}]" for key, samples in grouping.items()
    )
    return sha256_of_text("\n".join(entries) + "\n--\n" + "\n".join(grouped))


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _stable_float(value: float) -> str:
    """A round-trip-exact textual form for a float.

    Decimal truncation would let two genuinely different similarities collide
    in the digest, so the outcome hash uses float.hex(), which is exact and
    distinguishes negative zero from zero. The non-finite values have no hex
    form and are named explicitly."""
    # A not-a-number value is the only one that does not equal itself, which is
    # how it is recognised here. It has no hexadecimal form, so it is named.
    if value != value:
        return "float:nan"
    if value == math.inf:
        return "float:+inf"
    if value == -math.inf:
        return "float:-inf"
    # The hexadecimal form reproduces the stored number exactly, so two
    # similarities that differ in their final digits never share a digest.
    return f"float:{value.hex()}"


def _canonical_json(payload: Any) -> str:
    """Stable serialisation: UTF-8, sorted keys, exact float representation."""
    # Walk the whole structure and rewrite it into one fixed form. Two runs
    # that produced the same results must serialise to the same text, or the
    # digest built from it would differ for no scientific reason.
    def normalise(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        # Floats go through the exact form; decimal rounding here would let two
        # different similarities collide in the digest.
        if isinstance(value, float):
            return _stable_float(value)
        # Sorting the keys removes any dependence on insertion order.
        if isinstance(value, dict):
            return {k: normalise(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [normalise(v) for v in value]
        return value

    return json.dumps(normalise(payload), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def context_digest(context: Mapping[str, Any]) -> str:
    return sha256_of_text(_canonical_json(context))


# Timing fields are excluded because runtime varies between repeated executions
# and would change the digest without any scientific outcome changing. Every
# other field of every record contributes, so an edited decision is detected.
def canonical_run_digest(run: OpenSetRunResult) -> str:
    """Digest over every non-timing outcome field.

    Records are sorted before hashing, so an equivalent run in a different
    order produces the same digest while any changed decision does not."""
    search_rows = sorted(
        (
            {
                name: getattr(result, name)
                for name in (f.name for f in fields(result))
                if name not in _DIGEST_EXCLUDED_FIELDS
            }
            for result in run.search_results
        ),
        key=lambda row: str(row.get("sample_id")),
    )
    # Enrolment outcomes are keyed by identity, not by sample: sorting them on
    # a missing sample_id would leave them in input order and make the digest
    # depend on record ordering.
    enrolment_rows = sorted(
        (asdict(outcome) for outcome in run.enrolment_outcomes),
        key=lambda row: str(row.get("identity_hash")),
    )
    return sha256_of_text(_canonical_json({
        "search_results": search_rows,
        "enrolment_outcomes": enrolment_rows,
        "gallery_size": run.gallery_size,
        "comparisons_per_probe": run.comparisons_per_probe,
        "method": run.method,
        "partition": run.partition,
    }))


def _search_result_to_row(result: OpenSetSearchResult) -> Dict[str, Any]:
    return {f.name: getattr(result, f.name) for f in fields(result)}


def save_canonical_run(
    run: OpenSetRunResult,
    path: Path = CANONICAL_RUN_CACHE,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    """Persist a run with the context that produced it."""
    # The digest fingerprints the run's outcomes. Storing it beside the records
    # lets a later load prove the file has not been edited since it was written.
    digest = canonical_run_digest(run)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _restrict_permissions(path.parent, CANONICAL_CACHE_DIR_MODE)
    payload = {
        "cache_schema_version": CANONICAL_CACHE_SCHEMA_VERSION,
        "canonical_run_digest": digest,
        "canonical_context": dict(context) if context else None,
        "canonical_context_sha256": context_digest(context) if context else None,
        "created_at": utc_now_iso(),
        "method": run.method,
        "partition": run.partition,
        "gallery_size": run.gallery_size,
        "comparisons_per_probe": run.comparisons_per_probe,
        "enrolment_outcomes": [asdict(o) for o in run.enrolment_outcomes],
        "search_results": [_search_result_to_row(r) for r in run.search_results],
        "stage_times_seconds": run.stage_times_seconds,
        "privacy_note": CANONICAL_CACHE_PRIVACY_NOTE,
        "permission_note": _cache_permission_note(),
    }
    _atomic_private_write(
        path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return digest


def _restrict_permissions(target: Path, mode: int) -> bool:
    """Owner-only permissions where the platform supports them.

    Windows does not implement POSIX mode bits meaningfully. Failing the whole
    evaluation over that would be wrong, so the limitation is recorded in the
    cache instead of raised."""
    if os.name != "posix":
        return False
    try:
        os.chmod(target, mode)
        return True
    except OSError:
        return False


def _cache_permission_note() -> str:
    if os.name == "posix":
        return (
            "Written atomically via a temporary file and os.replace(). Directory "
            "restricted to 0700 and file to 0600, so only the owner can read the "
            "derived scores."
        )
    return (
        "Written atomically via a temporary file and os.replace(). POSIX mode bits "
        "are not applied on this platform, so the cache inherits the directory's "
        "access control instead; keep it inside access-restricted storage."
    )


def _atomic_private_write(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, then os.replace().

    A partially written cache must never be observable: os.replace() is atomic
    within a filesystem, so a reader sees either the previous complete file or
    the new complete one. The temporary file is created owner-only, so the
    contents are never briefly world-readable."""
    handle, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _restrict_permissions(temporary_path, CANONICAL_CACHE_FILE_MODE)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_canonical_run(path: Path = CANONICAL_RUN_CACHE) -> Optional[OpenSetRunResult]:
    """Reload a cached run, or return None when absent."""
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _run_from_cached_payload(payload)


def _run_from_cached_payload(payload: Mapping[str, Any]) -> OpenSetRunResult:
    return OpenSetRunResult(
        method=payload["method"],
        partition=payload["partition"],
        enrolment_outcomes=[EnrolmentOutcome(**o) for o in payload["enrolment_outcomes"]],
        search_results=[OpenSetSearchResult(**r) for r in payload["search_results"]],
        gallery_size=payload["gallery_size"],
        comparisons_per_probe=payload["comparisons_per_probe"],
        stage_times_seconds=payload.get("stage_times_seconds", {}),
    )


def cached_payload_integrity_reason(payload: Mapping[str, Any]) -> Optional[str]:
    """Why a cached payload is not internally sound, or None when it is.

    Checking the expected context against the stored context is not enough: it
    proves the cache was built for this configuration, not that its contents
    are still the ones that were built. A cache whose records were edited
    afterwards would otherwise be loaded silently and republished under a
    freshly computed digest; these checks prevent that.

    Names only the category or field at fault, never a stored value."""
    # Checked in four escalating stages, cheapest first, each of which is
    # sufficient on its own to reject the cache.

    # 1. Schema. A file written by a different version of this programme may
    #    use fields that no longer mean the same thing.
    schema = payload.get("cache_schema_version")
    if schema != CANONICAL_CACHE_SCHEMA_VERSION:
        return "cache schema version differs"
    # 2. Shape. A missing or retyped field means the file is not what this
    #    programme wrote, so it is rebuilt rather than parsed defensively.
    for field_name, expected_type in _CACHE_REQUIRED_FIELDS.items():
        if field_name not in payload or payload[field_name] is None:
            return f"cached field absent: {field_name}"
        if not isinstance(payload[field_name], expected_type):
            return f"cached field has the wrong type: {field_name}"
    # 3. Stored context against its own digest. This detects an edit to the
    #    recorded configuration itself, which the caller's comparison against
    #    the expected context could not see.
    stored_context = payload["canonical_context"]
    if context_digest(stored_context) != payload["canonical_context_sha256"]:
        return "stored context digest does not match the stored context"
    # 4. Stored records against the stored outcome digest. The digest is
    #    recomputed from the records themselves, so an edited similarity, rank
    #    or enrolment count is caught rather than republished under a new hash.
    try:
        rebuilt = _run_from_cached_payload(payload)
    except (TypeError, KeyError, ValueError):
        return "cached records could not be reconstructed"
    if canonical_run_digest(rebuilt) != payload["canonical_run_digest"]:
        return "stored outcome digest does not match the stored records"
    return None


def cache_invalidation_reason(
    path: Path, expected_context: Mapping[str, Any]
) -> Optional[str]:
    """Why a cache cannot be reused, or None when it can.

    Reports the differing field name only: the values can include private
    material, so they are never placed in a message."""
    path = Path(path)
    if not path.is_file():
        return "no cached run present"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "cached run could not be read"
    if not isinstance(payload, dict):
        return "cached run is not an object"
    # Internal soundness first: there is no point comparing configurations if
    # the file's own records no longer match its own digests.
    integrity = cached_payload_integrity_reason(payload)
    if integrity is not None:
        return integrity
    # Then the configuration that produced the cache against the one now in
    # force. Comparing digests rather than field by field means a newly added
    # context field cannot be silently ignored.
    stored = payload["canonical_context"]
    if context_digest(stored) == context_digest(expected_context):
        return None
    # Name the differing fields so a rebuild can be explained, taking the union
    # of both sides so an added or removed field is reported as well as a
    # changed one. Field names only: a value could carry private material.
    differing = sorted(
        key for key in set(stored) | set(expected_context)
        if stored.get(key) != expected_context.get(key)
    )
    return f"context differs in: {', '.join(differing) or 'unknown field'}"


# A cache is accepted only when the recomputed context matches the stored one
# and the stored payload is internally consistent. Anything else is rebuilt,
# because publishing a figure from a cache built under different inputs would
# misattribute the result.
def canonical_primary_run(
    protocol: OpenSetProtocol,
    *,
    partition: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    dataset: Optional[BfwDataset] = None,
    base_cache: Path = CANONICAL_RUN_CACHE,
    refresh: bool = False,
) -> Tuple[OpenSetRunResult, str, Dict[str, Any]]:
    """Return the canonical primary-pipeline run for one partition.

    Rebuilt whenever any input capable of changing the result differs from the
    cached context, so Experiments 6, 7 and 8 cannot diverge."""
    cache_path = canonical_cache_path(partition, base_cache)
    context = canonical_run_context(
        protocol, partition=partition, dataset=dataset, detector=detector, embedder=embedder
    )
    # No reason to reject means the cached run was produced under exactly this
    # configuration and its records still match their stored digests, so it can
    # be reused. Reuse is what keeps Experiments 6, 7 and 8 reporting the same
    # figures for the same method.
    reason = cache_invalidation_reason(cache_path, context)
    if not refresh and reason is None:
        cached = load_canonical_run(cache_path)
        if cached is not None:
            return cached, canonical_run_digest(cached), context
    # Otherwise the reason is announced, so a rebuild is never silent.
    if reason is not None and not refresh:
        announce(f"Rebuilding the {partition} canonical run: {reason}")

    # Score the partition afresh and store it with the configuration that
    # produced it, ready for the next experiment to reuse.
    run = run_open_set_method(
        protocol, partition=partition, method=METHOD_B, detector=detector, embedder=embedder
    )
    return run, save_canonical_run(run, cache_path, context), context


def canonical_primary_test_run(
    protocol: OpenSetProtocol,
    *,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    dataset: Optional[BfwDataset] = None,
    cache_path: Path = CANONICAL_RUN_CACHE,
    refresh: bool = False,
) -> Tuple[OpenSetRunResult, str, Dict[str, Any]]:
    """Held-out partition convenience wrapper."""
    return canonical_primary_run(
        protocol, partition="test", detector=detector, embedder=embedder,
        dataset=dataset, base_cache=cache_path, refresh=refresh,
    )


def open_set_coverage(run: OpenSetRunResult) -> Dict[str, Any]:
    """Coverage counts. Reported alongside every rate, because a rate measured
    over a small surviving fraction of the protocol is not comparable with one
    measured over nearly all of it."""
    intended_gallery = len(run.enrolment_outcomes)
    enrolled = sum(1 for o in run.enrolment_outcomes if o.enrolled)
    mated = [r for r in run.search_results if r.role == "mated_probe"]
    non_mated = [r for r in run.search_results if r.role == "non_mated_probe"]
    scored_mated = [r for r in mated if r.failure_code is None]
    scored_non_mated = [r for r in non_mated if r.failure_code is None]

    # Failures are grouped by their category only. A code may carry detail after
    # a colon, which is dropped here so the published breakdown counts kinds of
    # failure without disclosing anything about the individual image.
    failure_breakdown: Dict[str, int] = {}
    for row in run.search_results:
        if row.failure_code is None:
            continue
        key = row.failure_code.split(":", 1)[0]
        failure_breakdown[key] = failure_breakdown.get(key, 0) + 1
    enrolment_failures = sum(1 for o in run.enrolment_outcomes if not o.enrolled)

    top1_ms = [r.top1_time_seconds * 1000.0 for r in run.search_results if r.top1_time_seconds]
    top5_ms = [r.top5_time_seconds * 1000.0 for r in run.search_results if r.top5_time_seconds]

    return {
        "method": run.method,
        "partition": run.partition,
        "gallery_size": run.gallery_size,
        "similarity_comparisons": run.comparisons_per_probe * len(scored_mated + scored_non_mated),
        "intended_gallery_identities": intended_gallery,
        "enrolled_gallery_identities": enrolled,
        "gallery_enrolment_failure_count": enrolment_failures,
        "gallery_enrolment_failure_rate": (
            enrolment_failures / intended_gallery if intended_gallery else float("nan")
        ),
        "gallery_enrolment_coverage": (
            enrolled / intended_gallery if intended_gallery else float("nan")
        ),
        "intended_mated_probes": len(mated),
        "scored_mated_probes": len(scored_mated),
        "mated_extraction_failure_rate": (
            (len(mated) - len(scored_mated)) / len(mated) if mated else float("nan")
        ),
        "intended_non_mated_probes": len(non_mated),
        "scored_non_mated_probes": len(scored_non_mated),
        "non_mated_extraction_failure_rate": (
            (len(non_mated) - len(scored_non_mated)) / len(non_mated) if non_mated else float("nan")
        ),
        "probe_failure_breakdown": failure_breakdown,
        **{
            f"{stage.replace('_seconds', '')}_latency_{suffix}_ms": value
            for stage, samples in (run.stage_times_seconds or {}).items()
            for suffix, value in (
                ("mean", statistics.fmean([s * 1000.0 for s in samples]) if samples else float("nan")),
                ("p95", percentile([s * 1000.0 for s in samples], 95) if samples else float("nan")),
            )
        },
        "images_timed": len((run.stage_times_seconds or {}).get("complete_pipeline_seconds", [])),
        "top1_search_time_mean_ms": statistics.fmean(top1_ms) if top1_ms else float("nan"),
        "top1_search_time_p95_ms": percentile(top1_ms, 95) if top1_ms else float("nan"),
        "top5_search_time_mean_ms": statistics.fmean(top5_ms) if top5_ms else float("nan"),
        "top5_search_time_p95_ms": percentile(top5_ms, 95) if top5_ms else float("nan"),
    }


def open_set_duplicate_detection(
    run: OpenSetRunResult, threshold: float
) -> Dict[str, float]:
    """Conditional and end-to-end duplicate detection, defined exactly as in the
    corrected LFW gallery accounting so the two experiments stay comparable."""
    # Every mated probe the protocol intended, and the subset that survived
    # extraction. The two lists become the two denominators below.
    mated = [r for r in run.search_results if r.role == "mated_probe"]
    scored = [r for r in mated if r.failure_code is None]
    # A detection requires both conditions: the correct profile ranked first
    # *and* above threshold. Rank alone would count a case the reviewer never
    # sees, and threshold alone would credit a match against the wrong profile.
    detected = sum(
        1
        for r in scored
        if r.correct_rank == 1 and r.correct_similarity is not None
        and r.correct_similarity >= threshold
    )
    return {
        # Conditional: of the photographs the model could process. Answers how
        # well the comparison performs when it runs at all.
        "conditional_duplicate_detection_rate": (
            detected / len(scored) if scored else float("nan")
        ),
        # End-to-end: of every photograph intended. An extraction failure counts
        # against this rate, because operationally a duplicate that was never
        # scored is a duplicate that was never found.
        "end_to_end_duplicate_detection_rate": detected / len(mated) if mated else float("nan"),
    }


# =============================================================================
# 21. Open-set threshold development and freezing
# =============================================================================
#
# The threshold is developed on the BFW development partition only. Test scores
# never enter candidate generation or selection, and a test evaluation refuses
# any policy whose status is not exactly open_set_frozen.


class OpenSetPolicyError(RuntimeError):
    """Raised when an open-set policy is missing, malformed or not frozen."""


OPEN_SET_SELECTION_RULE = (
    "Among candidate thresholds whose development FPIR is no greater than the target, "
    "select the highest development TPIR at rank 1; ties broken by lower development "
    "FPIR, then by higher threshold, then by candidate name, for full determinism."
)


def open_set_threshold_candidates(
    results: Sequence[OpenSetSearchResult],
) -> List[Tuple[str, float]]:
    """Every distinct top score observed on the development partition, plus two
    sentinels that bracket the range so a target can always be met."""
    scores = sorted(
        {
            round(float(r.top_similarity), 12)
            for r in results
            if r.failure_code is None and r.top_similarity is not None
        }
    )
    candidates: List[Tuple[str, float]] = [("sentinel_reject_all", 1.0000000001)]
    for score in scores:
        candidates.append((f"score_{score:.12f}", score))
    candidates.append(("sentinel_accept_all", -1.0000000001))
    return candidates


def select_open_set_threshold(
    development_results: Sequence[OpenSetSearchResult],
    *,
    target_fpir: float,
) -> Dict[str, Any]:
    """Apply the selection rule at one target FPIR and return the full evidence."""
    evaluated: List[Dict[str, Any]] = []
    for name, threshold in open_set_threshold_candidates(development_results):
        rates = open_set_rates_at_threshold(development_results, threshold)
        evaluated.append({"candidate": name, **rates})

    admissible = [c for c in evaluated if c["fpir"] == c["fpir"] and c["fpir"] <= target_fpir]
    if not admissible:
        raise OpenSetPolicyError(
            f"No candidate threshold achieved a development FPIR at or below {target_fpir}. "
            f"The lowest observed was {min((c['fpir'] for c in evaluated), default=float('nan'))}."
        )

    def sort_key(candidate: Dict[str, Any]):
        tpir = candidate["tpir_rank1"]
        return (
            -(tpir if tpir == tpir else -1.0),  # highest TPIR@1; NaN sorts last
            candidate["fpir"],                   # then lower FPIR
            -candidate["threshold"],             # then higher threshold
            candidate["candidate"],              # then candidate name
        )

    chosen = sorted(admissible, key=sort_key)[0]
    return {
        "target_fpir": target_fpir,
        "selected_candidate": chosen["candidate"],
        "threshold": chosen["threshold"],
        "development_fpir": chosen["fpir"],
        "development_tpir_rank1": chosen["tpir_rank1"],
        "development_tpir_rank5": chosen["tpir_rank5"],
        "selection_rule": OPEN_SET_SELECTION_RULE,
        "candidates_evaluated": len(evaluated),
        "candidates_admissible": len(admissible),
    }


def require_frozen_open_set_policy(payload: Mapping[str, Any], *, context: str = "") -> float:
    """Refuse to score the held-out test partition with anything but a frozen
    policy. This is the structural guarantee that the reported test numbers were
    not tuned on the data they describe."""
    status = payload.get("status")
    if status != OPEN_SET_STATUS_FROZEN:
        # An unrecognised status is a different fault from a recognised but
        # wrong one — the first suggests a corrupt or foreign artefact, the
        # second an ordering mistake — so the message distinguishes them.
        detail = (
            f"Only {OPEN_SET_STATUS_FROZEN!r} is accepted; develop and freeze the policy "
            f"on the development partition first."
            if status in OPEN_SET_STATUSES
            else f"That is not a recognised open-set status; expected one of "
            f"{list(OPEN_SET_STATUSES)}. The artefact may be corrupt or from another tool."
        )
        raise OpenSetPolicyError(
            f"Refusing to run a held-out open-set evaluation with policy status {status!r}"
            f"{f' from {context}' if context else ''}. {detail}"
        )
    operating = payload.get("operating_points") or {}
    primary = operating.get(str(PRIMARY_FPIR_TARGET))
    if not primary or "threshold" not in primary:
        raise OpenSetPolicyError(
            f"Frozen open-set policy carries no threshold for the primary FPIR target "
            f"{PRIMARY_FPIR_TARGET}."
        )
    return float(primary["threshold"])


# =============================================================================
# 22. Cluster-bootstrap confidence intervals
# =============================================================================
#
# Several probes can belong to one identity, so resampling images independently
# would treat correlated observations as independent and produce intervals that
# are far too narrow. Identities are resampled instead, with subgroup
# stratification preserved, which is the standard cluster bootstrap.

##############
# Title: Bootstrap Methods: Another Look at the Jackknife
# Author: Efron, B., The Annals of Statistics, 7(1), pp. 1-26
# Date: 1979
# Availability: https://doi.org/10.1214/aos/1176344552
##############
# The resampling principle is Efron's. The clustering by identity, the subgroup
# stratification and the percentile interval below are written for this
# project; no external implementation is copied or adapted.

# Replicate count is fixed rather than tuned. A larger count narrows the Monte
# Carlo error of the interval endpoints, not the interval itself.
BOOTSTRAP_REPLICATES = 2000


def _percentile_interval(values: Sequence[float]) -> Tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        return (float("nan"), float("nan"))
    return (percentile(ordered, 2.5), percentile(ordered, 97.5))


def zero_event_identity_upper_bound(
    non_mated_counts: Mapping[str, Tuple[int, int]], *, false_referrals: int,
    confidence: float = 0.95,
) -> Optional[Dict[str, Any]]:
    """Supplement a degenerate bootstrap with the correct identity estimand.

    Each independent identity is one Bernoulli trial: did any of its intended
    probes produce a false referral? Dependence among its images is unrestricted.
    This is not a binomial bound treating images as independent, and it is not
    a bound on FPIR conditional on successful extraction.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly between zero and one.")
    counts = [pair for pair in non_mated_counts.values() if pair[0] > 0]
    if false_referrals != 0 or not counts:
        return None
    if any(scored < 0 or scored > intended for intended, scored in counts):
        raise ValueError("Scored counts must be between zero and intended counts.")
    upper = -math.expm1(math.log1p(-confidence) / len(counts))
    equal_intended = len({intended for intended, _scored in counts}) == 1
    return {
        "confidence_level": confidence,
        "one_sided_upper": upper,
        "independent_identities": len(counts),
        "intended_probes": sum(intended for intended, _scored in counts),
        "scored_probes": sum(scored for _intended, scored in counts),
        "observed_false_referrals": 0,
        "estimand": "Probability that an identity has any false referral in its probe set",
        "dependence_model": (
            "Independent, exchangeable identities under this fixed gallery and frozen "
            "policy; arbitrary dependence among images of one identity. Extraction "
            "failures remain unresolved and do not generate a referral."
        ),
        "formula": "1 - (1 - confidence) ** (1 / independent_identities)",
        "equal_intended_probes_per_identity": equal_intended,
        "intended_probe_false_referral_rate_upper": upper if equal_intended else None,
        "conditional_scored_probe_fpir_upper": None,
        "interpretation": (
            "For an equal fixed number of intended probes per identity, the false "
            "referral rate per intended probe is no greater than the probability "
            "of any referral per identity. This implication does not bound FPIR "
            "conditional on successful extraction or establish deployment performance."
        ),
    }


def _attach_zero_event_bound(
    interval: Dict[str, Any], counts: Mapping[str, Tuple[int, int]], false_referrals: int,
) -> None:
    bound = zero_event_identity_upper_bound(counts, false_referrals=false_referrals)
    if bound is not None:
        interval["bootstrap_degenerate_zero_events"] = True
        interval["zero_event_identity_bound"] = bound
        interval["interval_warning"] = (
            "The empirical bootstrap cannot generate unobserved events; [0, 0] "
            "does not bound population FPIR. The supplementary one-sided bound "
            "concerns any false referral per independent identity, not scored-probe FPIR."
        )


def cluster_bootstrap_intervals(
    results: Sequence[OpenSetSearchResult],
    *,
    threshold: float,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Dict[str, Any]]:
    """95% percentile intervals over identity-level resampling.

    A replicate in which a metric is undefined (no mated or no non-mated
    identity survived, say) is excluded from that metric's interval and counted,
    never silently replaced by zero."""
    by_identity: Dict[str, List[OpenSetSearchResult]] = {}
    subgroup_of: Dict[str, str] = {}
    for row in results:
        by_identity.setdefault(row.identity_hash, []).append(row)
        subgroup_of[row.identity_hash] = row.subgroup

    mated_ids = sorted({r.identity_hash for r in results if r.role == "mated_probe"})
    non_mated_ids = sorted({r.identity_hash for r in results if r.role == "non_mated_probe"})

    def stratify(identity_hashes: Sequence[str]) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = {}
        for identity_hash in identity_hashes:
            buckets.setdefault(subgroup_of[identity_hash], []).append(identity_hash)
        return {k: sorted(v) for k, v in sorted(buckets.items())}

    mated_strata = stratify(mated_ids)
    non_mated_strata = stratify(non_mated_ids)

    tracked = (
        "fpir",
        "fnir_rank1",
        "fnir_rank5",
        "tpir_rank1",
        "tpir_rank5",
        "end_to_end_duplicate_detection_rate",
        "extraction_coverage",
        # Bootstrap mated and non-mated coverage separately.
        "mated_extraction_coverage",
        "non_mated_extraction_coverage",
    )
    samples: Dict[str, List[float]] = {name: [] for name in tracked}

    rng = random.Random(seed)
    for _replicate in range(replicates):
        drawn: List[OpenSetSearchResult] = []
        for strata in (mated_strata, non_mated_strata):
            for _subgroup, members in strata.items():
                # With replacement, preserving each stratum's size.
                for _ in range(len(members)):
                    drawn.extend(by_identity[members[rng.randrange(len(members))]])

        rates = open_set_rates_at_threshold(drawn, threshold)
        mated_rows = [r for r in drawn if r.role == "mated_probe"]
        scored_mated = [r for r in mated_rows if r.failure_code is None]
        detected = sum(
            1
            for r in scored_mated
            if r.correct_rank == 1
            and r.correct_similarity is not None
            and r.correct_similarity >= threshold
        )
        # Preserve role-specific denominators within each identity.
        non_mated_rows = [r for r in drawn if r.role == "non_mated_probe"]
        scored_non_mated = sum(1 for r in non_mated_rows if r.failure_code is None)
        extras = {
            "end_to_end_duplicate_detection_rate": (
                detected / len(mated_rows) if mated_rows else float("nan")
            ),
            "extraction_coverage": (
                sum(1 for r in drawn if r.failure_code is None) / len(drawn)
                if drawn
                else float("nan")
            ),
            "mated_extraction_coverage": (
                len(scored_mated) / len(mated_rows) if mated_rows else float("nan")
            ),
            "non_mated_extraction_coverage": (
                scored_non_mated / len(non_mated_rows) if non_mated_rows else float("nan")
            ),
        }
        for name in tracked:
            value = extras.get(name, rates.get(name, float("nan")))
            if isinstance(value, float) and value == value:
                samples[name].append(value)

    intervals: Dict[str, Dict[str, Any]] = {}
    for name in tracked:
        low, high = _percentile_interval(samples[name])
        intervals[name] = {
            "lower_95": low,
            "upper_95": high,
            "valid_replicates": len(samples[name]),
            "requested_replicates": replicates,
        }
    counts = {
        identity: (len(rows), sum(row.failure_code is None for row in rows))
        for identity, all_rows in by_identity.items()
        if (rows := [row for row in all_rows if row.role == "non_mated_probe"])
    }
    false_referrals = sum(
        row.role == "non_mated_probe" and row.failure_code is None
        and row.top_similarity is not None and row.top_similarity >= threshold
        for row in results
    )
    _attach_zero_event_bound(intervals["fpir"], counts, false_referrals)
    return intervals


# =============================================================================
# 23. Demographic subgroup performance
# =============================================================================
#
# Subgroup figures are aggregate fairness reporting over benchmark identities.
# No subgroup label is ever published beside anything that identifies a person,
# no attribute is inferred with another model, and the primary result uses one
# global threshold — subgroup-specific thresholds would change what is being
# measured and are not applied to the held-out test.


# Subgroup membership is read only when reporting. It is never a classifier
# feature, a threshold input or a reason to apply a different decision policy.
def subgroup_open_set_metrics(
    results: Sequence[OpenSetSearchResult],
    *,
    threshold: float,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Dict[str, Any]]:
    per_subgroup: Dict[str, Dict[str, Any]] = {}
    # One breakdown per official subgroup. Every subgroup is measured at the
    # same frozen threshold: applying a different threshold to each would be
    # tuning the system per demographic group, which this project does not do.
    for subgroup in BFW_SUBGROUPS:
        rows = [r for r in results if r.subgroup == subgroup]
        if not rows:
            continue
        rates = open_set_rates_at_threshold(rows, threshold)
        mated = [r for r in rows if r.role == "mated_probe"]
        non_mated = [r for r in rows if r.role == "non_mated_probe"]
        intervals = cluster_bootstrap_intervals(
            rows, threshold=threshold, replicates=replicates, seed=seed
        )
        # Each rate is published with an interval beside it. A subgroup holds
        # only an eighth of the partition, so a point estimate on its own would
        # suggest more precision than the sample size supports.
        entry: Dict[str, Any] = {}
        for metric in ("fpir", "fnir_rank1", "fnir_rank5", "tpir_rank1", "tpir_rank5"):
            entry[metric] = rates[metric]
            band = intervals.get(metric, {})
            entry[f"{metric}_lower_95"] = band.get("lower_95", float("nan"))
            entry[f"{metric}_upper_95"] = band.get("upper_95", float("nan"))
        for label, rows_of_role, series in (
            ("mated_probe_coverage", mated, "mated_extraction_coverage"),
            ("non_mated_probe_coverage", non_mated, "non_mated_extraction_coverage"),
        ):
            scored = sum(1 for r in rows_of_role if r.failure_code is None)
            entry[label] = scored / len(rows_of_role) if rows_of_role else float("nan")
            band = intervals.get(series, {})
            entry[f"{label}_lower_95"] = band.get("lower_95", float("nan"))
            entry[f"{label}_upper_95"] = band.get("upper_95", float("nan"))
        entry["scored_mated_probes"] = rates["scored_mated_probes"]
        entry["scored_non_mated_probes"] = rates["scored_non_mated_probes"]
        entry["intended_mated_probes"] = len(mated)
        entry["intended_non_mated_probes"] = len(non_mated)
        per_subgroup[subgroup] = entry
    return per_subgroup


def subgroup_disparity_summary(per_subgroup: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Spread across subgroups. The max/min ratio is reported only when the
    denominator is non-zero; an absolute range is always reported, because a
    ratio against a zero rate is undefined rather than infinitely bad."""
    # Subgroups with no measurable rate are excluded. Comparing against a
    # not-a-number entry would silently poison the maximum and the minimum.
    fpirs = [v["fpir"] for v in per_subgroup.values() if v["fpir"] == v["fpir"]]
    tpirs = [v["tpir_rank1"] for v in per_subgroup.values() if v["tpir_rank1"] == v["tpir_rank1"]]
    if not fpirs:
        return {"subgroups_reported": len(per_subgroup)}
    max_fpir, min_fpir = max(fpirs), min(fpirs)
    summary: Dict[str, Any] = {
        "subgroups_reported": len(per_subgroup),
        "max_subgroup_fpir": max_fpir,
        "min_subgroup_fpir": min_fpir,
        "absolute_fpir_range": max_fpir - min_fpir,
    }
    # A ratio is only meaningful when every subgroup recorded some false
    # referrals. Where one recorded none, the absolute range is reported instead
    # of a figure that would read as an unbounded disparity.
    if min_fpir > 0:
        summary["max_to_min_fpir_ratio"] = max_fpir / min_fpir
    else:
        summary["max_to_min_fpir_ratio"] = None
        summary["max_to_min_fpir_ratio_note"] = (
            "Undefined: at least one subgroup recorded a zero FPIR, so the ratio is not "
            "reported. Use the absolute range instead."
        )
    if tpirs:
        summary["max_subgroup_tpir_rank1"] = max(tpirs)
        summary["min_subgroup_tpir_rank1"] = min(tpirs)
        summary["absolute_tpir_rank1_range"] = max(tpirs) - min(tpirs)
    return summary


# =============================================================================
# 24. Open-set experiment orchestration and artefacts
# =============================================================================
#
# Experiment 6 in full: load BFW, build the identity-disjoint partition, run the
# single-image control, develop and freeze a policy on the development
# partition, apply it unchanged to the held-out test partition, then compute
# confidence intervals and subgroup figures.
#
# Success criteria are declared here, in source, before any test result exists.
# They are research targets, not claims, and the report states plainly which
# were achieved, which were not, and which could not be measured.

OPEN_SET_SUCCESS_CRITERIA = {
    "held_out_fpir_max": 0.01,
    "target_fpir": PRIMARY_FPIR_TARGET,
    "tpir_rank1_min": 0.90,
    "tpir_rank5_min": 0.95,
    "gallery_enrolment_coverage_min": 0.90,
    "probe_extraction_coverage_min": 0.90,
}

OPEN_SET_LIMITATIONS = (
    "This remains a proof of concept. No result here proves fraud, misuse or "
    "misrepresentation by any person.",
    "No automatic sanction is applied. Every outcome opens a case for human review and "
    "nothing else. In this gallery-screening experiment the referral is triggered by a "
    "high similarity to another enrolled identity; profile-photo consistency refers a "
    "low similarity to the profile's own template instead, and an extraction failure "
    "makes no decision at all.",
    "The BFW open-set evaluation uses a protocol defined by this project. BFW publishes "
    "verification and bias-analysis protocols, not an open-set identification protocol.",
    "Development and test identities are completely disjoint, and the operating threshold "
    "was frozen before the held-out test partition was scored.",
    "Extraction failures are counted as coverage failures, never as genuine no-match "
    "decisions.",
    "Confidence intervals describe sampling uncertainty over these benchmark identities "
    "only. They do not extend to any other population.",
    "Benchmark demographics do not represent any real deployed user population, so "
    "subgroup figures must not be read as deployment estimates.",
)


def evaluate_open_set_success_criteria(
    coverage: Mapping[str, Any], rates: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare held-out results against the pre-declared targets. A metric that
    is undefined is reported as not measurable rather than as a pass."""

    # One comparison used for every criterion. Some targets are floors the
    # result must reach, others are ceilings it must stay under, so the caller
    # states which. A metric that could not be measured is never a pass.
    def verdict(actual: Any, threshold: float, *, minimum: bool) -> Dict[str, Any]:
        if not isinstance(actual, (int, float)) or actual != actual:
            return {"outcome": "not_measurable", "actual": None, "target": threshold}
        achieved = actual >= threshold if minimum else actual <= threshold
        return {
            "outcome": "achieved" if achieved else "not_achieved",
            "actual": float(actual),
            "target": threshold,
        }

    # The weaker of the two probe groups is taken, so a good result on one
    # group cannot disguise poor image processing on the other.
    probe_coverage = min(
        (
            1.0 - coverage.get("mated_extraction_failure_rate", float("nan")),
            1.0 - coverage.get("non_mated_extraction_failure_rate", float("nan")),
        ),
        default=float("nan"),
    )
    return {
        "criteria_declared_before_test": True,
        "fpir_at_or_below_1_percent": verdict(
            rates.get("fpir"), OPEN_SET_SUCCESS_CRITERIA["held_out_fpir_max"], minimum=False
        ),
        "tpir_rank1_at_least_90_percent": verdict(
            rates.get("tpir_rank1"), OPEN_SET_SUCCESS_CRITERIA["tpir_rank1_min"], minimum=True
        ),
        "tpir_rank5_at_least_95_percent": verdict(
            rates.get("tpir_rank5"), OPEN_SET_SUCCESS_CRITERIA["tpir_rank5_min"], minimum=True
        ),
        "gallery_enrolment_coverage_at_least_90_percent": verdict(
            coverage.get("gallery_enrolment_coverage"),
            OPEN_SET_SUCCESS_CRITERIA["gallery_enrolment_coverage_min"],
            minimum=True,
        ),
        "probe_extraction_coverage_at_least_90_percent": verdict(
            probe_coverage,
            OPEN_SET_SUCCESS_CRITERIA["probe_extraction_coverage_min"],
            minimum=True,
        ),
    }


def _open_set_provenance(
    dataset: BfwDataset,
    protocol: OpenSetProtocol,
    detector: Any,
    embedder: Any,
    description: Optional[PipelineDescription] = None,
) -> Dict[str, Any]:
    _identity = pipeline_identity_fields(detector, embedder, description)
    return {
        "schema_version": SCHEMA_VERSION,
        "methodology_revision": GALLERY_METHODOLOGY_REVISION,
        "opaque_id_version": OPAQUE_ID_VERSION,
        "created_at": utc_now_iso(),
        "software_environment": software_environment_report(),
        "pipeline_name": _identity["model_version"],
        **_identity,
        "dataset_provenance": bfw_dataset_provenance(dataset),
        "seed": protocol.seed,
        "policy_note": POLICY_NOTE,
        "limitations": list(OPEN_SET_LIMITATIONS),
    }


def run_open_set_experiment(
    *,
    output_root: Path = AGGREGATE_ROOT,
    raw_root: Path = RAW_ROOT,
    threshold_artifact: Optional[Path] = None,
    seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    """Experiment 6 end to end. Stops on the exact underlying blocker; never
    fabricates, hardcodes or approximates a result."""
    config = EnvironmentConfig.load()
    if not id_hmac_key_is_configured():
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} must be configured before identifiers are produced."
        )
    image_root, metadata_path = config.require_bfw_roots()

    detector, embedder = load_models(config.require_model_root())
    announce_stage(1, 6, "Loading and validating the BFW dataset",
                   "The pinned metadata schema is checked before anything is scored.")
    dataset = load_bfw_dataset(image_root, metadata_path)
    announce(f"BFW: {len(dataset.images)} images across {len(dataset.by_identity())} identities")

    protocol = build_open_set_protocol(dataset, seed=seed)
    private_manifest = Path(raw_root) / "bfw_open_set_manifest.json"
    write_open_set_private_manifest(protocol, private_manifest)
    announce(
        f"Wrote the private open-set manifest to {project_relative(private_manifest)} "
        f"(contains real image paths — kept out of Git)"
    )

    summary_payload = open_set_protocol_summary(
        protocol, dataset=dataset, detector=detector, embedder=embedder
    )
    write_json_artifact(output_root / "bfw_open_set_protocol_summary.json", summary_payload)

    provenance = _open_set_provenance(dataset, protocol, detector, embedder)
    public_manifest_sha256 = summary_payload["public_manifest_sha256"]

    # --- Method A control, at the untouched LFW 1:1 threshold -----------------
    threshold_artifact = threshold_artifact or (output_root / "calibrated_threshold.json")
    lfw_payload = read_json_artifact(threshold_artifact)
    lfw_threshold = require_frozen_threshold(
        lfw_payload, context=project_relative(threshold_artifact)
    )

    announce_stage(2, 6, "Building development and held-out identity groups",
                   "The groups do not share any person.")
    announce("Running Method A (single-image enrolment, LFW 1:1 threshold) on development")
    control_dev = run_open_set_method(
        protocol, partition="development", method=METHOD_A, detector=detector, embedder=embedder
    )

    # --- Method B development and freezing -----------------------------------
    announce_stage(3, 6, "Creating enrolled profile templates",
                   "Three photographs are combined to represent each profile.")
    announce("Running Method B (three-image template) on development")
    # Reuse the cache rather than force a rebuild. Experiment 6 is the producer
    # of the canonical runs, but forcing a refresh here re-rolled the primary
    # pipeline on every invocation, and OpenCV detection is not bit-reproducible
    # across processes, so a probe could change side between two runs of the
    # same unchanged code. The context digest still rebuilds whenever anything
    # capable of changing the result differs, including CANONICAL_PIPELINE_REVISION.
    proposed_dev, development_digest, development_context = canonical_primary_run(
        protocol, partition="development", detector=detector, embedder=embedder,
        dataset=dataset,
    )
    announce(f"Canonical development run digest {development_digest[:16]}")

    operating_points: Dict[str, Any] = {}
    for target in FPIR_TARGETS:
        operating_points[str(target)] = select_open_set_threshold(
            proposed_dev.search_results, target_fpir=target
        )

    policy_payload = {
        "artifact_type": "bfw_open_set_threshold",
        "status": OPEN_SET_STATUS_FROZEN,
        "method": METHOD_B,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "operating_points": operating_points,
        "selection_rule": OPEN_SET_SELECTION_RULE,
        "developed_on": "BFW development partition (identity-disjoint from test)",
        "public_manifest_sha256": public_manifest_sha256,
        "protocol_digest": summary_payload["public_manifest_sha256"],
        **provenance,
    }
    policy_path = output_root / "bfw_open_set_threshold.json"
    write_json_artifact(policy_path, policy_payload)
    announce_stage(4, 6, "Selecting the review threshold",
                   "Only development identities are used at this stage.")
    announce(f"Froze the open-set policy at {project_relative(policy_path)}")

    frozen_threshold = require_frozen_open_set_policy(
        read_json_artifact(policy_path), context=project_relative(policy_path)
    )

    development_payload = {
        "artifact_type": "bfw_open_set_development_metrics",
        "status": OPEN_SET_STATUS_DEVELOPMENT,
        "public_manifest_sha256": public_manifest_sha256,
        "protocol_digest": summary_payload["public_manifest_sha256"],
        "threshold_source": project_relative(policy_path),
        "methods": {
            METHOD_A: {
                "coverage": open_set_coverage(control_dev),
                "operating_threshold": lfw_threshold,
                "threshold_role": (
                    "Control only. This is the LFW 1:1 verification threshold reused "
                    "unchanged; it is not a valid open-set operating threshold."
                ),
                "rates": open_set_rates_at_threshold(control_dev.search_results, lfw_threshold),
                **open_set_duplicate_detection(control_dev, lfw_threshold),
            },
            METHOD_B: {
                "coverage": open_set_coverage(proposed_dev),
                "operating_points": {
                    str(t): open_set_rates_at_threshold(
                        proposed_dev.search_results, operating_points[str(t)]["threshold"]
                    )
                    for t in FPIR_TARGETS
                },
                "at_lfw_control_threshold": open_set_rates_at_threshold(
                    proposed_dev.search_results, lfw_threshold
                ),
            },
        },
        **provenance,
    }
    write_json_artifact(
        output_root / "bfw_open_set_development_metrics.json", development_payload
    )

    # --- Held-out test, frozen policy applied unchanged -----------------------
    announce_stage(5, 6, "Testing unseen identities",
                   "The frozen threshold is now applied to the held-out test group.")
    announce("Scoring the held-out test partition with the frozen policy")
    control_test = run_open_set_method(
        protocol, partition="test", method=METHOD_A, detector=detector, embedder=embedder
    )
    proposed_test, canonical_digest, test_context = canonical_primary_test_run(
        protocol, detector=detector, embedder=embedder, dataset=dataset
    )
    announce(f"Canonical primary-pipeline run digest {canonical_digest[:16]}")
    provenance["canonical_run_digest"] = canonical_digest
    provenance["canonical_test_run_digest"] = canonical_digest
    provenance["canonical_development_run_digest"] = development_digest
    provenance["canonical_test_context_sha256"] = context_digest(test_context)
    provenance["canonical_development_context_sha256"] = context_digest(development_context)
    provenance["cache_schema_version"] = CANONICAL_CACHE_SCHEMA_VERSION

    test_coverage = open_set_coverage(proposed_test)
    primary_rates = open_set_rates_at_threshold(proposed_test.search_results, frozen_threshold)

    test_payload = {
        "artifact_type": "bfw_open_set_test_metrics",
        "status": OPEN_SET_STATUS_TESTED,
        "public_manifest_sha256": public_manifest_sha256,
        "protocol_digest": summary_payload["public_manifest_sha256"],
        "threshold_source": project_relative(policy_path),
        "operating_threshold": frozen_threshold,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "methods": {
            METHOD_A: {
                "coverage": open_set_coverage(control_test),
                "operating_threshold": lfw_threshold,
                "threshold_role": (
                    "Control only. Not a valid open-set operating threshold."
                ),
                "rates": open_set_rates_at_threshold(control_test.search_results, lfw_threshold),
                **open_set_duplicate_detection(control_test, lfw_threshold),
            },
            METHOD_B: {
                "coverage": test_coverage,
                "at_lfw_control_threshold": open_set_rates_at_threshold(
                    proposed_test.search_results, lfw_threshold
                ),
                "operating_points": {
                    str(t): open_set_rates_at_threshold(
                        proposed_test.search_results, operating_points[str(t)]["threshold"]
                    )
                    for t in FPIR_TARGETS
                },
                "primary_operating_point": primary_rates,
                **open_set_duplicate_detection(proposed_test, frozen_threshold),
            },
        },
        "success_criteria": evaluate_open_set_success_criteria(test_coverage, primary_rates),
        **provenance,
    }
    write_json_artifact(output_root / "bfw_open_set_test_metrics.json", test_payload)

    # --- Confidence intervals and subgroup figures ---------------------------
    announce(f"Computing {bootstrap_replicates} cluster-bootstrap replicates")
    intervals = cluster_bootstrap_intervals(
        proposed_test.search_results,
        threshold=frozen_threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    write_json_artifact(
        output_root / "open_set_confidence_intervals.json",
        {
            "artifact_type": "open_set_confidence_intervals",
            "method": METHOD_B,
            "partition": "test",
            "operating_threshold": frozen_threshold,
            "replicates": bootstrap_replicates,
            "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
            "intervals": intervals,
            **provenance,
        },
    )

    per_subgroup = subgroup_open_set_metrics(
        proposed_test.search_results, threshold=frozen_threshold,
        replicates=bootstrap_replicates, seed=seed,
    )
    _write_subgroup_csv(output_root / "bfw_subgroup_metrics.csv", per_subgroup)
    write_json_artifact(
        output_root / "bfw_subgroup_confidence_intervals.json",
        {
            "artifact_type": "bfw_subgroup_confidence_intervals",
            "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
            "replicates": bootstrap_replicates,
            "operating_threshold": frozen_threshold,
            "subgroups": per_subgroup,
            **provenance,
        },
    )
    _write_method_comparison_csv(
        output_root / "open_set_method_comparison.csv",
        control_test=control_test,
        proposed_test=proposed_test,
        lfw_threshold=lfw_threshold,
        frozen_threshold=frozen_threshold,
    )

    # Optional extensions. Each either produces a real artefact or records why
    # it did not run; neither is allowed to interrupt the primary experiment.
    write_pipeline_comparison_csv(
        output_root / "pretrained_pipeline_comparison.csv",
        primary=primary_pipeline_description(detector, embedder),
        config=config,
    )
    consistency = profile_photo_consistency_summary(proposed_test, frozen_threshold)
    write_json_artifact(
        output_root / "profile_photo_consistency.json",
        {"artifact_type": "profile_photo_consistency", **consistency,
         **{k: v for k, v in provenance.items() if k != "policy_note"},
         "duplicate_screening_policy_note": (
             "Applies to gallery screening only, not to profile consistency: " + POLICY_NOTE
         )},
    )
    write_profile_consistency_artefacts(
        {MODEL_VERSION: consistency}, output_root, provenance=provenance
    )
    by_sex = sex_aggregated_metrics(
        proposed_test.search_results, threshold=frozen_threshold,
        replicates=bootstrap_replicates, seed=seed,
    )
    write_json_artifact(
        output_root / "bfw_sex_aggregated_metrics.json",
        {"artifact_type": "bfw_sex_aggregated_metrics", "operating_threshold": frozen_threshold,
         "aggregation": "pooled over identity outcomes, not averaged over subgroup percentages",
         "groups": by_sex, **provenance},
    )

    report = render_open_set_report(
        protocol_summary=summary_payload,
        development=development_payload,
        test=test_payload,
        intervals=intervals,
        per_subgroup=per_subgroup,
        disparity=subgroup_disparity_summary(per_subgroup),
    )
    (output_root / "OPEN_SET_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    announce("Wrote OPEN_SET_EVALUATION_REPORT.md")

    # Privacy validation over every artefact this experiment produced, including
    # the new ones. A leak stops the run rather than being reported afterwards.
    leaks = find_path_leaks(output_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise PrivacyLeakError(
            "Refusing to finish: open-set output(s) contain a personal/absolute path:\n"
            + "\n".join(f"  {redact_private_paths(leak)}" for leak in leaks)
        )
    assert_no_identifier_key_leak(output_root)
    announce("Privacy validation passed for every open-set artefact")
    announce_stage(6, 6, "Calculating uncertainty and writing reports",
                   f"{BOOTSTRAP_REPLICATES:,} identity-level bootstrap samples are used "
                   f"for confidence intervals.")

    return test_payload


# Profile-photo identity consistency reuses the architecture already in place:
# detection, embedding, normalised template, cosine comparison, calibrated
# decision, human review. It is not a separate identity-verification system.
#
# A photograph scoring below the operating threshold is inconsistent with the
# profile's enrolled template under this model and threshold. That is a review
# signal. It does not independently establish that the photograph belongs to
# another person: pose, lighting, occlusion, image quality, age difference,
# detection failure and model error all produce the same outcome.
###############################################################################
# Profile-photo consistency
###############################################################################
#
# A separate question from duplicate-profile screening, and the referral runs in
# the opposite direction: a photograph is referred when it is *dissimilar* to
# the profile's own enrolled template. The threshold is reused rather than
# recalibrated, so the whole analysis is exploratory.

PROFILE_CONSISTENCY_POLICY_NOTE = (
    "A photograph whose similarity to its own enrolled profile template is below the "
    "frozen consistency threshold opens an inconsistency review. A score at or above that "
    "threshold is facially consistent under this model and does not open an inconsistency "
    "case. An extraction failure is unresolved and must not be treated as either a match "
    "or mismatch."
)

PROFILE_CONSISTENCY_NOTE = (
    "A non-match indicates that the photograph is inconsistent with the enrolled facial "
    "template under the evaluated model and threshold. It does not prove that the "
    "photograph belongs to another person or that fraud occurred. Pose, lighting, "
    "occlusion, image quality, age difference, face-detection failure and model error can "
    "all produce the same result. An inconsistent photograph opens a human-review case; a "
    "consistent one does not, and an extraction failure resolves nothing."
)

# The whole consistency analysis reuses a threshold frozen for duplicate-profile
# screening. That is a defensible exploratory reuse, not a validated design.
PROFILE_CONSISTENCY_STATUS_NOTE = (
    "Exploratory threshold reuse. The operating threshold was frozen for open-set "
    "duplicate-profile screening and is applied here unchanged; no threshold was "
    "calibrated for profile-photo consistency and none of these figures has been "
    "separately validated. This is not a validated identity-authentication system "
    "and must not be reported as one."
)

CONSISTENCY_CONTROL_DEFINITIONS = {
    "open_set_non_mated_gallery_control": (
        "A non-mated probe searched against the complete gallery. This control tests "
        "whether a person absent from the gallery avoids matching any enrolled "
        "profile. It is stricter than, and structurally different from, comparing one "
        "photograph with one specified profile template."
    ),
    "wrong_profile_template_control": (
        "The same control photograph compared with exactly one wrong enrolled profile "
        "template, assigned deterministically from the protocol seed and the opaque "
        "sample identifier. This is the direct one-photograph-to-one-profile "
        "comparison. It is supplementary and does not replace the open-set control; "
        "both apply the same frozen pipeline-specific threshold, and no threshold was "
        "recalibrated for either."
    ),
}


def profile_photo_consistency_summary(
    run: OpenSetRunResult, threshold: float
) -> Dict[str, Any]:
    """Aggregate same-person consistency for photographs on one profile.

    This is a different decision from duplicate-profile screening and must not
    reuse its polarity. Screening asks whether a score *at or above* threshold
    indicates a possible existing profile. Consistency asks whether the
    correct-identity score falls *below* threshold, which makes the photograph
    inconsistent with the enrolled template and opens a consistency review.

    Each mated probe is a further photograph supplied for a profile whose
    template was built from its enrolment images, so its similarity to its own
    template is the consistency score."""
    consistent = inconsistent = 0
    extraction_failures = unavailable = 0
    scores: List[float] = []
    for row in run.search_results:
        if row.role != "mated_probe":
            continue
        if row.failure_code == GALLERY_REFERENCE_UNAVAILABLE:
            unavailable += 1
            continue
        if row.failure_code is not None or row.correct_similarity is None:
            extraction_failures += 1
            continue
        scores.append(float(row.correct_similarity))
        if row.correct_similarity >= threshold:
            consistent += 1
        else:
            inconsistent += 1

    # Open-set non-mated gallery control. A non-mated probe is enrolled nowhere,
    # so a top similarity below threshold means the search correctly returned no
    # profile. At or above threshold is a false-consistent result.
    #
    # This asks whether a person absent from the gallery avoids matching *any*
    # enrolled profile. It is stricter than, and structurally different from,
    # comparing one photograph with one specified profile template, and must not
    # be described as the latter.
    controls_correct = controls_false_consistent = control_failures = 0
    for row in run.search_results:
        if row.role != "non_mated_probe":
            continue
        if row.failure_code is not None or row.top_similarity is None:
            control_failures += 1
        elif row.top_similarity < threshold:
            controls_correct += 1
        else:
            controls_false_consistent += 1

    # Supplementary direct control: one photograph against exactly one wrong
    # profile template, assigned deterministically. Reported separately and
    # never in place of the open-set control above.
    wrong_correct = wrong_false_consistent = wrong_failures = 0
    for row in run.search_results:
        if row.role != "non_mated_probe":
            continue
        if row.failure_code is not None or row.assigned_wrong_template_similarity is None:
            wrong_failures += 1
        elif row.assigned_wrong_template_similarity < threshold:
            wrong_correct += 1
        else:
            wrong_false_consistent += 1

    intended_same_person = consistent + inconsistent + extraction_failures + unavailable
    scored_same_person = consistent + inconsistent
    intended_controls = controls_correct + controls_false_consistent + control_failures
    scored_controls = controls_correct + controls_false_consistent
    intended_wrong = wrong_correct + wrong_false_consistent + wrong_failures
    scored_wrong = wrong_correct + wrong_false_consistent

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else float("nan")

    return {
        "operating_threshold": threshold,
        # Same-person photographs.
        "intended_same_person_photographs": intended_same_person,
        "scored_same_person_photographs": scored_same_person,
        "consistent_same_person_photographs": consistent,
        "inconsistent_same_person_review_candidates": inconsistent,
        "same_person_extraction_failures": extraction_failures,
        "gallery_reference_unavailable": unavailable,
        # Open-set non-mated gallery control: absent person against the whole
        # gallery. Not a one-photograph-to-one-profile comparison.
        "intended_open_set_non_mated_gallery_controls": intended_controls,
        "scored_open_set_non_mated_gallery_controls": scored_controls,
        "open_set_non_mated_gallery_controls_correctly_identified": controls_correct,
        "open_set_non_mated_gallery_controls_false_consistent": controls_false_consistent,
        "open_set_non_mated_gallery_control_extraction_failures": control_failures,
        # Supplementary direct control: one photograph against exactly one
        # deterministically assigned wrong profile template.
        "wrong_profile_template_controls_scored": scored_wrong,
        "wrong_profile_template_controls_correctly_inconsistent": wrong_correct,
        "wrong_profile_template_controls_false_consistent": wrong_false_consistent,
        "wrong_profile_template_control_extraction_failures": wrong_failures,
        "wrong_profile_template_controls_intended": intended_wrong,
        "wrong_profile_template_mismatch_detection_conditional": rate(
            wrong_correct, scored_wrong
        ),
        "wrong_profile_template_mismatch_detection_end_to_end": rate(
            wrong_correct, intended_wrong
        ),
        "wrong_profile_template_false_consistency_conditional": rate(
            wrong_false_consistent, scored_wrong
        ),
        "wrong_profile_template_false_consistency_end_to_end": rate(
            wrong_false_consistent, intended_wrong
        ),
        "wrong_profile_template_extraction_coverage": rate(scored_wrong, intended_wrong),
        # Conditional rates divide by what was scored; end-to-end rates divide
        # by what the protocol intended, so extraction failures reduce them.
        "same_person_consistency_rate_conditional": rate(consistent, scored_same_person),
        "same_person_false_inconsistency_rate_conditional": rate(
            inconsistent, scored_same_person
        ),
        "same_person_consistency_rate_end_to_end": rate(consistent, intended_same_person),
        "same_person_false_inconsistency_rate_end_to_end": rate(
            inconsistent, intended_same_person
        ),
        "open_set_non_mated_gallery_detection_conditional": rate(
            controls_correct, scored_controls
        ),
        "open_set_non_mated_gallery_false_consistency_conditional": rate(
            controls_false_consistent, scored_controls
        ),
        "open_set_non_mated_gallery_detection_end_to_end": rate(
            controls_correct, intended_controls
        ),
        "open_set_non_mated_gallery_false_consistency_end_to_end": rate(
            controls_false_consistent, intended_controls
        ),
        "same_person_extraction_coverage": rate(scored_same_person, intended_same_person),
        "open_set_non_mated_gallery_control_extraction_coverage": rate(
            scored_controls, intended_controls
        ),
        "consistency_score_mean": statistics.fmean(scores) if scores else float("nan"),
        "consistency_score_median": statistics.median(scores) if scores else float("nan"),
        "outcome_policy": (
            "A consistent photograph does not open a case. An inconsistent photograph "
            "opens a consistency review. An extraction failure resolves nothing and is a "
            "separate unresolved outcome, not a decision."
        ),
        "control_definitions": CONSISTENCY_CONTROL_DEFINITIONS,
        "analysis_status": PROFILE_CONSISTENCY_STATUS_NOTE,
        "interpretation_note": PROFILE_CONSISTENCY_NOTE,
        # The duplicate-screening note has the opposite referral direction and
        # must not be attached to a consistency result.
        "profile_consistency_policy_note": PROFILE_CONSISTENCY_POLICY_NOTE,
        # Older names, kept only so an existing reader does not break. Every one
        # of them hides its denominator, which is why the authoritative fields
        # above state "conditional" or "end_to_end" explicitly. Do not cite
        # these in a result.
        "deprecated_compatibility_aliases": {
            "photographs_assessed": intended_same_person,
            "inconsistent_review_candidates": inconsistent,
            "extraction_failures": extraction_failures,
            "consistency_rate": rate(consistent, intended_same_person),
            "review_referral_rate": rate(inconsistent, intended_same_person),
            "deprecation_note": (
                "Superseded by the explicit conditional, end-to-end and coverage "
                "fields. Retained for backwards compatibility only; each of these "
                "names leaves its denominator unstated."
            ),
        },
    }


def sex_aggregated_metrics(
    results: Sequence[OpenSetSearchResult],
    *,
    threshold: float,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate over the four female and four male subgroups.

    Pooled from the underlying identity outcomes rather than by averaging
    subgroup percentages, which would weight a small subgroup as heavily as a
    large one. Sex is an evaluation dimension only and is never a predictor."""
    grouped: Dict[str, List[OpenSetSearchResult]] = {"female": [], "male": []}
    for row in results:
        if row.subgroup.endswith("_females"):
            grouped["female"].append(row)
        elif row.subgroup.endswith("_males"):
            grouped["male"].append(row)

    summary: Dict[str, Dict[str, Any]] = {}
    for label, rows in grouped.items():
        if not rows:
            continue
        rates = open_set_rates_at_threshold(rows, threshold)
        intervals = cluster_bootstrap_intervals(
            rows, threshold=threshold, replicates=replicates, seed=seed
        )
        mated = [r for r in rows if r.role == "mated_probe"]
        non_mated = [r for r in rows if r.role == "non_mated_probe"]
        entry: Dict[str, Any] = {
            "subgroups_pooled": sorted({r.subgroup for r in rows}),
            "identities": len({r.identity_hash for r in rows}),
        }
        for metric in ("fpir", "tpir_rank1", "tpir_rank5", "fnir_rank1"):
            entry[metric] = rates[metric]
            entry[f"{metric}_lower_95"] = intervals[metric]["lower_95"]
            entry[f"{metric}_upper_95"] = intervals[metric]["upper_95"]
        for label_key, rows_of_role, series in (
            ("mated_probe_coverage", mated, "mated_extraction_coverage"),
            ("non_mated_probe_coverage", non_mated, "non_mated_extraction_coverage"),
        ):
            scored = sum(1 for r in rows_of_role if r.failure_code is None)
            entry[label_key] = scored / len(rows_of_role) if rows_of_role else float("nan")
            entry[f"{label_key}_lower_95"] = intervals[series]["lower_95"]
            entry[f"{label_key}_upper_95"] = intervals[series]["upper_95"]
        entry["intended_mated_probes"] = len(mated)
        entry["intended_non_mated_probes"] = len(non_mated)
        # A coverage percentage alone cannot be checked. Publishing the scored
        # counts and the failures beside it lets a reader confirm that
        # scored + failures = intended and that coverage = scored / intended,
        # rather than taking the ratio on trust.
        entry["scored_mated_probes"] = sum(1 for r in mated if r.failure_code is None)
        entry["scored_non_mated_probes"] = sum(
            1 for r in non_mated if r.failure_code is None
        )
        entry["mated_extraction_failures"] = sum(
            1 for r in mated if r.failure_code is not None
        )
        entry["non_mated_extraction_failures"] = sum(
            1 for r in non_mated if r.failure_code is not None
        )
        entry["population_note"] = (
            "These benchmark categories do not represent every identity or any real "
            "deployed population."
        )
        summary[label] = entry
    return summary


def _write_subgroup_csv(path: Path, per_subgroup: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_SUBGROUP_CSV_COLUMNS)
        for subgroup in sorted(per_subgroup):
            row = per_subgroup[subgroup]
            writer.writerow([subgroup] + [row.get(n, "") for n in _SUBGROUP_CSV_COLUMNS[1:]])


def _write_method_comparison_csv(
    path: Path,
    *,
    control_test: OpenSetRunResult,
    proposed_test: OpenSetRunResult,
    lfw_threshold: float,
    frozen_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (METHOD_A, "lfw_1to1_control", lfw_threshold, control_test),
        (METHOD_B, f"open_set_fpir_{PRIMARY_FPIR_TARGET}", frozen_threshold, proposed_test),
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method", "operating_point", "threshold", "gallery_size",
                "fpir", "fnir_rank1", "tpir_rank1", "tpir_rank5",
                "false_reviews_per_1000_non_mated",
                "gallery_enrolment_coverage", "scored_mated_probes", "scored_non_mated_probes",
                "end_to_end_duplicate_detection_rate",
            ]
        )
        for method, point, threshold, run in rows:
            rates = open_set_rates_at_threshold(run.search_results, threshold)
            coverage = open_set_coverage(run)
            detection = open_set_duplicate_detection(run, threshold)
            writer.writerow(
                [
                    method, point, threshold, run.gallery_size,
                    rates["fpir"], rates["fnir_rank1"], rates["tpir_rank1"], rates["tpir_rank5"],
                    rates["false_reviews_per_1000_non_mated"],
                    coverage["gallery_enrolment_coverage"],
                    rates["scored_mated_probes"], rates["scored_non_mated_probes"],
                    detection["end_to_end_duplicate_detection_rate"],
                ]
            )


def render_open_set_report(
    *,
    protocol_summary: Mapping[str, Any],
    development: Mapping[str, Any],
    test: Mapping[str, Any],
    intervals: Mapping[str, Any],
    per_subgroup: Mapping[str, Mapping[str, Any]],
    disparity: Mapping[str, Any],
) -> str:
    """Every number is read from the artefacts, never restated from memory."""
    method_b_test = test["methods"][METHOD_B]
    method_a_test = test["methods"][METHOD_A]
    primary = method_b_test["primary_operating_point"]
    coverage = method_b_test["coverage"]
    control_rates = method_a_test["rates"]

    lines = [
        "# Open-set duplicate-profile evaluation (Experiment 6)",
        "",
        f"Auto-generated by `ACP_arden.py --mode open-set` on {test['created_at']}.",
        "",
        "## Research question",
        "",
        "To what extent can gallery-specific threshold calibration and multi-image profile "
        "enrolment reduce false duplicate-profile reviews while retaining duplicate-detection "
        "performance in an open-set face-verification proof of concept evaluated on real "
        "public benchmark datasets?",
        "",
        "## Protocol",
        "",
        f"- Development identities: {protocol_summary['development']['identities']}; "
        f"held-out test identities: {protocol_summary['test']['identities']} (disjoint).",
        f"- Seed {protocol_summary['seed']}, stratified by the eight BFW subgroups.",
        f"- {protocol_summary['odd_count_rule']}",
        "",
        "## Method A — control: 1:1 threshold reused for 1:N search",
        "",
        f"- Operating threshold {format_number(method_a_test['operating_threshold'], 6)} "
        f"({method_a_test['threshold_role']})",
        f"- FPIR **{format_percentage(control_rates['fpir'])}**, "
        f"TPIR@1 {format_percentage(control_rates['tpir_rank1'])}",
        f"- False reviews per 1,000 non-mated searches: "
        f"{format_number(control_rates['false_reviews_per_1000_non_mated'], 1)}",
        "",
        "## Method B — proposed: three-image template, open-set calibrated",
        "",
        f"- Frozen threshold {format_number(test['operating_threshold'], 6)} at target FPIR "
        f"{test['primary_fpir_target']}",
        f"- FPIR **{format_percentage(primary['fpir'])}** "
        f"(95% CI {format_percentage(intervals['fpir']['lower_95'])} – "
        f"{format_percentage(intervals['fpir']['upper_95'])})",
        f"- TPIR@1 **{format_percentage(primary['tpir_rank1'])}** "
        f"(95% CI {format_percentage(intervals['tpir_rank1']['lower_95'])} – "
        f"{format_percentage(intervals['tpir_rank1']['upper_95'])})",
        f"- TPIR@5 {format_percentage(primary['tpir_rank5'])} "
        f"(CMC rank-1 {format_percentage(primary['cmc_rank1'])}, rank-5 "
        f"{format_percentage(primary['cmc_rank5'])})",
        f"- False reviews per 1,000 non-mated searches: "
        f"{format_number(primary['false_reviews_per_1000_non_mated'], 1)}",
        "",
        "TPIR@5 and TPIR@1 can coincide even though the CMC figures differ, and that is "
        "not an error: TPIR requires the mate to be both within rank *k* **and** above "
        "threshold. The CMC gap shows the mate does surface at ranks 2-5 for a small "
        "number of probes, but those scores fall below the operating threshold, so "
        "returning five candidates instead of one adds no true identification at this "
        "operating point. It would still change the reviewer's workload.",
        "",
        "## Attributing the effect: enrolment or calibration?",
        "",
        "Method B changes two things at once, so the comparison above cannot say which "
        "one helped. Holding the enrolment at three images and reverting only the "
        "threshold separates them:",
        "",
        "| Configuration | FPIR | TPIR@1 |",
        "| --- | --- | --- |",
        f"| A: one image, LFW 1:1 threshold (control) | "
        f"{format_percentage(control_rates['fpir'])} | "
        f"{format_percentage(control_rates['tpir_rank1'])} |",
        f"| B: three images, LFW 1:1 threshold (enrolment change only) | "
        f"{format_percentage(method_b_test['at_lfw_control_threshold']['fpir'])} | "
        f"{format_percentage(method_b_test['at_lfw_control_threshold']['tpir_rank1'])} |",
        f"| B: three images, calibrated threshold (both changes) | "
        f"{format_percentage(primary['fpir'])} | "
        f"{format_percentage(primary['tpir_rank1'])} |",
        "",
        "The middle row isolates the observed enrolment change at the transferred "
        "threshold. Averaged templates are renormalised to unit length, so geometry "
        "alone does not establish whether genuine or impostor scores should increase. "
        "The final row measures recalibration with the same three-image templates. "
        "To assess enrolment at an equal development FPIR target, run the comparison "
        "diagnostics: these calibrate the one-image and three-image methods separately. "
        "The three rows above alone do not establish which enrolment method is better "
        "when both receive appropriate calibration.",
        "",
        "## Operating points and the development-to-test gap",
        "",
        "| Target FPIR | Threshold | Achieved test FPIR | TPIR@1 |",
        "| --- | --- | --- | --- |",
        *[
            f"| {target} | "
            f"{format_number(method_b_test['operating_points'][str(target)].get('threshold'), 4)} | "
            f"{format_percentage(method_b_test['operating_points'][str(target)]['fpir'])} | "
            f"{format_percentage(method_b_test['operating_points'][str(target)]['tpir_rank1'])} |"
            for target in FPIR_TARGETS
        ],
        "",
        "Each achieved test FPIR exceeds the target it was calibrated for. That gap is the "
        "cost of holding the threshold fixed across disjoint identities and is the reason "
        "the policy is frozen on development data before the test partition is scored — "
        "recalibrating on these figures would make them meaningless.",
        "",
        "### Coverage (quoted with every rate above)",
        "",
        f"- Gallery enrolment coverage "
        f"{format_percentage(coverage['gallery_enrolment_coverage'])} "
        f"({coverage['enrolled_gallery_identities']} of "
        f"{coverage['intended_gallery_identities']} identities)",
        f"- Mated probe extraction failure "
        f"{format_percentage(coverage['mated_extraction_failure_rate'])}; non-mated "
        f"{format_percentage(coverage['non_mated_extraction_failure_rate'])}",
        "",
        "## Pre-declared success criteria",
        "",
    ]
    for name, verdict in test["success_criteria"].items():
        if name == "criteria_declared_before_test":
            continue
        actual = verdict.get("actual")
        rendered = "not measurable" if actual is None else format_percentage(actual)
        lines.append(
            f"- {name}: **{verdict['outcome'].replace('_', ' ')}** "
            f"(target {verdict['target']}, achieved {rendered})"
        )

    lines += ["", "## Demographic subgroup performance", "", "| Subgroup | FPIR | TPIR@1 |", "| --- | --- | --- |"]
    for subgroup in sorted(per_subgroup):
        row = per_subgroup[subgroup]
        lines.append(
            f"| {subgroup} | {format_percentage(row['fpir'])} | "
            f"{format_percentage(row['tpir_rank1'])} |"
        )
    ratio = disparity.get("max_to_min_fpir_ratio")
    lines += [
        "",
        f"Absolute FPIR range across subgroups: "
        f"{format_percentage(disparity.get('absolute_fpir_range'))}."
        + (
            f" Max/min ratio {format_number(ratio, 2)}."
            if ratio is not None
            else " Max/min ratio not reported (a subgroup recorded zero FPIR)."
        ),
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in OPEN_SET_LIMITATIONS]
    return "\n".join(lines) + "\n"


def render_open_set_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    """Terminal summary for --mode open-set-summary, read from the artefacts."""
    test_path = output_root / "bfw_open_set_test_metrics.json"
    if not test_path.is_file():
        return (
            "No open-set results found. Run `python ACP_arden.py --mode open-set` first.\n"
            f"That requires the official BFW dataset and {BFW_ROOT_VARIABLE}."
        )
    test = read_json_artifact(test_path)
    intervals = read_json_artifact(output_root / "open_set_confidence_intervals.json").get(
        "intervals", {}
    )
    method_b = test["methods"][METHOD_B]
    method_a = test["methods"][METHOD_A]
    primary = method_b["primary_operating_point"]
    coverage = method_b["coverage"]

    def interval(name: str) -> str:
        row = intervals.get(name)
        if not row:
            return ""
        return (
            f" [95% CI {format_percentage(row['lower_95'])} – "
            f"{format_percentage(row['upper_95'])}]"
        )

    lines = [
        f"{PROGRAMME_TITLE} — open-set results summary",
        "",
        "Experiment 6 — BFW open-set duplicate-profile evaluation (held-out test)",
        f"  Frozen threshold: {format_number(test.get('operating_threshold'), 6)} "
        f"(target FPIR {test.get('primary_fpir_target')}, status {test.get('status')})",
        "",
        "  Method A — control, LFW 1:1 threshold reused for 1:N search",
        f"    FPIR: {format_percentage(method_a['rates']['fpir'])}",
        f"    TPIR@1: {format_percentage(method_a['rates']['tpir_rank1'])}",
        "    NOTE: a control, not a valid open-set operating threshold.",
        "",
        "  Method B — proposed, three-image template with open-set calibration",
        f"    FPIR: {format_percentage(primary['fpir'])}{interval('fpir')}",
        f"    TPIR@1: {format_percentage(primary['tpir_rank1'])}{interval('tpir_rank1')}",
        f"    TPIR@5: {format_percentage(primary['tpir_rank5'])}",
        f"    False reviews per 1,000 non-mated searches: "
        f"{format_number(primary['false_reviews_per_1000_non_mated'], 1)}",
        f"    Gallery enrolment coverage: "
        f"{format_percentage(coverage['gallery_enrolment_coverage'])}",
        f"    Mated extraction failure: "
        f"{format_percentage(coverage['mated_extraction_failure_rate'])}",
        f"    Non-mated extraction failure: "
        f"{format_percentage(coverage['non_mated_extraction_failure_rate'])}",
        "    LIMITATION: every rate above is conditional on the coverage figures printed "
        "with it. An FPIR measured over a small surviving fraction of the protocol is not "
        "comparable with one measured over nearly all of it.",
        "",
        "  Pre-declared success criteria:",
    ]
    for name, verdict in test.get("success_criteria", {}).items():
        if name == "criteria_declared_before_test":
            continue
        lines.append(f"    {name}: {verdict['outcome'].replace('_', ' ')}")
    lines += ["", "Policy: " + POLICY_NOTE, "", "Full write-up: results/aggregate/OPEN_SET_EVALUATION_REPORT.md"]
    return "\n".join(lines)


def report_optional_dataset_status() -> List[str]:
    """Concise skipped-with-reason lines for the optional extensions. Never
    fabricates a result for a dataset or model that is not present."""
    config = EnvironmentConfig.load()
    lines: List[str] = []
    status = pipeline_comparison_status(config)
    # Reported either way. An optional experiment that did not run must say so
    # with its reason, so an absent comparison is never mistaken for one that
    # ran and found nothing.
    if status["comparison_run"]:
        lines.append(
            "Higher-capacity pipeline comparison: configured and verified against pinned "
            "digests. Reported as a complete-pipeline comparison."
        )
    else:
        lines.append(
            f"Higher-capacity pipeline comparison (InsightFace SCRFD + ArcFace "
            f"{ARCFACE_MODEL_PACK}): NOT RUN [{status['status']}] — {status['reason']} "
            f"This is a technical precondition, not a licensing obstacle: the evaluation "
            f"is non-commercial academic research, which the official research terms "
            f"permit. No substitute model was used."
        )
    return lines


# =============================================================================
# 25. Pipeline description and the optional higher-capacity comparison
# =============================================================================
#
# A "pipeline" is the whole chain — detector, preprocessing and embedding model
# — not the embedding model alone. Two pipelines that differ in any of those
# three cannot attribute a difference in results to the embedding, which is why
# the record below names every component and why the comparison is always
# labelled a complete-pipeline comparison.
#
# The comparator is InsightFace SCRFD detection with ArcFace buffalo_l
# recognition. Its pretrained weights are published for non-commercial research,
# which this MSc artefact is, so those terms permit the evaluation. The weights
# are loaded from private local storage by their exact verified path, are never
# redistributed and are never downloaded automatically. No substitute model is
# used in place of the approved one.
##############
# Title: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
# Author: Deng, J., Guo, J., Xue, N. and Zafeiriou, S., Proceedings of the
#         IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
# Date: 2019
# Availability: https://doi.org/10.1109/CVPR.2019.00482
# GitHub: https://github.com/deepinsight/insightface
##############
##############
# Title: Sample and Computation Redistribution for Efficient Face Detection (SCRFD)
# Author: Guo, J., Deng, J., Lattas, A. and Zafeiriou, S., International
#         Conference on Learning Representations (ICLR)
# Date: 2022
# Availability: https://arxiv.org/abs/2105.04714
# GitHub: https://github.com/deepinsight/insightface
##############
##############
# Title: InsightFace: 2D and 3D Face Analysis Project
# Author: InsightFace contributors (deepinsight/insightface)
# Date: 2021 onwards; buffalo_l model pack released 2021
# Availability: https://github.com/deepinsight/insightface
##############
# The detector and recognition weights are external pretrained artefacts,
# created and trained by the InsightFace project. Nothing here trains or
# fine-tunes them, and no weight file is redistributed by this repository.

ARCFACE_MODEL_ROOT_VARIABLE = "FACE_ARCFACE_MODEL_ROOT"
ARCFACE_MODEL_PACK = "buffalo_l"
ARCFACE_DETECTOR_FILENAME = "det_10g.onnx"
ARCFACE_RECOGNITION_FILENAME = "w600k_r50.onnx"

# Digests are pinned in source, never accepted from the command line, so a
# reportable evaluation cannot be pointed at an unverified weight file. They are
# filled in only once the exact approved files have been obtained; until then
# the adapter refuses rather than trusting whatever is on disk.
ARCFACE_DETECTOR_SHA256: Optional[str] = (
    "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
)
ARCFACE_RECOGNITION_SHA256: Optional[str] = (
    "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
)

# Status vocabulary. A licensing status is reserved for the case where the
# official terms do not clearly permit non-commercial academic research; it is
# never used merely because commercial deployment would need separate
# permission, which is irrelevant to this project's purpose.
PIPELINE_STATUS_EVALUATED = "evaluated_non_commercial_academic_research"
PIPELINE_STATUS_NOT_CONFIGURED = "not_run_model_files_not_configured"
PIPELINE_STATUS_SOURCE_UNVERIFIED = "not_run_official_model_source_unverified"
PIPELINE_STATUS_DIGEST_NOT_PINNED = "not_run_model_digest_not_pinned"
PIPELINE_STATUS_DEPENDENCIES_MISSING = "not_run_dependencies_not_installed"
PIPELINE_STATUS_TERMS_UNCLEAR = "not_run_research_terms_not_established"

ARCFACE_LICENCE_NOTE = (
    "InsightFace publishes its pretrained models for non-commercial research use. This "
    "project is an MSc academic research artefact: it is not a commercial service, is not "
    "deployed to real users, makes no commercial decisions, and neither sells, licenses "
    "nor redistributes the pretrained weights. The evaluation is local and non-commercial "
    "and publishes only aggregate metrics, so it falls within those research terms. The "
    "models were created and trained externally by the InsightFace project; no "
    "face-recognition network is trained or fine-tuned here. The MIT licence covering "
    "InsightFace source code does "
    "not automatically extend to every pretrained weight file, and no ownership of the "
    "models, their training data or their weights is claimed here."
)

# Tense follows the real status: the past tense is only correct once held-out
# metrics exist, so the wording is generated rather than fixed.
ARCFACE_USE_STATEMENT_NOT_RUN = (
    "The InsightFace pipeline is intended for evaluation solely within non-commercial MSc "
    "research. The pretrained model files are stored outside the public repository and are "
    "not redistributed. The project publishes only aggregate benchmark results and "
    "provides full attribution to the original model, software and research publications."
)

ARCFACE_USE_STATEMENT_EVALUATED = (
    "The InsightFace pipeline was evaluated solely within non-commercial MSc research. The "
    "pretrained model files were stored outside the public repository and were not "
    "redistributed. The project publishes only aggregate benchmark results and provides "
    "full attribution to the original model, software and research publications."
)


def arcface_use_statement(evaluated: bool) -> str:
    return ARCFACE_USE_STATEMENT_EVALUATED if evaluated else ARCFACE_USE_STATEMENT_NOT_RUN


class PipelineUnavailableError(RuntimeError):
    """Raised when an optional pipeline cannot be configured, verified or loaded."""


class PipelineComparisonError(RuntimeError):
    """Raised when a comparison would be recorded without real held-out metrics."""


# SCRFD rescales the input to a fixed square before detection. InsightFace
# defaults to 640, which detects nothing on BFW's face crops: their longest
# side is around one hundred pixels, so a 640 canvas leaves the face far below
# the smallest anchor. A 320 canvas detects reliably. This is a preprocessing
# scale, not a decision threshold; the detection threshold stays at the
# published default so coverage is not inflated by lowering the bar.
ARCFACE_DETECTION_INPUT_SIZE = 320
ARCFACE_DETECTION_THRESHOLD = 0.5


class ArcFaceDetector:
    """Adapter exposing InsightFace detection through the project's protocol.

    Requires exactly one face, matching the rule the primary pipeline applies,
    so the two pipelines remain comparable."""

    def __init__(self, model: Any, model_sha256: str):
        self._model = model
        self.model_sha256 = model_sha256
        self._last_landmarks: Optional[np.ndarray] = None

    def detect_single_face(self, bgr: np.ndarray) -> np.ndarray:
        boxes, landmarks = self._model.detect(bgr, max_num=0, metric="default")
        count = 0 if boxes is None else len(boxes)
        if count != 1:
            self._last_landmarks = None
            raise FaceCountError(count)
        box = np.asarray(boxes[0], dtype=np.float64)  # type: ignore[index]
        self._last_landmarks = (
            np.asarray(landmarks[0], dtype=np.float64) if landmarks is not None else None
        )
        # Fifteen columns matching the YuNet row shape: box, five landmarks,
        # then the detection score last.
        row = np.zeros(15, dtype=np.float64)
        row[0], row[1] = box[0], box[1]
        row[2], row[3] = box[2] - box[0], box[3] - box[1]
        if self._last_landmarks is not None:
            row[4:14] = self._last_landmarks.reshape(-1)[:10]
        row[14] = float(box[4]) if box.shape[0] > 4 else 0.0
        return row

    def last_landmarks(self) -> Optional[np.ndarray]:
        return self._last_landmarks


class ArcFaceEmbedder:
    """Align by the detected landmarks, then embed with ArcFace."""

    def __init__(
        self, model: Any, detector: ArcFaceDetector, model_sha256: str, dimensions: int = 512
    ):
        self._model = model
        self._detector = detector
        self.model_sha256 = model_sha256
        self._dimensions = dimensions

    @staticmethod
    def _landmarks_from_row(face_row: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """The five landmarks a fifteen-column detector row carries, or None.

        Columns 4 to 13 hold them as x,y pairs. An all-zero block means the
        detector supplied none and the caller should look elsewhere."""
        if face_row is None:
            return None
        row = np.asarray(face_row, dtype=np.float64).reshape(-1)
        if row.shape[0] < 14:
            return None
        landmarks = row[4:14].reshape(5, 2)
        return landmarks if np.any(landmarks) else None

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        from insightface.utils import face_align  # type: ignore[import-not-found]

        # Read the landmarks from the row first, so the embedder works with any
        # detector that fills the fifteen columns rather than only the one it
        # was constructed with. The detector's own record is the fallback for
        # rows that carry no landmarks.
        landmarks = self._landmarks_from_row(face_row)
        if landmarks is None:
            landmarks = self._detector.last_landmarks()
        if landmarks is None:
            raise SimilarityError("No landmarks were available for alignment.")
        aligned = face_align.norm_crop(bgr, landmark=landmarks, image_size=112)
        embedding = np.asarray(self._model.get_feat(aligned), dtype=np.float64).reshape(-1)
        if embedding.shape[0] != self._dimensions:
            # Refuse an unexpected width rather than compare templates drawn
            # from two different spaces.
            raise SimilarityError(
                f"Expected {self._dimensions}-dimensional embeddings, got {embedding.shape[0]}."
            )
        return embedding


@dataclass(frozen=True)
class PipelineDescription:
    """Everything that must match before two runs are comparable. Published in
    every artefact so a reader can tell which chain produced a number."""

    pipeline_name: str
    detector_name: str
    embedding_model_name: str
    embedding_dimensions: int
    preprocessing_revision: str
    model_sha256: Dict[str, str]
    licence_note: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "detector_name": self.detector_name,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimensions": self.embedding_dimensions,
            "preprocessing_revision": self.preprocessing_revision,
            "model_sha256": dict(self.model_sha256),
            "licence_note": self.licence_note,
            "comparison_scope": (
                "Complete pipeline: detection, preprocessing and embedding. A difference "
                "between pipelines cannot be attributed to the embedding model alone."
            ),
        }


def pipeline_identity_fields(
    detector: Any = None,
    embedder: Any = None,
    description: Optional[PipelineDescription] = None,
) -> Dict[str, Any]:
    """The model identity an artefact records: name, preprocessing and digests.

    ``description`` must be supplied whenever the models are not the baseline
    pair. Without it the fields name YuNet and SFace unconditionally, which
    stamps a comparison run with the wrong model names under the right digests
    and leaves the artefact contradicting itself."""
    if description is not None:
        return {
            "model_version": description.pipeline_name,
            "preprocessing_revision": description.preprocessing_revision,
            "model_sha256": dict(description.model_sha256),
        }
    return {
        "model_version": MODEL_VERSION,
        "preprocessing_revision": PREPROCESSING_REVISION,
        # Record the verified model digests. The synthetic stand-ins used by
        # the self-tests carry none; the real wrappers always do, and it is
        # their verified value that reaches the artefact.
        "model_sha256": {
            "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
            "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
        },
    }


def primary_pipeline_description(detector: Any = None, embedder: Any = None) -> PipelineDescription:
    """The mandatory YuNet + SFace pipeline."""
    return PipelineDescription(
        pipeline_name=MODEL_VERSION,
        detector_name="OpenCV YuNet 2023mar",
        embedding_model_name="OpenCV SFace 2021dec",
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        preprocessing_revision=PREPROCESSING_REVISION,
        model_sha256={
            "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
            "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
        },
        licence_note=(
            "OpenCV Zoo release: YuNet weights MIT, SFace weights Apache-2.0. "
            "Both are redistributable for research use."
        ),
    )


def arcface_preconditions(config: Optional[EnvironmentConfig] = None) -> Dict[str, Any]:
    """Diagnose the comparator in a fixed order and name the first real blocker.

    Every status returned is technical. Commercial-use restrictions are not a
    blocker here: this artefact is non-commercial academic research, which the
    official research terms permit."""
    config = config or EnvironmentConfig.load()
    checks: Dict[str, bool] = {}

    # Four separate conditions must all hold before the optional comparison can
    # run: the libraries are installed, a model directory is configured, both
    # weight files are present, and their digests are pinned in source.
    # Recording each one individually lets the report state exactly which is
    # missing rather than only that the comparison did not run.
    missing_dependencies = [
        name for name in ("onnxruntime", "insightface")
        if importlib.util.find_spec(name) is None
    ]
    checks["dependencies_installed"] = not missing_dependencies
    checks["model_root_configured"] = config.arcface_model_root is not None

    root = Path(config.arcface_model_root) if config.arcface_model_root else None
    detector_path = root / ARCFACE_DETECTOR_FILENAME if root else None
    recognition_path = root / ARCFACE_RECOGNITION_FILENAME if root else None
    checks["model_files_present"] = bool(
        detector_path and recognition_path
        and detector_path.is_file() and recognition_path.is_file()
    )
    checks["digests_pinned"] = (
        ARCFACE_DETECTOR_SHA256 is not None and ARCFACE_RECOGNITION_SHA256 is not None
    )
    checks["research_terms_established"] = True

    if not checks["research_terms_established"]:
        status, reason = PIPELINE_STATUS_TERMS_UNCLEAR, (
            "The official terms do not clearly permit non-commercial academic research."
        )
    elif not checks["model_root_configured"]:
        status, reason = PIPELINE_STATUS_NOT_CONFIGURED, (
            f"{ARCFACE_MODEL_ROOT_VARIABLE} is not set, so no model files are available. "
            f"Obtain the official {ARCFACE_MODEL_PACK} pack from the InsightFace project and "
            f"store it in private local research storage; nothing is downloaded automatically."
        )
    elif not checks["model_files_present"]:
        status, reason = PIPELINE_STATUS_SOURCE_UNVERIFIED, (
            f"{ARCFACE_MODEL_ROOT_VARIABLE} is set but {ARCFACE_DETECTOR_FILENAME} and "
            f"{ARCFACE_RECOGNITION_FILENAME} were not both found, so the model source "
            f"cannot be verified."
        )
    elif not checks["digests_pinned"]:
        status, reason = PIPELINE_STATUS_DIGEST_NOT_PINNED, (
            "The model files are present but their SHA-256 digests are not pinned in "
            "source. Compute the digests of the exact approved files and set "
            "ARCFACE_DETECTOR_SHA256 and ARCFACE_RECOGNITION_SHA256; digests are never "
            "accepted as command-line arguments for a reportable evaluation."
        )
    elif not checks["dependencies_installed"]:
        status, reason = PIPELINE_STATUS_DEPENDENCIES_MISSING, (
            f"Required package(s) not installed: {', '.join(missing_dependencies)}. "
            f"Install them from requirements-comparison.txt."
        )
    else:
        status, reason = PIPELINE_STATUS_EVALUATED, ""

    return {
        "status": status,
        "reason": reason,
        "checks": checks,
        "missing_dependencies": missing_dependencies,
        "ready": status == PIPELINE_STATUS_EVALUATED,
    }


def arcface_pipeline_description(config: Optional[EnvironmentConfig] = None) -> PipelineDescription:
    """Describe the comparator, or refuse with the precise technical blocker."""
    config = config or EnvironmentConfig.load()
    preconditions = arcface_preconditions(config)
    if not preconditions["ready"]:
        raise PipelineUnavailableError(
            f"[{preconditions['status']}] {preconditions['reason']}"
        )
    root = Path(config.arcface_model_root)  # type: ignore[arg-type]
    return PipelineDescription(
        pipeline_name=f"insightface-scrfd-arcface-{ARCFACE_MODEL_PACK}",
        detector_name="InsightFace SCRFD (det_10g)",
        embedding_model_name=f"InsightFace ArcFace {ARCFACE_MODEL_PACK} (w600k_r50)",
        embedding_dimensions=512,
        preprocessing_revision="insightface-arcface-112x112-v1",
        model_sha256={
            "detector": verify_model_file(
                root / ARCFACE_DETECTOR_FILENAME, str(ARCFACE_DETECTOR_SHA256)
            ),
            "recognition": verify_model_file(
                root / ARCFACE_RECOGNITION_FILENAME, str(ARCFACE_RECOGNITION_SHA256)
            ),
        },
        licence_note=ARCFACE_LICENCE_NOTE,
    )


SCRFD_WARNING_EXPLANATION = (
    "Loading SCRFD + ArcFace.\n"
    "\n"
    "The SCRFD model uses dynamic output dimensions. ONNX Runtime may print\n"
    "technical output-shape warnings while adapting the model to the selected\n"
    "image size. These warnings are expected in this verified configuration and\n"
    "do not indicate that the evaluation has failed.\n"
    "\n"
    "They are withheld inside this detector's own ONNX Runtime session only.\n"
    "Errors are not withheld: a model-loading failure, an invalid tensor, a\n"
    "provider failure or a missing output is still reported.\n"
    "\n"
    "The programme will stop if the returned detector outputs are invalid."
)


def _quiet_scrfd_detector(model_path: Path) -> Any:
    """Load SCRFD with its own ONNX Runtime session at ERROR severity.

    The dynamic SCRFD graph makes ONNX Runtime print one output-shape warning
    per output on every detection call, which buries the programme's own
    progress messages. The severity is raised on *this session only*, never
    globally and never for the recognition model, so warnings from any other
    session still appear.

    ERROR severity is deliberate: model-loading failures, invalid tensors,
    provider failures and missing outputs are errors and continue to surface.
    Only the informational shape warning is withheld, and the detector's actual
    outputs are validated immediately afterwards regardless."""
    import onnxruntime as ort
    from insightface.model_zoo.scrfd import SCRFD  # type: ignore[import-not-found]

    options = ort.SessionOptions()
    # 0 verbose, 1 info, 2 warning, 3 error, 4 fatal.
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return SCRFD(model_file=str(model_path), session=session)


def validate_scrfd_outputs(detector_model: Any) -> None:
    """Check that the loaded detector returns usable geometry.

    The output-shape warnings are only acceptable because this runs afterwards:
    a detector that warns and then returns malformed boxes must stop the
    evaluation rather than be explained away. A synthetic image is used, so no
    benchmark photograph is needed to validate the load."""
    probe = np.zeros(
        (ARCFACE_DETECTION_INPUT_SIZE, ARCFACE_DETECTION_INPUT_SIZE, 3), dtype=np.uint8
    )
    try:
        outputs = detector_model.detect(probe, max_num=0, metric="default")
    except Exception as exc:  # pragma: no cover - depends on the optional model
        raise PipelineUnavailableError(
            f"[{PIPELINE_STATUS_DEPENDENCIES_MISSING}] the SCRFD detector failed to "
            f"produce outputs during validation: {exc}"
        ) from exc

    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise PipelineUnavailableError(
            f"[{PIPELINE_STATUS_DEPENDENCIES_MISSING}] the SCRFD detector returned "
            f"an unexpected number of outputs."
        )
    boxes, landmarks = outputs
    # An empty detection on a blank image is the correct outcome; only the
    # shape and value contract is being checked here.
    if boxes is not None and len(boxes):
        array = np.asarray(boxes, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] < 4:
            raise PipelineUnavailableError(
                f"[{PIPELINE_STATUS_DEPENDENCIES_MISSING}] the SCRFD detector returned "
                f"bounding boxes of an unexpected rank or width."
            )
        if not np.isfinite(array).all():
            raise PipelineUnavailableError(
                f"[{PIPELINE_STATUS_DEPENDENCIES_MISSING}] the SCRFD detector returned "
                f"non-finite bounding-box values."
            )
        widths = array[:, 2] - array[:, 0]
        heights = array[:, 3] - array[:, 1]
        if float(np.min(widths)) <= 0.0 or float(np.min(heights)) <= 0.0:
            raise PipelineUnavailableError(
                f"[{PIPELINE_STATUS_DEPENDENCIES_MISSING}] the SCRFD detector returned "
                f"a bounding box without positive width and height."
            )
    if landmarks is not None and len(landmarks):
        points = np.asarray(landmarks, dtype=np.float64)
        if points.ndim != 3 or points.shape[1:] != (5, 2):
            raise PipelineUnavailableError(
                f"[{PIPELINE_STATUS_DEPENDENCIES_MISSING}] the SCRFD detector returned "
                f"landmarks of an unexpected shape."
            )
        if not np.isfinite(points).all():
            raise PipelineUnavailableError(
                f"[{PIPELINE_STATUS_DEPENDENCIES_MISSING}] the SCRFD detector returned "
                f"non-finite landmark values."
            )
    announce("SCRFD output validation passed.")


def load_arcface_pipeline(config: Optional[EnvironmentConfig] = None):
    """Load the comparator through InsightFace's documented API.

    ``download=False`` and an explicit local root are required: the pretrained
    weights must come from private research storage that the researcher
    populated, never from an automatic fetch."""
    config = config or EnvironmentConfig.load()
    description = arcface_pipeline_description(config)
    try:
        # Optional dependency, absent by default; the ImportError below is the
        # supported path when the comparison is not being run.
        from insightface.model_zoo import get_model  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PipelineUnavailableError(
            f"[{PIPELINE_STATUS_DEPENDENCIES_MISSING}] insightface is not installed."
        ) from exc

    # Each ONNX file is loaded by its exact verified path. FaceAnalysis is
    # deliberately not used: it resolves models through a cache directory and
    # fetches the pack from the network when that directory is empty, which
    # would both download automatically and evaluate files other than the
    # pinned ones.
    # Explained before loading, because ONNX Runtime prints output-shape
    # warnings while adapting the dynamic SCRFD graph to the selected input
    # size. Those are expected here and are not evaluation failures.
    print("")
    print(SCRFD_WARNING_EXPLANATION)
    print("")

    root = Path(config.arcface_model_root)  # type: ignore[arg-type]
    detector_model = cast(Any, _quiet_scrfd_detector(root / ARCFACE_DETECTOR_FILENAME))
    detector_model.prepare(
        ctx_id=-1,
        input_size=(ARCFACE_DETECTION_INPUT_SIZE, ARCFACE_DETECTION_INPUT_SIZE),
        det_thresh=ARCFACE_DETECTION_THRESHOLD,
    )
    recognition_model = cast(Any, get_model(str(root / ARCFACE_RECOGNITION_FILENAME)))
    recognition_model.prepare(ctx_id=-1)

    detector = ArcFaceDetector(detector_model, description.model_sha256["detector"])
    # The warning is tolerated only because the detector's actual outputs are
    # checked. A malformed box, a non-finite value or a missing output stops
    # the run rather than being explained away.
    validate_scrfd_outputs(detector_model)
    embedder = ArcFaceEmbedder(
        recognition_model, detector, description.model_sha256["recognition"],
        dimensions=description.embedding_dimensions,
    )
    return (detector, embedder), description


def pipeline_comparison_status(
    config: Optional[EnvironmentConfig] = None,
) -> Dict[str, Any]:
    """Whether the comparison can run, and if not, precisely why."""
    preconditions = arcface_preconditions(config)
    # Not-run is a recorded outcome with a stated reason, never a silent skip:
    # a reader must be able to tell an unavailable comparison from one that ran
    # and found no difference. The reason is redacted, because the usual cause
    # is a missing file in private research storage.
    if not preconditions["ready"]:
        return {
            "comparison_run": False,
            "status": preconditions["status"],
            "reason": redact_private_paths(preconditions["reason"]),
            "preconditions": preconditions["checks"],
            "substitute_model_used": False,
            "licence_note": ARCFACE_LICENCE_NOTE,
            "use_statement": arcface_use_statement(False),
        }
    description = arcface_pipeline_description(config)
    return {
        "comparison_run": True,
        "status": PIPELINE_STATUS_EVALUATED,
        "pipeline": description.as_dict(),
        "preconditions": preconditions["checks"],
        "substitute_model_used": False,
        "licence_note": ARCFACE_LICENCE_NOTE,
        "use_statement": arcface_use_statement(True),
    }


def _model_file_sizes(root: Optional[Path]) -> Dict[str, Any]:
    """Byte and megabyte sizes of the evaluated weight files."""
    sizes: Dict[str, Any] = {}
    if root is None:
        return sizes
    for name in (ARCFACE_DETECTOR_FILENAME, ARCFACE_RECOGNITION_FILENAME):
        path = Path(root) / name
        if path.is_file():
            size = path.stat().st_size
            sizes[name] = {"bytes": size, "megabytes": round(size / (1024 * 1024), 2)}
    return sizes


def _primary_model_file_sizes(model_root: Optional[Path]) -> Dict[str, Any]:
    sizes: Dict[str, Any] = {}
    if model_root is None:
        return sizes
    for name in (YUNET_FILENAME, SFACE_FILENAME):
        path = Path(model_root) / name
        if path.is_file():
            size = path.stat().st_size
            sizes[name] = {"bytes": size, "megabytes": round(size / (1024 * 1024), 2)}
    return sizes


def write_pipeline_performance_csv(
    path: Path, *, payload: Mapping[str, Any]
) -> None:
    """Performance comparison table, written only from computed metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "pipeline", "detector", "recognition_model", "embedding_dimensions", "threshold",
        "fpir", "tpir_rank1", "tpir_rank5", "fnir_rank1", "fnir_rank5",
        "false_reviews_per_1000_non_mated", "end_to_end_duplicate_detection_rate",
        "gallery_enrolment_coverage", "mated_extraction_coverage",
        "non_mated_extraction_coverage",
        "embedding_latency_mean_ms", "embedding_latency_p95_ms",
        "complete_pipeline_latency_mean_ms", "complete_pipeline_latency_p95_ms",
        "top1_search_time_mean_ms", "top1_search_time_p95_ms",
        "detector_file_size_mb", "recognition_file_size_mb", "status",
    ]
    held_out = payload.get("held_out_metrics") or {}
    descriptions = {
        payload["primary_pipeline"]["pipeline_name"]: payload["primary_pipeline"],
    }
    if payload.get("comparison_pipeline"):
        descriptions[payload["comparison_pipeline"]["pipeline_name"]] = (
            payload["comparison_pipeline"]
        )
    sizes = payload.get("model_file_sizes", {})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        if not held_out:
            for name, description in descriptions.items():
                writer.writerow(
                    [name, description["detector_name"], description["embedding_model_name"],
                     description["embedding_dimensions"]] + [""] * 19 + [payload["status"]]
                )
            return
        for name, metrics in held_out.items():
            description = descriptions.get(name, {})
            rates, coverage = metrics["rates"], metrics["coverage"]
            group = "primary" if name == payload["primary_pipeline"]["pipeline_name"] else "comparison"
            file_sizes = list((sizes.get(group) or {}).values())
            writer.writerow([
                name, description.get("detector_name", ""),
                description.get("embedding_model_name", ""),
                description.get("embedding_dimensions", ""),
                metrics["development_threshold"],
                rates["fpir"], rates["tpir_rank1"], rates["tpir_rank5"],
                rates["fnir_rank1"], rates["fnir_rank5"],
                rates["false_reviews_per_1000_non_mated"],
                metrics.get("end_to_end_duplicate_detection_rate"),
                coverage["gallery_enrolment_coverage"],
                1.0 - coverage["mated_extraction_failure_rate"],
                1.0 - coverage["non_mated_extraction_failure_rate"],
                # Embedding timing, never gallery-search timing.
                coverage.get("embedding_latency_mean_ms"),
                coverage.get("embedding_latency_p95_ms"),
                coverage.get("complete_pipeline_latency_mean_ms"),
                coverage.get("complete_pipeline_latency_p95_ms"),
                coverage.get("top1_search_time_mean_ms"),
                coverage.get("top1_search_time_p95_ms"),
                file_sizes[0]["megabytes"] if len(file_sizes) > 0 else "",
                file_sizes[1]["megabytes"] if len(file_sizes) > 1 else "",
                payload["status"],
            ])


def write_pipeline_subgroup_csv(path: Path, *, payload: Mapping[str, Any]) -> None:
    """One row per pipeline and subgroup once the comparison has been run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    held_out = payload.get("held_out_metrics") or {}
    columns = ["pipeline"] + list(_SUBGROUP_CSV_COLUMNS) + ["status"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        if not held_out:
            # The comparison pipeline still gets its rows, left blank and
            # carrying the status. A reader sees that it was not evaluated
            # rather than finding it silently absent from the table.
            for subgroup in BFW_SUBGROUPS:
                writer.writerow(
                    [f"insightface-scrfd-arcface-{ARCFACE_MODEL_PACK}", subgroup]
                    + [""] * (len(_SUBGROUP_CSV_COLUMNS) - 1) + [payload["status"]]
                )
            return
        for name, metrics in held_out.items():
            for subgroup in sorted(metrics.get("subgroups", {})):
                row = metrics["subgroups"][subgroup]
                writer.writerow(
                    [name, subgroup]
                    + [row.get(c, "") for c in _SUBGROUP_CSV_COLUMNS[1:]]
                    + [payload["status"]]
                )


def write_pipeline_comparison_csv(
    path: Path, *, primary: PipelineDescription, config: Optional[EnvironmentConfig] = None
) -> None:
    """Emit the comparison table. When the comparator is unavailable the file
    still records the primary pipeline and states plainly that no second
    pipeline was evaluated, so absence is visible rather than inferred."""
    status = pipeline_comparison_status(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["pipeline_name", "detector_name", "embedding_model_name", "embedding_dimensions",
             "preprocessing_revision", "evaluated", "note"]
        )
        writer.writerow(
            [primary.pipeline_name, primary.detector_name, primary.embedding_model_name,
             primary.embedding_dimensions, primary.preprocessing_revision, "yes",
             "Mandatory primary pipeline."]
        )
        if status["comparison_run"]:
            other = status["pipeline"]
            writer.writerow(
                [other["pipeline_name"], other["detector_name"], other["embedding_model_name"],
                 other["embedding_dimensions"], other["preprocessing_revision"], "yes",
                 other["comparison_scope"]]
            )
        else:
            writer.writerow(
                ["insightface-scrfd-arcface-buffalo_l", "InsightFace SCRFD / RetinaFace",
                 "InsightFace ArcFace buffalo_l", 512, "insightface-arcface-112x112-v1",
                 "no", status["reason"]]
            )


# =============================================================================
# 26. Experiment 7 — interpretable machine-learning review classifier
# =============================================================================
#
# Asks whether a small classifier over ranking context can refer fewer innocent
# registrations for human review than a single similarity threshold, without
# losing duplicate detection.
#
# Logistic regression is the primary model: its coefficients are directly
# readable, it is reproducible under a fixed seed, it suits a nine-feature
# problem, and it emits a probability that can be calibrated to a target FPIR.
# No face-recognition model is trained or fine-tuned anywhere in this section.
#
# The output opens a human-review case and nothing else. It is not evidence of
# duplication, fraud or misuse.
##############
# Title: The Regression Analysis of Binary Sequences
# Author: Cox, D.R., Journal of the Royal Statistical Society, Series B, 20(2),
#         pp. 215-242
# Date: 1958
# Availability: https://doi.org/10.1111/j.2517-6161.1958.tb00292.x
##############
##############
# Title: Scikit-learn: Machine Learning in Python (LogisticRegression and
#        StandardScaler APIs)
# Author: Pedregosa, F. et al., Journal of Machine Learning Research, 12,
#         pp. 2825-2830
# Date: 2011
# Availability: https://jmlr.org/papers/v12/pedregosa11a.html
##############

ML_REVIEW_STATUS_FROZEN = "ml_review_frozen"
ML_REVIEW_METHOD = "logistic_regression_review_classifier"

# Fixed feature order. Serialised with the model so that a stored coefficient
# can never be applied to a differently-ordered vector.
ML_REVIEW_FEATURES = (
    "top1_similarity",
    "top2_similarity",
    "top1_top2_margin",
    "top5_similarity_mean",
    "top5_similarity_stdev",
    "top1_gallery_image_count",
    "gallery_size",
    "probe_detection_confidence",
    "probe_face_area_ratio",
)

# Declared as typed constants rather than read back out of the dictionary
# below. scikit-learn types solver and class_weight as literals, so recovering
# them with str() would erase the literal and defeat the type checker. The
# dictionary is built from these constants, keeping one source of truth for
# both the estimator call and the published provenance.
ML_REVIEW_SOLVER: Literal["lbfgs"] = "lbfgs"
ML_REVIEW_CLASS_WEIGHT: Literal["balanced"] = "balanced"
# Ridge penalty expressed as l1_ratio=0; the older penalty="l2" spelling is
# deprecated from scikit-learn 1.8 and removed at 1.10.
ML_REVIEW_L1_RATIO = 0.0
ML_REVIEW_C = 1.0
ML_REVIEW_MAX_ITER = 1000
ML_REVIEW_FIT_INTERCEPT = True

ML_REVIEW_HYPERPARAMETERS = {
    "estimator": "sklearn.linear_model.LogisticRegression",
    "l1_ratio": ML_REVIEW_L1_RATIO,
    "C": ML_REVIEW_C,
    "solver": ML_REVIEW_SOLVER,
    "max_iter": ML_REVIEW_MAX_ITER,
    "class_weight": ML_REVIEW_CLASS_WEIGHT,
    "random_state": DEFAULT_RANDOM_SEED,
    "fit_intercept": ML_REVIEW_FIT_INTERCEPT,
}

# Proportion of development identities used for fitting; the remainder
# calibrates the probability threshold.
ML_REVIEW_TRAINING_FRACTION = 0.70

ML_REVIEW_SUCCESS_CRITERIA = {
    "fpir_max": 0.01,
    "target_fpir": PRIMARY_FPIR_TARGET,
    "tpir_rank1_min": 0.90,
    "end_to_end_detection_tolerance": 0.02,
    "coverage_min": 0.90,
}


class MlReviewError(RuntimeError):
    """Raised when the review classifier cannot be built, frozen or applied."""


@dataclass(frozen=True)
class ReviewFeatureRow:
    """One search rendered as classifier input plus its supervision label.

    ``correct_rank`` and ``correct_similarity`` are evaluation metadata drawn
    from benchmark ground truth. They are never placed in the feature vector."""

    sample_id: str
    identity_hash: str
    subgroup: str
    role: str
    features: Dict[str, float]
    # The label represents a correct rank-one referral.
    label: int
    correct_rank: Optional[int] = None
    correct_similarity: Optional[float] = None


@dataclass(frozen=True)
class ReviewIdentityOutcome:
    """Complete intended protocol outcome for one identity.

    Retained so the end-to-end denominator keeps probes that never produced a
    score; filtering to scored rows would silently make it conditional."""

    identity_hash: str
    subgroup: str
    intended_mated: int
    scored_mated: int
    intended_non_mated: int
    scored_non_mated: int
    mated_extraction_failures: int
    non_mated_extraction_failures: int
    gallery_reference_unavailable: int


def split_development_identities_for_classifier(
    protocol: OpenSetProtocol,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    training_fraction: float = ML_REVIEW_TRAINING_FRACTION,
) -> Tuple[List[str], List[str]]:
    """Divide development identities into training and calibration groups.

    Split by identity, never by image, so no photograph of one person can
    appear on both sides. The held-out test identities are not touched."""
    development = sorted({e.identity for e in protocol.partition("development")})
    subgroup_of = {e.identity: e.subgroup for e in protocol.partition("development")}

    training: List[str] = []
    calibration: List[str] = []
    by_subgroup: Dict[str, List[str]] = {}
    for identity in development:
        by_subgroup.setdefault(subgroup_of[identity], []).append(identity)

    # Stratified so every subgroup is represented in both groups.
    for subgroup in sorted(by_subgroup):
        members = sorted(by_subgroup[subgroup])
        rng = random.Random(f"{seed}:ml-review:{subgroup}")
        rng.shuffle(members)
        cut = round(len(members) * training_fraction)
        cut = min(max(cut, 1), len(members) - 1) if len(members) > 1 else len(members)
        training.extend(members[:cut])
        calibration.extend(members[cut:])

    training, calibration = sorted(training), sorted(calibration)
    if set(training) & set(calibration):
        raise MlReviewError("Classifier training and calibration identities overlap.")
    if not training or not calibration:
        raise MlReviewError("Too few development identities to form both classifier groups.")
    return training, calibration


def build_review_feature_rows(
    results: Sequence[OpenSetSearchResult],
    *,
    identities: Optional[Set[str]] = None,
    identity_of_sample: Optional[Mapping[str, str]] = None,
) -> Tuple[List[ReviewFeatureRow], Dict[str, int]]:
    """Turn scored searches into labelled feature rows.

    A row is emitted only when every feature is available; records with a
    missing feature are counted and excluded rather than imputed, because a
    silently invented value would propagate into the coefficients."""
    rows: List[ReviewFeatureRow] = []
    excluded = {"unscored": 0, "missing_feature": 0, "outside_partition": 0}

    for result in results:
        if identities is not None and identity_of_sample is not None:
            private = identity_of_sample.get(result.sample_id)
            if private is None or private not in identities:
                excluded["outside_partition"] += 1
                continue
        if result.failure_code is not None or result.top_similarity is None:
            # An extraction failure is a coverage outcome, never a negative
            # decision, so it must not become a training example.
            excluded["unscored"] += 1
            continue

        top1 = float(result.top_similarity)
        top2 = result.top2_similarity
        values = {
            "top1_similarity": top1,
            "top2_similarity": None if top2 is None else float(top2),
            "top1_top2_margin": None if top2 is None else top1 - float(top2),
            "top5_similarity_mean": result.top5_similarity_mean,
            "top5_similarity_stdev": result.top5_similarity_stdev,
            "top1_gallery_image_count": result.top1_gallery_image_count,
            "gallery_size": result.gallery_size,
            "probe_detection_confidence": result.probe_detection_confidence,
            "probe_face_area_ratio": result.probe_face_area_ratio,
        }
        if any(values[name] is None for name in ML_REVIEW_FEATURES):
            excluded["missing_feature"] += 1
            continue

        # Ground truth is used for supervision, never as an input feature.
        # A referral to the wrong identity is not a true identification, so a
        # mated probe is positive only when its own identity ranks first.
        positive = result.role == "mated_probe" and result.correct_rank == 1
        rows.append(
            ReviewFeatureRow(
                sample_id=result.sample_id,
                identity_hash=result.identity_hash,
                subgroup=result.subgroup,
                role=result.role,
                features={name: float(values[name]) for name in ML_REVIEW_FEATURES},
                label=1 if positive else 0,
                correct_rank=result.correct_rank,
                correct_similarity=result.correct_similarity,
            )
        )
    return rows, excluded


def _feature_matrix(rows: Sequence[ReviewFeatureRow], *, features: Sequence[str] = ML_REVIEW_FEATURES) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.array([[r.features[name] for name in features] for r in rows], dtype=float).reshape(len(rows), len(features))
    labels = np.array([r.label for r in rows], dtype=int)
    return matrix, labels


@dataclass(frozen=True)
class ReviewClassifier:
    """A fitted classifier stored as plain numbers.

    Coefficients, intercept and scaler parameters are published as JSON rather
    than pickled: a pickle would be unsafe to load and opaque to a reader."""

    feature_order: Tuple[str, ...]
    coefficients: List[float]
    intercept: float
    scaler_mean: List[float]
    scaler_scale: List[float]

    def probabilities(self, matrix: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.scaler_mean, dtype=float)
        scale = np.asarray(self.scaler_scale, dtype=float)
        standardised = (matrix - mean) / scale
        logits = standardised @ np.asarray(self.coefficients, dtype=float) + self.intercept
        # Logistic link, evaluated in a form that does not overflow for large
        # negative logits.
        return np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": ML_REVIEW_METHOD,
            "feature_order": list(self.feature_order),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "hyperparameters": dict(ML_REVIEW_HYPERPARAMETERS),
            "class_imbalance_handling": "class_weight='balanced', declared before fitting",
            "serialisation": "plain JSON numerics; no pickle is written or read",
        }


# Fitted on the development identities only, and on search-derived features
# alone. No demographic attribute is a feature, and no held-out identity is seen
# during fitting or during the probability-threshold calibration that follows.
def fit_review_classifier(rows: Sequence[ReviewFeatureRow], *, features: Sequence[str] = ML_REVIEW_FEATURES) -> ReviewClassifier:
    """Fit the logistic regression on training identities only."""
    if not rows:
        raise MlReviewError("No training rows available for the review classifier.")
    labels_present = {r.label for r in rows}
    if labels_present != {0, 1}:
        raise MlReviewError(
            f"Training rows carry labels {sorted(labels_present)}; both classes are required."
        )
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - pinned dependency
        raise MlReviewError(
            "scikit-learn is required for the review classifier. Install it from "
            "requirements.txt before running --mode ml-review."
        ) from exc

    if not features or len(set(features)) != len(features) or not set(features) <= set(ML_REVIEW_FEATURES):
        raise MlReviewError("Classifier feature set must be a nonempty unique subset of the declared features.")
    matrix, labels = _feature_matrix(rows, features=features)
    # Fit the scaler on training identities only; calibration and test rows are
    # transformed with these same parameters.
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        l1_ratio=ML_REVIEW_L1_RATIO,
        C=ML_REVIEW_C,
        solver=ML_REVIEW_SOLVER,
        max_iter=ML_REVIEW_MAX_ITER,
        class_weight=ML_REVIEW_CLASS_WEIGHT,
        random_state=DEFAULT_RANDOM_SEED,
        fit_intercept=ML_REVIEW_FIT_INTERCEPT,
    ).fit(scaler.transform(matrix), labels)

    scale = np.asarray(scaler.scale_, dtype=float)
    # A constant feature yields a zero scale; guard so division stays defined.
    scale = np.where(scale == 0.0, 1.0, scale)
    return ReviewClassifier(
        feature_order=tuple(features),
        coefficients=[float(c) for c in np.asarray(model.coef_).ravel()],
        intercept=float(np.asarray(model.intercept_).ravel()[0]),
        scaler_mean=[float(m) for m in np.asarray(scaler.mean_, dtype=float)],
        scaler_scale=[float(s) for s in scale],
    )


def build_review_identity_outcomes(
    results: Sequence[OpenSetSearchResult],
) -> Dict[str, ReviewIdentityOutcome]:
    """Aggregate the complete intended outcome per identity.

    Retain failed probes in the end-to-end denominator; excluding them would
    turn an end-to-end rate into a conditional one."""
    # Group every probe outcome under the person it belongs to. The bootstrap
    # resamples whole identities, so each identity must carry its complete set
    # of outcomes rather than a set of loose photographs.
    scratch: Dict[str, Dict[str, Any]] = {}
    for result in results:
        row = scratch.setdefault(
            result.identity_hash,
            {"subgroup": result.subgroup, "intended_mated": 0, "scored_mated": 0,
             "intended_non_mated": 0, "scored_non_mated": 0, "mated_fail": 0,
             "non_mated_fail": 0, "unavailable": 0},
        )
        # "Scored" means the photograph reached a comparison at all. Both the
        # intended and the scored counts are kept, because the two support the
        # end-to-end and conditional denominators respectively.
        scored = result.failure_code is None and result.top_similarity is not None
        if result.role == "mated_probe":
            row["intended_mated"] += 1
            row["scored_mated"] += scored
            if not scored:
                row["mated_fail"] += 1
                # A missing gallery reference is recorded separately: the
                # person's own profile could not be enrolled, so the search
                # never had anything to find.
                if result.failure_code == GALLERY_REFERENCE_UNAVAILABLE:
                    row["unavailable"] += 1
        elif result.role == "non_mated_probe":
            row["intended_non_mated"] += 1
            row["scored_non_mated"] += scored
            if not scored:
                row["non_mated_fail"] += 1
    return {
        identity: ReviewIdentityOutcome(
            identity_hash=identity, subgroup=row["subgroup"],
            intended_mated=row["intended_mated"], scored_mated=row["scored_mated"],
            intended_non_mated=row["intended_non_mated"],
            scored_non_mated=row["scored_non_mated"],
            mated_extraction_failures=row["mated_fail"],
            non_mated_extraction_failures=row["non_mated_fail"],
            gallery_reference_unavailable=row["unavailable"],
        )
        for identity, row in sorted(scratch.items())
    }


def review_rates_at_probability(
    rows: Sequence[ReviewFeatureRow],
    probabilities: np.ndarray,
    threshold: float,
    *,
    outcomes: Optional[Mapping[str, ReviewIdentityOutcome]] = None,
) -> Dict[str, Any]:
    """Rank-aware FPIR, TPIR and the decision counts at one threshold.

    Require the correct identity at rank one: a mated probe referred to some
    other identity is a referral, but it is not an identification."""
    rank1 = rank5 = wrong_identity = not_referred = 0
    non_mated_referred = non_mated_correct = 0
    for row, probability in zip(rows, probabilities):
        referred = bool(probability >= threshold)
        if row.role == "mated_probe":
            if not referred:
                not_referred += 1
                continue
            if row.correct_rank == 1:
                rank1 += 1
                rank5 += 1
            elif row.correct_rank is not None and row.correct_rank <= 5:
                rank5 += 1
                wrong_identity += 1
            else:
                wrong_identity += 1
        else:
            non_mated_referred += referred
            non_mated_correct += not referred

    scored_mated = rank1 + wrong_identity + not_referred
    scored_non_mated = non_mated_referred + non_mated_correct
    fpir = non_mated_referred / scored_non_mated if scored_non_mated else float("nan")
    tpir1 = rank1 / scored_mated if scored_mated else float("nan")
    tpir5 = rank5 / scored_mated if scored_mated else float("nan")

    # Intended denominators come from the protocol, not the surviving rows.
    intended_mated = (
        sum(o.intended_mated for o in outcomes.values()) if outcomes else scored_mated
    )
    intended_non_mated = (
        sum(o.intended_non_mated for o in outcomes.values()) if outcomes else scored_non_mated
    )
    mated_failures = (
        sum(o.mated_extraction_failures for o in outcomes.values()) if outcomes else 0
    )
    non_mated_failures = (
        sum(o.non_mated_extraction_failures for o in outcomes.values()) if outcomes else 0
    )
    unavailable = (
        sum(o.gallery_reference_unavailable for o in outcomes.values()) if outcomes else 0
    )
    return {
        "probability_threshold": threshold,
        "fpir": fpir,
        "tpir_rank1": tpir1,
        "tpir_rank5": tpir5,
        "fnir_rank1": 1.0 - tpir1 if tpir1 == tpir1 else float("nan"),
        "fnir_rank5": 1.0 - tpir5 if tpir5 == tpir5 else float("nan"),
        "false_reviews_per_1000_non_mated": fpir * 1000.0 if fpir == fpir else float("nan"),
        "mated_correct_rank1_referred": rank1,
        "mated_correct_rank5_referred": rank5,
        "mated_wrong_identity_referred": wrong_identity,
        "mated_not_referred": not_referred,
        "non_mated_incorrectly_referred": non_mated_referred,
        "non_mated_correctly_not_referred": non_mated_correct,
        "scored_mated_probes": scored_mated,
        "scored_non_mated_probes": scored_non_mated,
        "intended_mated_probes": intended_mated,
        "intended_non_mated_probes": intended_non_mated,
        "mated_extraction_failures": mated_failures,
        "non_mated_extraction_failures": non_mated_failures,
        "gallery_reference_unavailable": unavailable,
        "mated_extraction_coverage": (
            scored_mated / intended_mated if intended_mated else float("nan")
        ),
        "non_mated_extraction_coverage": (
            scored_non_mated / intended_non_mated if intended_non_mated else float("nan")
        ),
        "extraction_coverage": (
            (scored_mated + scored_non_mated) / (intended_mated + intended_non_mated)
            if (intended_mated + intended_non_mated)
            else float("nan")
        ),
        "end_to_end_duplicate_detection_rate": (
            rank1 / intended_mated if intended_mated else float("nan")
        ),
        "conditional_duplicate_detection_rate": tpir1,
    }


def select_review_probability_threshold(
    rows: Sequence[ReviewFeatureRow],
    probabilities: np.ndarray,
    *,
    target_fpir: float,
) -> Dict[str, Any]:
    """Choose the probability threshold on calibration identities only.

    Same deterministic rule as the similarity threshold: admissible by FPIR,
    then highest TPIR@1, then lower FPIR, then the higher threshold."""
    # Every probability the classifier actually produced is a candidate, plus
    # one above 1.0 representing "refer nothing at all".
    candidates = sorted({round(float(p), 12) for p in probabilities}) + [1.0000000001]
    evaluated = [review_rates_at_probability(rows, probabilities, c) for c in candidates]
    # Only thresholds meeting the false-referral budget may be considered. The
    # budget is fixed in advance, so detection cannot be bought by allowing
    # more unnecessary reviews than the experiment declared.
    admissible = [e for e in evaluated if e["fpir"] == e["fpir"] and e["fpir"] <= target_fpir]
    if not admissible:
        raise MlReviewError(
            f"No probability threshold reached a calibration FPIR at or below {target_fpir}."
        )
    # Among those, the best detection wins; ties go to the lower false-referral
    # rate and then to the stricter threshold, so the choice is reproducible.
    chosen = sorted(
        admissible,
        key=lambda e: (
            -(e["tpir_rank1"] if e["tpir_rank1"] == e["tpir_rank1"] else -1.0),
            e["fpir"],
            -e["probability_threshold"],
        ),
    )[0]
    return {
        "target_fpir": target_fpir,
        "probability_threshold": chosen["probability_threshold"],
        "calibration_fpir": chosen["fpir"],
        "calibration_tpir_rank1": chosen["tpir_rank1"],
        "selection_rule": (
            "Among probability thresholds whose calibration FPIR is no greater than the "
            "target, select the highest calibration TPIR at rank 1; ties broken by lower "
            "calibration FPIR, then by higher probability threshold."
        ),
        "candidates_evaluated": len(evaluated),
        "candidates_admissible": len(admissible),
    }


def require_frozen_review_policy(payload: Mapping[str, Any], *, context: str = "") -> float:
    """Refuse held-out evaluation unless the classifier policy is frozen."""
    # Two separate guards. First the policy must be frozen: fitting or
    # recalibrating against held-out identities would invalidate the result,
    # so the refusal lives in code rather than in procedure.
    status = payload.get("status")
    if status != ML_REVIEW_STATUS_FROZEN:
        raise MlReviewError(
            f"Refusing to evaluate held-out identities with review-policy status {status!r}"
            f"{f' from {context}' if context else ''}. Only {ML_REVIEW_STATUS_FROZEN!r} is "
            f"accepted; fit on training identities and calibrate on calibration identities "
            f"first."
        )
    # Second, the frozen policy must actually carry a probability for the
    # primary FPIR target. A policy marked frozen but missing its operating
    # point would otherwise let the held-out evaluation proceed with no
    # threshold at all.
    operating = payload.get("operating_points") or {}
    primary = operating.get(str(PRIMARY_FPIR_TARGET))
    if not primary or "probability_threshold" not in primary:
        raise MlReviewError(
            f"Frozen review policy carries no probability threshold for the primary FPIR "
            f"target {PRIMARY_FPIR_TARGET}."
        )
    return float(primary["probability_threshold"])


def review_cluster_bootstrap(
    rows: Sequence[ReviewFeatureRow],
    probabilities: np.ndarray,
    threshold: float,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
    outcomes: Optional[Mapping[str, ReviewIdentityOutcome]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Identity-cluster bootstrap over classifier decisions.

    Resample identities with their complete protocol outcomes, so a replicate's
    end-to-end denominator counts probes that failed extraction as well as
    those that were scored."""
    by_identity: Dict[str, List[int]] = {}
    subgroup_of: Dict[str, str] = {}
    for index, row in enumerate(rows):
        by_identity.setdefault(row.identity_hash, []).append(index)
        subgroup_of[row.identity_hash] = row.subgroup
    # An identity whose every probe failed contributes no feature row, yet still
    # belongs in the denominator, so it must be resampled too.
    if outcomes:
        for identity, outcome in outcomes.items():
            by_identity.setdefault(identity, [])
            subgroup_of.setdefault(identity, outcome.subgroup)

    strata: Dict[str, List[str]] = {}
    for identity_hash in sorted(by_identity):
        strata.setdefault(subgroup_of[identity_hash], []).append(identity_hash)

    tracked = (
        "fpir", "tpir_rank1", "tpir_rank5", "fnir_rank1", "fnir_rank5",
        "end_to_end_duplicate_detection_rate", "extraction_coverage",
        "mated_extraction_coverage", "non_mated_extraction_coverage",
    )
    samples: Dict[str, List[float]] = {name: [] for name in tracked}
    rng = random.Random(seed)
    for _ in range(replicates):
        indices: List[int] = []
        drawn_identities: List[str] = []
        for _subgroup, members in sorted(strata.items()):
            for _ in range(len(members)):
                chosen = members[rng.randrange(len(members))]
                drawn_identities.append(chosen)
                indices.extend(by_identity[chosen])
        drawn = [rows[i] for i in indices]
        drawn_probabilities = probabilities[np.asarray(indices, dtype=int)] if indices else (
            np.asarray([], dtype=float)
        )
        replicate_outcomes: Optional[Dict[str, ReviewIdentityOutcome]] = None
        if outcomes:
            # Accumulate per draw, since one identity may be drawn repeatedly.
            totals: Dict[str, ReviewIdentityOutcome] = {}
            for position, identity in enumerate(drawn_identities):
                outcome = outcomes.get(identity)
                if outcome is not None:
                    totals[f"{identity}:{position}"] = outcome
            replicate_outcomes = totals
        rates = review_rates_at_probability(
            drawn, drawn_probabilities, threshold, outcomes=replicate_outcomes
        )
        for name in tracked:
            value = rates.get(name, float("nan"))
            if isinstance(value, float) and value == value:
                samples[name].append(value)

    intervals: Dict[str, Dict[str, Any]] = {}
    for name in tracked:
        low, high = _percentile_interval(samples[name])
        intervals[name] = {
            "lower_95": low,
            "upper_95": high,
            "valid_replicates": len(samples[name]),
            "requested_replicates": replicates,
        }
    counts = {}
    if outcomes is not None:
        counts = {identity: (o.intended_non_mated, o.scored_non_mated)
                  for identity, o in outcomes.items() if o.intended_non_mated}
    else:
        for row in rows:
            if row.role == "non_mated_probe":
                n = counts.get(row.identity_hash, (0, 0))[0] + 1
                counts[row.identity_hash] = (n, n)
    false_referrals = sum(row.role == "non_mated_probe" and probability >= threshold
                          for row, probability in zip(rows, probabilities))
    _attach_zero_event_bound(intervals["fpir"], counts, int(false_referrals))
    return intervals


def review_subgroup_metrics(
    rows: Sequence[ReviewFeatureRow],
    probabilities: np.ndarray,
    threshold: float,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
    outcomes: Optional[Mapping[str, ReviewIdentityOutcome]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-subgroup rates with confidence intervals.

    Subgroup is used for evaluation, never prediction."""
    per_subgroup: Dict[str, Dict[str, Any]] = {}
    # Every subgroup is scored at the same frozen probability. The breakdown
    # reports how the one policy performs across groups; it never gives a group
    # its own threshold.
    for subgroup in BFW_SUBGROUPS:
        # Positions are collected rather than the rows themselves, because the
        # matching classifier probabilities must be selected by the same index.
        indices = [i for i, r in enumerate(rows) if r.subgroup == subgroup]
        subset_outcomes = (
            {k: v for k, v in outcomes.items() if v.subgroup == subgroup} if outcomes else None
        )
        # A subgroup with no scored rows may still have intended outcomes, if
        # every one of its photographs failed extraction. That case is still
        # reported, because the end-to-end rate must account for it.
        if not indices and not subset_outcomes:
            continue
        subset = [rows[i] for i in indices]
        subset_probabilities = (
            probabilities[np.asarray(indices, dtype=int)] if indices
            else np.asarray([], dtype=float)
        )
        rates = review_rates_at_probability(
            subset, subset_probabilities, threshold, outcomes=subset_outcomes
        )
        intervals = review_cluster_bootstrap(
            subset, subset_probabilities, threshold, replicates=replicates, seed=seed,
            outcomes=subset_outcomes,
        )
        entry: Dict[str, Any] = {}
        for metric in ("fpir", "fnir_rank1", "fnir_rank5", "tpir_rank1", "tpir_rank5"):
            entry[metric] = rates[metric]
            entry[f"{metric}_lower_95"] = intervals[metric]["lower_95"]
            entry[f"{metric}_upper_95"] = intervals[metric]["upper_95"]
        for metric, key in (
            ("mated_probe_coverage", "mated_extraction_coverage"),
            ("non_mated_probe_coverage", "non_mated_extraction_coverage"),
        ):
            entry[metric] = rates[key]
            entry[f"{metric}_lower_95"] = intervals[key]["lower_95"]
            entry[f"{metric}_upper_95"] = intervals[key]["upper_95"]
        entry["scored_mated_probes"] = rates["scored_mated_probes"]
        entry["scored_non_mated_probes"] = rates["scored_non_mated_probes"]
        entry["intended_mated_probes"] = rates["intended_mated_probes"]
        entry["intended_non_mated_probes"] = rates["intended_non_mated_probes"]
        per_subgroup[subgroup] = entry
    return per_subgroup


def evaluate_review_success_criteria(
    classifier_rates: Mapping[str, Any],
    coverage: Mapping[str, Any],
    baseline_rates: Mapping[str, Any],
    baseline_detection: Mapping[str, Any],
    classifier_detection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare against criteria declared before the held-out run."""

    def verdict(
        actual: Any, target: float, *, minimum: bool, strict: bool = False
    ) -> Dict[str, Any]:
        if not isinstance(actual, (int, float)) or actual != actual:
            return {"outcome": "not_measurable", "actual": None, "target": target}
        if minimum:
            ok = actual > target if strict else actual >= target
        else:
            ok = actual < target if strict else actual <= target
        return {"outcome": "achieved" if ok else "not_achieved", "actual": float(actual),
                "target": target}

    baseline_e2e = baseline_detection.get("end_to_end_duplicate_detection_rate", float("nan"))
    classifier_e2e = classifier_detection.get("end_to_end_duplicate_detection_rate", float("nan"))
    detection_gap = (
        baseline_e2e - classifier_e2e
        if isinstance(baseline_e2e, float) and isinstance(classifier_e2e, float)
        and baseline_e2e == baseline_e2e and classifier_e2e == classifier_e2e
        else float("nan")
    )
    probe_coverage = min(
        (
            1.0 - coverage.get("mated_extraction_failure_rate", float("nan")),
            1.0 - coverage.get("non_mated_extraction_failure_rate", float("nan")),
        ),
        default=float("nan"),
    )
    return {
        "criteria_declared_before_test": True,
        "primary_fpir_at_or_below_1_percent": verdict(
            classifier_rates.get("fpir"), ML_REVIEW_SUCCESS_CRITERIA["fpir_max"], minimum=False
        ),
        "tpir_rank1_at_least_90_percent": verdict(
            classifier_rates.get("tpir_rank1"),
            ML_REVIEW_SUCCESS_CRITERIA["tpir_rank1_min"],
            minimum=True,
        ),
        "end_to_end_detection_within_2pp_of_threshold_method": verdict(
            detection_gap,
            ML_REVIEW_SUCCESS_CRITERIA["end_to_end_detection_tolerance"],
            minimum=False,
        ),
        # "Lower than" means strictly lower. Matching the comparator is not a
        # reduction, and recording it as one would overstate the finding.
        "fewer_false_reviews_than_threshold_method": verdict(
            (
                baseline_rates.get("false_reviews_per_1000_non_mated", float("nan"))
                - classifier_rates.get("false_reviews_per_1000_non_mated", float("nan"))
            ),
            0.0,
            minimum=True,
            strict=True,
        ),
        "gallery_enrolment_coverage_at_least_90_percent": verdict(
            coverage.get("gallery_enrolment_coverage"),
            ML_REVIEW_SUCCESS_CRITERIA["coverage_min"],
            minimum=True,
        ),
        "probe_extraction_coverage_at_least_90_percent": verdict(
            probe_coverage, ML_REVIEW_SUCCESS_CRITERIA["coverage_min"], minimum=True
        ),
    }


# =============================================================================
# 27. Experiment 7 and 8 orchestration, figures and reports
# =============================================================================
#
# Experiment 7 fits and freezes the review classifier before the held-out BFW
# identities are scored once. Experiment 8 compares complete pretrained
# pipelines, each calibrated on its own development data, and records an
# explicit not-run status when the stronger pipeline is unlicensed or absent.
#
# Every figure is drawn from the JSON and CSV artefacts, never from values
# typed by hand, so a figure cannot drift from the numbers it illustrates.

FIGURES_ROOT = RESULTS_ROOT / "figures"

ML_REVIEW_LIMITATIONS = (
    "A benchmark-validated, human-review-only research proof of concept. Nothing here "
    "is production-ready, unbiased, secure, or capable of proving fraudulent behaviour.",
    "The classifier predicts whether a search should open a human-review case. Its output "
    "is not evidence of duplication, fraud or misuse, and no sanction follows from it.",
    "Coefficients describe association within this classifier on these benchmark "
    "identities. They are not causal, and they do not transfer to another population.",
    "Demographic subgroup is never a classifier input. It is used only for post-hoc "
    "fairness reporting.",
    "Extraction failures are reported separately and are never counted as genuine "
    "negative decisions.",
)


def _identity_of_sample(protocol: OpenSetProtocol) -> Dict[str, str]:
    # Private identity per sample, used only to enforce partition membership.
    return {e.sample_id: e.identity for e in protocol.entries}


def _provenance_block(
    dataset: BfwDataset, protocol: OpenSetProtocol, summary: Mapping[str, Any],
    detector: Any, embedder: Any, *, artifact_type: str,
    description: Optional[PipelineDescription] = None,
) -> Dict[str, Any]:
    """Provenance carried by every artefact this section writes.

    ``description`` names the pipeline that produced the result. It must be
    supplied whenever the models are not the baseline pair, because
    ``primary_pipeline_description`` reports the OpenCV names unconditionally
    and would otherwise label a comparison run as YuNet + SFace."""
    pipeline = description or primary_pipeline_description(detector, embedder)
    return {
        "artifact_type": artifact_type,
        "schema_version": SCHEMA_VERSION,
        "opaque_id_version": OPAQUE_ID_VERSION,
        "created_at": utc_now_iso(),
        "seed": protocol.seed,
        "dataset_name": "BFW",
        "protocol_version": BFW_PROTOCOL_VERSION,
        "protocol_digest": summary["public_manifest_sha256"],
        "public_manifest_digest": summary["public_manifest_sha256"],
        "evaluated_image_set_sha256": summary.get("evaluated_image_set_sha256")
        or bfw_dataset_provenance(dataset)["evaluated_image_set_sha256"],
        "dataset_provenance": bfw_dataset_provenance(dataset),
        "pipeline": pipeline.as_dict(),
        "preprocessing_revision": PREPROCESSING_REVISION,
        "software_environment": software_environment_report(),
        "dependency_versions": _reported_dependency_versions(),
        "opencv_distribution": opencv_distribution_report(),
        "opencv_execution": configure_deterministic_opencv(),
        "policy_note": POLICY_NOTE,
        "limitations": list(ML_REVIEW_LIMITATIONS),
    }


# Core packages plus every optional dependency the comparison pipeline needs,
# so an Experiment 8 artefact records the environment that produced it.
_REPORTED_PACKAGES = (
    "numpy", "opencv-python-headless", "Pillow", "scikit-learn", "matplotlib",
    "insightface", "onnxruntime", "onnx", "scipy", "scikit-image", "tqdm",
    "easydict", "prettytable", "requests",
)

# insightface 1.0.1 imports neither of these on the code paths this artefact
# uses (model_zoo.get_model and utils.face_align). Recording them as absent
# with an unexplained null would read as a broken environment, so their status
# is stated instead.
_NOT_REQUIRED_BY_INSIGHTFACE = frozenset({"easydict", "prettytable"})
NOT_REQUIRED_STATUS = "not_required_by_insightface_1_0_1"


def _reported_dependency_versions() -> Dict[str, Optional[str]]:
    """Installed versions, with a stated reason for a deliberate absence.

    A null version is never published without an explanation: an absent package
    is either a recorded non-requirement or a genuine gap worth seeing."""
    reported: Dict[str, Optional[str]] = {}
    for package in _REPORTED_PACKAGES:
        version = _package_version(package)
        if version is None and package in _NOT_REQUIRED_BY_INSIGHTFACE:
            version = NOT_REQUIRED_STATUS
        reported[package] = version
    return reported


def opencv_distribution_report() -> Dict[str, Any]:
    """Which OpenCV is effective, without publishing a local path."""
    import cv2

    return {
        "opencv_distribution": (
            "opencv-python-headless" if _package_version("opencv-python-headless")
            else "opencv-python"
        ),
        "conflicting_opencv_python_present": _package_version("opencv-python") is not None,
        "imported_cv2_version": str(getattr(cv2, "__version__", "")),
    }


def run_ml_review_experiment(
    *, output_root: Path = AGGREGATE_ROOT, seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    detector: Optional[FaceDetector] = None,
    embedder: Optional[FaceEmbedder] = None,
    description: Optional[PipelineDescription] = None,
    base_cache: Path = CANONICAL_RUN_CACHE,
) -> Dict[str, Any]:
    """Experiment 7 end to end. Stops on the exact blocker; fabricates nothing.

    Every optional parameter defaults to the baseline behaviour, so calling
    this with no arguments is Experiment 7 exactly as before. Supplying a
    different pipeline runs the same method over different models, which is
    Experiment 11.

    ``base_cache`` must change with the pipeline. The canonical cache is keyed
    by partition alone, so a second pipeline sharing the default path would
    overwrite the baseline cache and silently invalidate Experiments 6 to 9."""
    config = EnvironmentConfig.load()
    if not id_hmac_key_is_configured():
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} must be configured before identifiers are produced."
        )
    image_root, metadata_path = config.require_bfw_roots()
    if detector is None or embedder is None:
        detector, embedder = load_models(config.require_model_root())

    announce_stage(1, 4, "Loading BFW and rebuilding the identity groups",
                   "Training, calibration and held-out identities share no person.")
    announce("Loading BFW and rebuilding the identity-disjoint protocol")
    dataset = load_bfw_dataset(image_root, metadata_path)
    protocol = build_open_set_protocol(dataset, seed=seed)
    summary = open_set_protocol_summary(
        protocol, dataset=dataset, detector=detector, embedder=embedder
    )
    identity_of_sample = _identity_of_sample(protocol)

    training_ids, calibration_ids = split_development_identities_for_classifier(
        protocol, seed=seed
    )
    test_ids = {e.identity for e in protocol.partition("test")}
    if (set(training_ids) | set(calibration_ids)) & test_ids:
        raise MlReviewError("Classifier identities overlap the held-out test partition.")

    announce(
        f"Classifier identities: {len(training_ids)} training, {len(calibration_ids)} "
        f"calibration, {len(test_ids)} held-out test (disjoint)"
    )

    announce_stage(2, 4, "Measuring each development photograph",
                   "Search and image-quality measurements only; no demographic "
                   "attribute is used.")
    announce("Scoring the development partition once for both classifier groups")
    development, development_digest, development_context = canonical_primary_run(
        protocol, partition="development", detector=detector, embedder=embedder,
        dataset=dataset, base_cache=base_cache,
    )
    announce(f"Using canonical development run {development_digest[:16]}")

    training_rows, training_excluded = build_review_feature_rows(
        development.search_results, identities=set(training_ids),
        identity_of_sample=identity_of_sample,
    )
    calibration_rows, calibration_excluded = build_review_feature_rows(
        development.search_results, identities=set(calibration_ids),
        identity_of_sample=identity_of_sample,
    )
    calibration_outcomes = {
        identity_hash: outcome
        for identity_hash, outcome in build_review_identity_outcomes(
            [
                r for r in development.search_results
                if identity_of_sample.get(r.sample_id) in set(calibration_ids)
            ]
        ).items()
    }

    classifier = fit_review_classifier(training_rows)
    calibration_matrix, _ = _feature_matrix(calibration_rows)
    calibration_probabilities = classifier.probabilities(calibration_matrix)

    operating_points: Dict[str, Any] = {}
    for target in FPIR_TARGETS:
        operating_points[str(target)] = select_review_probability_threshold(
            calibration_rows, calibration_probabilities, target_fpir=target
        )

    provenance = _provenance_block(
        dataset, protocol, summary, detector, embedder,
        artifact_type="ml_review_threshold", description=description,
    )
    subgroup_counts = lambda ids: {
        s: sum(1 for i in ids if _subgroup_of_identity(protocol)[i] == s) for s in BFW_SUBGROUPS
    }
    write_json_artifact(
        output_root / "ml_review_protocol_summary.json",
        {
            **_provenance_block(dataset, protocol, summary, detector, embedder,

                                artifact_type="ml_review_protocol_summary",
                                description=description),
            "training_identities": len(training_ids),
            "calibration_identities": len(calibration_ids),
            "held_out_test_identities": len(test_ids),
            "training_fraction": ML_REVIEW_TRAINING_FRACTION,
            "split_rule": (
                "Development identities split by identity, stratified by subgroup, under "
                "the research seed. No image of one person appears in both groups, and the "
                "held-out test identities are untouched."
            ),
            "training_subgroup_counts": subgroup_counts(training_ids),
            "calibration_subgroup_counts": subgroup_counts(calibration_ids),
            "training_identity_hashes": sorted(opaque_id(f"bfw-identity:{i}") for i in training_ids),
            "calibration_identity_hashes": sorted(
                opaque_id(f"bfw-identity:{i}") for i in calibration_ids
            ),
            "excluded_records": {"training": training_excluded, "calibration": calibration_excluded},
        },
    )

    write_json_artifact(
        output_root / "ml_review_model.json",
        {**_provenance_block(dataset, protocol, summary, detector, embedder,
                             artifact_type="ml_review_model", description=description),
         "model": classifier.as_dict(),
         "feature_definitions": _feature_definitions(),
         "training_rows": len(training_rows),
         "trained_on": "BFW classifier-training identities only"},
    )

    policy_payload = {
        **provenance,
        "status": ML_REVIEW_STATUS_FROZEN,
        "method": ML_REVIEW_METHOD,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "operating_points": operating_points,
        "threshold_source": "BFW classifier-calibration identities only",
        "threshold_selection_rule": operating_points[str(PRIMARY_FPIR_TARGET)]["selection_rule"],
        "threshold_status": ML_REVIEW_STATUS_FROZEN,
        "classifier_hyperparameters": dict(ML_REVIEW_HYPERPARAMETERS),
    }
    policy_path = output_root / "ml_review_threshold.json"
    write_json_artifact(policy_path, policy_payload)
    frozen_probability = require_frozen_review_policy(
        read_json_artifact(policy_path), context=project_relative(policy_path)
    )
    announce_stage(3, 4, "Selecting the referral probability",
                   "Separate calibration identities are used; no held-out identity "
                   "is seen.")
    announce(f"Froze the review classifier at probability {frozen_probability:.6f}")

    write_json_artifact(
        output_root / "ml_review_development_metrics.json",
        {**_provenance_block(dataset, protocol, summary, detector, embedder,

                             artifact_type="ml_review_development_metrics",
                             description=description),
         "status": "ml_review_development",
         "calibration_operating_points": {
             str(t): review_rates_at_probability(
                 calibration_rows, calibration_probabilities,
                 operating_points[str(t)]["probability_threshold"],
                 outcomes=calibration_outcomes)
             for t in FPIR_TARGETS},
         "training_rows": len(training_rows),
         "calibration_rows": len(calibration_rows)},
    )

    # --- Held-out test, scored once ------------------------------------------
    announce_stage(4, 4, "Testing unseen identities",
                   "The frozen referral probability is applied without further "
                   "adjustment.")
    announce("Scoring the held-out test partition")
    test_run, canonical_digest, test_context = canonical_primary_test_run(
        protocol, detector=detector, embedder=embedder, dataset=dataset,
        cache_path=base_cache,
    )
    announce(f"Using canonical primary-pipeline run {canonical_digest[:16]}")
    test_rows, test_excluded = build_review_feature_rows(test_run.search_results)
    test_matrix, _ = _feature_matrix(test_rows)
    test_probabilities = classifier.probabilities(test_matrix)
    classifier_timing = measure_classifier_latency(classifier, test_matrix)

    coverage = open_set_coverage(test_run)
    # Retain failed probes in the intended denominator.
    test_outcomes = build_review_identity_outcomes(test_run.search_results)
    classifier_rates = review_rates_at_probability(
        test_rows, test_probabilities, frozen_probability, outcomes=test_outcomes
    )

    # The comparator, on identical identities, gallery, probes and accounting.
    baseline_policy = read_json_artifact(output_root / "bfw_open_set_threshold.json")
    baseline_threshold = require_frozen_open_set_policy(baseline_policy)
    baseline_rates = open_set_rates_at_threshold(test_run.search_results, baseline_threshold)
    baseline_detection = open_set_duplicate_detection(test_run, baseline_threshold)

    intervals = review_cluster_bootstrap(
        test_rows, test_probabilities, frozen_probability,
        replicates=bootstrap_replicates, seed=seed, outcomes=test_outcomes,
    )
    per_subgroup = review_subgroup_metrics(
        test_rows, test_probabilities, frozen_probability,
        replicates=bootstrap_replicates, seed=seed, outcomes=test_outcomes,
    )

    test_payload = {
        **_provenance_block(dataset, protocol, summary, detector, embedder,
                            artifact_type="ml_review_test_metrics", description=description),
        "status": "ml_review_tested",
        "threshold_source": project_relative(policy_path),
        "threshold_status": ML_REVIEW_STATUS_FROZEN,
        "operating_probability_threshold": frozen_probability,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "classifier_hyperparameters": dict(ML_REVIEW_HYPERPARAMETERS),
        "feature_definitions": _feature_definitions(),
        "classifier": classifier_rates,
        **classifier_timing,
        "comparator_three_image_open_set_calibrated": {
            "operating_threshold": baseline_threshold,
            "rates": baseline_rates,
            **baseline_detection,
        },
        "coverage": coverage,
        "excluded_records": test_excluded,
        "canonical_run_digest": canonical_digest,
        "canonical_test_run_digest": canonical_digest,
        "canonical_development_run_digest": development_digest,
        "canonical_test_context_sha256": context_digest(test_context),
        "canonical_development_context_sha256": context_digest(development_context),
        "cache_schema_version": CANONICAL_CACHE_SCHEMA_VERSION,
        "success_criteria": evaluate_review_success_criteria(
            classifier_rates, coverage, baseline_rates, baseline_detection, classifier_rates
        ),
    }
    write_json_artifact(output_root / "ml_review_test_metrics.json", test_payload)
    write_json_artifact(
        output_root / "ml_review_confidence_intervals.json",
        {**_provenance_block(dataset, protocol, summary, detector, embedder,

                             artifact_type="ml_review_confidence_intervals",
                             description=description),
         "replicates": bootstrap_replicates,
         "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
         "intervals": intervals},
    )
    write_json_artifact(
        output_root / "ml_review_subgroup_metrics.json",
        {**_provenance_block(dataset, protocol, summary, detector, embedder,

                             artifact_type="ml_review_subgroup_metrics",
                             description=description),
         "replicates": bootstrap_replicates,
         "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
         "subgroups": per_subgroup},
    )
    _write_review_csvs(output_root, classifier_rates, baseline_rates, baseline_detection,
                       per_subgroup, test_rows, test_probabilities, frozen_probability)

    report = render_ml_review_report(test_payload, intervals, per_subgroup, classifier)
    (output_root / "ML_REVIEW_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")

    leaks = find_path_leaks(output_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise PrivacyLeakError(
            "Refusing to finish: review output(s) contain a personal/absolute path:\n"
            + "\n".join(f"  {redact_private_paths(leak)}" for leak in leaks)
        )
    assert_no_identifier_key_leak(output_root)
    announce("Privacy validation passed for every review artefact")
    return test_payload


###############################################################################
# Experiment 11: the review classifier on the higher-capacity pipeline
###############################################################################
#
# Experiment 7 fitted the classifier on the baseline pipeline and Experiment 8
# compared the pipelines without it, so the framework's most elaborate addition
# and its strongest components were never combined. This runs the identical
# method over SCRFD + ArcFace search results, which is what allows the question
# "does the framework still add value once the components are better?" to be
# answered rather than assumed.
#
# The identity split uses the same seed, so the training, calibration and
# held-out groups are exactly those of Experiment 7 and the two classifiers are
# directly comparable.

ARCFACE_REVIEW_DIRNAME = "arcface_review"

# A separate cache base. The canonical cache is keyed by partition alone, so
# sharing the default path would overwrite the baseline runs that Experiments 6
# to 9 depend on.
ARCFACE_RUN_CACHE = RAW_ROOT / "canonical_arcface_run.json"


def run_arcface_review_experiment(
    output_root: Path = AGGREGATE_ROOT, *, seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Fit and evaluate the review classifier on SCRFD + ArcFace searches.

    The classifier is reported against a similarity-threshold comparator, and
    that comparator must be this pipeline's own threshold: Experiment 6 froze
    0.477 for SFace, which means nothing in ArcFace's embedding space. An
    open-set policy is therefore developed and frozen here first, using exactly
    the rule Experiment 6 uses and the same development identities."""
    config = EnvironmentConfig.load()
    (detector, embedder), description = load_arcface_pipeline(config)
    sub_root = Path(output_root) / ARCFACE_REVIEW_DIRNAME
    sub_root.mkdir(parents=True, exist_ok=True)

    image_root, metadata_path = config.require_bfw_roots()
    dataset = load_bfw_dataset(image_root, metadata_path)
    protocol = build_open_set_protocol(dataset, seed=seed)
    summary = open_set_protocol_summary(
        protocol, dataset=dataset, detector=detector, embedder=embedder
    )

    announce_stage(1, 2, "Freezing an operating threshold for SCRFD + ArcFace",
                   "Development identities only, using the rule Experiment 6 uses.")
    # Cached, so the classifier stage below reuses this scoring rather than
    # repeating it.
    development, _digest, _context = canonical_primary_run(
        protocol, partition="development", detector=detector, embedder=embedder,
        dataset=dataset, base_cache=ARCFACE_RUN_CACHE,
    )
    operating_points = {
        str(target): select_open_set_threshold(
            development.search_results, target_fpir=target
        )
        for target in FPIR_TARGETS
    }
    policy_path = sub_root / "bfw_open_set_threshold.json"
    write_json_artifact(policy_path, {
        "artifact_type": "bfw_open_set_threshold",
        "status": OPEN_SET_STATUS_FROZEN,
        "method": METHOD_B,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "operating_points": operating_points,
        "selection_rule": OPEN_SET_SELECTION_RULE,
        "developed_on": "BFW development partition (identity-disjoint from test)",
        "public_manifest_sha256": summary["public_manifest_sha256"],
        "protocol_digest": summary["public_manifest_sha256"],
        **_provenance_block(
            dataset, protocol, summary, detector, embedder,
            artifact_type="bfw_open_set_threshold", description=description,
        ),
    })
    announce(f"Froze the SCRFD + ArcFace open-set policy at "
             f"{project_relative(policy_path)}")

    announce_stage(2, 2, "Fitting and evaluating the review classifier",
                   "Same features, same seed and same identity groups as "
                   "Experiment 7.")
    return run_ml_review_experiment(
        output_root=sub_root, seed=seed,
        detector=detector, embedder=embedder, description=description,
        base_cache=ARCFACE_RUN_CACHE,
    )


def render_arcface_review_summary(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """The classifier's effect on each pipeline, side by side.

    The question is not which column is best but whether the classifier moves
    the review burden in the same direction on both pipelines. A consistent
    direction is a finding about the framework; an inconsistent one means the
    effect depends on the components underneath it."""
    root = Path(aggregate_root)
    baseline = _load_optional(root, "ml_review_test_metrics.json")
    compare = _load_optional(root / ARCFACE_REVIEW_DIRNAME, "ml_review_test_metrics.json")
    if not compare:
        return missing_artefact_message(
            "classifier pipeline comparison", "option 14 (--mode arcface-review)"
        )

    def cells(getter) -> List[str]:
        out: List[str] = []
        for payload in (baseline, compare):
            if not payload:
                out += ["not available", "not available"]
                continue
            # Each artefact carries both the similarity-threshold comparator
            # and the classifier fitted on the same pipeline.
            out.append(getter(payload["comparator_three_image_open_set_calibrated"]["rates"]))
            out.append(getter(payload["classifier"]))
        return out

    rows = [
        ["Known duplicates detected (TPIR@1)",
         *cells(lambda r: _percentage_of(r.get("tpir_rank1")))],
        ["New profiles wrongly referred (FPIR)",
         *cells(lambda r: _percentage_of(r.get("fpir")))],
        ["Reviews per 1,000 new profiles",
         *cells(lambda r: f"{r.get('false_reviews_per_1000_non_mated', float('nan')):.1f}")],
    ]

    def direction(payload: Optional[Mapping[str, Any]]) -> Optional[float]:
        if not payload:
            return None
        before = payload["comparator_three_image_open_set_calibrated"]["rates"].get(
            "false_reviews_per_1000_non_mated")
        after = payload["classifier"].get("false_reviews_per_1000_non_mated")
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            return None
        return after - before

    moved_baseline, moved_compare = direction(baseline), direction(compare)
    lines = [
        "EXPERIMENT 11 - THE REVIEW CLASSIFIER ON BOTH PIPELINES",
        "",
        "Dataset:",
        "BFW held-out identities. Both classifiers are fitted on the same",
        "development identities under the same seed, and each is frozen on its",
        "own calibration group before the held-out identities are scored.",
        "",
        render_plain_pipeline_table(
            ["Metric", "SFace threshold", "SFace + classifier",
             "ArcFace threshold", "ArcFace + classifier"],
            rows,
        ),
        "",
        "Outcome:",
        "",
    ]
    if moved_baseline is not None and moved_compare is not None:
        same_direction = (moved_baseline > 0) == (moved_compare > 0)
        verb = lambda d: "raised" if d > 0 else "reduced"
        lines.append(wrap_plain(
            f"The classifier {verb(moved_baseline)} the review burden on "
            f"YuNet + SFace by {abs(moved_baseline):.1f} per 1,000, and "
            f"{verb(moved_compare)} it on SCRFD + ArcFace by "
            f"{abs(moved_compare):.1f} per 1,000."
        ))
        lines += ["", wrap_plain(
            "The direction is the same on both pipelines, so the effect is a "
            "property of the classifier rather than of the models it runs on."
            if same_direction else
            "The direction differs between pipelines, so the classifier's effect "
            "depends on the components underneath it and does not generalise."
        )]
    # A rate of zero is an observation, not a guarantee. Saying so matters most
    # here, because the classifier's best-looking column is the zero one.
    zero_observed = [
        payload for payload in (baseline, compare)
        if payload and payload["classifier"].get("false_reviews_per_1000_non_mated") == 0
    ]
    if zero_observed:
        scored = zero_observed[0]["classifier"].get("scored_non_mated_probes")
        lines += ["", wrap_plain(
            f"A rate of zero means no false referral was observed among the "
            f"{scored:,} new profiles scored. It does not establish that the "
            f"population rate is zero. An empirical zero-event bootstrap interval cannot bound population FPIR."
        )]
    lines += ["", wrap_plain(DENOMINATOR_NOTE), "", wrap_plain(REFERRAL_DISCLAIMER)]
    return "\n".join(lines)

# --- Experiment 12: crossing the two pipelines -------------------------------
#
# Experiments 6 and 8 each keep a detector and an embedder together, so the
# advantage the comparison pipeline showed could belong to either component or
# only to the pairing. Crossing them fills the two empty cells of the design:
#
#                   SFace embedder      ArcFace embedder
#   YuNet detector   Experiment 6          crossing
#   SCRFD detector     crossing          Experiment 8
#
# Each crossing is calibrated afresh on the development identities. A cosine
# similarity means something different in every embedding space, so no frozen
# threshold transfers between these four columns.

MIXED_PIPELINE_DIRNAME = "mixed_pipelines"

# Detector first, embedder second, matching the order the names are read in.
MIXED_CROSSINGS = ("scrfd-sface", "yunet-arcface")

MIXED_LICENCE_NOTE = (
    "This pipeline combines an OpenCV Zoo model with an InsightFace model. Both "
    "licences therefore apply together: the OpenCV weights are redistributable "
    "for research use, and the InsightFace weights are used here only for "
    "local, non-commercial academic evaluation that publishes aggregate metrics "
    "alone. Neither model is trained, fine-tuned or redistributed."
)


def mixed_pipeline_description(
    crossing: str, config: Optional[EnvironmentConfig] = None
) -> PipelineDescription:
    """Name, dimension and verified digests for one crossing.

    The digests are re-verified here rather than copied, so a crossing cannot
    be recorded against a model file that has since changed."""
    if crossing not in MIXED_CROSSINGS:
        raise ValueError(f"Unknown crossing {crossing!r}; expected one of {MIXED_CROSSINGS}.")
    config = config or EnvironmentConfig.load()
    arcface = arcface_pipeline_description(config)
    primary_root = Path(config.require_model_root())
    if crossing == "scrfd-sface":
        return PipelineDescription(
            pipeline_name="mixed-scrfd-sface",
            detector_name="InsightFace SCRFD (det_10g)",
            embedding_model_name="OpenCV SFace 2021dec",
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            preprocessing_revision="mixed-scrfd-sface-112x112-v1",
            model_sha256={
                "detector": arcface.model_sha256["detector"],
                "recognition": verify_model_file(
                    primary_root / SFACE_FILENAME, SFACE_SHA256
                ),
            },
            licence_note=MIXED_LICENCE_NOTE,
        )
    return PipelineDescription(
        pipeline_name="mixed-yunet-arcface",
        detector_name="OpenCV YuNet 2023mar",
        embedding_model_name=f"InsightFace ArcFace {ARCFACE_MODEL_PACK} (w600k_r50)",
        embedding_dimensions=512,
        preprocessing_revision="mixed-yunet-arcface-112x112-v1",
        model_sha256={
            "detector": verify_model_file(primary_root / YUNET_FILENAME, YUNET_SHA256),
            "recognition": arcface.model_sha256["recognition"],
        },
        licence_note=MIXED_LICENCE_NOTE,
    )


def load_mixed_pipeline(
    crossing: str, config: Optional[EnvironmentConfig] = None
) -> Tuple[Tuple[FaceDetector, FaceEmbedder], PipelineDescription]:
    """One detector and one embedder drawn from different pipelines.

    Both pipelines are loaded because the InsightFace loader prepares the
    detector and the recognition model together; only the two halves named by
    the crossing are returned."""
    config = config or EnvironmentConfig.load()
    description = mixed_pipeline_description(crossing, config)
    yunet, sface = load_models(config.require_model_root())
    (scrfd, arcface), _ = load_arcface_pipeline(config)
    if crossing == "scrfd-sface":
        return (scrfd, sface), description
    return (yunet, arcface), description


def mixed_run_cache(crossing: str) -> Path:
    """A cache base of its own for each crossing.

    The canonical cache is keyed by partition alone, so every pipeline needs a
    separate base or one would silently overwrite another's scored run."""
    return RAW_ROOT / f"canonical_mixed_{crossing.replace('-', '_')}_run.json"


def run_mixed_pipeline_experiment(
    crossing: str,
    output_root: Path = AGGREGATE_ROOT,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    """Develop, freeze and test one crossing on the BFW open-set protocol.

    The stages are the ones Experiment 6 obeys: a threshold selected on the
    development identities alone by the same rule, then applied unchanged to
    the held-out identities. Only the three-image template method is run. The
    single-image control needs an LFW 1:1 threshold calibrated for this exact
    embedding space, which no crossing has, and reusing another pipeline's
    would produce a control that means nothing."""
    config = EnvironmentConfig.load()
    (detector, embedder), description = load_mixed_pipeline(crossing, config)
    sub_root = Path(output_root) / MIXED_PIPELINE_DIRNAME / crossing
    sub_root.mkdir(parents=True, exist_ok=True)
    base_cache = mixed_run_cache(crossing)

    image_root, metadata_path = config.require_bfw_roots()
    dataset = load_bfw_dataset(image_root, metadata_path)
    protocol = build_open_set_protocol(dataset, seed=seed)
    summary = open_set_protocol_summary(
        protocol, dataset=dataset, detector=detector, embedder=embedder,
        description=description,
    )
    write_json_artifact(sub_root / "bfw_open_set_protocol_summary.json", summary)

    announce_stage(1, 3, f"Freezing an operating threshold for {description.detector_name} "
                         f"+ {description.embedding_model_name}",
                   "Development identities only, using the rule Experiment 6 uses.")
    development, development_digest, development_context = canonical_primary_run(
        protocol, partition="development", detector=detector, embedder=embedder,
        dataset=dataset, base_cache=base_cache,
    )
    operating_points = {
        str(target): select_open_set_threshold(development.search_results, target_fpir=target)
        for target in FPIR_TARGETS
    }
    policy_path = sub_root / "bfw_open_set_threshold.json"
    write_json_artifact(policy_path, {
        "artifact_type": "bfw_open_set_threshold",
        "status": OPEN_SET_STATUS_FROZEN,
        "method": METHOD_B,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "operating_points": operating_points,
        "selection_rule": OPEN_SET_SELECTION_RULE,
        "developed_on": "BFW development partition (identity-disjoint from test)",
        "public_manifest_sha256": summary["public_manifest_sha256"],
        "protocol_digest": summary["public_manifest_sha256"],
        **_provenance_block(
            dataset, protocol, summary, detector, embedder,
            artifact_type="bfw_open_set_threshold", description=description,
        ),
    })
    frozen_threshold = require_frozen_open_set_policy(
        read_json_artifact(policy_path), context=project_relative(policy_path)
    )
    announce(f"Froze the {description.pipeline_name} open-set policy at "
             f"{project_relative(policy_path)}")

    announce_stage(2, 3, "Testing unseen identities",
                   "The frozen threshold is applied unchanged to the held-out group.")
    test_run, test_digest, test_context = canonical_primary_run(
        protocol, partition="test", detector=detector, embedder=embedder,
        dataset=dataset, base_cache=base_cache,
    )
    coverage = open_set_coverage(test_run)
    rates = open_set_rates_at_threshold(test_run.search_results, frozen_threshold)

    provenance = _provenance_block(
        dataset, protocol, summary, detector, embedder,
        artifact_type="bfw_open_set_test_metrics", description=description,
    )
    provenance["canonical_test_run_digest"] = test_digest
    provenance["canonical_development_run_digest"] = development_digest
    provenance["canonical_test_context_sha256"] = context_digest(test_context)
    provenance["canonical_development_context_sha256"] = context_digest(development_context)
    provenance["cache_schema_version"] = CANONICAL_CACHE_SCHEMA_VERSION

    test_payload = {
        "status": OPEN_SET_STATUS_TESTED,
        "crossing": crossing,
        "public_manifest_sha256": summary["public_manifest_sha256"],
        "protocol_digest": summary["public_manifest_sha256"],
        "threshold_source": project_relative(policy_path),
        "operating_threshold": frozen_threshold,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "methods": {
            METHOD_B: {
                "coverage": coverage,
                "operating_points": {
                    str(target): open_set_rates_at_threshold(
                        test_run.search_results, operating_points[str(target)]["threshold"]
                    )
                    for target in FPIR_TARGETS
                },
                "primary_operating_point": rates,
                **open_set_duplicate_detection(test_run, frozen_threshold),
            },
        },
        "success_criteria": evaluate_open_set_success_criteria(coverage, rates),
        **provenance,
    }
    write_json_artifact(sub_root / "bfw_open_set_test_metrics.json", test_payload)

    announce_stage(3, 3, f"Computing {bootstrap_replicates} cluster-bootstrap replicates",
                   "Identities, not images, are resampled.")
    intervals = cluster_bootstrap_intervals(
        test_run.search_results, threshold=frozen_threshold,
        replicates=bootstrap_replicates, seed=seed,
    )
    write_json_artifact(sub_root / "bfw_open_set_confidence_intervals.json", {
        "replicates": bootstrap_replicates,
        "seed": seed,
        "operating_threshold": frozen_threshold,
        "intervals": intervals,
        **_provenance_block(
            dataset, protocol, summary, detector, embedder,
            artifact_type="bfw_open_set_confidence_intervals", description=description,
        ),
    })

    # Published artefacts are scanned like every other, so a private storage
    # path cannot reach a result through this chain either.
    leaks = find_path_leaks(sub_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise PrivacyLeakError(
            f"{len(leaks)} personal or absolute path(s) found in the {crossing} artefacts."
        )
    announce(f"Privacy validation passed for every {crossing} artefact")
    return test_payload


def run_mixed_pipeline_matrix(
    output_root: Path = AGGREGATE_ROOT,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    """Both crossings, completing the two-by-two against Experiments 6 and 8."""
    results: Dict[str, Any] = {}
    for index, crossing in enumerate(MIXED_CROSSINGS, start=1):
        announce_stage(index, len(MIXED_CROSSINGS), f"Crossed pipeline: {crossing}")
        results[crossing] = run_mixed_pipeline_experiment(
            crossing, output_root, seed=seed, bootstrap_replicates=bootstrap_replicates
        )
    return results

def render_mixed_pipeline_summary(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """The detector and the embedder varied one at a time.

    Experiments 6 and 8 change both components together, so the gap between
    them cannot say which one earned it. Reading the rows of this table gives
    the detector's contribution at a fixed embedder, and the columns give the
    embedder's at a fixed detector."""
    root = Path(aggregate_root)
    pipeline = _load_optional(root, "pipeline_comparison_metrics.json")
    held_out = (pipeline or {}).get("held_out_metrics") or {}

    def matched(fragment: str) -> Optional[Mapping[str, Any]]:
        for name, metrics in held_out.items():
            if fragment in name:
                rates = metrics.get("rates")
                return {**rates, "end_to_end_duplicate_detection_rate": metrics.get("end_to_end_duplicate_detection_rate")} if isinstance(rates, Mapping) else None
        return None

    def crossed(crossing: str) -> Optional[Mapping[str, Any]]:
        payload = _load_optional(
            root / MIXED_PIPELINE_DIRNAME / crossing, "bfw_open_set_test_metrics.json"
        )
        if not payload:
            return None
        point = ((payload.get("methods") or {}).get(METHOD_B) or {}).get(
            "primary_operating_point"
        )
        return {**point, "end_to_end_duplicate_detection_rate": payload["methods"][METHOD_B].get("end_to_end_duplicate_detection_rate")} if isinstance(point, Mapping) else None

    cells = {
        ("yunet", "sface"): matched("opencv"),
        ("scrfd", "arcface"): matched("insightface"),
        ("scrfd", "sface"): crossed("scrfd-sface"),
        ("yunet", "arcface"): crossed("yunet-arcface"),
    }
    if cells[("scrfd", "sface")] is None and cells[("yunet", "arcface")] is None:
        return missing_artefact_message(
            "crossed pipelines", "option 16 (--mode mixed-pipelines)"
        )

    def cell(detector: str, embedder: str, render) -> str:
        rates = cells[(detector, embedder)]
        return render(rates) if rates else "not run"

    def grid(title: str, render) -> List[str]:
        rows = [
            [name,
             cell(key, "sface", render),
             cell(key, "arcface", render)]
            for key, name in (("yunet", "YuNet detector"), ("scrfd", "SCRFD detector"))
        ]
        return [title, "",
                render_plain_pipeline_table(
                    ["", "SFace embedder (128)", "ArcFace embedder (512)"], rows
                ), ""]

    lines = ["THE DETECTOR AND THE EMBEDDER, CROSSED", ""]
    lines += [wrap_plain(
        "Each cell is one pipeline on the same BFW held-out identities, each at "
        "its own threshold frozen on the development identities by the same rule. "
        "The diagonal cells are Experiments 6 and 8; the other two are run only "
        "to separate the two components."
    ), ""]
    lines += grid("Known duplicates detected (TPIR@1, higher is better):",
                  lambda r: _percentage_of(r.get("tpir_rank1")))
    lines += grid("New profiles wrongly referred, per 1,000 (lower is better):",
                  lambda r: (f"{r['false_reviews_per_1000_non_mated']:.1f}"
                             if isinstance(r.get("false_reviews_per_1000_non_mated"),
                                           (int, float)) else "not available"))

    lines += grid("Duplicates detected across ALL intended probes (end-to-end):",
                  lambda r: _percentage_of(r.get("end_to_end_duplicate_detection_rate")))
    lines += [wrap_plain(
        "Conditional TPIR uses each pipeline's successfully scored subset. Differences "
        "between these rates alone do not isolate component effects. Compare end-to-end "
        "detection and coverage, then the paired common-success analysis below."
    ), ""]
    if (root / "comparative_statistics.json").is_file():
        lines += [render_paired_comparison_report(root), ""]

    lines += [wrap_plain(
        "No threshold is shared between cells. A cosine similarity from a "
        "512-dimensional ArcFace vector does not mean what the same number means "
        "in SFace's 128-dimensional space, so each pipeline is calibrated afresh "
        "and only the resulting rates are comparable."
    ), "", wrap_plain(DENOMINATOR_NOTE), "", wrap_plain(REFERRAL_DISCLAIMER)]
    return "\n".join(lines)




def _subgroup_of_identity(protocol: OpenSetProtocol) -> Dict[str, str]:
    return {e.identity: e.subgroup for e in protocol.entries}


def _feature_definitions() -> Dict[str, str]:
    return {
        "top1_similarity": "Cosine similarity to the highest-ranked gallery template.",
        "top2_similarity": "Cosine similarity to the second-ranked gallery template.",
        "top1_top2_margin": "top1_similarity minus top2_similarity; ranking decisiveness.",
        "top5_similarity_mean": "Mean similarity across the five highest-ranked templates.",
        "top5_similarity_stdev": "Sample standard deviation across those five similarities.",
        "top1_gallery_image_count": "Images that contributed to the top-ranked template.",
        "gallery_size": "Enrolled identities searched, which scales impostor exposure.",
        "probe_detection_confidence": "YuNet detection score for the probe image.",
        "probe_face_area_ratio": "Detected face box area divided by whole-image area.",
    }


# Shared column order for both experiments' subgroup files, so a reader can
# compare them directly.
_SUBGROUP_CSV_COLUMNS = (
    "subgroup",
    "fpir", "fpir_lower_95", "fpir_upper_95",
    "fnir_rank1", "fnir_rank1_lower_95", "fnir_rank1_upper_95",
    "fnir_rank5", "fnir_rank5_lower_95", "fnir_rank5_upper_95",
    "tpir_rank1", "tpir_rank1_lower_95", "tpir_rank1_upper_95",
    "tpir_rank5", "tpir_rank5_lower_95", "tpir_rank5_upper_95",
    "mated_probe_coverage", "mated_probe_coverage_lower_95", "mated_probe_coverage_upper_95",
    "non_mated_probe_coverage", "non_mated_probe_coverage_lower_95",
    "non_mated_probe_coverage_upper_95",
    "scored_mated_probes", "scored_non_mated_probes",
    "intended_mated_probes", "intended_non_mated_probes",
)


def _write_review_csvs(
    output_root: Path, classifier_rates: Mapping[str, Any], baseline_rates: Mapping[str, Any],
    baseline_detection: Mapping[str, Any], per_subgroup: Mapping[str, Mapping[str, Any]],
    rows: Sequence[ReviewFeatureRow], probabilities: np.ndarray, threshold: float,
) -> None:
    with open(output_root / "ml_review_method_comparison.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "fpir", "tpir_rank1", "tpir_rank5",
                         "false_reviews_per_1000_non_mated",
                         "end_to_end_duplicate_detection_rate"])
        writer.writerow(["three_image_open_set_calibrated", baseline_rates["fpir"],
                         baseline_rates["tpir_rank1"], baseline_rates["tpir_rank5"],
                         baseline_rates["false_reviews_per_1000_non_mated"],
                         baseline_detection["end_to_end_duplicate_detection_rate"]])
        writer.writerow([ML_REVIEW_METHOD, classifier_rates["fpir"],
                         classifier_rates["tpir_rank1"], classifier_rates["tpir_rank5"],
                         classifier_rates["false_reviews_per_1000_non_mated"],
                         classifier_rates["end_to_end_duplicate_detection_rate"]])

    with open(output_root / "ml_review_subgroup_metrics.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = _SUBGROUP_CSV_COLUMNS
        writer.writerow(columns)
        for subgroup in sorted(per_subgroup):
            r = per_subgroup[subgroup]
            writer.writerow([subgroup] + [r.get(name, "") for name in columns[1:]])

    # Aggregate only: no per-image score, identifier or path is published.
    referred = int(sum(1 for p in probabilities if p >= threshold))
    with open(output_root / "ml_review_predictions_summary.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["statistic", "value"])
        writer.writerow(["scored_records", len(rows)])
        writer.writerow(["referred_for_review", referred])
        writer.writerow(["not_referred", len(rows) - referred])
        writer.writerow(["probability_threshold", threshold])
        writer.writerow(["mean_probability", float(np.mean(probabilities)) if len(rows) else ""])


def render_ml_review_report(
    test: Mapping[str, Any], intervals: Mapping[str, Any],
    per_subgroup: Mapping[str, Mapping[str, Any]], classifier: ReviewClassifier,
) -> str:
    """Every figure below is read from the artefacts, never restated."""
    c = test["classifier"]
    b = test["comparator_three_image_open_set_calibrated"]["rates"]
    coverage = test["coverage"]
    lines = [
        "# Machine-learning review classifier (Experiment 7)",
        "",
        f"Auto-generated by `ACP_arden.py --mode ml-review` on {test['created_at']}.",
        "",
        "> Can an interpretable machine-learning review classifier trained on BFW "
        "development identities reduce false duplicate-profile review referrals while "
        "retaining duplicate-detection performance compared with the existing single "
        "calibrated similarity threshold?",
        "",
        "## Held-out comparison",
        "",
        "| Method | FPIR | TPIR@1 | TPIR@5 | False reviews / 1,000 |",
        "| --- | --- | --- | --- | --- |",
        f"| Calibrated similarity threshold | {format_percentage(b['fpir'])} | "
        f"{format_percentage(b['tpir_rank1'])} | {format_percentage(b['tpir_rank5'])} | "
        f"{format_number(b['false_reviews_per_1000_non_mated'], 1)} |",
        f"| Logistic-regression classifier | {format_percentage(c['fpir'])} | "
        f"{format_percentage(c['tpir_rank1'])} | {format_percentage(c['tpir_rank5'])} | "
        f"{format_number(c['false_reviews_per_1000_non_mated'], 1)} |",
        "",
        "Both use the same rank-aware TPIR definition: the correct identity must be "
        "ranked first (or within five) *and* above the operating point. A referral to "
        "another identity is not an identification.",
        "",
        f"Classifier FPIR 95% CI {format_percentage(intervals['fpir']['lower_95'])} – "
        f"{format_percentage(intervals['fpir']['upper_95'])}; TPIR@1 95% CI "
        f"{format_percentage(intervals['tpir_rank1']['lower_95'])} – "
        f"{format_percentage(intervals['tpir_rank1']['upper_95'])} "
        f"(2,000 identity-cluster replicates).",
        "",
        format_zero_event_note(intervals["fpir"]) if intervals["fpir"].get("bootstrap_degenerate_zero_events") else "",
        "### Decision counts",
        "",
        f"- Mated probes referred with the correct identity at rank one: "
        f"{c['mated_correct_rank1_referred']}; within rank five: "
        f"{c['mated_correct_rank5_referred']}",
        f"- Mated probes referred to the wrong identity (a referral, not an "
        f"identification): {c['mated_wrong_identity_referred']}",
        f"- Mated probes not referred: {c['mated_not_referred']}",
        f"- Non-mated probes referred in error: {c['non_mated_incorrectly_referred']}; "
        f"correctly not referred: {c['non_mated_correctly_not_referred']}",
        f"- Gallery-reference-unavailable failures: {c['gallery_reference_unavailable']}",
        f"- Extraction failures (never counted as negatives): "
        f"{coverage['intended_mated_probes'] - coverage['scored_mated_probes']} mated, "
        f"{coverage['intended_non_mated_probes'] - coverage['scored_non_mated_probes']} non-mated",
        "",
        "### Coverage",
        "",
        f"- Gallery enrolment coverage "
        f"{format_percentage(coverage['gallery_enrolment_coverage'])}",
        f"- Mated extraction failure "
        f"{format_percentage(coverage['mated_extraction_failure_rate'])}; non-mated "
        f"{format_percentage(coverage['non_mated_extraction_failure_rate'])}",
        "",
        "## Pre-declared success criteria",
        "",
    ]
    for name, verdict in test["success_criteria"].items():
        if name == "criteria_declared_before_test":
            continue
        actual = verdict.get("actual")
        lines.append(
            f"- {name}: **{verdict['outcome'].replace('_', ' ')}** "
            f"(target {verdict['target']}, achieved "
            f"{'not measurable' if actual is None else format_number(actual, 4)})"
        )

    lines += ["", "## Standardised coefficients", "",
              "Positive coefficients raise the probability of opening a review case; negative "
              "coefficients lower it. These describe association inside this classifier, not "
              "causation.", "", "| Feature | Coefficient |", "| --- | --- |"]
    for name, weight in zip(classifier.feature_order, classifier.coefficients):
        lines.append(f"| {name} | {weight:+.4f} |")

    lines += ["", "## Subgroup performance (95% confidence intervals)", "",
              "| Subgroup | FPIR | TPIR@1 |", "| --- | --- | --- |"]
    for subgroup in sorted(per_subgroup):
        r = per_subgroup[subgroup]
        lines.append(
            f"| {subgroup} | {format_percentage(r['fpir'])} "
            f"[{format_percentage(r['fpir_lower_95'])}–{format_percentage(r['fpir_upper_95'])}] | "
            f"{format_percentage(r['tpir_rank1'])} "
            f"[{format_percentage(r['tpir_rank1_lower_95'])}–"
            f"{format_percentage(r['tpir_rank1_upper_95'])}] |"
        )
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in ML_REVIEW_LIMITATIONS]
    return "\n".join(lines) + "\n"


def render_ml_review_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    path = output_root / "ml_review_test_metrics.json"
    if not path.is_file():
        return (
            "No review-classifier results found. Run `python ACP_arden.py --mode ml-review`.\n"
            f"That requires the official BFW dataset and {BFW_ROOT_VARIABLE}."
        )
    test = read_json_artifact(path)
    c = test["classifier"]
    b = test["comparator_three_image_open_set_calibrated"]["rates"]
    coverage = test["coverage"]
    lines = [
        f"{PROGRAMME_TITLE} — review-classifier summary",
        "",
        "Experiment 7 — logistic-regression review classifier (held-out BFW test)",
        f"  Frozen probability threshold: "
        f"{format_number(test.get('operating_probability_threshold'), 6)} "
        f"(target FPIR {test.get('primary_fpir_target')}, status {test.get('status')})",
        "",
        "  Calibrated similarity threshold (comparator)",
        f"    FPIR: {format_percentage(b['fpir'])}   TPIR@1: "
        f"{format_percentage(b['tpir_rank1'])}",
        f"    False reviews per 1,000: "
        f"{format_number(b['false_reviews_per_1000_non_mated'], 1)}",
        "",
        "  Logistic-regression classifier",
        f"    FPIR: {format_percentage(c['fpir'])}   TPIR@1: "
        f"{format_percentage(c['tpir_rank1'])}",
        f"    False reviews per 1,000: "
        f"{format_number(c['false_reviews_per_1000_non_mated'], 1)}",
        f"    Gallery enrolment coverage: "
        f"{format_percentage(coverage['gallery_enrolment_coverage'])}",
        f"    Mated extraction failure: "
        f"{format_percentage(coverage['mated_extraction_failure_rate'])}; non-mated "
        f"{format_percentage(coverage['non_mated_extraction_failure_rate'])}",
        "    LIMITATION: every rate is conditional on the coverage printed with it, and a "
        "referral opens human review only.",
        "",
        "  Pre-declared success criteria:",
    ]
    for name, verdict in test.get("success_criteria", {}).items():
        if name == "criteria_declared_before_test":
            continue
        lines.append(f"    {name}: {verdict['outcome'].replace('_', ' ')}")
    lines += ["", "Policy: " + POLICY_NOTE, "",
              "Full write-up: results/aggregate/ML_REVIEW_EVALUATION_REPORT.md"]
    return "\n".join(lines)


def evaluate_one_pipeline_for_comparison(
    protocol: OpenSetProtocol,
    *,
    name: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    canonical_development_run: Optional[OpenSetRunResult] = None,
    canonical_test_run: Optional[OpenSetRunResult] = None,
) -> Dict[str, Any]:
    """Develop, freeze and evaluate one pipeline over the shared protocol.

    Each pipeline receives its own development-only threshold: similarity
    scores from different embedding models are not interchangeable."""
    # The primary pipeline's threshold must be selected from the same canonical
    # development outcomes Experiments 6 and 7 used, or the three would disagree.
    development = canonical_development_run if canonical_development_run is not None else (
        run_open_set_method(
            protocol, partition="development", method=METHOD_B,
            detector=detector, embedder=embedder,
        )
    )
    operating_points = {
        str(target): select_open_set_threshold(
            development.search_results, target_fpir=target
        )
        for target in FPIR_TARGETS
    }
    # Freeze each pipeline threshold before held-out evaluation.
    frozen = float(operating_points[str(PRIMARY_FPIR_TARGET)]["threshold"])

    # The primary pipeline reuses the canonical run so its numbers cannot
    # differ from the ones Experiments 6 and 7 report.
    test = canonical_test_run if canonical_test_run is not None else run_open_set_method(
        protocol, partition="test", method=METHOD_B, detector=detector, embedder=embedder,
    )
    coverage = open_set_coverage(test)
    rates = open_set_rates_at_threshold(test.search_results, frozen)
    intervals = cluster_bootstrap_intervals(
        test.search_results, threshold=frozen, replicates=bootstrap_replicates, seed=seed
    )
    subgroup_replicates = bootstrap_replicates
    per_subgroup = subgroup_open_set_metrics(
        test.search_results, threshold=frozen, replicates=subgroup_replicates, seed=seed,
    )

    failures = {"zero_faces": 0, "multiple_faces": 0, "image_error": 0,
                GALLERY_REFERENCE_UNAVAILABLE: 0}
    for row in test.search_results:
        if row.failure_code is None:
            continue
        key = row.failure_code.split(":", 1)[0]
        failures[key] = failures.get(key, 0) + 1
    for outcome in test.enrolment_outcomes:
        if not outcome.enrolled and outcome.failure_code:
            failures[outcome.failure_code] = failures.get(outcome.failure_code, 0) + 1

    # Aggregate histograms only: no individual score, identifier or path.
    def histogram(rows: Sequence[OpenSetSearchResult], attribute: str) -> Dict[str, Any]:
        values = [
            float(getattr(r, attribute)) for r in rows
            if r.failure_code is None and getattr(r, attribute) is not None
        ]
        if not values:
            return {"bin_edges": [], "counts": [], "n": 0}
        counts, edges = np.histogram(values, bins=40, range=(-1.0, 1.0))
        return {"bin_edges": [float(e) for e in edges],
                "counts": [int(c) for c in counts], "n": len(values)}

    mated_rows = [r for r in test.search_results if r.role == "mated_probe"]
    non_mated_rows = [r for r in test.search_results if r.role == "non_mated_probe"]

    return {
        "pipeline_name": name,
        "development_threshold": frozen,
        "threshold_status": OPEN_SET_STATUS_FROZEN,
        "operating_points": operating_points,
        "rates": rates,
        "coverage": coverage,
        "confidence_intervals": intervals,
        "subgroups": per_subgroup,
        "subgroup_bootstrap_replicates": subgroup_replicates,
        "global_bootstrap_replicates": bootstrap_replicates,
        "failure_breakdown": failures,
        "profile_photo_consistency": profile_photo_consistency_summary(test, frozen),
        "sex_aggregated": sex_aggregated_metrics(
            test.search_results, threshold=frozen, replicates=bootstrap_replicates, seed=seed,
        ),
        "similarity_histograms": {
            "mated_correct_identity": histogram(mated_rows, "correct_similarity"),
            "non_mated_top1": histogram(non_mated_rows, "top_similarity"),
        },
        **open_set_duplicate_detection(test, frozen),
    }


def run_pipeline_comparison(*, output_root: Path = AGGREGATE_ROOT) -> Dict[str, Any]:
    """Records a precise technical status when the local comparison pipeline is
    unavailable and performs the complete held-out comparison when all verified
    preconditions are satisfied."""
    config = EnvironmentConfig.load()
    detector, embedder = load_models(config.require_model_root())
    primary = primary_pipeline_description(detector, embedder)
    status = pipeline_comparison_status(config)

    # Real held-out metrics, computed only when every precondition holds.
    comparison_metrics: Optional[Dict[str, Any]] = None
    canonical_digests: Dict[str, Any] = {}
    protocol_digest: Optional[str] = None
    evaluated_image_digest: Optional[str] = None
    dataset: Optional[BfwDataset] = None
    if status["comparison_run"]:
        image_root, metadata_path = config.require_bfw_roots()
        dataset = load_bfw_dataset(image_root, metadata_path)
        protocol = build_open_set_protocol(dataset, seed=DEFAULT_RANDOM_SEED)
        summary = open_set_protocol_summary(
            protocol, dataset=dataset, detector=detector, embedder=embedder
        )
        protocol_digest = summary["public_manifest_sha256"]
        evaluated_image_digest = bfw_dataset_provenance(dataset)["evaluated_image_set_sha256"]

        (arcface_detector, arcface_embedder), arcface_description = load_arcface_pipeline(config)
        primary_development_run, primary_development_digest, primary_development_context = (
            canonical_primary_run(
                protocol, partition="development", detector=detector, embedder=embedder,
                dataset=dataset,
            )
        )
        primary_test_run, primary_test_digest, primary_test_context = (
            canonical_primary_test_run(
                protocol, detector=detector, embedder=embedder, dataset=dataset
            )
        )
        canonical_digests = {
            "canonical_development_run_digest": primary_development_digest,
            "canonical_test_run_digest": primary_test_digest,
            "canonical_development_context_sha256": context_digest(primary_development_context),
            "canonical_test_context_sha256": context_digest(primary_test_context),
            "cache_schema_version": CANONICAL_CACHE_SCHEMA_VERSION,
        }
        arcface_development_run, _, _ = canonical_primary_run(
            protocol, partition="development", detector=arcface_detector,
            embedder=arcface_embedder, dataset=dataset, base_cache=ARCFACE_RUN_CACHE,
        )
        arcface_test_run, _, _ = canonical_primary_run(
            protocol, partition="test", detector=arcface_detector,
            embedder=arcface_embedder, dataset=dataset, base_cache=ARCFACE_RUN_CACHE,
        )
        # Both pipelines traverse the identical protocol: same identities, same
        # split, same roles, same failure taxonomy.
        comparison_metrics = {
            primary.pipeline_name: evaluate_one_pipeline_for_comparison(
                protocol, name=primary.pipeline_name, detector=detector, embedder=embedder,
                canonical_development_run=primary_development_run,
                canonical_test_run=primary_test_run,
            ),
            arcface_description.pipeline_name: evaluate_one_pipeline_for_comparison(
                protocol, name=arcface_description.pipeline_name,
                detector=arcface_detector, embedder=arcface_embedder,
                canonical_development_run=arcface_development_run,
                canonical_test_run=arcface_test_run,
            ),
        }
        if not comparison_metrics:
            raise PipelineComparisonError(
                "The comparison cannot be marked as evaluated without held-out metrics."
            )

    payload: Dict[str, Any] = {
        "artifact_type": "pipeline_comparison_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "seed": DEFAULT_RANDOM_SEED,
        "dataset_name": "BFW",
        "protocol_version": BFW_PROTOCOL_VERSION,
        "primary_pipeline": primary.as_dict(),
        "comparison_scope": (
            "Complete pretrained pipelines. Detection, landmarking, alignment, "
            "preprocessing, embedding dimensionality and runtime all differ, so no "
            "difference may be attributed to the embedding model alone."
        ),
        "training_performed": False,
        "fine_tuning_performed": False,
        "separate_calibration_required": (
            "Similarity scores from different embedding models are not interchangeable, so "
            "each pipeline must receive its own development-only threshold. The SFace "
            "threshold is never applied to ArcFace."
        ),
        "software_environment": software_environment_report(),
        "dependency_versions": _reported_dependency_versions(),
        # Which OpenCV actually served the import. Package metadata alone is not
        # enough: opencv-python and opencv-python-headless ship the same cv2
        # module, so a shadowing install can change numerics while the pinned
        # version still reads correctly.
        "opencv_distribution": opencv_distribution_report(),
        "policy_note": POLICY_NOTE,
    }
    payload["licence_note"] = status["licence_note"]
    payload["use_statement"] = status["use_statement"]
    payload["preconditions"] = status["preconditions"]
    payload["substitute_model_used"] = False
    payload["model_provenance"] = {
        "created_and_trained_by": "InsightFace project; not trained or fine-tuned here",
        "model_pack": ARCFACE_MODEL_PACK,
        "weights_redistributed": False,
        "weights_committed_to_git": False,
        "weights_downloaded_automatically": False,
        "ownership_claimed": False,
    }
    # evaluated=yes requires real held-out metrics, never readiness alone.
    if status["comparison_run"] and comparison_metrics:
        payload["evaluated"] = "yes"
        payload["status"] = status["status"]
        payload["comparison_pipeline"] = status["pipeline"]
        payload["held_out_metrics"] = comparison_metrics
    else:
        payload["evaluated"] = "no"
        payload["status"] = status["status"]
        payload["reason"] = status["reason"] or (
            "Preconditions were met but no held-out metrics were produced."
        )
        payload["comparison_pipeline"] = None
        payload["held_out_metrics"] = None
    payload["protocol_digest"] = protocol_digest
    payload["public_manifest_digest"] = protocol_digest
    payload["evaluated_image_set_sha256"] = evaluated_image_digest
    payload["preprocessing_revision"] = PREPROCESSING_REVISION
    payload["model_filenames"] = {
        "primary": [YUNET_FILENAME, SFACE_FILENAME],
        "comparison": [ARCFACE_DETECTOR_FILENAME, ARCFACE_RECOGNITION_FILENAME],
    }
    payload["model_digests"] = {
        "primary": primary.model_sha256,
        "comparison": (
            payload["comparison_pipeline"]["model_sha256"]
            if payload["comparison_pipeline"] else
            {"detector": ARCFACE_DETECTOR_SHA256, "recognition": ARCFACE_RECOGNITION_SHA256}
        ),
    }
    payload["threshold_policy"] = (
        "Each pipeline is calibrated on the BFW development partition only and its "
        "threshold frozen before the held-out identities are scored once."
    )
    payload["calibration_partition"] = (
        "BFW development identities, identity-disjoint from the held-out test partition."
    )
    payload["held_out_partition"] = (
        "BFW test identities, never used for fitting or threshold selection."
    )
    payload["limitations"] = list(OPEN_SET_LIMITATIONS)
    payload.update(canonical_digests)
    payload["model_file_sizes"] = {
        "primary": _primary_model_file_sizes(config.model_root),
        "comparison": _model_file_sizes(config.arcface_model_root),
    }
    payload["model_storage"] = (
        "Weight files are held in private local research storage, are excluded from Git "
        "and from any release, and are never downloaded automatically."
    )
    payload["redistribution"] = "No pretrained weight file is redistributed by this project."
    payload["dataset_provenance"] = (
        bfw_dataset_provenance(dataset) if dataset is not None else None
    )
    payload["development_partition"] = (
        "BFW development identities; each pipeline calibrated here only."
    )
    payload["held_out_partition"] = (
        "BFW test identities, identity-disjoint, scored once per pipeline."
    )
    payload["frozen_thresholds"] = {
        name: metrics["development_threshold"]
        for name, metrics in (comparison_metrics or {}).items()
    }

    write_json_artifact(output_root / "pipeline_comparison_metrics.json", payload)
    write_json_artifact(
        output_root / "pipeline_comparison_protocol.json",
        {"artifact_type": "pipeline_comparison_protocol", "schema_version": SCHEMA_VERSION,
         "created_at": utc_now_iso(), "seed": DEFAULT_RANDOM_SEED,
         "shared_protocol": (
             "Both pipelines use the same BFW identities, development and test split, "
             "subgroup stratification, gallery image count, mated and non-mated probes, "
             "target FPIR, bootstrap procedure, seed, success criteria and failure taxonomy."
         ),
         "status": payload["status"], "evaluated": payload["evaluated"],
         "dataset_name": "BFW", "dataset_provenance": payload["dataset_provenance"],
         "protocol_version": BFW_PROTOCOL_VERSION,
         "protocol_digest": protocol_digest,
         "public_manifest_digest": protocol_digest,
         "evaluated_image_set_sha256": evaluated_image_digest,
         "development_partition": payload["development_partition"],
         "held_out_partition": payload["held_out_partition"],
         "primary_pipeline": payload["primary_pipeline"],
         "comparison_pipeline": payload.get("comparison_pipeline"),
         "model_filenames": payload["model_filenames"],
         "model_digests": payload["model_digests"],
         "model_file_sizes": payload["model_file_sizes"],
         "dependency_versions": payload["dependency_versions"],
         "opencv_distribution": payload["opencv_distribution"],
         "software_environment": payload["software_environment"],
         "preprocessing_revision": payload["preprocessing_revision"],
         "threshold_policy": payload["threshold_policy"],
         "frozen_thresholds": payload["frozen_thresholds"],
         "canonical_development_run_digest": payload.get("canonical_development_run_digest"),
         "canonical_test_run_digest": payload.get("canonical_test_run_digest"),
         "canonical_development_context_sha256": payload.get(
             "canonical_development_context_sha256"),
         "canonical_test_context_sha256": payload.get("canonical_test_context_sha256"),
         "cache_schema_version": payload.get("cache_schema_version"),
         "licence_note": payload["licence_note"],
         "limitations": payload["limitations"],
         "policy_note": POLICY_NOTE},
    )
    per_pipeline_consistency = {
        name: metrics["profile_photo_consistency"]
        for name, metrics in (comparison_metrics or {}).items()
        if metrics.get("profile_photo_consistency")
    }
    if per_pipeline_consistency:
        write_profile_consistency_artefacts(
            per_pipeline_consistency, output_root,
            provenance={"status": payload["status"], "seed": DEFAULT_RANDOM_SEED,
                        "created_at": utc_now_iso(),
                        "interpretation_note": PROFILE_CONSISTENCY_NOTE},
        )

    write_pipeline_sex_aggregates(payload, output_root)
    write_pipeline_performance_csv(
        output_root / "pretrained_pipeline_comparison.csv", payload=payload
    )
    write_pipeline_subgroup_csv(
        output_root / "pretrained_pipeline_subgroup_metrics.csv", payload=payload
    )

    # The interval and subgroup artefacts are written whether or not the
    # comparison ran. A silently absent file is indistinguishable from one that
    # was forgotten; an empty file carrying the blocking status is not.
    evaluated = payload["evaluated"] == "yes"
    write_json_artifact(
        output_root / "pipeline_comparison_confidence_intervals.json",
        {
            "artifact_type": "pipeline_comparison_confidence_intervals",
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "seed": DEFAULT_RANDOM_SEED,
            "status": payload["status"],
            "evaluated": payload["evaluated"],
            "replicates": BOOTSTRAP_REPLICATES if evaluated else 0,
            "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
            "intervals": {
                name: metrics["confidence_intervals"]
                for name, metrics in (comparison_metrics or {}).items()
            },
            "note": (
                "Populated for both pipelines once evaluated. "
                + ("" if evaluated else payload.get("reason", ""))
            ),
            "seed": DEFAULT_RANDOM_SEED,
            "dataset_name": "BFW",
            "protocol_version": BFW_PROTOCOL_VERSION,
            "protocol_digest": protocol_digest,
            "public_manifest_digest": protocol_digest,
            "evaluated_image_set_sha256": evaluated_image_digest,
            "primary_pipeline": payload["primary_pipeline"],
            "comparison_pipeline": payload.get("comparison_pipeline"),
            "model_filenames": payload["model_filenames"],
            "model_digests": payload["model_digests"],
            "model_file_sizes": payload["model_file_sizes"],
            "dependency_versions": payload["dependency_versions"],
            "opencv_distribution": payload["opencv_distribution"],
            "software_environment": payload["software_environment"],
            "preprocessing_revision": payload["preprocessing_revision"],
            "threshold_policy": payload["threshold_policy"],
            "frozen_thresholds": payload["frozen_thresholds"],
            "canonical_development_run_digest": payload.get(
                "canonical_development_run_digest"),
            "canonical_test_run_digest": payload.get("canonical_test_run_digest"),
            "canonical_development_context_sha256": payload.get(
                "canonical_development_context_sha256"),
            "canonical_test_context_sha256": payload.get("canonical_test_context_sha256"),
            "cache_schema_version": payload.get("cache_schema_version"),
            "development_partition": payload["development_partition"],
            "held_out_partition": payload["held_out_partition"],
            "licence_note": payload["licence_note"],
            "limitations": payload["limitations"],
            "policy_note": POLICY_NOTE,
        },
    )
    report = [
        "# Pretrained pipeline comparison (Experiment 8)",
        "",
        f"Auto-generated by `ACP_arden.py --mode pipeline-compare` on {payload['created_at']}.",
        "",
        "> Does a stronger pretrained detection and face-embedding pipeline improve "
        "extraction coverage, open-set duplicate detection and subgroup consistency "
        "compared with YuNet + SFace under the same BFW protocol?",
        "",
        f"## Status: {payload['status']}",
        "",
    ]
    report += [
        "## Licensing position",
        "",
        payload["licence_note"],
        "",
        payload["use_statement"],
        "",
    ]
    if payload["evaluated"] == "no":
        report += [
            f"## Why it did not run: `{payload['status']}`",
            "",
            f"{payload['reason']}",
            "",
            "**This is a technical precondition, not a licensing obstacle.** The official "
            "terms permit non-commercial research, and this evaluation is non-commercial "
            "academic research, so commercial-use restrictions do not apply to it.",
            "",
            "Outstanding preconditions:",
            "",
        ]
        report += [
            f"- `{name}`: {'satisfied' if ok else 'NOT satisfied'}"
            for name, ok in sorted(payload["preconditions"].items())
        ]
        report += [
            "",
            "No substitute model was used and no performance figures are reported. "
            "Replacing the approved comparator with a different model in order to produce "
            "a number would make the comparison meaningless.",
        ]
    else:
        report += [
            "Both pipelines were evaluated under the shared protocol above, each with its "
            "own development-only threshold.",
        ]
    held_out = payload.get("held_out_metrics") or {}
    if held_out:
        report += _render_comparison_result_sections(payload, held_out)

    report += [
        "",
        "## Why this is not an embedding-only comparison",
        "",
        payload["comparison_scope"],
        "",
        "Each pipeline receives its own development-only threshold. " +
        payload["separate_calibration_required"],
        "",
        "Neither pipeline was trained or fine-tuned.",
    ]
    (output_root / "PRETRAINED_PIPELINE_COMPARISON_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    # Readable wording in the terminal; the exact internal status stays in JSON.
    announce(f"Pipeline comparison: {plain_status(payload['status'])}")
    return payload


def _render_comparison_result_sections(
    payload: Mapping[str, Any], held_out: Mapping[str, Any]
) -> List[str]:
    """Numerical sections of the Experiment 8 report, read from the artefact."""
    pct = format_percentage
    names = sorted(held_out, key=lambda n: "opencv" not in n)

    def band(name: str, metric: str) -> str:
        row = (held_out[name].get("confidence_intervals") or {}).get(metric) or {}
        low, high = row.get("lower_95"), row.get("upper_95")
        return "" if low is None else f" [{pct(low)}–{pct(high)}]"

    lines = [
        "", "## Held-out results", "",
        "Both pipelines were evaluated on the held-out identities under thresholds "
        "frozen using development data. The evaluation was repeated only to test "
        "computational reproducibility; no repeated held-out result influenced model "
        "selection, threshold selection or reported policy. Each pipeline was evaluated "
        "under its own frozen development threshold. Intervals are 95% percentile "
        "bounds from a "
        f"{payload.get('held_out_metrics', {}).get(names[0], {}).get('global_bootstrap_replicates', 2000)}"
        "-replicate identity-cluster bootstrap, resampling identities with their complete "
        "protocol outcomes and preserving subgroup stratification.",
        "",
        "| Metric | " + " | ".join(n for n in names) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]
    rows = [
        ("Frozen threshold", lambda n: format_number(held_out[n]["development_threshold"], 6)),
        ("Scored mated probes", lambda n: str(held_out[n]["rates"]["scored_mated_probes"])),
        ("Scored non-mated probes", lambda n: str(held_out[n]["rates"]["scored_non_mated_probes"])),
        ("FPIR", lambda n: pct(held_out[n]["rates"]["fpir"]) + band(n, "fpir")),
        ("TPIR@1", lambda n: pct(held_out[n]["rates"]["tpir_rank1"]) + band(n, "tpir_rank1")),
        ("TPIR@5", lambda n: pct(held_out[n]["rates"]["tpir_rank5"]) + band(n, "tpir_rank5")),
        ("CMC rank-1", lambda n: pct(held_out[n]["rates"]["cmc_rank1"])),
        ("CMC rank-5", lambda n: pct(held_out[n]["rates"]["cmc_rank5"])),
        ("False reviews / 1,000",
         lambda n: format_number(held_out[n]["rates"]["false_reviews_per_1000_non_mated"], 2)),
        ("End-to-end detection",
         lambda n: pct(held_out[n].get("end_to_end_duplicate_detection_rate"))
         + band(n, "end_to_end_duplicate_detection_rate")),
        ("Gallery coverage",
         lambda n: pct(held_out[n]["coverage"]["gallery_enrolment_coverage"])),
        ("Mated coverage",
         lambda n: pct(1.0 - held_out[n]["coverage"]["mated_extraction_failure_rate"])),
        ("Non-mated coverage",
         lambda n: pct(1.0 - held_out[n]["coverage"]["non_mated_extraction_failure_rate"])),
        ("Zero-face failures",
         lambda n: str(held_out[n]["failure_breakdown"].get("zero_faces", 0))),
        ("Multiple-face failures",
         lambda n: str(held_out[n]["failure_breakdown"].get("multiple_faces", 0))),
        ("Gallery reference unavailable",
         lambda n: str(held_out[n]["failure_breakdown"].get(GALLERY_REFERENCE_UNAVAILABLE, 0))),
    ]
    for label, getter in rows:
        lines.append(f"| {label} | " + " | ".join(getter(n) for n in names) + " |")

    lines += ["", "## Cost", "",
              "| Measure | " + " | ".join(names) + " |",
              "| --- | " + " | ".join("---" for _ in names) + " |"]
    sizes = payload.get("model_file_sizes") or {}
    def size_of(name: str, which: int) -> str:
        group = sizes.get("primary" if "opencv" in name else "comparison") or {}
        values = list(group.values())
        return f"{values[which]['megabytes']} MB" if len(values) > which else "n/a"
    cost_rows = [
        ("Embedding dimensions",
         lambda n: str(held_out[n].get("embedding_dimensions")
                       or (128 if "opencv" in n else 512))),
        ("Detector model size", lambda n: size_of(n, 0)),
        ("Recognition model size", lambda n: size_of(n, 1)),
    ]
    for key, label in (("image_load_latency", "Image load"),
                       ("detection_latency", "Detection"),
                       ("embedding_latency", "Embedding"),
                       ("complete_pipeline_latency", "Complete pipeline")):
        cost_rows.append((f"{label} mean (ms)",
                          lambda n, k=key: format_number(
                              held_out[n]["coverage"].get(f"{k}_mean_ms"), 3)))
        cost_rows.append((f"{label} p95 (ms)",
                          lambda n, k=key: format_number(
                              held_out[n]["coverage"].get(f"{k}_p95_ms"), 3)))
    cost_rows += [
        ("Gallery search mean (ms)",
         lambda n: format_number(held_out[n]["coverage"].get("top1_search_time_mean_ms"), 3)),
        ("Gallery search p95 (ms)",
         lambda n: format_number(held_out[n]["coverage"].get("top1_search_time_p95_ms"), 3)),
    ]
    for label, getter in cost_rows:
        lines.append(f"| {label} | " + " | ".join(getter(n) for n in names) + " |")

    for sex in ("female", "male"):
        lines += ["", f"## {sex.capitalize()} aggregate", "",
                  "Scored counts are published beside every coverage figure, so that "
                  "scored + failures = intended and coverage = scored / intended can "
                  "both be checked rather than taken on trust.",
                  "",
                  "| Pipeline | FPIR | TPIR@1 | TPIR@5 | Mated scored/intended | "
                  "Mated failures | Mated coverage | Non-mated scored/intended | "
                  "Non-mated failures | Non-mated coverage |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name in names:
            entry = (held_out[name].get("sex_aggregated") or {}).get(sex)
            if not entry:
                continue
            lines.append(
                f"| {name} | {pct(entry['fpir'])} "
                f"[{pct(entry['fpir_lower_95'])}–{pct(entry['fpir_upper_95'])}] | "
                f"{pct(entry['tpir_rank1'])} "
                f"[{pct(entry['tpir_rank1_lower_95'])}–{pct(entry['tpir_rank1_upper_95'])}] | "
                f"{pct(entry['tpir_rank5'])} | "
                f"{entry['scored_mated_probes']}/{entry['intended_mated_probes']} | "
                f"{entry['mated_extraction_failures']} | "
                f"{pct(entry['mated_probe_coverage'])} | "
                f"{entry['scored_non_mated_probes']}/{entry['intended_non_mated_probes']} | "
                f"{entry['non_mated_extraction_failures']} | "
                f"{pct(entry['non_mated_probe_coverage'])} |"
            )

    lines += ["", "## Subgroup extremes", ""]
    for metric, label, best_is_low in (("fpir", "FPIR", True),
                                       ("tpir_rank1", "TPIR@1", False),
                                       ("mated_probe_coverage", "Mated coverage", False)):
        for name in names:
            subgroups = held_out[name].get("subgroups") or {}
            valid = {k: v[metric] for k, v in subgroups.items()
                     if isinstance(v.get(metric), float) and v[metric] == v[metric]}
            if not valid:
                continue
            best = min(valid, key=lambda k: valid[k]) if best_is_low else max(valid, key=lambda k: valid[k])
            worst = max(valid, key=lambda k: valid[k]) if best_is_low else min(valid, key=lambda k: valid[k])
            lines.append(
                f"- {label}, {name}: strongest **{best}** ({pct(valid[best])}), "
                f"weakest **{worst}** ({pct(valid[worst])})"
            )

    lines += ["", "## Profile-photo consistency", "",
              f"*{PROFILE_CONSISTENCY_STATUS_NOTE}*", "",
              "| Pipeline | Consistency (cond.) | Consistency (end-to-end) | "
              "Same-person coverage |",
              "| --- | --- | --- | --- |"]
    for name in names:
        c = held_out[name].get("profile_photo_consistency")
        if not c:
            continue
        lines.append(
            f"| {name} | {pct(c['same_person_consistency_rate_conditional'])} | "
            f"{pct(c['same_person_consistency_rate_end_to_end'])} | "
            f"{pct(c['same_person_extraction_coverage'])} |"
        )

    # The two controls answer different questions and are never merged.
    lines += ["", "### Open-set non-mated gallery control", "",
              CONSISTENCY_CONTROL_DEFINITIONS["open_set_non_mated_gallery_control"], "",
              "| Pipeline | Scored | Correctly identified | False-consistent | "
              "Detection (cond.) | Detection (end-to-end) | False-consistency (cond.) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        c = held_out[name].get("profile_photo_consistency")
        if not c:
            continue
        lines.append(
            f"| {name} | {c['scored_open_set_non_mated_gallery_controls']} | "
            f"{c['open_set_non_mated_gallery_controls_correctly_identified']} | "
            f"{c['open_set_non_mated_gallery_controls_false_consistent']} | "
            f"{pct(c['open_set_non_mated_gallery_detection_conditional'])} | "
            f"{pct(c['open_set_non_mated_gallery_detection_end_to_end'])} | "
            f"{pct(c['open_set_non_mated_gallery_false_consistency_conditional'])} |"
        )
    lines += ["", "### Wrong-profile-template control (supplementary)", "",
              CONSISTENCY_CONTROL_DEFINITIONS["wrong_profile_template_control"], "",
              "| Pipeline | Scored | Correctly inconsistent | False-consistent | "
              "Failures | Detection (cond.) | Detection (end-to-end) | "
              "False-consistency (cond.) | False-consistency (end-to-end) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        c = held_out[name].get("profile_photo_consistency")
        if not c:
            continue
        lines.append(
            f"| {name} | {c['wrong_profile_template_controls_scored']} | "
            f"{c['wrong_profile_template_controls_correctly_inconsistent']} | "
            f"{c['wrong_profile_template_controls_false_consistent']} | "
            f"{c['wrong_profile_template_control_extraction_failures']} | "
            f"{pct(c['wrong_profile_template_mismatch_detection_conditional'])} | "
            f"{pct(c['wrong_profile_template_mismatch_detection_end_to_end'])} | "
            f"{pct(c['wrong_profile_template_false_consistency_conditional'])} | "
            f"{pct(c['wrong_profile_template_false_consistency_end_to_end'])} |"
        )
    lines += ["", PROFILE_CONSISTENCY_POLICY_NOTE, ""]

    lines += [
        "## Canonical primary-pipeline provenance", "",
        f"- Development run digest: `{payload.get('canonical_development_run_digest')}`",
        f"- Held-out run digest: `{payload.get('canonical_test_run_digest')}`",
        f"- Development context digest: `{payload.get('canonical_development_context_sha256')}`",
        f"- Held-out context digest: `{payload.get('canonical_test_context_sha256')}`",
        "",
        "The YuNet + SFace figures above derive from the same canonical runs used by "
        "Experiments 6 and 7, so all three report identical scored counts.",
    ]

    # Only claim an improvement the generated values actually support.
    primary, comparator = names[0], names[-1]
    if len(names) == 2:
        better = (
            held_out[comparator]["coverage"]["gallery_enrolment_coverage"]
            >= held_out[primary]["coverage"]["gallery_enrolment_coverage"]
            and held_out[comparator]["rates"]["fpir"] <= held_out[primary]["rates"]["fpir"]
            and (held_out[comparator].get("end_to_end_duplicate_detection_rate") or 0)
            >= (held_out[primary].get("end_to_end_duplicate_detection_rate") or 0)
        )
        costlier = (
            held_out[comparator]["coverage"].get("complete_pipeline_latency_mean_ms", 0)
            > held_out[primary]["coverage"].get("complete_pipeline_latency_mean_ms", 0)
        )
        lines += ["", "## Conclusion", ""]
        if better and costlier:
            lines.append(
                "SCRFD + ArcFace improved extraction coverage, held-out duplicate detection "
                "and false-review performance under this BFW protocol, but required "
                "substantially greater model storage and complete-pipeline processing time. "
                "Because detection, alignment, preprocessing and embedding all differ, the "
                "result cannot be attributed solely to ArcFace."
            )
        else:
            lines.append(
                "The generated values do not support a uniform improvement, so none is "
                "claimed. Compare the tables above directly."
            )
        lines.append(
            "This is not a claim of universal superiority, deployment readiness or absence "
            "of bias."
        )
    return lines


def render_pipeline_comparison_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    path = output_root / "pipeline_comparison_metrics.json"
    if not path.is_file():
        return "No pipeline-comparison record found. Run `--mode pipeline-compare`."
    payload = read_json_artifact(path)
    lines = [
        f"{PROGRAMME_TITLE} — pretrained pipeline comparison",
        "",
        f"  Primary pipeline: {payload['primary_pipeline']['pipeline_name']} "
        f"({payload['primary_pipeline']['embedding_dimensions']}-dimensional)",
        f"  Evaluated: {payload['evaluated']}",
        f"  Status: {plain_status(payload['status'])}",
        f"  Internal status value (as stored in JSON): {payload['status']}",
    ]
    if payload.get("reason"):
        lines.append(f"  Reason: {payload['reason']}")
    lines += [
        "  No model was trained or fine-tuned; no substitute model was used.",
        "",
        "Full write-up: results/aggregate/PRETRAINED_PIPELINE_COMPARISON_REPORT.md",
    ]
    return "\n".join(lines)


# Supplementary paired comparisons; thresholds and galleries remain fixed.
COMPARISON_PIPELINE_ORDER = ("YuNet + SFace", "SCRFD + SFace", "YuNet + ArcFace", "SCRFD + ArcFace")


def paired_pipeline_bootstrap(
    runs: Mapping[str, OpenSetRunResult], thresholds: Mapping[str, float],
    *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Paired, role/subgroup-stratified identity bootstrap of method differences.

    Common-success results use the same successfully extracted probes in every
    pipeline. End-to-end results always retain every intended mated probe.
    """
    if len(runs) < 2 or replicates < 2 or set(runs) != set(thresholds):
        raise ValueError("At least two aligned runs, their thresholds and two replicates are required.")
    names = list(runs)
    keyed = {name: {r.sample_id: r for r in run.search_results} for name, run in runs.items()}
    reference = keyed[names[0]]
    if not reference:
        raise ProtocolError("Paired comparison requires intended probes.")
    if any(len(keyed[name]) != len(runs[name].search_results) for name in names):
        raise ProtocolError("Duplicate probe identifiers in paired comparison.")
    for name in names:
        if set(keyed[name]) != set(reference):
            raise ProtocolError("Paired methods must contain exactly the same intended probes.")
        if not math.isfinite(thresholds[name]):
            raise ValueError("Paired thresholds must be finite.")
        for sample, row in keyed[name].items():
            ref = reference[sample]
            if (row.identity_hash, row.subgroup, row.role) != (ref.identity_hash, ref.subgroup, ref.role):
                raise ProtocolError("Paired methods disagree about probe identity, role or subgroup.")
    identities = sorted({r.identity_hash for r in reference.values()})
    position = {identity: i for i, identity in enumerate(identities)}
    identity_strata: Dict[str, Tuple[str, str]] = {}
    for row in reference.values():
        value = (row.role, row.subgroup)
        if identity_strata.setdefault(row.identity_hash, value) != value:
            raise ProtocolError("An identity crosses roles or subgroups in a paired comparison.")
    common = {sample for sample in reference
              if all(keyed[name][sample].failure_code is None for name in names)}
    strata: Dict[Tuple[str, str], List[int]] = {}
    for identity in identities:
        strata.setdefault(identity_strata[identity], []).append(position[identity])
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(identities)), dtype=float)
    for members in strata.values():
        weights[:, members] = rng.multinomial(len(members), [1 / len(members)] * len(members), size=replicates)
    metrics = ("fpir", "tpir_rank1", "end_to_end_duplicate_detection_rate",
               "mated_extraction_coverage", "non_mated_extraction_coverage")

    def rates(counts: np.ndarray) -> np.ndarray:
        # Columns: intended mated, scored mated, detected, intended new, scored new, false referrals.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.stack((counts[..., 5] / counts[..., 4], counts[..., 2] / counts[..., 1],
                             counts[..., 2] / counts[..., 0], counts[..., 1] / counts[..., 0],
                             counts[..., 4] / counts[..., 3]), axis=-1)

    def interval(values: np.ndarray, estimate: float) -> Dict[str, Any]:
        valid = values[np.isfinite(values)]
        low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (float("nan"), float("nan"))
        return {"estimate": float(estimate), "lower_95": float(low), "upper_95": float(high),
                "valid_replicates": len(valid), "requested_replicates": replicates}

    analyses: Dict[str, Any] = {}
    for scope in ("full_protocol", "common_success"):
        observed: Dict[str, np.ndarray] = {}
        draws: Dict[str, np.ndarray] = {}
        point_results: Dict[str, Any] = {}
        for name in names:
            counts = np.zeros((len(identities), 6), dtype=float)
            for sample, row in keyed[name].items():
                if scope == "common_success" and sample not in common:
                    continue
                i = position[row.identity_hash]
                offset = 0 if row.role == "mated_probe" else 3
                counts[i, offset] += 1
                if row.failure_code is not None:
                    continue
                counts[i, offset + 1] += 1
                if offset == 0:
                    counts[i, 2] += int(row.correct_rank == 1 and row.correct_similarity is not None
                                        and row.correct_similarity >= thresholds[name])
                else:
                    counts[i, 5] += int(row.top_similarity is not None and row.top_similarity >= thresholds[name])
            observed[name] = rates(counts.sum(axis=0))
            draws[name] = rates(weights @ counts)
            point_results[name] = {metric: interval(draws[name][:, j], observed[name][j])
                                   for j, metric in enumerate(metrics)}
            point_results[name]["counts"] = dict(zip(
                ("intended_mated", "scored_mated", "detected", "intended_non_mated", "scored_non_mated", "false_referrals"),
                [int(x) for x in counts.sum(axis=0)]))
        differences = []
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                differences.append({"left": left, "right": right, "direction": "right minus left",
                    "metrics": {metric: interval(draws[right][:, j] - draws[left][:, j], observed[right][j] - observed[left][j])
                                for j, metric in enumerate(metrics)}})
        analysis: Dict[str, Any] = {"pipelines": point_results, "paired_differences": differences}
        if all(name in runs for name in COMPARISON_PIPELINE_ORDER):
            a, b, c, d = COMPARISON_PIPELINE_ORDER
            analysis["detector_embedder_interaction"] = {
                "contrast": "(SCRFD+ArcFace - YuNet+ArcFace) - (SCRFD+SFace - YuNet+SFace)",
                "metrics": {metric: interval(draws[d][:, j] - draws[c][:, j] - draws[b][:, j] + draws[a][:, j],
                                            observed[d][j] - observed[c][j] - observed[b][j] + observed[a][j])
                            for j, metric in enumerate(metrics)}}
        analyses[scope] = analysis
    return {"artifact_type": "paired_pipeline_comparison", "seed": seed, "replicates": replicates,
            "independent_identities": len(identities), "intended_probes": len(reference),
            "common_success_probes": len(common), "thresholds": dict(thresholds), "analyses": analyses,
            "method": "Paired percentile identity bootstrap stratified by role and subgroup; identical identity draws across methods",
            "limitations": "Exploratory unadjusted 95% intervals. Fixed galleries and frozen policies; uncertainty excludes model fitting, threshold estimation, gallery selection and population shift. Common-success analysis describes a selected subset only. Zero-event bootstrap intervals cannot bound population FPIR."}


def generate_comparative_statistics(output_root: Path = AGGREGATE_ROOT) -> Dict[str, Any]:
    """Verify canonical records and publish paired comparisons without changing policies."""
    config = EnvironmentConfig.load()
    dataset = load_bfw_dataset(*config.require_bfw_roots())
    protocol = build_open_set_protocol(dataset)
    yunet, sface = load_models(config.require_model_root())
    (scrfd, arcface), _description = load_arcface_pipeline(config)
    specs = (
        ("YuNet + SFace", CANONICAL_RUN_CACHE, output_root, yunet, sface),
        ("SCRFD + SFace", mixed_run_cache("scrfd-sface"), output_root / MIXED_PIPELINE_DIRNAME / "scrfd-sface", scrfd, sface),
        ("YuNet + ArcFace", mixed_run_cache("yunet-arcface"), output_root / MIXED_PIPELINE_DIRNAME / "yunet-arcface", yunet, arcface),
        ("SCRFD + ArcFace", ARCFACE_RUN_CACHE, output_root / ARCFACE_REVIEW_DIRNAME, scrfd, arcface),
    )
    runs: Dict[str, OpenSetRunResult] = {}
    thresholds: Dict[str, float] = {}
    provenance: Dict[str, str] = {}
    for name, base, root, detector, embedder in specs:
        path = canonical_cache_path("test", base)
        if not path.is_file():
            raise PipelineComparisonError("Paired statistics require completed Experiments 6, 8, 11 and 12.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        context = canonical_run_context(protocol, partition="test", dataset=dataset, detector=detector, embedder=embedder)
        reason = cache_invalidation_reason(path, context)
        if reason:
            raise PipelineComparisonError(f"Paired comparison cache for {name} is stale: {reason}. Rerun that experiment.")
        policy_path = root / "bfw_open_set_threshold.json"
        policy = read_json_artifact(policy_path)
        if policy.get("public_manifest_sha256", policy.get("protocol_digest")) != public_manifest_digest(protocol):
            raise PipelineComparisonError(f"Frozen policy for {name} belongs to another protocol.")
        thresholds[name] = require_frozen_open_set_policy(policy)
        runs[name] = _run_from_cached_payload(payload)
        provenance[name] = canonical_run_digest(runs[name])
    result = paired_pipeline_bootstrap(runs, thresholds)
    result["canonical_run_digests"] = provenance
    result["public_manifest_sha256"] = public_manifest_digest(protocol)
    write_json_artifact(output_root / "comparative_statistics.json", result)
    return result


def format_zero_event_note(interval: Mapping[str, Any]) -> str:
    bound = interval.get("zero_event_identity_bound")
    if not bound:
        return "Zero-event bootstrap intervals such as 0%–0% cannot bound population FPIR."
    return (f"No false referrals were observed across {bound['independent_identities']} independent identity clusters. "
            f"The 95% one-sided upper bound for an identity having any false referral in its probe set is "
            f"{format_percentage(bound['one_sided_upper'])}, assuming independent exchangeable identities. "
            "This is not a bound on conditional per-image FPIR; within-identity image dependence is unrestricted.")


def measure_classifier_latency(classifier: ReviewClassifier, matrix: np.ndarray) -> Dict[str, Any]:
    """Warm single-query timings; exclude feature extraction and model fitting."""
    if matrix.ndim != 2 or len(matrix) == 0:
        raise MlReviewError("Cannot time the classifier without a nonempty feature matrix.")
    for row in matrix[:min(10, len(matrix))]:
        classifier.probabilities(row[None, :])
    timings = []
    for row in matrix:
        start = time.perf_counter()
        classifier.probabilities(row[None, :])
        timings.append((time.perf_counter() - start) * 1000.0)
    return {"classifier_decision_latency_mean_ms": statistics.fmean(timings),
            "classifier_decision_latency_p95_ms": percentile(timings, 95),
            "classifier_latency_measurement": "One warm single-query call per scored probe; inference only; wall-clock milliseconds",
            "classifier_latency_samples": len(timings)}


DIAGNOSTIC_REVISION = "calibrated-controls-gallery-features-quality-v1"
DIAGNOSTIC_GALLERY_SIZES = (25, 50, 100, 200)
DIAGNOSTIC_GALLERY_SEEDS = (20260917, 20260918, 20260919)
DIAGNOSTIC_FEATURE_SETS = {
    "similarity_only": ("top1_similarity",),
    "similarity_and_margin": ("top1_similarity", "top1_top2_margin"),
    "all_features": ML_REVIEW_FEATURES,
}


def gallery_sensitivity_protocol(
    protocol: OpenSetProtocol, size: int, *, seed: int, probe_core_size: int = 25,
) -> OpenSetProtocol:
    """Nested balanced galleries with a fixed probe core; only distractors change."""
    if size < probe_core_size or probe_core_size < 1:
        raise ProtocolError("Gallery size must be at least the positive fixed probe-core size.")
    entries: List[OpenSetEntry] = []
    for partition in ("development", "test"):
        source = protocol.partition(partition)
        subgroup_of = {e.identity: e.subgroup for e in source if e.role == "gallery_enrolment"}
        buckets: Dict[str, List[str]] = {}
        for identity in sorted(subgroup_of):
            buckets.setdefault(subgroup_of[identity], []).append(identity)
        rng = random.Random(f"{seed}:gallery:{partition}")
        groups = sorted(buckets)
        rng.shuffle(groups)
        for group in groups:
            rng.shuffle(buckets[group])
        ordered = [buckets[group][index] for index in range(max(map(len, buckets.values()), default=0))
                   for group in groups if index < len(buckets[group])]
        if size > len(ordered):
            raise ProtocolError(f"Requested gallery size {size} exceeds available {partition} identities.")
        gallery_ids, probe_ids = set(ordered[:size]), set(ordered[:probe_core_size])
        entries.extend(e for e in source if e.role == "non_mated_probe"
                       or (e.role == "gallery_enrolment" and e.identity in gallery_ids)
                       or (e.role == "mated_probe" and e.identity in probe_ids))
    _assert_protocol_invariants(entries)
    return OpenSetProtocol(entries, protocol.seed, {
        **protocol.provenance, "diagnostic_gallery_size": size,
        "gallery_sampling_seed": seed, "fixed_mated_probe_core": probe_core_size,
    })


def _diagnostic_metrics(run: OpenSetRunResult, threshold: float) -> Dict[str, Any]:
    return {"rates": open_set_rates_at_threshold(run.search_results, threshold),
            "coverage": open_set_coverage(run), **open_set_duplicate_detection(run, threshold)}


def classifier_feature_ablation(
    protocol: OpenSetProtocol, development: OpenSetRunResult, test: OpenSetRunResult,
    *, target_fpir: float = PRIMARY_FPIR_TARGET,
) -> Dict[str, Any]:
    training, calibration = split_development_identities_for_classifier(protocol)
    train_names, calibration_names = set(training), set(calibration)
    training_ids = {e.identity_hash for e in protocol.partition("development") if e.identity in train_names}
    calibration_ids = {e.identity_hash for e in protocol.partition("development") if e.identity in calibration_names}
    dev_rows, _ = build_review_feature_rows(development.search_results)
    test_rows, test_excluded = build_review_feature_rows(test.search_results)
    train_rows = [r for r in dev_rows if r.identity_hash in training_ids]
    cal_rows = [r for r in dev_rows if r.identity_hash in calibration_ids]
    outcomes = build_review_identity_outcomes(test.search_results)
    if training_ids & calibration_ids or (training_ids | calibration_ids) & {r.identity_hash for r in test.search_results}:
        raise ProtocolError("Classifier ablation partitions share identities.")
    fitted: Dict[str, Tuple[ReviewClassifier, float, Dict[str, Any]]] = {}
    # Freeze every variant before inspecting any test prediction.
    for name, features in DIAGNOSTIC_FEATURE_SETS.items():
        classifier = fit_review_classifier(train_rows, features=features)
        cal_matrix, _ = _feature_matrix(cal_rows, features=features)
        selection = select_review_probability_threshold(cal_rows, classifier.probabilities(cal_matrix), target_fpir=target_fpir)
        fitted[name] = (classifier, selection["probability_threshold"], selection)
    result: Dict[str, Any] = {}
    for name, (classifier, threshold, selection) in fitted.items():
        matrix, _ = _feature_matrix(test_rows, features=classifier.feature_order)
        probabilities = classifier.probabilities(matrix)
        result[name] = {"features": list(classifier.feature_order), "model": classifier.as_dict(),
            "calibration": selection, "threshold_status": ML_REVIEW_STATUS_FROZEN,
            "rates": review_rates_at_probability(test_rows, probabilities, threshold, outcomes=outcomes),
            "training_rows": len(train_rows), "calibration_rows": len(cal_rows),
            "test_rows": len(test_rows), "test_excluded_records": test_excluded}
    return result


def quality_error_analysis(
    development: OpenSetRunResult, test: OpenSetRunResult,
    quality: Mapping[str, Mapping[str, float]], threshold: float,
) -> Dict[str, Any]:
    """Observational strata defined from development quartiles, with failures retained."""
    report: Dict[str, Any] = {}
    for feature in ("probe_detection_confidence", "probe_face_area_ratio", "image_short_side_pixels", "laplacian_variance"):
        values = [quality.get(r.sample_id, {}).get(feature, float("nan")) for r in development.search_results]
        finite = [x for x in values if math.isfinite(x)]
        if not finite:
            report[feature] = {"status": "not_measurable"}
            continue
        cutoffs = [float(x) for x in np.unique(np.percentile(finite, [25, 50, 75]))]
        buckets: Dict[str, List[OpenSetSearchResult]] = {str(i): [] for i in range(len(cutoffs) + 1)}
        buckets["unavailable"] = []
        for row in test.search_results:
            value = quality.get(row.sample_id, {}).get(feature, float("nan"))
            key = str(int(np.searchsorted(cutoffs, value, side="right"))) if math.isfinite(value) else "unavailable"
            buckets[key].append(row)
        summaries = {}
        for label, rows in buckets.items():
            failures: Dict[str, int] = {}
            for row in rows:
                if row.failure_code:
                    code = row.failure_code.split(":", 1)[0]
                    failures[code] = failures.get(code, 0) + 1
            intended_mated = sum(r.role == "mated_probe" for r in rows)
            detected = sum(r.role == "mated_probe" and r.failure_code is None and r.correct_rank == 1
                           and r.correct_similarity is not None and r.correct_similarity >= threshold for r in rows)
            summaries[label] = {"intended_probes": len(rows), "failure_count": sum(failures.values()),
                "failure_breakdown": failures, "rates": open_set_rates_at_threshold(rows, threshold),
                "intended_mated": intended_mated,
                "end_to_end_duplicate_detection_rate": detected / intended_mated if intended_mated else None}
        report[feature] = {"development_cutoffs": cutoffs, "bins_low_to_high": summaries}
    return report


def run_comparison_diagnostics(output_root: Path = AGGREGATE_ROOT) -> Dict[str, Any]:
    """Add controls, gallery sensitivity, feature ablations and error analysis.

    These analyses are exploratory: they were designed after the benchmark
    test results had been inspected. No existing frozen policy is overwritten.
    """
    config = EnvironmentConfig.load()
    dataset = load_bfw_dataset(*config.require_bfw_roots())
    protocol = build_open_set_protocol(dataset)
    root = output_root / "comparison_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    plan = {"artifact_type": "comparison_diagnostics_protocol", "revision": DIAGNOSTIC_REVISION,
        "target_fpir": PRIMARY_FPIR_TARGET, "gallery_sizes": list(DIAGNOSTIC_GALLERY_SIZES),
        "gallery_sampling_seeds": list(DIAGNOSTIC_GALLERY_SEEDS), "fixed_probe_core_size": 25,
        "feature_sets": {k: list(v) for k, v in DIAGNOSTIC_FEATURE_SETS.items()},
        "status": "exploratory_protocol_frozen_before_diagnostic_scoring",
        "limitations": "Designed after inspection of the benchmark test results. Exploratory analyses without independent confirmatory data. Gallery repeats vary galleries, not the development/test identity split.",
        "protocol_digest": public_manifest_digest(protocol),
        "image_content_sha256": image_content_digest([e.image_path for e in protocol.entries]),
        "software_environment": software_environment_report()}
    write_json_artifact(root / "protocol.json", plan)
    pipeline_specs = [("YuNet + SFace", "yunet-sface"), ("SCRFD + ArcFace", "scrfd-arcface")]
    outputs: Dict[str, Any] = {}
    for display, slug in pipeline_specs:
        announce(f"Comparison diagnostics: {display}")
        if slug == "yunet-sface":
            detector, embedder = load_models(config.require_model_root())
            description = primary_pipeline_description(detector, embedder)
        else:
            if not pipeline_comparison_status(config)["comparison_run"]:
                outputs[display] = {"status": "not_run_models_unavailable"}
                continue
            (detector, embedder), description = load_arcface_pipeline(config)
        memory: Dict[str, Any] = {}

        def memo_embed(entry, active_detector, active_embedder):
            if entry.sample_id not in memory:
                memory[entry.sample_id] = _embed_open_set_entry(entry, active_detector, active_embedder)
            return memory[entry.sample_id]

        def score(selected: OpenSetProtocol, partition: str, method: str, label: str) -> OpenSetRunResult:
            context = canonical_run_context(selected, partition=partition, dataset=dataset, detector=detector, embedder=embedder)
            context.update({"diagnostic_revision": DIAGNOSTIC_REVISION, "enrolment_method": method,
                            "image_content_sha256": plan["image_content_sha256"]})
            path = RAW_ROOT / "comparison_diagnostics" / slug / f"{label}_{partition}.json"
            if cache_invalidation_reason(path, context) is None:
                cached = load_canonical_run(path)
                if cached is not None:
                    announce(f"  Reusing verified {label} {partition} scores")
                    return cached
            announce(f"  Scoring {label}, {partition}")
            run = run_open_set_method(selected, partition=partition, method=method,
                                      detector=detector, embedder=embedder, embed_fn=memo_embed)
            save_canonical_run(run, path, context)
            return run

        development: Dict[str, OpenSetRunResult] = {}
        policies: Dict[str, Any] = {}
        for name, method in (("one_image", METHOD_A), ("three_images", METHOD_B)):
            development[name] = score(protocol, "development", method, name)
            policies[name] = select_open_set_threshold(development[name].search_results, target_fpir=PRIMARY_FPIR_TARGET)
        write_json_artifact(root / f"{slug}_enrolment_policy.json", {
            "status": OPEN_SET_STATUS_FROZEN, "developed_on": "development identities only",
            "policies": policies, "diagnostic_revision": DIAGNOSTIC_REVISION})
        test_runs = {name: score(protocol, "test", method, name)
                     for name, method in (("one_image", METHOD_A), ("three_images", METHOD_B))}
        thresholds = {name: policy["threshold"] for name, policy in policies.items()}
        controls = {name: {"calibration": policies[name], **_diagnostic_metrics(run, thresholds[name])}
                    for name, run in test_runs.items()}
        paired = paired_pipeline_bootstrap(test_runs, thresholds)
        gallery_rows = []
        for seed in DIAGNOSTIC_GALLERY_SEEDS:
            for size in DIAGNOSTIC_GALLERY_SIZES:
                selected = gallery_sensitivity_protocol(protocol, size, seed=seed)
                label = f"gallery_{size}_seed_{seed}"
                dev = score(selected, "development", METHOD_B, label)
                policy = select_open_set_threshold(dev.search_results, target_fpir=PRIMARY_FPIR_TARGET)
                # Persist the selected policy before test scoring.
                write_json_artifact(root / f"{slug}_{label}_policy.json", {
                    "status": OPEN_SET_STATUS_FROZEN, "calibration": policy,
                    "protocol_digest": public_manifest_digest(selected)})
                test = score(selected, "test", METHOD_B, label)
                gallery_rows.append({"gallery_size_intended": size, "gallery_seed": seed,
                    "calibration": policy, **_diagnostic_metrics(test, policy["threshold"])})
        announce("  Comparing classifier feature sets on identical training/calibration/test groups")
        ablations = classifier_feature_ablation(protocol, development["three_images"], test_runs["three_images"])
        # Image-quality features are observational; no pose or lighting ground truth is invented.
        quality: Dict[str, Dict[str, float]] = {}
        for run in (development["three_images"], test_runs["three_images"]):
            for row in run.search_results:
                quality[row.sample_id] = {k: float(value) for k in ("probe_detection_confidence", "probe_face_area_ratio")
                                         if (value := getattr(row, k)) is not None}
        import cv2
        announce("  Analysing resolution, blur proxies and extraction failures")
        for entry in protocol.entries:
            if entry.role == "gallery_enrolment":
                continue
            try:
                loaded = load_image_bgr(entry.image_path)
                height, width = loaded.bgr.shape[:2]
                gray = cv2.cvtColor(loaded.bgr, cv2.COLOR_BGR2GRAY)
                # Standardised size makes the blur proxy less dependent on raw dimensions.
                gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
                quality[entry.sample_id]["image_short_side_pixels"] = float(min(height, width))
                quality[entry.sample_id]["laplacian_variance"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            except ImageLoadError:
                pass
        outputs[display] = {"status": "evaluated", "pipeline": description.as_dict(),
            "enrolment_controls": controls, "enrolment_paired_statistics": paired,
            "gallery_sensitivity": gallery_rows, "feature_ablations": ablations,
            "quality_analysis": quality_error_analysis(development["three_images"], test_runs["three_images"], quality, thresholds["three_images"]),
            "quality_note": "Development quartile bins; whole-image Laplacian variance at 128x128 is a blur proxy, not a causal or face-quality label. Missing detection metadata retains unresolved failures. Pose is assessed separately by CPLFW; lighting is not labelled.",
            "timing_note": "Extraction times originate from the first in-memory extraction per image; each gallery search is timed separately. Gallery-size results describe this Python exact-search implementation."}
        write_json_artifact(root / f"{slug}_metrics.json", outputs[display])
        memory.clear()
    payload = {"artifact_type": "comparison_diagnostics", "protocol": plan, "pipelines": outputs}
    write_json_artifact(root / "comparison_diagnostics.json", payload)
    write_markdown_artifact(root / "COMPARISON_DIAGNOSTICS_REPORT.md", render_comparison_diagnostics_summary(output_root))
    if find_path_leaks(root, forbidden_substrings=default_forbidden_path_substrings()):
        raise PrivacyLeakError("Private paths detected in comparison diagnostics.")
    assert_no_identifier_key_leak(root)
    return payload


def render_comparison_diagnostics_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    path = output_root / "comparison_diagnostics" / "comparison_diagnostics.json"
    if not path.is_file():
        return "Comparison diagnostics are not available. Run option 18 or --mode comparison-diagnostics."
    payload = read_json_artifact(path)
    lines = ["# Comparison diagnostics", "", payload["protocol"]["limitations"], "",
             f"All variants use development FPIR target {format_percentage(payload['protocol']['target_fpir'])}; actual test FPIR can differ."]
    for name in ("YuNet + SFace", "SCRFD + ArcFace"):
        result = payload["pipelines"][name]
        lines += ["", f"## {name}", ""]
        if result["status"] != "evaluated":
            lines.append(plain_status(result["status"]))
            continue
        lines += ["### Calibrated enrolment", "",
                  "| Images | Test FPIR | Conditional TPIR@1 | End-to-end detection | Mated coverage |",
                  "| --- | --- | --- | --- | --- |"]
        for variant, item in result["enrolment_controls"].items():
            rates, coverage = item["rates"], item["coverage"]
            lines.append(f"| {variant.replace('_', ' ')} | {format_percentage(rates['fpir'])} | {format_percentage(rates['tpir_rank1'])} | "
                         f"{format_percentage(item['end_to_end_duplicate_detection_rate'])} | "
                         f"{format_percentage(1 - coverage['mated_extraction_failure_rate'])} |")
        delta = result["enrolment_paired_statistics"]["analyses"]["full_protocol"]["paired_differences"][0]["metrics"]["end_to_end_duplicate_detection_rate"]
        lines += ["", f"Three minus one image, end-to-end difference: {100*delta['estimate']:+.2f} percentage points "
                  f"(paired 95% CI {100*delta['lower_95']:+.2f} to {100*delta['upper_95']:+.2f}).", "",
                  "### Gallery size", "", "Fixed 25-identity mated probe core and all protocol non-mated probes within each repeat. Three seeds; ranges are repeat ranges, not confidence intervals.", "",
                  "| Intended identities | Mean test FPIR (range) | Mean end-to-end detection | Mean search ms |",
                  "| --- | --- | --- | --- |"]
        for size in DIAGNOSTIC_GALLERY_SIZES:
            rows = [r for r in result["gallery_sensitivity"] if r["gallery_size_intended"] == size]
            fpir = [r["rates"]["fpir"] for r in rows]
            times = [r["coverage"]["top1_search_time_mean_ms"] for r in rows]
            lines.append(f"| {size} | {format_percentage(statistics.fmean(fpir))} ({format_percentage(min(fpir))}–{format_percentage(max(fpir))}) | "
                         f"{format_percentage(statistics.fmean(r['end_to_end_duplicate_detection_rate'] for r in rows))} | {statistics.fmean(times):.3f} |")
        lines += ["", "### Classifier features", "", "Same train/calibration identities, complete feature rows and fixed hyperparameters; outputs are referral scores, not calibrated real-world probabilities.", "",
                  "| Features | Test FPIR | Conditional TPIR@1 | End-to-end detection |", "| --- | --- | --- | --- |"]
        for variant in DIAGNOSTIC_FEATURE_SETS:
            item = result["feature_ablations"][variant]
            rates = item["rates"]
            lines.append(f"| {variant.replace('_', ' ')} | {format_percentage(rates['fpir'])} | {format_percentage(rates['tpir_rank1'])} | {format_percentage(rates['end_to_end_duplicate_detection_rate'])} |")
        lines += ["", "### Error analysis", "", result["quality_note"], "",
                  "| Feature | Development-defined range | Intended probes | Processing failures | Conditional TPIR@1 | End-to-end detection |", "| --- | --- | --- | --- | --- | --- |"]
        for feature, analysis in result["quality_analysis"].items():
            cutoffs = analysis.get("development_cutoffs", [])
            for label, item in analysis.get("bins_low_to_high", {}).items():
                if label == "unavailable":
                    interval_label = "Measurement unavailable"
                else:
                    index = int(label)
                    low = f"{cutoffs[index - 1]:.3g} ≤ value" if index else "value"
                    interval_label = low + (f" < {cutoffs[index]:.3g}" if index < len(cutoffs) else "")
                lines.append(f"| {feature.replace('_', ' ')} | {interval_label} | {item['intended_probes']} | {item['failure_count']} | {format_percentage(item['rates']['tpir_rank1'])} | {format_percentage(item['end_to_end_duplicate_detection_rate'])} |")
    lines += ["", REFERRAL_DISCLAIMER]
    return "\n".join(lines) + "\n"


# =============================================================================
# 28. Figure generation
# =============================================================================
#
# Every figure is built from the published JSON and CSV artefacts, so a chart
# cannot drift from the numbers it illustrates. Axes start at zero unless a log
# scale is stated, no three-dimensional effects are used, and PNG text metadata
# is stripped before publication so a renderer cannot leak a local path.
##############
# Title: Matplotlib: A 2D Graphics Environment
# Author: Hunter, J.D., Computing in Science and Engineering, 9(3), pp. 90-95
# Date: 2007
# Availability: https://doi.org/10.1109/MCSE.2007.55
##############

FIGURE_DPI = 300


def _figure_backend():
    """Import matplotlib with a headless backend, or explain the blocker."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - pinned dependency
        raise ArtifactError(
            "matplotlib is required to generate figures. Install it from requirements.txt."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, path: Path) -> None:
    """Write PNG and SVG, then strip PNG text metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", metadata={"Creator": None})
    _strip_png_text_metadata(path)


def _strip_png_text_metadata(path: Path) -> None:
    """Rewrite the PNG without tEXt/iTXt chunks, which can carry a local path."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pinned dependency
        return
    with Image.open(path) as image:
        pixels = image.copy()
    pixels.save(path, format="PNG", optimize=True)


def _percent(value: Any) -> float:
    return float(value) * 100.0 if isinstance(value, (int, float)) and value == value else float("nan")


def _write_figure_captions(
    aggregate_root: Path, figures_root: Path, written: Sequence[Path]
) -> None:
    """Caption every figure with its denominator and an interpretation.

    A chart without its sample size invites over-reading, so the denominators
    are stated beside each figure rather than left in the JSON."""
    def load(name: str) -> Optional[Dict[str, Any]]:
        path = aggregate_root / name
        return read_json_artifact(path) if path.is_file() else None

    open_set = load("bfw_open_set_test_metrics.json")
    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")

    mated = non_mated = enrolled = intended_gallery = "n/a"
    if open_set:
        coverage = open_set["methods"][METHOD_B]["coverage"]
        mated = coverage["scored_mated_probes"]
        non_mated = coverage["scored_non_mated_probes"]
        enrolled = coverage["enrolled_gallery_identities"]
        # Coverage is enrolled over *intended*; using the enrolled count as the
        # denominator would state a rate of one by construction.
        intended_gallery = coverage["intended_gallery_identities"]

    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")
    consistency = load("profile_photo_consistency.json")
    evaluated = bool(pipeline and pipeline.get("evaluated") == "yes")

    lines = [
        "# Figure captions",
        "",
        "Generated from the published JSON and CSV artefacts by `ACP_arden.py`. No value is "
        "typed by hand. Every figure is PNG at 300 dpi plus SVG, with PNG text metadata "
        "stripped, and passes the project privacy scan.",
        "",
        f"**Denominators (BFW held-out test):** {enrolled} of {intended_gallery} gallery "
        f"identities enrolled, {mated} scored mated probes, {non_mated} scored non-mated "
        f"probes.",
        "",
        "**Metric definitions.** FPIR is the proportion of non-mated searches returning at "
        "least one candidate above threshold — a 1:N quantity that compounds with gallery "
        "size, and never interchangeable with the 1:1 false-match rate. TPIR@k is the "
        "proportion of mated searches placing the correct identity within rank k *and* above "
        "threshold; a referral to another identity is a referral, not an identification. "
        "End-to-end detection divides by every intended mated probe, so extraction failures "
        "reduce it; conditional rates divide by those actually scored.",
        "",
        "**Confidence intervals.** All intervals are 95% percentile bounds from an "
        "identity-cluster bootstrap: identities are resampled with replacement, carrying "
        "their complete protocol outcomes, with subgroup stratification preserved. Images "
        "are never resampled independently, which would treat correlated probes of one "
        "person as independent observations and understate the intervals.",
        "",
        "**Coefficients describe association inside the fitted classifier on these "
        "benchmark identities. They are not causal and do not transfer to another "
        "population.**",
        "",
        f"**Experiment 8 status:** `{pipeline['status'] if pipeline else 'not run'}`.",
        "",
        "## Result order",
        "",
        "Captions follow the order in which the study developed, so each layer's intent is "
        "visible before the pretrained comparison.",
        "",
        "### 1. LFW 1:1 verification",
        "",
        "Official ten-fold pairwise cross-validation with a training-only threshold per fold, reported as accuracy, FMR and "
        "FNMR over scored pairs. A 1:1 quantity that never appears on an FPIR axis: one "
        "comparison, no competing candidates, no ranking. Conditional on scored pairs.",
        "",
        "### 2. CPLFW cross-pose evaluation",
        "",
        "The same frozen threshold on raw cross-pose images. Conditional accuracy only, "
        "always quoted with its extraction-failure rate; cross-pose detection rather than "
        "comparison is the dominant effect.",
        "",
        "## Implementation layers (results 3-6)",
        "",
        "The five layers share the BFW open-set protocol and show the order the project "
        "developed them. Coverage and calibration can also differ; adjacent contrasts "
        "are descriptive. The separately calibrated enrolment comparison and the "
        "crossed detector/recogniser experiment address these confounders:",
        "",
        "3. Single-image gallery, transferred 1:1 threshold",
        "4. Three-image gallery, transferred 1:1 threshold — observed TPIR and FPIR both "
        "increase under this transferred policy; this does not establish an inherent "
        "disadvantage of averaging normalised embeddings",
        "5. Three-image gallery, BFW development calibration — isolates the policy "
        "change while keeping the three-image representation fixed",
        "6. Logistic-regression review classifier, frozen probability threshold",
        "7. SCRFD + ArcFace, its own frozen BFW development calibration",
        "",
        "LFW and CPLFW are 1:1 verification and are deliberately absent from this series: "
        "mixing an FMR into an FPIR axis would compare different quantities.",
        "",
        "- **implementation_layers_fpir** — false review referrals per 1,000 non-mated "
        f"searches over {non_mated} scored probes. Lower is better.",
        "- **implementation_layers_duplicate_detection** — TPIR@1, TPIR@5 and end-to-end "
        "detection, kept as separate bars because they use different denominators. Higher "
        "is better.",
        "- **implementation_layers_coverage** — gallery, mated and non-mated coverage. The "
        "remainder in each bar is extraction failure, which is shown rather than hidden.",
        "- **implementation_layers_performance_latency** — end-to-end duplicate detection "
        "is plotted against mean complete-pipeline latency in milliseconds. Marker area "
        "represents false reviews per 1,000 non-mated searches, so a larger marker "
        "means more false reviews. Colour identifies the pipeline: layers 1-4 share "
        "YuNet + SFace, layer 5 is SCRFD + ArcFace. Each point is labelled with its "
        "layer number and a short method name. Better operating points lie towards "
        "the upper-left with smaller markers. The latency axis begins at zero so the "
        "cost difference reads as a ratio; the detection axis is padded around the "
        "observed values rather than spanning 0-100%, because every layer lies above "
        "85% and a full range would hide the differences. Layers 2, 3 and 4 record "
        "an identical latency because they reuse the same extraction pipeline and "
        "differ only in threshold or decision rule, so their markers share one "
        "horizontal position. Latency excludes one-time model loading and is "
        "specific to the recorded local evaluation environment; it is not a "
        "portable performance claim. A layer without a measured complete-pipeline "
        "latency is omitted from this figure rather than given an invented value.",
        "",
        "## Same-person and profile-photo figures (Figures E-F)",
        "",
        "- **mated_non_mated_similarity_distributions** — one panel per evaluated "
        "pipeline, each marking its own frozen threshold. Mated scores are similarity to "
        "the probe's own enrolled template; non-mated scores are top-1 similarity against "
        "a gallery the person is not in. Aggregate histograms only: bin edges and counts, "
        "never an individual score, identifier or path.",
    ]
    if consistency:
        lines.append(
            f"- **profile_photo_consistency_outcomes** — every evaluated pipeline, over "
            f"{consistency['intended_same_person_photographs']} same-person photographs "
            f"per pipeline. Outcomes are not equivalent: a **consistent** photograph "
            f"opens no case; an **inconsistent** one — a *low* similarity to the "
            f"profile's own enrolled template — opens a consistency review; an "
            f"**extraction failure** resolves nothing and is a separate unresolved "
            f"outcome rather than a decision. Two different controls appear separately. "
            f"The **open-set control** searches a person absent from the gallery against "
            f"every enrolled profile, and is the stricter test; the **wrong-template "
            f"control** compares one photograph with exactly one deterministically "
            f"assigned wrong profile, and is the direct one-to-one comparison. An "
            f"inconsistent result is **not** proof of photo theft or fraud: pose, "
            f"lighting, occlusion, image quality, age difference, detection failure and "
            f"model error all produce it."
        )

    lines += [
        "",
        "## 7-8. Female and male subgroup evaluation",
        "",
        "Sex is an evaluation dimension only: never a classifier feature, threshold input, "
        "calibration variable, or reason to apply a different decision policy. The female "
        "panel covers asian, black, indian and white females; the male panel covers the "
        "same four categories. Both use identical metric order, units and interval format "
        "so they compare fairly.",
        "",
        "FPIR is plotted on its own axis with a metric-specific upper bound, not on the "
        "0-100% axis used for TPIR and coverage: these FPIR values are fractions of one "
        "per cent, and compressing them against a 0-100% scale would flatten every bar to "
        "the baseline and hide the difference the experiment is about. The female and male "
        "companion figures share identical FPIR axis limits, computed across both sexes "
        "and both pipelines before either figure is drawn.",
        "",
        "- **female_subgroup_pipeline_comparison** / **male_subgroup_pipeline_comparison** — "
        "FPIR (lower better, own axis), TPIR@1 and TPIR@5 (higher better), mated coverage "
        "and non-mated coverage, each with 95% identity-cluster bounds.",
        "- **female_male_aggregate_comparison** — pooled from underlying identity outcomes, "
        "not by averaging four subgroup percentages, which would weight a small subgroup as "
        "heavily as a large one. FPIR occupies a separate panel for the same reason.",
        "",
        "> A zero-event percentile-bootstrap interval such as 0%–0% means that no false "
        "referral was observed among the resampled benchmark identities. It does not "
        "establish that the population error probability is exactly zero.",
        "",
        "These are binary dataset categories. They do not represent the full range of "
        "gender identities, every identity, or any real deployed population.",
        "",
        "## 9. Profile-photo consistency analysis",
        "",
        "A same-identity probe stands for a photograph belonging to the enrolled person. "
        "Two different controls appear, and they are not interchangeable. The **open-set "
        "non-mated gallery control** searches a person absent from the gallery against "
        "every enrolled profile, testing whether they avoid matching any of them; it is "
        "the stricter test. The **wrong-profile-template control** compares one photograph "
        "with exactly one deterministically assigned wrong profile, and is the direct "
        "one-to-one comparison. Referral is the correct outcome for both.",
        "",
        "> A non-match indicates that the photograph is inconsistent with the enrolled "
        "facial template under the evaluated model and threshold. It does not prove that "
        "the photograph belongs to another person or that fraud occurred.",
        "",
        "## 10-11. Pipeline comparison and the latency trade-off",
        "",
        "- **pipeline_coverage_and_latency** — both pipelines once Experiment 8 is "
        "evaluated, each with its own frozen development threshold; the SFace threshold is "
        "never applied to ArcFace. A complete-pipeline comparison: detection, alignment, "
        "preprocessing and embedding width all differ, so no difference is attributable to "
        "the embedding model alone.",
        "- **implementation_layers_performance_latency** — end-to-end duplicate detection "
        "is plotted against mean complete-pipeline latency in milliseconds. Marker area "
        "represents false reviews per 1,000 non-mated searches, so a larger marker "
        "means more false reviews. Colour identifies the pipeline: layers 1-4 share "
        "YuNet + SFace, layer 5 is SCRFD + ArcFace. Each point is labelled with its "
        "layer number and a short method name. Better operating points lie towards "
        "the upper-left with smaller markers. The latency axis begins at zero so the "
        "cost difference reads as a ratio; the detection axis is padded around the "
        "observed values rather than spanning 0-100%, because every layer lies above "
        "85% and a full range would hide the differences. Layers 2, 3 and 4 record "
        "an identical latency because they reuse the same extraction pipeline and "
        "differ only in threshold or decision rule, so their markers share one "
        "horizontal position. Latency excludes one-time model loading and is "
        "specific to the recorded local evaluation environment; it is not a "
        "portable performance claim. A layer without a measured complete-pipeline "
        "latency is omitted from this figure rather than given an invented value.",
        "",
        "## The later comparisons (Experiments 9 to 12)",
        "",
        "- **pipeline_across_datasets** — both pipelines on one-to-one verification, "
        "accuracy on the left and the share of photographs reaching comparison on the "
        "right, for LFW and CPLFW. The two panels move in opposite directions on LFW "
        "and must not be read as one quantity: the accuracy panel is conditional on "
        "scored pairs, while coverage counts every intended pair. The coverage "
        "difference on LFW includes rejections under the exactly-one-face rule; the "
        "failure categories do not establish which detections are true faces.",
        "- **detector_embedder_crossed** — the four detector and embedder combinations "
        "on the same held-out identities, each at a threshold frozen on the "
        "development identities. Colour identifies the embedder, so a pair of bars of "
        "one colour shows the detector varied at a fixed embedder. Detection carries "
        "95% identity-cluster bootstrap bounds. Referral points are descriptive; "
        "paired referral differences and their intervals appear in the paired report. "
        "The conditional detection axis starts at 85%, explicitly truncating the "
        "scale. Scored subsets differ, so read this alongside the next figure.",
        "- **paired_pipeline_comparison** — all four pipelines on a common 0–100% "
        "scale: mated extraction coverage, detection across all intended mated probes, "
        "and TPIR@1 within the common-success subset. Bars show 95% identity-cluster "
        "intervals. The accompanying paired report estimates differences using the "
        "same identity draws for every pipeline; interval overlap alone is not a test.",
        "",
        "## Open-set operating points and the review classifier",
        "",
        "- **open_set_operating_curve** — TPIR@1 (per cent, higher is better) against "
        "FPIR (a proportion of non-mated searches, log scale, lower is better). Two "
        "series: development, on which the threshold was selected, and the held-out "
        "test, which never influenced selection. The log axis cannot show zero, so an "
        "observed FPIR of zero is drawn at 1e-4; such a point marks the absence of an "
        "observed false referral, not a measured rate of 1e-4.",
        "- **duplicate_detection_by_method** — conditional TPIR@1, end-to-end duplicate "
        "detection and gallery enrolment coverage as separate bars on a 0-100% axis. "
        "They are kept apart because they use different denominators: conditional rates "
        "divide by what was scored, end-to-end rates by every intended probe. Higher is "
        "better in all three.",
        "- **false_reviews_per_1000_by_method** — false human-review referrals per 1,000 "
        "non-mated searches, comparing the single-image control with the three-image "
        "proposed method. Lower is better. The count, not a percentage, is the "
        "operationally meaningful quantity for a review queue.",
        "- **ml_review_classifier_coefficients** — standardised logistic-regression "
        "coefficients, one horizontal bar per feature. Blue is a positive coefficient, "
        "which raises the modelled referral probability; red is negative, which lowers "
        "it. Bar length is the magnitude of the standardised coefficient, so features "
        "are comparable with one another. Feature names are the model's own, as "
        "published in ml_review_model.json. These are associations within this "
        "benchmark and are not causal claims; no demographic attribute is a feature.",
        "- **subgroup_fpir_tpir_with_confidence_intervals** — the review classifier's "
        "FPIR and TPIR@1 for each of the eight BFW subgroups, with 95% identity-cluster "
        "bootstrap bounds. Each panel is bounded by its own observed interval rather "
        "than a shared 0-100% axis, on which a sub-one-per-cent FPIR and a 95% TPIR "
        "would both be unreadable; the axis range therefore differs between the two "
        "panels and should be read from the tick labels. Subgroups appear in a fixed "
        "alphabetical order shared with the other subgroup figures.",
        "",
        "## 12. Limitations",
        "",
        "## Limitations common to every figure",
        "",
        "- Each rate is conditional on the coverage reported beside it.",
        "- Subgroup intervals are wide once the partition is divided eight ways; "
        "overlapping intervals are not evidence of equality.",
        "- These are benchmark identities, not a user population.",
        "- A referral opens human review only. Nothing here proves duplication, fraud, "
        "ownership or identity.",
    ]
    if review:
        c = review["classifier"]
        lines += [
            "",
            "## Note on the classifier",
            "",
            f"The classifier referred {c['non_mated_incorrectly_referred']} non-mated "
            f"searches in error against the calibrated threshold's "
            f"{review['comparator_three_image_open_set_calibrated']['rates'].get('false_reviews_per_1000_non_mated', 0) * (c['scored_non_mated_probes'] / 1000):.0f}. "
            f"Its primary hypothesis — fewer false referrals — is not achieved.",
        ]
    if not evaluated:
        lines += [
            "",
            "## Note on the absent comparison",
            "",
            f"No stronger-pipeline series appears: the comparison did not run "
            f"(`{pipeline['status'] if pipeline else 'unknown'}`). Nothing is estimated in "
            f"its place.",
        ]

    (figures_root / "FIGURE_CAPTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



# The five implementation layers, in the order the project developed them.
# Together they are the experiment behind the research objective: each layer
# adds one component to the previous combination, so comparing consecutive
# layers can also reflect calibration and coverage. All five are measured on the
# same BFW open-set protocol and are therefore directly comparable. LFW and
# CPLFW are 1:1 verification and are never mixed into this series: an FMR and
# an FPIR are not the same quantity.
IMPLEMENTATION_LAYERS = (
    "Layer 1\nsingle image\ntransferred threshold",
    "Layer 2\nthree images\ntransferred threshold",
    "Layer 3\nthree images\nBFW calibration",
    "Layer 4\nreview classifier",
    "Layer 5\nSCRFD + ArcFace\nown calibration",
)

# Presentation only. Short forms for direct point labelling, where the full
# three-line names above would not fit beside a marker.
LAYER_SHORT_LABELS = (
    "Single image + LFW threshold",
    "Three images + LFW threshold",
    "Three images + BFW threshold",
    "Logistic classifier",
    "SCRFD + ArcFace",
)

# One display name per pipeline, used in every figure so a reader does not have
# to match a package identifier to a model pair.
PIPELINE_DISPLAY_NAMES = {
    "opencv": "YuNet + SFace",
    "insightface": "SCRFD + ArcFace",
}

LAYER_PIPELINE_COLOURS = ("#4C72B0", "#DD8452")

# Fixed label offsets in points, chosen because layers 2-4 record an identical
# latency and layers 1 and 3 an identical detection rate: automatic placement
# would stack the labels on top of one another.
LAYER_LABEL_OFFSETS = (
    (34, -26, "left"),
    (34, 26, "left"),
    (34, -52, "left"),
    (34, 0, "left"),
    (-34, 20, "right"),
)

# Reference values for the marker-area legend, spanning the observed range of
# false-review burdens across the five layers.
REVIEW_LEGEND_VALUES = (5.0, 50.0, 100.0, 150.0)


def _review_marker_area(reviews: float) -> float:
    """Marker area in points squared for a false-review rate.

    Area rather than radius is made proportional to the rate, because a reader
    compares the visual area of two markers. The floor keeps a near-zero rate
    visible without suggesting it is larger than it is."""
    if not isinstance(reviews, (int, float)) or reviews != reviews:
        return 40.0
    return 40.0 + float(reviews) * 9.0


def pipeline_display_name(name: str) -> str:
    """Map an internal pipeline identifier to its published display name.

    Figures and reports must not label one pipeline two different ways, so the
    model pair is named rather than the package that provides it."""
    lowered = name.lower()
    for key, label in PIPELINE_DISPLAY_NAMES.items():
        if key in lowered:
            return label
    return name


def subgroup_display_name(subgroup: str) -> str:
    """Render a BFW subgroup key as readable axis text.

    The stored keys use underscores, which read poorly on an axis. Only the
    presentation changes; the key itself remains the value used in every
    artefact."""
    return subgroup.replace("_", " ").strip().capitalize()


# Repeated verbatim wherever a subgroup interval is drawn. A percentile
# bootstrap that observed no event returns 0%-0%, which describes the resampled
# benchmark identities and not the population error probability.
ZERO_EVENT_INTERVAL_NOTE = (
    "A 0%–0% interval means no event was observed among the resampled benchmark "
    "identities. It does not establish a population error probability of zero."
)


def _collect_implementation_layers(aggregate_root: Path) -> List[Dict[str, Any]]:
    """Assemble the layer series from published artefacts only."""
    def load(name: str) -> Optional[Dict[str, Any]]:
        path = aggregate_root / name
        return read_json_artifact(path) if path.is_file() else None

    open_set = load("bfw_open_set_test_metrics.json")
    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")
    layers: List[Dict[str, Any]] = []
    if not open_set:
        return layers

    control = open_set["methods"][METHOD_A]
    proposed = open_set["methods"][METHOD_B]
    layers.append({
        "name": IMPLEMENTATION_LAYERS[0],
        "rates": control["rates"], "coverage": control["coverage"],
        "end_to_end": control.get("end_to_end_duplicate_detection_rate"),
    })
    # Layer 2 reuses the three-image gallery at the transferred threshold, so
    # its end-to-end rate divides the same numerator by every intended mated
    # probe rather than by those that were scored.
    layer2 = proposed["at_lfw_control_threshold"]
    intended_mated = proposed["coverage"]["intended_mated_probes"]
    scored_mated = layer2.get("scored_mated_probes") or 0
    tpir1 = layer2.get("tpir_rank1")
    layers.append({
        "name": IMPLEMENTATION_LAYERS[1],
        "rates": layer2, "coverage": proposed["coverage"],
        "end_to_end": (
            (tpir1 * scored_mated) / intended_mated
            if isinstance(tpir1, (int, float)) and tpir1 == tpir1 and intended_mated
            else None
        ),
    })
    layers.append({
        "name": IMPLEMENTATION_LAYERS[2],
        "rates": proposed["primary_operating_point"], "coverage": proposed["coverage"],
        "end_to_end": proposed.get("end_to_end_duplicate_detection_rate"),
    })
    if review:
        layers.append({
            "name": IMPLEMENTATION_LAYERS[3],
            "rates": review["classifier"], "coverage": review["coverage"],
            "end_to_end": review["classifier"].get("end_to_end_duplicate_detection_rate"),
        })
    if pipeline and pipeline.get("evaluated") == "yes":
        held_out = pipeline.get("held_out_metrics") or {}
        arcface = next(
            (v for k, v in held_out.items() if "arcface" in k.lower()), None
        )
        if arcface:
            layers.append({
                "name": IMPLEMENTATION_LAYERS[4],
                "rates": arcface["rates"], "coverage": arcface["coverage"],
                "end_to_end": arcface.get("end_to_end_duplicate_detection_rate"),
            })
    return layers


def write_implementation_layer_artefacts(
    aggregate_root: Path = AGGREGATE_ROOT
) -> Optional[List[Dict[str, Any]]]:
    """Publish the layer series as machine-readable files.

    Assembled from existing frozen policies and search results; no method or
    threshold is recalculated."""
    layers = _collect_implementation_layers(aggregate_root)
    if not layers:
        return None
    sources = [
        "LFW 1:1 frozen threshold (transferred)",
        "LFW 1:1 frozen threshold (transferred)",
        "BFW development calibration, frozen",
        "BFW classifier-calibration identities, frozen probability",
        "BFW development calibration for SCRFD + ArcFace, frozen",
    ]
    # One row per implementation layer, in the order the project developed
    # them, so the table reads as the progression from the transferred 1:1
    # threshold through to the higher-capacity pipeline.
    rows: List[Dict[str, Any]] = []
    for index, layer in enumerate(layers):
        rates, coverage = layer["rates"], layer["coverage"]
        rows.append({
            "layer": index + 1,
            "method": layer["name"].replace("\n", " "),
            "fpir": rates.get("fpir"),
            "false_reviews_per_1000_non_mated": rates.get("false_reviews_per_1000_non_mated"),
            "tpir_rank1": rates.get("tpir_rank1"),
            "tpir_rank5": rates.get("tpir_rank5"),
            "end_to_end_duplicate_detection_rate": layer["end_to_end"],
            "gallery_enrolment_coverage": coverage.get("gallery_enrolment_coverage"),
            # Coverage is stored as a failure rate, so it is inverted here to
            # report the share of photographs that were processed successfully.
            "mated_probe_coverage": (
                1.0 - coverage["mated_extraction_failure_rate"]
                if isinstance(coverage.get("mated_extraction_failure_rate"), (int, float))
                else None
            ),
            "non_mated_probe_coverage": (
                1.0 - coverage["non_mated_extraction_failure_rate"]
                if isinstance(coverage.get("non_mated_extraction_failure_rate"), (int, float))
                else None
            ),
            "complete_pipeline_latency_mean_ms": coverage.get(
                "complete_pipeline_latency_mean_ms"
            ),
            "search_latency_mean_ms": coverage.get("top1_search_time_mean_ms"),
            "threshold_source": sources[index] if index < len(sources) else "",
        })

    with open(aggregate_root / "implementation_layer_comparison.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_artifact(
        aggregate_root / "implementation_layer_comparison.json",
        {
            "artifact_type": "implementation_layer_comparison",
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "seed": DEFAULT_RANDOM_SEED,
            "layers": rows,
            "note": (
                "All layers share the BFW open-set protocol and are directly comparable. "
                "LFW and CPLFW are 1:1 verification and are deliberately excluded: an FMR "
                "and an FPIR are different quantities. No layer improved every metric."
            ),
            "policy_note": POLICY_NOTE,
        },
    )
    return rows


CONSISTENCY_OUTCOME_FIELDS = (
    "intended_same_person_photographs",
    "scored_same_person_photographs",
    "consistent_same_person_photographs",
    "inconsistent_same_person_review_candidates",
    "same_person_extraction_failures",
    "gallery_reference_unavailable",
    "intended_open_set_non_mated_gallery_controls",
    "scored_open_set_non_mated_gallery_controls",
    "open_set_non_mated_gallery_controls_correctly_identified",
    "open_set_non_mated_gallery_controls_false_consistent",
    "open_set_non_mated_gallery_control_extraction_failures",
    "wrong_profile_template_controls_intended",
    "wrong_profile_template_controls_scored",
    "wrong_profile_template_controls_correctly_inconsistent",
    "wrong_profile_template_controls_false_consistent",
    "wrong_profile_template_control_extraction_failures",
)

CONSISTENCY_RATE_FIELDS = (
    "same_person_consistency_rate_conditional",
    "same_person_false_inconsistency_rate_conditional",
    "same_person_consistency_rate_end_to_end",
    "same_person_false_inconsistency_rate_end_to_end",
    "open_set_non_mated_gallery_detection_conditional",
    "open_set_non_mated_gallery_false_consistency_conditional",
    "open_set_non_mated_gallery_detection_end_to_end",
    "open_set_non_mated_gallery_false_consistency_end_to_end",
    "wrong_profile_template_mismatch_detection_conditional",
    "wrong_profile_template_mismatch_detection_end_to_end",
    "wrong_profile_template_false_consistency_conditional",
    "wrong_profile_template_false_consistency_end_to_end",
    "same_person_extraction_coverage",
    "open_set_non_mated_gallery_control_extraction_coverage",
    "wrong_profile_template_extraction_coverage",
)


def write_profile_consistency_artefacts(
    per_pipeline: Mapping[str, Mapping[str, Any]],
    aggregate_root: Path = AGGREGATE_ROOT,
    provenance: Optional[Mapping[str, Any]] = None,
) -> None:
    """Publish consistency outcomes for every evaluated pipeline.

    Aggregate counts only; no individual photograph, score or identity."""
    write_json_artifact(
        aggregate_root / "profile_photo_consistency_metrics.json",
        {
            "artifact_type": "profile_photo_consistency_metrics",
            "pipelines": {name: dict(entry) for name, entry in per_pipeline.items()},
            "outcome_fields": list(CONSISTENCY_OUTCOME_FIELDS),
            "rate_fields": list(CONSISTENCY_RATE_FIELDS),
            "profile_consistency_policy_note": PROFILE_CONSISTENCY_POLICY_NOTE,
            # The generic screening note has the opposite referral direction, so
            # it is carried only under its own labelled field, never as the
            # policy for a consistency result.
            "duplicate_screening_policy_note": (
                "Applies to gallery screening only, not to profile consistency: " + POLICY_NOTE
            ),
            **{k: v for k, v in (provenance or {}).items() if k != "policy_note"},
        },
    )
    with open(aggregate_root / "profile_photo_consistency_metrics.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # One row per outcome rather than one column per outcome, so a new
        # measure can be added later without changing the file's shape.
        writer.writerow(["pipeline", "outcome", "count"])
        for name, entry in per_pipeline.items():
            # Counts and rates are both written, so a reader can check any
            # published rate against the counts it was derived from.
            for field_name in CONSISTENCY_OUTCOME_FIELDS + CONSISTENCY_RATE_FIELDS:
                writer.writerow([name, field_name, entry.get(field_name, "")])


def write_pipeline_sex_aggregates(
    payload: Mapping[str, Any], aggregate_root: Path = AGGREGATE_ROOT
) -> None:
    """Dedicated per-pipeline female and male aggregates.

    Pooled from identity outcomes, never by averaging subgroup percentages,
    which would weight a small subgroup as heavily as a large one."""
    held_out = payload.get("held_out_metrics") or {}
    # One entry per pipeline and sex, giving four rows once both pipelines have
    # been evaluated. Sex is an evaluation dimension only; it is never used to
    # decide anything about an individual case.
    groups: Dict[str, Dict[str, Any]] = {}
    for name, metrics in held_out.items():
        for sex, entry in (metrics.get("sex_aggregated") or {}).items():
            groups[f"{name} — {sex}"] = {
                "pipeline": name,
                "sex": sex,
                "pipeline_threshold": metrics["development_threshold"],
                "bootstrap_replicates": metrics.get("global_bootstrap_replicates"),
                "seed": DEFAULT_RANDOM_SEED,
                **entry,
            }
    write_json_artifact(
        aggregate_root / "pipeline_sex_aggregated_metrics.json",
        {
            "artifact_type": "pipeline_sex_aggregated_metrics",
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "seed": DEFAULT_RANDOM_SEED,
            "aggregation": "pooled over identity outcomes, not averaged over subgroups",
            "status": payload.get("status"),
            "groups": groups,
            "limitations": (
                "These are binary dataset categories. They do not represent the full range "
                "of gender identities or any real deployed population."
            ),
            "policy_note": POLICY_NOTE,
        },
    )
    columns = [
        "pipeline", "sex", "identities", "pipeline_threshold", "bootstrap_replicates", "seed",
        "intended_mated_probes", "scored_mated_probes", "mated_extraction_failures",
        "intended_non_mated_probes", "scored_non_mated_probes",
        "non_mated_extraction_failures",
        "fpir", "fpir_lower_95", "fpir_upper_95",
        "tpir_rank1", "tpir_rank1_lower_95", "tpir_rank1_upper_95",
        "tpir_rank5", "tpir_rank5_lower_95", "tpir_rank5_upper_95",
        "mated_probe_coverage", "mated_probe_coverage_lower_95", "mated_probe_coverage_upper_95",
        "non_mated_probe_coverage", "non_mated_probe_coverage_lower_95",
        "non_mated_probe_coverage_upper_95",
    ]
    with open(aggregate_root / "pipeline_sex_aggregated_metrics.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for entry in groups.values():
            writer.writerow([entry.get(c, "") for c in columns])


###############################################################################
# Report generation
###############################################################################
#
# Every report is rendered from the published artefacts rather than from live
# objects, so a figure, a table and a JSON file cannot disagree about the same
# quantity.


def render_paired_comparison_report(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    path = aggregate_root / "comparative_statistics.json"
    if not path.is_file():
        return "Paired comparisons are not available. Run option 20 or --mode comparative-statistics."
    payload = read_json_artifact(path)
    full = payload["analyses"]["full_protocol"]
    common = payload["analyses"]["common_success"]
    lines = ["Same intended identities and pipeline-specific development thresholds. "
             "End-to-end detection retains failed extraction and failed enrolment in its denominator.", "",
             "| Pipeline | Conditional TPIR@1 | End-to-end detection | Mated coverage | False reviews / 1,000 scored new probes | Common-success TPIR@1 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name in COMPARISON_PIPELINE_ORDER:
        metrics = full["pipelines"][name]
        lines.append(f"| {name} | {format_percentage(metrics['tpir_rank1']['estimate'])} | "
                     f"{format_percentage(metrics['end_to_end_duplicate_detection_rate']['estimate'])} | "
                     f"{format_percentage(metrics['mated_extraction_coverage']['estimate'])} | "
                     f"{1000*metrics['fpir']['estimate']:.2f} | {format_percentage(common['pipelines'][name]['tpir_rank1']['estimate'])} |")
    lines += ["", f"Common-success subset: {payload['common_success_probes']:,} of {payload['intended_probes']:,} intended probes. "
              "Conditional comparisons across different surviving subsets cannot isolate a detector effect.", "",
              "Paired changes below are right minus left in percentage points. Every bootstrap replicate uses the same identity draws across all methods.", "",
              "| Left → right | End-to-end change (95% CI) | FPIR change (95% CI) |", "| --- | --- | --- |"]
    def band(entry: Mapping[str, Any]) -> str:
        return f"{100*entry['estimate']:+.2f} [{100*entry['lower_95']:+.2f}, {100*entry['upper_95']:+.2f}]"
    for comparison in full["paired_differences"]:
        metrics = comparison["metrics"]
        lines.append(f"| {comparison['left']} → {comparison['right']} | {band(metrics['end_to_end_duplicate_detection_rate'])} | {band(metrics['fpir'])} |")
    interaction = full.get("detector_embedder_interaction")
    if interaction:
        lines += ["", "Detector-by-embedder interaction in end-to-end detection: "
                  + band(interaction["metrics"]["end_to_end_duplicate_detection_rate"]) + " percentage points. "
                  "Contrast: " + interaction["contrast"] + "."]
    lines += ["", payload["limitations"], "",
              "An interval containing zero does not establish equivalence or absence of a contribution. "
              "The effects describe the complete configured detector, alignment, recogniser and calibrated-policy combinations."]
    return "\n".join(lines)


def render_research_report(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """Consolidated write-up, ordered so each layer's intent is visible.

    Every figure is read from the published artefacts. No layer is claimed to
    have improved every metric."""
    def load(name: str, base: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        # ``base`` lets the later sections read the comparison chains, which
        # write into their own sub-directories to avoid overwriting the
        # baseline artefacts of the same name.
        path = (base or aggregate_root) / name
        return read_json_artifact(path) if path.is_file() else None

    final = load("lfw_final_metrics.json")
    cplfw = load("cplfw_metrics.json")
    open_set = load("bfw_open_set_test_metrics.json")
    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")
    consistency = load("profile_photo_consistency.json")
    by_sex = load("bfw_sex_aggregated_metrics.json")
    pct = format_percentage

    lines = [
        "# COM7014 Advanced Computing Project — research report",
        "",
        "Auto-generated from the published artefacts. Ordered to show what each layer was "
        "intended to improve, and where it did not.",
        "",
        "**Research objective.** Assess how detector and recogniser choice, enrolment "
        "images and threshold calibration affect duplicate detection, false-review "
        "burden, subgroup performance and computational cost. Controlled comparisons "
        "separate these factors where the protocol permits. The face networks remain "
        "pretrained and frozen; only the review classifier is trained here.",
        "",
        "## 1. LFW 1:1 verification",
        "",
    ]
    if final:
        lines += [
            f"LFW mean fold accuracy {pct(final.get('mean_fold_accuracy', final.get('accuracy')))}, pooled FMR {pct(final.get('false_match_rate'))}, "
            f"FNMR {pct(final.get('false_non_match_rate'))}, EER "
            f"{pct(final.get('equal_error_rate'))}, extraction failure "
            f"{pct(final.get('failure_rate'))}. This is a 1:1 pair task and its FMR is not "
            f"comparable with the 1:N FPIR figures below. The official ten-fold protocol "
            f"fits a threshold using the other nine folds for each held-out fold; "
            f"the separate development-frozen transfer threshold is not used here.",
        ]
    lines += ["", "## 2. CPLFW cross-pose transfer", ""]
    if cplfw:
        lines += [
            f"Conditional accuracy {pct(cplfw.get('accuracy'))} over "
            f"{format_count(cplfw.get('scored_pairs'))} scored pairs, with "
            f"{pct(cplfw.get('failure_rate'))} of the protocol never reaching comparison. "
            f"Cross-pose *detection*, not comparison, is the dominant finding.",
        ]

    if open_set:
        control = open_set["methods"][METHOD_A]
        proposed = open_set["methods"][METHOD_B]
        primary = proposed["primary_operating_point"]
        lines += [
            "", "## 3. BFW single-image open-set control", "",
            f"FPIR {pct(control['rates']['fpir'])}, TPIR@1 "
            f"{pct(control['rates']['tpir_rank1'])}, "
            f"{format_number(control['rates']['false_reviews_per_1000_non_mated'], 1)} false "
            f"reviews per 1,000. Reusing a 1:1 threshold for 1:N search refers a large share "
            f"of genuinely new identities.",
            "", "## 4. BFW three-image template, same threshold", "",
            f"FPIR {pct(proposed['at_lfw_control_threshold']['fpir'])}, TPIR@1 "
            f"{pct(proposed['at_lfw_control_threshold']['tpir_rank1'])}. Averaging three "
            f"images changes both genuine and impostor score distributions. These "
            f"renormalised templates do not necessarily become closer to every face. "
            f"The separately calibrated one-versus-three-image comparison appears in "
            f"the supplementary diagnostics.",
            "", "## 5. BFW gallery-specific calibration", "",
            f"FPIR {pct(primary['fpir'])}, TPIR@1 {pct(primary['tpir_rank1'])}, "
            f"{format_number(primary['false_reviews_per_1000_non_mated'], 1)} false reviews "
            f"per 1,000. Comparing this row with the preceding three-image row isolates "
            f"the threshold change on the same templates.",
        ]

    if review:
        c = review["classifier"]
        b = review["comparator_three_image_open_set_calibrated"]["rates"]
        achieved = review["success_criteria"].get(
            "fewer_false_reviews_than_threshold_method", {}
        ).get("outcome")
        lines += [
            "", "## 6. Logistic-regression review classifier", "",
            f"FPIR {pct(c['fpir'])} against the threshold method's {pct(b['fpir'])}; TPIR@1 "
            f"{pct(c['tpir_rank1'])} against {pct(b['tpir_rank1'])}; "
            f"{format_number(c['false_reviews_per_1000_non_mated'], 2)} false reviews per "
            f"1,000 against {format_number(b['false_reviews_per_1000_non_mated'], 2)}.",
            "",
            f"The primary hypothesis was that the classifier would reduce false review "
            f"referrals while retaining detection. That criterion is "
            f"**{str(achieved).replace('_', ' ')}**. The classifier raises identification "
            f"while referring more innocent registrations, which is a trade-off rather than "
            f"an improvement.",
        ]

    held_out_all = (pipeline or {}).get("held_out_metrics") or {}
    sex_sources: Dict[str, Dict[str, Any]] = {}
    for name, metrics in held_out_all.items():
        for sex, entry in (metrics.get("sex_aggregated") or {}).items():
            sex_sources.setdefault(sex, {})[name] = entry
    if not sex_sources and by_sex and by_sex.get("groups"):
        for sex, entry in by_sex["groups"].items():
            sex_sources.setdefault(sex, {})[MODEL_VERSION] = entry

    for sex, heading in (("female", "7. Female subgroup analysis"),
                         ("male", "8. Male subgroup analysis")):
        entries = sex_sources.get(sex)
        if not entries:
            continue
        lines += [
            "", f"## {heading}", "",
            "Pooled over identity outcomes, not by averaging subgroup percentages.",
            "",
            "| Pipeline | Identities | FPIR | TPIR@1 | TPIR@5 | "
            "Mated scored/intended | Mated failures | Mated coverage | "
            "Non-mated scored/intended | Non-mated failures | Non-mated coverage |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, entry in entries.items():
            lines.append(
                f"| {name} | {entry['identities']} | {pct(entry['fpir'])} "
                f"[{pct(entry['fpir_lower_95'])}–{pct(entry['fpir_upper_95'])}] | "
                f"{pct(entry['tpir_rank1'])} "
                f"[{pct(entry['tpir_rank1_lower_95'])}–{pct(entry['tpir_rank1_upper_95'])}] | "
                f"{pct(entry['tpir_rank5'])} "
                f"[{pct(entry['tpir_rank5_lower_95'])}–{pct(entry['tpir_rank5_upper_95'])}] | "
                f"{entry.get('scored_mated_probes', '')}/"
                f"{entry.get('intended_mated_probes', '')} | "
                f"{entry.get('mated_extraction_failures', '')} | "
                f"{pct(entry['mated_probe_coverage'])} "
                f"[{pct(entry['mated_probe_coverage_lower_95'])}–"
                f"{pct(entry['mated_probe_coverage_upper_95'])}] | "
                f"{entry.get('scored_non_mated_probes', '')}/"
                f"{entry.get('intended_non_mated_probes', '')} | "
                f"{entry.get('non_mated_extraction_failures', '')} | "
                f"{pct(entry['non_mated_probe_coverage'])} "
                f"[{pct(entry['non_mated_probe_coverage_lower_95'])}–"
                f"{pct(entry['non_mated_probe_coverage_upper_95'])}] |"
            )

    if consistency:
        # Report every evaluated pipeline, and the mismatched controls beside
        # the same-person outcomes: the control is what shows whether the
        # threshold separates the two cases at all.
        per_pipeline = {
            name: metrics["profile_photo_consistency"]
            for name, metrics in ((pipeline or {}).get("held_out_metrics") or {}).items()
            if metrics.get("profile_photo_consistency")
        } or {MODEL_VERSION: consistency}
        lines += ["", "## 9. Profile-photo identity consistency", "",
                  PROFILE_CONSISTENCY_STATUS_NOTE, "",
                  "| Pipeline | Consistency (cond.) | Consistency (end-to-end) | "
                  "Open-set control detection (cond.) | Wrong-template detection (cond.) | "
                  "Wrong-template false-consistency (cond.) | Same-person coverage |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, entry in per_pipeline.items():
            lines.append(
                f"| {name} | {pct(entry.get('same_person_consistency_rate_conditional'))} | "
                f"{pct(entry.get('same_person_consistency_rate_end_to_end'))} | "
                f"{pct(entry.get('open_set_non_mated_gallery_detection_conditional'))} | "
                f"{pct(entry.get('wrong_profile_template_mismatch_detection_conditional'))} | "
                f"{pct(entry.get('wrong_profile_template_false_consistency_conditional'))} | "
                f"{pct(entry.get('same_person_extraction_coverage'))} |"
            )
        lines += [
            "",
            "The outcomes are not equivalent. A consistent photograph opens no case. An "
            "inconsistent one — a *low* similarity to the profile's own template — opens "
            "a consistency review. An extraction failure resolves nothing and is an "
            "unresolved outcome rather than a decision. Duplicate screening runs in the "
            "opposite direction: there a *high* similarity to another enrolled identity "
            "opens the review.",
            "",
            "The two controls are also different questions. The open-set control searches "
            "an absent person against the whole gallery, which is the stricter test. The "
            "wrong-template control is the direct one-photograph-to-one-profile "
            "comparison and is supplementary.",
            "", consistency["interpretation_note"],
        ]

    if pipeline:
        lines += ["", "## 10. YuNet + SFace against SCRFD + ArcFace", ""]
        held_out = pipeline.get("held_out_metrics") or {}
        if held_out:
            lines += ["| Pipeline | Threshold | FPIR | TPIR@1 | Reviews/1,000 | Coverage |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for name, m in held_out.items():
                lines.append(
                    f"| {name} | {format_number(m['development_threshold'], 6)} | "
                    f"{pct(m['rates']['fpir'])} | {pct(m['rates']['tpir_rank1'])} | "
                    f"{format_number(m['rates']['false_reviews_per_1000_non_mated'], 2)} | "
                    f"{pct(m['coverage']['gallery_enrolment_coverage'])} |"
                )
            lines += ["", "| Pipeline | End-to-end (95% CI) | Zero-face | Multiple-face | "
                      "Embed mean | Complete mean | Model size |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            sizes = pipeline.get("model_file_sizes") or {}
            for name, m in held_out.items():
                ci = (m.get("confidence_intervals") or {}).get(
                    "end_to_end_duplicate_detection_rate") or {}
                group = sizes.get("primary" if "opencv" in name else "comparison") or {}
                total = sum(v["megabytes"] for v in group.values()) if group else float("nan")
                lines.append(
                    f"| {name} | {pct(m.get('end_to_end_duplicate_detection_rate'))} "
                    f"[{pct(ci.get('lower_95'))}–{pct(ci.get('upper_95'))}] | "
                    f"{m['failure_breakdown'].get('zero_faces', 0)} | "
                    f"{m['failure_breakdown'].get('multiple_faces', 0)} | "
                    f"{format_number(m['coverage'].get('embedding_latency_mean_ms'), 2)} ms | "
                    f"{format_number(m['coverage'].get('complete_pipeline_latency_mean_ms'), 2)} ms | "
                    f"{format_number(total, 1)} MB |"
                )
            lines += [
                "",
                "Each pipeline was calibrated on its own development scores; the SFace "
                "threshold is never applied to ArcFace. This is a complete-pipeline "
                "comparison — detection, alignment, preprocessing, embedding width and "
                "runtime all differ — so no difference is attributable to the embedding "
                "model alone.",
            ]
        else:
            lines.append(
                f"Not run: `{pipeline['status']}`. {pipeline.get('reason', '')} No figures "
                f"are invented."
            )
        lines += [
            "", "## 10a. The same two pipelines on one-to-one verification", "",
        ]
        verif = aggregate_root / VERIFICATION_COMPARISON_DIRNAME
        comp_lfw = load("lfw_final_metrics.json", verif) if verif.is_dir() else None
        comp_cplfw = load("cplfw_metrics.json", verif) if verif.is_dir() else None
        if comp_lfw and comp_cplfw and final and cplfw:
            def coverage(payload: Mapping[str, Any]) -> str:
                return pct(1.0 - payload.get("failure_rate", float("nan")))

            def faces(payload: Mapping[str, Any], kind: str) -> int:
                breakdown = payload.get("failure_breakdown") or {}
                return sum(v for k, v in breakdown.items() if k.startswith(kind))

            lines += [
                "Section 10 compared the pipelines on gallery search alone, so the "
                "conclusion rested on a single task. Experiments 9 and 10 put the "
                "comparison pipeline through the same one-to-one protocols. LFW uses official "
                "ten-fold cross-validation, fitting each threshold on the other nine "
                "folds. CPLFW uses each pipeline's separate frozen LFW development "
                "threshold. No threshold is shared between pipelines.",
                "",
                "| Metric | LFW YuNet+SFace | LFW SCRFD+ArcFace | "
                "CPLFW YuNet+SFace | CPLFW SCRFD+ArcFace |",
                "| --- | --- | --- | --- | --- |",
                f"| Correct among scored pairs | {pct(final.get('accuracy'))} | "
                f"{pct(comp_lfw.get('accuracy'))} | {pct(cplfw.get('accuracy'))} | "
                f"{pct(comp_cplfw.get('accuracy'))} |",
                f"| Reached comparison | {coverage(final)} | {coverage(comp_lfw)} | "
                f"{coverage(cplfw)} | {coverage(comp_cplfw)} |",
                f"| Zero-face failures | {faces(final, 'zero_faces'):,} | "
                f"{faces(comp_lfw, 'zero_faces'):,} | "
                f"{faces(cplfw, 'zero_faces'):,} | "
                f"{faces(comp_cplfw, 'zero_faces'):,} |",
                f"| Multiple-face failures | {faces(final, 'multiple_faces'):,} | "
                f"{faces(comp_lfw, 'multiple_faces'):,} | "
                f"{faces(cplfw, 'multiple_faces'):,} | "
                f"{faces(comp_cplfw, 'multiple_faces'):,} |",
                "",
                "Conditional accuracy must be read alongside extraction coverage: the "
                "pipelines score different subsets of pairs. The exactly-one-face rule "
                "rejects both zero-face and multiple-face detections. The breakdown "
                "records these outcomes but cannot establish whether additional detections "
                "are bystanders or false positives without manual annotation. These "
                "results compare complete pipelines and do not isolate the recogniser.",
                "",
            ]
        else:
            lines += [
                "Not run in this checkout. No one-to-one comparison figures are "
                "invented in its place.", "",
            ]

        arcface_review = (
            load("ml_review_test_metrics.json", aggregate_root / ARCFACE_REVIEW_DIRNAME)
            if (aggregate_root / ARCFACE_REVIEW_DIRNAME).is_dir() else None
        )
        lines += ["", "## 10b. The review classifier on both pipelines", ""]
        if arcface_review and review:
            def burden(payload: Mapping[str, Any], key: str) -> float:
                node = (payload["classifier"] if key == "classifier"
                        else payload["comparator_three_image_open_set_calibrated"]["rates"])
                value = node.get("false_reviews_per_1000_non_mated")
                return float(value) if isinstance(value, (int, float)) else float("nan")

            base_before, base_after = burden(review, "threshold"), burden(review, "classifier")
            comp_before = burden(arcface_review, "threshold")
            comp_after = burden(arcface_review, "classifier")
            scored = arcface_review["classifier"].get("scored_non_mated_probes", 0)
            lines += [
                "Section 6 fitted the classifier on the baseline pipeline and section "
                "10 compared the pipelines without it, so the framework's most "
                "elaborate addition and its strongest components were never combined. "
                "Experiment 11 runs the identical method over the comparison "
                "pipeline, under the same seed and therefore the same identity "
                "groups.",
                "",
                "| Review burden per 1,000 new profiles | Threshold alone | "
                "With the classifier |",
                "| --- | --- | --- |",
                f"| YuNet + SFace | {base_before:.1f} | {base_after:.1f} |",
                f"| SCRFD + ArcFace | {comp_before:.1f} | {comp_after:.1f} |",
                "",
                "The classifier moves the burden in **opposite directions** on the two "
                "pipelines. Its effect is therefore a property of the components it "
                "runs on rather than of the classifier alone. The negative result in "
                "section 6 stands for the baseline pipeline, but it cannot be stated "
                "as a general finding about the method.",
                "",
                f"The zero on the second row is an observation over {scored:,} scored "
                "new profiles, not a demonstration that the population rate is zero; "
                "the empirical zero-event bootstrap interval cannot bound population FPIR. A supplementary identity-level upper bound is reported with its assumptions. Detection fell "
                f"from {pct(arcface_review['comparator_three_image_open_set_calibrated']['rates'].get('tpir_rank1'))} "
                f"to {pct(arcface_review['classifier'].get('tpir_rank1'))} in exchange.",
                "",
            ]
        else:
            lines += [
                "Not run in this checkout. No cross-pipeline classifier result is "
                "invented in its place.", "",
            ]

        lines += ["", "## 10c. Detector, recogniser and coverage contributions", "",
                  render_paired_comparison_report(aggregate_root)]

        lines += [
            "", "## 11. Performance against cost", "",
            "A stronger pipeline is not free. Where it improves extraction and "
            "identification it also costs disk and latency, and the trade-off is shown in "
            "`implementation_layers_performance_latency` rather than omitted.",
        ]

    diagnostics_path = aggregate_root / "comparison_diagnostics" / "comparison_diagnostics.json"
    if diagnostics_path.is_file():
        lines += ["", "## 11a. Additional controlled diagnostics", "",
                  "The separate COMPARISON_DIAGNOSTICS_REPORT.md reports calibrated one- versus "
                  "three-image enrolment, gallery-size sensitivity, classifier feature ablations "
                  "and error analysis. These extensions were designed after the benchmark test "
                  "results were inspected and are exploratory, not independent confirmation."]
    lines += [
        "", "## 12. Limitations and policy", "",
        "This is a benchmark-validated, human-review-only academic face-comparison "
        "study. It evaluates duplicate-profile screening and profile-photo "
        "facial consistency using frozen pretrained face-recognition pipelines and an "
        "identity-disjoint logistic-regression review classifier.",
        "",
        "The two tasks refer in opposite directions, and a single threshold statement "
        "would misdescribe one of them:",
        "",
        "- **Duplicate-profile screening** — a *high* similarity to some other enrolled "
        "identity opens a duplicate-profile review.",
        "- **Profile-photo consistency** — a *low* similarity to the profile's own "
        "enrolled template opens an inconsistency review.",
        "",
        "Neither is proof of fraud, ownership or identity, and an extraction failure "
        "resolves nothing in either direction.",
        "",
        "No face-detection or face-recognition network is trained or fine-tuned. Experiment "
        "7 trains a small logistic-regression review classifier on identity-disjoint BFW "
        "development data and evaluates it on untouched held-out identities.",
        "",
    ]
    lines += [f"- {item}" for item in OPEN_SET_LIMITATIONS]
    # Two further caveats follow from sections 10a and 10b, which compare the
    # same additions across both pipelines. They are stated here rather than in
    # OPEN_SET_LIMITATIONS because they describe this comparison, not the
    # open-set protocol that every artefact carries.
    lines += [
        "- The review classifier was fitted separately on each pipeline and moved the "
        "review burden in opposite directions on the two. Its contribution therefore "
        "depends on the components underneath it, and neither result should be read as "
        "a general property of the method.",
        "- Coverage differences between the two detectors are shaped by this protocol's "
        "requirement that exactly one face be found. A detector that recovers faint "
        "faces also recovers bystanders, so a coverage loss is not on its own evidence "
        "of weaker detection.",
    ]
    return "\n".join(lines) + "\n"


def generate_figures(
    *, aggregate_root: Path = AGGREGATE_ROOT, figures_root: Path = FIGURES_ROOT
) -> List[Path]:
    """Produce every dissertation figure the available artefacts support."""
    plt = _figure_backend()
    # Imported here rather than at module scope, because matplotlib is only
    # required when figures are actually generated.
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    figures_root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def load(name: str) -> Optional[Dict[str, Any]]:
        path = aggregate_root / name
        return read_json_artifact(path) if path.is_file() else None

    def load_from(base: Path, name: str) -> Optional[Dict[str, Any]]:
        """The same, for the sub-directories the later experiments write into."""
        path = base / name
        return read_json_artifact(path) if path.is_file() else None

    open_set = load("bfw_open_set_test_metrics.json")
    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")
    intervals = load("ml_review_confidence_intervals.json")


    # --- Figures A-D: implementation layers ----------------------------------
    layers = _collect_implementation_layers(aggregate_root)
    if layers:
        names = [layer["name"] for layer in layers]
        denominator = layers[0]["rates"].get("scored_non_mated_probes", "n/a")

        values = [layer["rates"].get("false_reviews_per_1000_non_mated", float("nan"))
                  for layer in layers]
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        ax.bar(names, values, color="#4C72B0")
        ax.set_ylabel("False reviews per 1,000 non-mated searches (lower is better)")
        ax.set_title(
            f"Implementation layers: false human-review referrals "
            f"(n={denominator} scored non-mated probes)"
        )
        ax.set_ylim(bottom=0)
        for index, value in enumerate(values):
            if value == value:
                ax.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        path = figures_root / "implementation_layers_fpir.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        width = 0.26
        positions = np.arange(len(layers))
        for offset, key, label, colour in (
            (-width, "tpir_rank1", "TPIR@1 (conditional)", "#4C72B0"),
            (0.0, "tpir_rank5", "TPIR@5 (conditional)", "#55A868"),
            (width, None, "End-to-end duplicate detection", "#DD8452"),
        ):
            series = [
                _percent(layer["end_to_end"]) if key is None
                else _percent(layer["rates"].get(key))
                for layer in layers
            ]
            ax.bar(positions + offset, series, width, label=label, color=colour)
        ax.set_xticks(positions); ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("Per cent (higher is better)"); ax.set_ylim(0, 100)
        ax.set_title("Implementation layers: duplicate detection, conditional and end-to-end")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        path = figures_root / "implementation_layers_duplicate_detection.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for offset, key, label, colour in (
            (-width, "gallery_enrolment_coverage", "Gallery enrolment", "#55A868"),
            (0.0, "mated_extraction_failure_rate", "Mated probe", "#4C72B0"),
            (width, "non_mated_extraction_failure_rate", "Non-mated probe", "#DD8452"),
        ):
            series = []
            for layer in layers:
                value = layer["coverage"].get(key)
                if key.endswith("failure_rate") and isinstance(value, (int, float)):
                    value = 1.0 - value
                series.append(_percent(value))
            ax.bar(positions + offset, series, width, label=label, color=colour)
        ax.set_xticks(positions); ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("Coverage (%) — extraction failure is the remainder")
        ax.set_ylim(0, 100)
        ax.set_title("Implementation layers: extraction coverage")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        path = figures_root / "implementation_layers_coverage.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

        # Three dimensions share one panel, so each is stated explicitly rather
        # than left to be inferred: horizontal position is cost, vertical
        # position is benefit, and marker area is the human-review burden.
        fig, ax = plt.subplots(figsize=(12.0, 7.0))
        plotted: List[Tuple[int, float, float, float]] = []
        for index, layer in enumerate(layers):
            latency = layer["coverage"].get("complete_pipeline_latency_mean_ms")
            detection = _percent(layer["end_to_end"])
            reviews = layer["rates"].get("false_reviews_per_1000_non_mated", float("nan"))
            # A layer without a measured complete-pipeline latency is omitted
            # here and explained in the caption; no value is invented for it.
            if not (isinstance(latency, (int, float)) and latency == latency
                    and detection == detection):
                continue
            plotted.append((index, float(latency), detection, float(reviews)))

        # Draw the largest markers first so a small marker sitting inside a
        # large one stays visible. Layers 1 and 3 differ by a factor of
        # seventeen in review burden at almost the same coordinates.
        for index, latency, detection, reviews in sorted(
            plotted, key=lambda row: -row[3]
        ):
            # Layers 1-4 share the YuNet + SFace extraction pipeline; layer 5 is
            # the InsightFace comparison. Colour therefore identifies the
            # pipeline, and the label identifies the layer within it.
            colour = LAYER_PIPELINE_COLOURS[1] if index == 4 else LAYER_PIPELINE_COLOURS[0]
            ax.scatter(latency, detection, s=_review_marker_area(reviews), alpha=0.55,
                       color=colour, edgecolors="#333333", linewidths=0.6, zorder=3)
            # Fixed per-layer offsets with a leader line. Layers 2-4 record an
            # identical latency and layers 1 and 3 an identical detection rate,
            # so automatic placement would overlap.
            dx, dy, ha = LAYER_LABEL_OFFSETS[index]
            ax.annotate(
                f"L{index + 1}: {LAYER_SHORT_LABELS[index]}",
                (latency, detection), xytext=(dx, dy), textcoords="offset points",
                fontsize=9, ha=ha, va="center", zorder=4,
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#666666",
                            "shrinkA": 0, "shrinkB": 2},
            )

        latencies = [row[1] for row in plotted]
        detections = [row[2] for row in plotted]
        # The latency axis begins at zero so the fourfold cost difference is
        # read as a ratio. The detection axis is not zero-based: every layer
        # lies above 85%, and a 0-100% range would compress the differences the
        # figure exists to show, so it is padded around the observed values.
        ax.set_xlim(0, max(latencies) * 1.30 if latencies else 1.0)
        span = (max(detections) - min(detections)) if detections else 1.0
        ax.set_ylim(min(detections) - max(span * 0.45, 1.5),
                    min(100.0, max(detections) + max(span * 0.45, 1.5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.0f}%"))
        ax.set_xlabel(
            "Mean complete-pipeline latency per image (milliseconds) — lower is better",
            fontsize=10,
        )
        ax.set_ylabel(
            "End-to-end duplicate detection (%) — higher is better", fontsize=10
        )
        ax.set_title(
            "Implementation layers: detection against cost\n"
            "Marker area is false reviews per 1,000 non-mated searches; "
            "larger markers mean more false reviews",
            fontsize=11,
        )
        ax.grid(alpha=0.3, zorder=0)

        # Two legends: one naming the pipeline behind each colour, one giving
        # the marker-area scale in the units it encodes.
        pipeline_handles = [
            Line2D([], [], marker="o", linestyle="none", markersize=9,
                   markerfacecolor=LAYER_PIPELINE_COLOURS[0], markeredgecolor="#333333",
                   alpha=0.55, label=f"{PIPELINE_DISPLAY_NAMES['opencv']} (layers 1-4)"),
            Line2D([], [], marker="o", linestyle="none", markersize=9,
                   markerfacecolor=LAYER_PIPELINE_COLOURS[1], markeredgecolor="#333333",
                   alpha=0.55, label=f"{PIPELINE_DISPLAY_NAMES['insightface']} (layer 5)"),
        ]
        pipeline_legend = ax.legend(
            handles=pipeline_handles, loc="lower right", fontsize=9,
            title="Pipeline", title_fontsize=9, framealpha=0.95,
        )
        ax.add_artist(pipeline_legend)
        size_handles = [
            Line2D([], [], marker="o", linestyle="none",
                   markersize=(_review_marker_area(value) ** 0.5) / 2.0,
                   markerfacecolor="#BBBBBB", markeredgecolor="#333333",
                   alpha=0.55, label=f"{value:g}")
            for value in REVIEW_LEGEND_VALUES
        ]
        ax.legend(
            handles=size_handles, loc="upper left", fontsize=9, labelspacing=1.5,
            borderpad=1.0, handletextpad=1.6, framealpha=0.95,
            title="False reviews per 1,000\nnon-mated searches", title_fontsize=9,
        )
        # State the preferred direction rather than leaving it to be deduced.
        # Placed lower-left, which the data leaves empty, so it covers neither
        # legend nor any plotted point.
        ax.text(
            0.02, 0.03,
            "Preferred direction: left (lower latency), upward (higher detection),\n"
            "smaller marker (fewer false reviews)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                  "edgecolor": "#999999", "alpha": 0.95},
        )
        path = figures_root / "implementation_layers_performance_latency.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

    # --- Figure E: mated and non-mated similarity distributions --------------
    held_out_all = (pipeline or {}).get("held_out_metrics") or {}
    with_histograms = {
        n: m for n, m in held_out_all.items() if m.get("similarity_histograms")
    }
    if with_histograms:
        names = sorted(with_histograms, key=lambda n: "opencv" not in n)
        fig, axes = plt.subplots(1, len(names), figsize=(6.0 * len(names), 4.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            metrics = with_histograms[name]
            hists = metrics["similarity_histograms"]
            for key, label, colour in (
                ("mated_correct_identity",
                 "Mated: similarity to the probe's own enrolled template", "#4C72B0"),
                ("non_mated_top1",
                 "Non-mated: highest similarity against the gallery", "#DD8452"),
            ):
                h = hists.get(key) or {}
                counts, edges = h.get("counts") or [], h.get("bin_edges") or []
                if not counts:
                    continue
                centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
                ax.bar(centres, counts, width=(edges[1] - edges[0]), alpha=0.6,
                       label=f"{label} (n={h.get('n', 0)})", color=colour)
            threshold = metrics["development_threshold"]
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
                       label=f"Frozen threshold {threshold:.3f}")
            ax.set_xlabel("Cosine similarity (dimensionless, -1 to 1)")
            ax.set_title(pipeline_display_name(name), fontsize=11)
            # Observed similarities occupy roughly the upper half of the range;
            # plotting the full -1 to 1 span would leave most of the axis blank.
            ax.set_xlim(0.0, 1.0)
            # Headroom above the tallest bin, so the legend sits clear of the
            # non-mated peak rather than on top of it.
            ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("Probe count")
        fig.suptitle(
            "Mated and non-mated similarity distributions, each pipeline at its own "
            "frozen threshold (aggregate histograms; no individual score published)",
            y=1.02, fontsize=10,
        )
        path = figures_root / "mated_non_mated_similarity_distributions.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

    # --- Figure F: profile-photo consistency ---------------------------------
    per_pipeline_consistency = {
        n: m["profile_photo_consistency"] for n, m in held_out_all.items()
        if m.get("profile_photo_consistency")
    }
    consistency_path = aggregate_root / "profile_photo_consistency.json"
    if not per_pipeline_consistency and consistency_path.is_file():
        per_pipeline_consistency = {MODEL_VERSION: read_json_artifact(consistency_path)}

    if per_pipeline_consistency:
        names = sorted(per_pipeline_consistency, key=lambda n: "opencv" not in n)
        outcomes = [
            ("consistent_same_person_photographs", "Consistent\nwith own profile"),
            ("inconsistent_same_person_review_candidates", "Inconsistent\nreview opened"),
            ("open_set_non_mated_gallery_controls_correctly_identified",
             "Correctly matched\nno enrolled profile"),
            ("open_set_non_mated_gallery_controls_false_consistent",
             "Wrongly matched\nan enrolled profile"),
            ("wrong_profile_template_controls_correctly_inconsistent",
             "Correctly inconsistent\nwith wrong profile"),
            ("wrong_profile_template_controls_false_consistent",
             "Wrongly consistent\nwith wrong profile"),
            ("same_person_extraction_failures", "Extraction\nfailure"),
            ("gallery_reference_unavailable", "Reference\nunavailable"),
        ]
        # The eight outcomes answer four separate questions. Grouping them
        # visually stops a reader comparing a same-person count directly with a
        # control count, which have different denominators.
        outcome_groups = (
            ("Same-person photographs", 0, 2),
            ("Open-set non-mated gallery control", 2, 4),
            ("Wrong-profile-template control", 4, 6),
            ("Unresolved", 6, 8),
        )
        fig, ax = plt.subplots(figsize=(15.0, 7.0))
        positions = np.arange(len(outcomes))
        width = 0.8 / max(len(names), 1)
        for index, name in enumerate(names):
            entry = per_pipeline_consistency[name]
            offset = (index - (len(names) - 1) / 2) * width
            values = [entry.get(key, 0) for key, _ in outcomes]
            bars = ax.bar(positions + offset, values, width,
                          label=pipeline_display_name(name),
                          color=("#4C72B0", "#DD8452")[index % 2])
            # Several outcomes are two orders of magnitude smaller than the
            # largest bar, so the count is printed rather than left to be read
            # off an axis on which the bar is invisible.
            ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)

        ceiling = max(
            (entry.get(key, 0) or 0)
            for entry in per_pipeline_consistency.values() for key, _ in outcomes
        )
        # Headroom for the printed counts and the group headings above them.
        ax.set_ylim(0, ceiling * 1.28 if ceiling else 1.0)
        for label, start, stop in outcome_groups:
            centre = (start + stop - 1) / 2.0
            ax.text(centre, ceiling * 1.20, label, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")
            if stop < len(outcomes):
                # Separator sits between two groups, never through a bar.
                ax.axvline(stop - 0.5, color="#BBBBBB", linewidth=0.8, zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels([label for _, label in outcomes], fontsize=8)
        ax.set_ylabel("Photographs (count)")
        ax.set_title(
            "Profile-photo consistency outcomes by category\n"
            "An inconsistent result is a review signal, not proof of photo theft or fraud",
            fontsize=11,
        )
        # Below the group headings and inside the empty left region, so it
        # covers neither a heading nor a bar.
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.88),
                  framealpha=0.95)
        ax.grid(axis="y", alpha=0.3)
        path = figures_root / "profile_photo_consistency_outcomes.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

    # --- Figures G-I: sex-separated results ----------------------------------
    # Prefer the two-pipeline subgroup file so both pipelines are compared;
    # fall back to the primary-only file before Experiment 8 has run.
    pipeline_subgroups = aggregate_root / "pretrained_pipeline_subgroup_metrics.csv"
    primary_subgroups = aggregate_root / "bfw_subgroup_metrics.csv"
    by_pipeline: Dict[str, Dict[str, Dict[str, str]]] = {}
    if pipeline_subgroups.is_file():
        for row in csv.DictReader(open(pipeline_subgroups, encoding="utf-8")):
            if row.get("fpir"):
                by_pipeline.setdefault(row["pipeline"], {})[row["subgroup"]] = row
    if not by_pipeline and primary_subgroups.is_file():
        by_pipeline[MODEL_VERSION] = {
            r["subgroup"]: r for r in csv.DictReader(open(primary_subgroups, encoding="utf-8"))
        }

    if by_pipeline:
        # Identical metric order, axis limits, units, pipeline order and
        # interval format in both figures, so they compare fairly.
        sex_metrics = (
            ("fpir", "FPIR\n(lower better)"),
            ("tpir_rank1", "TPIR@1\n(higher better)"),
            ("tpir_rank5", "TPIR@5\n(higher better)"),
            ("mated_probe_coverage", "Mated\ncoverage"),
            ("non_mated_probe_coverage", "Non-mated\ncoverage"),
        )
        pipeline_order = sorted(by_pipeline, key=lambda n: "opencv" not in n)
        colours = ("#4C72B0", "#DD8452")
        # FPIR here is a fraction of one per cent. Plotting it on the 0-100%
        # axis the coverage panels need would flatten every bar to the
        # baseline and hide the difference the experiment is about, so it gets
        # its own axis with a metric-specific bound. The bound is computed
        # across both sexes and both pipelines first, so the female and male
        # companion figures remain directly comparable.
        fpir_values = [
            _percent(float(row["fpir_upper_95"] or 0) or 0)
            for rows in by_pipeline.values() for row in rows.values()
            if row.get("fpir_upper_95")
        ] + [
            _percent(float(row["fpir"] or 0) or 0)
            for rows in by_pipeline.values() for row in rows.values()
            if row.get("fpir")
        ]
        fpir_limit = max(0.5, math.ceil(max(fpir_values or [0.0]) * 1.15 * 4) / 4)
        for sex, suffix in (("female", "_females"), ("male", "_males")):
            members = sorted(
                {s for rows in by_pipeline.values() for s in rows if s.endswith(suffix)}
            )
            if not members:
                continue
            fig, axes = plt.subplots(1, len(sex_metrics), figsize=(16.0, 4.6))
            positions = np.arange(len(members))
            offsets = np.linspace(-0.16, 0.16, len(pipeline_order))
            for ax, (metric, title) in zip(axes, sex_metrics):
                for offset, name, colour in zip(offsets, pipeline_order, colours):
                    rows = by_pipeline[name]
                    centre, lower, upper = [], [], []
                    for subgroup in members:
                        row = rows.get(subgroup)
                        if not row or not row.get(metric):
                            centre.append(float("nan")); lower.append(0); upper.append(0)
                            continue
                        value = _percent(float(row[metric]))
                        centre.append(value)
                        lower.append(max(value - _percent(float(row[f"{metric}_lower_95"])), 0))
                        upper.append(max(_percent(float(row[f"{metric}_upper_95"])) - value, 0))
                    ax.errorbar(positions + offset, centre, yerr=[lower, upper], fmt="o",
                                capsize=3, markersize=4, color=colour,
                                label=pipeline_display_name(name))
                ax.set_xticks(positions)
                # Strip the sex suffix, which the panel title already states,
                # and render the remaining category as readable axis text.
                ax.set_xticklabels(
                    [subgroup_display_name(m.replace(suffix, "")) for m in members],
                    rotation=30, ha="right", fontsize=9,
                )
                ax.set_title(title, fontsize=9)
                ax.grid(axis="y", alpha=0.3)
                # FPIR gets its own bound; every other panel stays on the common
                # 0-100% scale so they remain comparable with one another.
                ax.set_ylim(0, fpir_limit if metric == "fpir" else 100)
                # Fix the metric for this panel, so each axis keeps its own
                # decimal precision rather than the last panel's.
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda v, _pos, is_fpir=(metric == "fpir"):
                    f"{v:.2f}%" if is_fpir else f"{v:.0f}%"
                ))
                ax.tick_params(axis="y", labelsize=8)
                if metric == "fpir":
                    ax.set_ylabel("Per cent (95% identity-cluster CI)", fontsize=9)
            axes[1].set_ylabel("Per cent (95% identity-cluster CI)", fontsize=9)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles[:len(pipeline_order)], labels[:len(pipeline_order)],
                       loc="lower center", ncol=len(pipeline_order), fontsize=9,
                       bbox_to_anchor=(0.5, -0.10), frameon=False)
            fig.suptitle(
                f"{sex.capitalize()} subgroup performance, BFW held-out test", y=1.03
            )
            # Stated on the figure itself, because a zero-width interval is
            # easily over-read as proof of a zero population error rate.
            fig.text(0.5, -0.17, ZERO_EVENT_INTERVAL_NOTE, ha="center", fontsize=8,
                     color="#444444")
            path = figures_root / f"{sex}_subgroup_pipeline_comparison.png"
            _save_figure(fig, path); plt.close(fig); written.append(path)

    # Four series: each pipeline for each sex, pooled from identity outcomes.
    sex_groups: Dict[str, Dict[str, Any]] = {}
    if pipeline and (pipeline.get("held_out_metrics") or {}):
        for name, metrics in pipeline["held_out_metrics"].items():
            for sex, entry in (metrics.get("sex_aggregated") or {}).items():
                sex_groups[f"{pipeline_display_name(name)} — {sex}"] = entry
    else:
        sex_path = aggregate_root / "bfw_sex_aggregated_metrics.json"
        if sex_path.is_file():
            for sex, entry in read_json_artifact(sex_path).get("groups", {}).items():
                sex_groups[f"{pipeline_display_name(MODEL_VERSION)} — {sex}"] = entry

    if sex_groups:
        # Same pipeline order as every other comparison figure: the baseline
        # YuNet + SFace pair first, then the InsightFace comparison.
        names = sorted(
            sex_groups,
            key=lambda label: (PIPELINE_DISPLAY_NAMES["opencv"] not in label, label),
        )
        palette = ("#4C72B0", "#8FB2D9", "#DD8452", "#EFB48C")
        width = 0.8 / max(len(names), 1)

        # Same reasoning as the subgroup companions: a sub-one-per-cent FPIR
        # compressed onto a 0-100% axis shows nothing at all, so it is given
        # its own panel and its own bound.
        aggregate_fpir = [
            _percent(entry.get(key))
            for entry in sex_groups.values() for key in ("fpir", "fpir_upper_95")
            if entry.get(key) is not None
        ]
        aggregate_limit = max(
            0.5, math.ceil(max(aggregate_fpir or [0.0]) * 1.15 * 4) / 4
        )

        panels = (
            (("fpir",), ["FPIR\n(lower better)"], aggregate_limit, "FPIR"),
            (("tpir_rank1", "mated_probe_coverage", "non_mated_probe_coverage"),
             ["TPIR@1\n(higher better)", "Mated coverage", "Non-mated coverage"],
             100.0, "Identification and coverage"),
        )
        fig, axes = plt.subplots(
            1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [1, 3]}
        )
        for ax, (metrics_shown, labels, limit, title) in zip(axes, panels):
            positions = np.arange(len(metrics_shown))
            for index, name in enumerate(names):
                entry = sex_groups[name]
                offset = (index - (len(names) - 1) / 2) * width
                centre = [_percent(entry.get(m)) for m in metrics_shown]
                lower = [max(centre[i] - _percent(entry.get(f"{m}_lower_95")), 0)
                         for i, m in enumerate(metrics_shown)]
                upper = [max(_percent(entry.get(f"{m}_upper_95")) - centre[i], 0)
                         for i, m in enumerate(metrics_shown)]
                ax.bar(positions + offset, centre, width, label=name,
                       color=palette[index % len(palette)], yerr=[lower, upper],
                       capsize=3)
            ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylim(0, limit); ax.set_title(title, fontsize=10)
            ax.set_ylabel("Per cent (95% identity-cluster CI)", fontsize=9)
            # Fix the bound for this panel, so the FPIR and percentage axes
            # keep their own decimal precision.
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _pos, fine=(limit < 10):
                f"{v:.2f}%" if fine else f"{v:.0f}%"
            ))
            ax.grid(axis="y", alpha=0.3)
        # Placed below the panels so it cannot cover a bar or an interval.
        handles, legend_labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=9,
                   bbox_to_anchor=(0.5, -0.14), frameon=False)
        fig.suptitle("Aggregate female against male, pooled over identity outcomes")
        fig.text(0.5, -0.21, ZERO_EVENT_INTERVAL_NOTE, ha="center", fontsize=8,
                 color="#444444")
        path = figures_root / "female_male_aggregate_comparison.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

    # --- Figure 1: false review referrals by method --------------------------
    if open_set:
        labels, values = [], []
        control = open_set["methods"][METHOD_A]["rates"]
        proposed = open_set["methods"][METHOD_B]["primary_operating_point"]
        labels.append("Single-image\n1:1 threshold\n(control)")
        values.append(control["false_reviews_per_1000_non_mated"])
        labels.append("Three-image\ncalibrated")
        values.append(proposed["false_reviews_per_1000_non_mated"])
        if review:
            labels.append("Logistic-regression\nclassifier")
            values.append(review["classifier"]["false_reviews_per_1000_non_mated"])
        arcface = next(
            (v for k, v in ((pipeline or {}).get("held_out_metrics") or {}).items()
             if "arcface" in k.lower()), None
        )
        if arcface:
            labels.append("SCRFD +\nArcFace")
            values.append(arcface["rates"]["false_reviews_per_1000_non_mated"])

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(labels, values, color="#4C72B0")
        ax.set_ylabel("False reviews per 1,000 non-mated searches")
        ax.set_title("False human-review referrals by method (BFW held-out test)")
        ax.set_ylim(bottom=0)
        for index, value in enumerate(values):
            if value == value:
                ax.text(index, value, f"{value:.1f}", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.3)
        path = figures_root / "false_reviews_per_1000_by_method.png"
        _save_figure(fig, path)
        plt.close(fig)
        written.append(path)

    # --- Figure 2: detection and coverage ------------------------------------
    if open_set:
        proposed = open_set["methods"][METHOD_B]
        primary = proposed["primary_operating_point"]
        coverage = proposed["coverage"]
        groups = ["Conditional\nTPIR@1", "End-to-end\nduplicate detection", "Gallery enrolment\ncoverage"]
        values = [
            _percent(primary["tpir_rank1"]),
            _percent(proposed.get("end_to_end_duplicate_detection_rate")),
            _percent(coverage["gallery_enrolment_coverage"]),
        ]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(groups, values, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_ylabel("Per cent")
        ax.set_ylim(0, 100)
        ax.set_title("Detection and coverage, clearly separated (BFW held-out test)")
        for index, value in enumerate(values):
            if value == value:
                ax.text(index, value, f"{value:.1f}%", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.3)
        path = figures_root / "duplicate_detection_by_method.png"
        _save_figure(fig, path)
        plt.close(fig)
        written.append(path)

    # --- Figure 3: open-set operating curve ----------------------------------
    if open_set:
        development = load("bfw_open_set_development_metrics.json")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        if development:
            points = development["methods"][METHOD_B]["operating_points"]
            xs = [max(points[k]["fpir"], 1e-4) for k in sorted(points)]
            ys = [_percent(points[k]["tpir_rank1"]) for k in sorted(points)]
            ax.plot(xs, ys, "o--", label="Development (threshold selected here)", color="#4C72B0")
        points = open_set["methods"][METHOD_B]["operating_points"]
        xs = [max(points[k]["fpir"], 1e-4) for k in sorted(points)]
        ys = [_percent(points[k]["tpir_rank1"]) for k in sorted(points)]
        ax.plot(xs, ys, "s-", label="Held-out test (never used to select)", color="#C44E52")
        ax.set_xscale("log")
        ax.set_xlabel(
            "FPIR — false positive identification rate, proportion of non-mated "
            "searches (log scale) — lower is better"
        )
        ax.set_ylabel("TPIR@1 (%) — higher is better")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.0f}%"))
        ax.set_ylim(0, 100)
        ax.set_title("Open-set operating points: TPIR@1 against FPIR")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        path = figures_root / "open_set_operating_curve.png"
        _save_figure(fig, path)
        plt.close(fig)
        written.append(path)

    # --- Figure 4: subgroup performance with intervals -----------------------
    subgroup_csv = aggregate_root / "ml_review_subgroup_metrics.csv"
    if subgroup_csv.is_file():
        rows = list(csv.DictReader(open(subgroup_csv, encoding="utf-8")))
        if rows:
            names = [r["subgroup"] for r in rows]
            positions = list(range(len(names)))
            fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
            for ax, key, title in (
                (axes[0], "fpir", "FPIR by subgroup, lower is better (95% CI)"),
                (axes[1], "tpir_rank1", "TPIR@1 by subgroup, higher is better (95% CI)"),
            ):
                centre = [_percent(float(r[key])) for r in rows]
                lower = [max(centre[i] - _percent(float(rows[i][f"{key}_lower_95"])), 0) for i in positions]
                upper = [max(_percent(float(rows[i][f"{key}_upper_95"])) - centre[i], 0) for i in positions]
                ax.errorbar(positions, centre, yerr=[lower, upper], fmt="o", capsize=4,
                            color="#4C72B0")
                ax.set_ylabel("Per cent (95% identity-cluster CI)", fontsize=9)
                ax.set_title(title, fontsize=10)
                # Each metric is bounded by its own observed interval rather
                # than a shared 0-100% axis, on which a sub-one-per-cent FPIR
                # and a 95% TPIR would both be unreadable.
                top = max(centre[i] + upper[i] for i in positions)
                ax.set_ylim(0 if key == "fpir"
                            else max(0.0, min(centre[i] - lower[i] for i in positions) - 5.0),
                            top * 1.10 if key == "fpir" else min(100.5, top + 2.0))
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda v, _pos, is_fpir=(key == "fpir"):
                    f"{v:.1f}%" if is_fpir else f"{v:.0f}%"
                ))
                ax.grid(axis="y", alpha=0.3)
            axes[1].set_xticks(positions)
            axes[1].set_xticklabels(
                [subgroup_display_name(n) for n in names], rotation=30, ha="right",
                fontsize=9,
            )
            fig.suptitle("Subgroup performance of the review classifier", y=0.98)
            fig.text(0.5, -0.06, ZERO_EVENT_INTERVAL_NOTE, ha="center", fontsize=8,
                     color="#444444")
            path = figures_root / "subgroup_fpir_tpir_with_confidence_intervals.png"
            _save_figure(fig, path)
            plt.close(fig)
            written.append(path)

    # --- Figure 5: classifier coefficients -----------------------------------
    model = load("ml_review_model.json")
    if model:
        order = model["model"]["feature_order"]
        weights = model["model"]["coefficients"]
        colours = ["#C44E52" if w < 0 else "#4C72B0" for w in weights]
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.barh(order, weights, color=colours)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Standardised logistic-regression coefficient")
        ax.set_title(
            "Review-classifier coefficients (association, not causation)\n"
            "Positive raises the referral probability; negative lowers it. Feature "
            "names are the model's own, as published in ml_review_model.json",
            fontsize=10,
        )
        ax.grid(axis="x", alpha=0.3)
        path = figures_root / "ml_review_classifier_coefficients.png"
        _save_figure(fig, path)
        plt.close(fig)
        written.append(path)

    # --- Figure 6: pipeline coverage and latency -----------------------------
    held_out = (pipeline or {}).get("held_out_metrics") or {}
    if held_out:
        # Both evaluated pipelines, never the primary alone once ArcFace exists.
        names = sorted(held_out, key=lambda n: "opencv" not in n)
        fig, (left, right) = plt.subplots(1, 2, figsize=(14.0, 5.4))
        coverage_labels = ["Gallery", "Mated probe", "Non-mated probe"]
        positions = np.arange(len(coverage_labels)); width = 0.35
        for offset, name, colour in zip((-width / 2, width / 2), names, ("#4C72B0", "#DD8452")):
            c = held_out[name]["coverage"]
            bars = left.bar(positions + offset, [
                _percent(c["gallery_enrolment_coverage"]),
                _percent(1.0 - c["mated_extraction_failure_rate"]),
                _percent(1.0 - c["non_mated_extraction_failure_rate"]),
            ], width, label=pipeline_display_name(name), color=colour)
            # Coverage sits near the ceiling. The axis stays 0-100% so the bars
            # are not visually exaggerated, and the value is printed so the
            # difference between 94% and 100% is still legible.
            left.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
        left.set_xticks(positions); left.set_xticklabels(coverage_labels)
        left.set_ylabel("Coverage (%) — higher is better")
        # Ticks stop at 100%, but the limit leaves room for the printed values
        # so they do not run into the panel title.
        left.set_ylim(0, 110); left.set_yticks(list(range(0, 101, 20)))
        left.set_title("Extraction coverage")
        # Coverage is near the ceiling, so a legend inside the axes would sit
        # on top of the bars.
        left.legend(fontsize=9, loc="lower left", framealpha=0.95)
        left.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.0f}%"))
        left.grid(axis="y", alpha=0.3)

        latency_keys = [
            ("embedding_latency_mean_ms", "Embedding, mean"),
            ("embedding_latency_p95_ms", "Embedding, 95th pct"),
            ("complete_pipeline_latency_mean_ms", "Complete pipeline, mean"),
            ("complete_pipeline_latency_p95_ms", "Complete pipeline, 95th pct"),
            ("top1_search_time_mean_ms", "Gallery search, mean"),
            ("top1_search_time_p95_ms", "Gallery search, 95th pct"),
        ]
        positions = np.arange(len(latency_keys))
        for offset, name, colour in zip((-width / 2, width / 2), names, ("#4C72B0", "#DD8452")):
            c = held_out[name]["coverage"]
            values = [c.get(k) if isinstance(c.get(k), (int, float)) else float("nan")
                      for k, _ in latency_keys]
            right.bar(positions + offset, values, width,
                      label=pipeline_display_name(name), color=colour)
        right.set_xticks(positions)
        right.set_xticklabels([lab for _, lab in latency_keys], fontsize=9,
                              rotation=25, ha="right")
        right.set_ylabel("Milliseconds per image — lower is better")
        right.set_ylim(bottom=0); right.set_title("Latency (model loading excluded)")
        right.legend(fontsize=8); right.grid(axis="y", alpha=0.3)
        fig.suptitle("Complete-pipeline coverage and latency", y=1.0)
        path = figures_root / "pipeline_coverage_and_latency.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

        # --- Both pipelines across all three datasets (Experiments 9 and 10) --
        # The single most informative comparison in the project, because the
        # coverage difference reverses between LFW and CPLFW. Drawn only when
        # the 1:1 comparison has been run.
        verif = aggregate_root / VERIFICATION_COMPARISON_DIRNAME
        base_lfw = load("lfw_final_metrics.json")
        base_cplfw = load("cplfw_metrics.json")
        comp_lfw = read_json_artifact(verif / "lfw_final_metrics.json") \
            if (verif / "lfw_final_metrics.json").is_file() else None
        comp_cplfw = read_json_artifact(verif / "cplfw_metrics.json") \
            if (verif / "cplfw_metrics.json").is_file() else None
        if base_lfw and base_cplfw and comp_lfw and comp_cplfw:
            datasets = ["LFW", "CPLFW"]
            series = {
                PIPELINE_DISPLAY_NAMES["opencv"]: (base_lfw, base_cplfw),
                PIPELINE_DISPLAY_NAMES["insightface"]: (comp_lfw, comp_cplfw),
            }
            fig, (acc_ax, cov_ax) = plt.subplots(1, 2, figsize=(12.0, 5.0))
            positions = np.arange(len(datasets))
            width = 0.35
            for index, (name, payloads) in enumerate(series.items()):
                offset = (index - 0.5) * width
                colour = LAYER_PIPELINE_COLOURS[index % 2]
                accuracy = [_percent(d.get("accuracy")) for d in payloads]
                coverage = [
                    _percent(1.0 - d.get("failure_rate", float("nan"))) for d in payloads
                ]
                for ax, values in ((acc_ax, accuracy), (cov_ax, coverage)):
                    bars = ax.bar(positions + offset, values, width, label=name,
                                  color=colour)
                    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
            for ax, title in (
                (acc_ax, "Correct decisions among scored pairs"),
                (cov_ax, "Pairs reaching comparison"),
            ):
                ax.set_xticks(positions); ax.set_xticklabels(datasets, fontsize=10)
                # Headroom for the printed values; ticks still stop at 100%.
                ax.set_ylim(0, 112); ax.set_yticks(list(range(0, 101, 20)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
                ax.set_title(title, fontsize=10)
                ax.grid(axis="y", alpha=0.3)
            acc_ax.set_ylabel("Per cent", fontsize=9)
            acc_ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
            fig.suptitle(
                "Both pipelines on 1:1 verification, by dataset\n"
                "Accuracy favours SCRFD + ArcFace on both; coverage reverses "
                "between them",
                fontsize=11, y=1.04,
            )
            path = figures_root / "pipeline_across_datasets.png"
            _save_figure(fig, path); plt.close(fig); written.append(path)
    elif open_set:
        coverage = open_set["methods"][METHOD_B]["coverage"]
        fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.2))
        bars = ["Gallery", "Mated probe", "Non-mated probe"]
        values = [
            _percent(coverage["gallery_enrolment_coverage"]),
            _percent(1.0 - coverage["mated_extraction_failure_rate"]),
            _percent(1.0 - coverage["non_mated_extraction_failure_rate"]),
        ]
        left.bar(bars, values, color="#55A868")
        left.set_ylabel("Coverage (%)"); left.set_ylim(0, 100)
        left.set_title("Extraction coverage"); left.grid(axis="y", alpha=0.3)
        latencies = [coverage["top1_search_time_mean_ms"], coverage["top5_search_time_p95_ms"]]
        right.bar(["Search mean", "Search p95"], latencies, color="#DD8452")
        right.set_ylabel("Milliseconds"); right.set_ylim(bottom=0)
        right.set_title("Search latency"); right.grid(axis="y", alpha=0.3)
        pipeline_name = open_set.get("pipeline_name") or MODEL_VERSION
        fig.suptitle(f"{pipeline_name} coverage and latency", y=1.0)
        path = figures_root / "pipeline_coverage_and_latency.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)


    # --- Experiment 12: the two components varied one at a time --------------
    crossed = {
        name: load_from(aggregate_root / MIXED_PIPELINE_DIRNAME / name,
                        "bfw_open_set_test_metrics.json")
        for name in ("scrfd-sface", "yunet-arcface")
    }
    crossed_bands = {
        name: load_from(aggregate_root / MIXED_PIPELINE_DIRNAME / name,
                        "bfw_open_set_confidence_intervals.json")
        for name in ("scrfd-sface", "yunet-arcface")
    }
    if all(crossed.values()) and pipeline:
        held = pipeline.get("held_out_metrics") or {}

        def matched(fragment: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            for key, metrics in held.items():
                if fragment in key:
                    return (metrics.get("rates") or {},
                            metrics.get("confidence_intervals") or {})
            return {}, {}

        def crossing(name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            payload = crossed[name] or {}
            point = ((payload.get("methods") or {}).get(METHOD_B) or {}).get(
                "primary_operating_point") or {}
            return point, ((crossed_bands[name] or {}).get("intervals") or {})

        cells = [
            ("YuNet\n+ SFace", *matched("opencv")),
            ("SCRFD\n+ SFace", *crossing("scrfd-sface")),
            ("YuNet\n+ ArcFace", *crossing("yunet-arcface")),
            ("SCRFD\n+ ArcFace", *matched("insightface")),
        ]
        if all(rates for _n, rates, _b in cells):
            names = [name for name, _r, _b in cells]
            detection = [_percent(rates.get("tpir_rank1")) for _n, rates, _b in cells]
            # Asymmetric error bars, because a percentile interval is not
            # centred on its point estimate.
            lower, upper = [], []
            for (_name, rates, bands), point in zip(cells, detection):
                band = bands.get("tpir_rank1") or {}
                low, high = band.get("lower_95"), band.get("upper_95")
                if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                    lower.append(max(0.0, point - low * 100.0))
                    upper.append(max(0.0, high * 100.0 - point))
                else:
                    lower.append(0.0); upper.append(0.0)
            burden = [rates.get("false_reviews_per_1000_non_mated", float("nan"))
                      for _n, rates, _b in cells]
            # SFace cells first, then ArcFace, so the embedder groups read as pairs.
            colours = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]

            fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 5.0))
            positions = np.arange(len(names))
            bars = left.bar(positions, detection, 0.6, color=colours,
                            yerr=[lower, upper], capsize=4, ecolor="#444444")
            left.bar_label(bars, fmt="%.2f%%", fontsize=8, padding=6)
            left.set_ylim(85, 102); left.set_yticks(list(range(85, 101, 5)))
            left.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
            left.set_title("Known duplicates detected (TPIR@1), with 95% intervals",
                           fontsize=10)
            left.set_ylabel("Per cent", fontsize=9)

            bars = right.bar(positions, burden, 0.6, color=colours)
            right.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
            right.set_ylim(bottom=0)
            right.set_title("New profiles wrongly referred, per 1,000", fontsize=10)
            right.set_ylabel("Referrals per 1,000", fontsize=9)

            for ax in (left, right):
                ax.set_xticks(positions)
                ax.set_xticklabels(names, fontsize=9)
                ax.grid(axis="y", alpha=0.3)
            legend = [
                Line2D([0], [0], color="#4C72B0", lw=8, label="SFace embedder (128)"),
                Line2D([0], [0], color="#DD8452", lw=8, label="ArcFace embedder (512)"),
            ]
            left.legend(handles=legend, fontsize=9, loc="lower left", framealpha=0.95)
            fig.suptitle(
                "The detector and the embedder varied one at a time\n"
                "Conditional detection and false referrals describe different scored subsets",
                fontsize=11, y=1.04,
            )
            path = figures_root / "detector_embedder_crossed.png"
            _save_figure(fig, path); plt.close(fig); written.append(path)

    paired = load_from(aggregate_root, "comparative_statistics.json")
    if paired:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, scope, metric, title in zip(
            axes,
            ("full_protocol", "full_protocol", "common_success"),
            ("mated_extraction_coverage", "end_to_end_duplicate_detection_rate", "tpir_rank1"),
            ("Mated probes reaching comparison", "Duplicates detected / all intended mated probes",
             "TPIR@1 / common successfully scored probes"),
        ):
            rows = [paired["analyses"][scope]["pipelines"][name][metric]
                    for name in COMPARISON_PIPELINE_ORDER]
            points = [100 * row["estimate"] for row in rows]
            errors = [[max(0, point - 100 * row["lower_95"]) for point, row in zip(points, rows)],
                      [max(0, 100 * row["upper_95"] - point) for point, row in zip(points, rows)]]
            bars = ax.bar(np.arange(4), points, color=["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"],
                          yerr=errors, capsize=3)
            ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=5)
            ax.set_xticks(np.arange(4), [name.replace(" + ", "\n+ ") for name in COMPARISON_PIPELINE_ORDER])
            ax.set_ylim(0, 110)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Per cent")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Detector and recogniser comparison: coverage, end-to-end detection and common subset\n"
                     "95% identity-cluster intervals; paired differences are reported separately", fontsize=11)
        fig.subplots_adjust(top=0.80, wspace=0.30)
        path = figures_root / "paired_pipeline_comparison.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

    write_implementation_layer_artefacts(aggregate_root)
    _write_figure_captions(aggregate_root, figures_root, written)
    (aggregate_root / "RESEARCH_REPORT.md").write_text(
        render_research_report(aggregate_root), encoding="utf-8"
    )

    # Figures are published artefacts and are scanned like any other.
    leaks = find_path_leaks(figures_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise PrivacyLeakError(
            "Refusing to publish figures containing a personal/absolute path:\n"
            + "\n".join(f"  {redact_private_paths(leak)}" for leak in leaks)
        )
    return written


# =============================================================================
# 29. Synthetic self-test mode
# =============================================================================
#
# Deterministic checks that need no model binary, no dataset and no network.
# Detection and embedding are stood in for by small fakes keyed off image
# content, so every behaviour below is fully reproducible.


class SelfTestFailure(AssertionError):
    """Raised by a self-test assertion helper when an expectation does not hold."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SelfTestFailure(message)


def _assert_close(actual: float, expected: float, message: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise SelfTestFailure(f"{message} (expected {expected!r}, got {actual!r})")


def _assert_raises(
    exception: type[BaseException], callable_: Callable[[], Any], message: str
) -> None:
    try:
        callable_()
    except exception:
        return
    except Exception as exc:  # noqa: BLE001 - the wrong exception is still a failure
        raise SelfTestFailure(f"{message} (raised {type(exc).__name__} instead)") from exc
    raise SelfTestFailure(f"{message} (nothing was raised)")


def _image_key(bgr: np.ndarray) -> str:
    return hashlib.sha256(bgr.tobytes()).hexdigest()


class SyntheticDetector:
    """Duck-types YuNetDetector.detect_single_face without a real model."""

    def __init__(self, face_counts: Optional[Dict[str, int]] = None, default_count: int = 1):
        self.face_counts = face_counts or {}
        self.default_count = default_count

    def detect_single_face(self, bgr: np.ndarray) -> np.ndarray:
        count = self.face_counts.get(_image_key(bgr), self.default_count)
        if count != 1:
            raise FaceCountError(count)
        return np.array(
            [0, 0, bgr.shape[1], bgr.shape[0], 0, 0, 0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32
        )


class SyntheticEmbedder:
    """Duck-types SFaceEmbedder.embed: one deterministic vector per image."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS):
        self.dimensions = dimensions

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        seed = int(_image_key(bgr)[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.dimensions)


def _write_synthetic_image(directory: Path, name: str, fill: int) -> Tuple[Path, str]:
    """A tiny, lossless, uniform-colour PNG. Every channel holds the same value,
    so the bytes are identical read back as RGB or BGR and the returned key
    matches what load_image_bgr will see after the round trip."""
    from PIL import Image

    array = np.full((8, 8, 3), fill, dtype=np.uint8)
    path = directory / name
    Image.fromarray(array, mode="RGB").save(path, format="PNG")
    return path, _image_key(array)


def _self_test_cosine_similarity() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    _assert_close(cosine_similarity(a, a), 1.0, "identical vectors must score 1.0")
    _assert_close(cosine_similarity(a, b), 0.0, "orthogonal vectors must score 0.0")
    _assert_close(cosine_similarity(a, -a), -1.0, "opposed vectors must score -1.0")
    _assert_close(
        cosine_similarity(a, 5.0 * a), 1.0, "cosine similarity must be scale-invariant"
    )
    _assert_raises(
        SimilarityError, lambda: cosine_similarity(a, np.array([1.0, 0.0])), "ragged input"
    )
    _assert_raises(
        SimilarityError,
        lambda: cosine_similarity(a, np.array([np.nan, 0.0, 0.0])),
        "non-finite input",
    )


def _self_test_l2_normalisation() -> None:
    vector = np.array([3.0, 4.0])
    normalized = l2_normalize(vector)
    _assert_close(float(np.linalg.norm(normalized)), 1.0, "unit norm after normalisation")
    _assert_close(float(normalized[0]), 0.6, "direction preserved")
    _assert_close(float(normalized[1]), 0.8, "direction preserved")
    _assert_raises(SimilarityError, lambda: l2_normalize(np.zeros(4)), "zero vector")
    _assert_raises(
        SimilarityError, lambda: l2_normalize(np.array([np.inf, 1.0])), "non-finite vector"
    )


def _self_test_confusion_matrix_accounting() -> None:
    scores = [0.9, 0.8, 0.4, 0.1]
    labels = [1, 0, 1, 0]
    matrix = confusion_matrix(scores, labels, 0.5)
    _assert(matrix.true_positive == 1, "one true positive expected")
    _assert(matrix.false_positive == 1, "one false positive expected")
    _assert(matrix.true_negative == 1, "one true negative expected")
    _assert(matrix.false_negative == 1, "one false negative expected")
    _assert(matrix.total == len(scores), "every scored pair must land in exactly one cell")
    # The decision rule is inclusive: a score exactly on the threshold matches.
    boundary = confusion_matrix([0.5, 0.1], [1, 0], 0.5)
    _assert(boundary.true_positive == 1, "score == threshold must count as a predicted match")
    _assert_raises(MetricsError, lambda: confusion_matrix([0.5], [1], 0.5), "single-class labels")


def _self_test_rate_derivation() -> None:
    matrix = ConfusionMatrix(true_positive=8, false_positive=1, true_negative=9, false_negative=2)
    rates = rates_from_confusion(matrix)
    _assert_close(rates["accuracy"], 17 / 20, "accuracy")
    _assert_close(rates["precision"], 8 / 9, "precision")
    _assert_close(rates["recall"], 8 / 10, "recall")
    _assert_close(rates["f1"], 2 * (8 / 9) * 0.8 / ((8 / 9) + 0.8), "F1")
    _assert_close(rates["false_match_rate"], 1 / 10, "false match rate")
    _assert_close(rates["false_non_match_rate"], 2 / 10, "false non-match rate")
    # An undefined rate must propagate as NaN rather than a misleading zero.
    empty = rates_from_confusion(
        ConfusionMatrix(true_positive=0, false_positive=0, true_negative=5, false_negative=0)
    )
    _assert(empty["recall"] != empty["recall"], "recall must be NaN with no positives")
    _assert(empty["f1"] != empty["f1"], "F1 must be NaN when a component is undefined")


def _self_test_roc_auc() -> None:
    _assert_close(roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0, "perfect separation")
    _assert_close(roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0, "inverted separation")
    # Fully tied scores carry no ranking information at all.
    _assert_close(roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5, "tied scores")
    _assert_close(roc_auc([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0]), 0.75, "partial separation")
    _assert_raises(MetricsError, lambda: roc_auc([0.9, 0.8], [1, 1]), "single-class input")


def _self_test_equal_error_rate() -> None:
    perfect = equal_error_rate([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    _assert_close(perfect["equal_error_rate"], 0.0, "separable scores give a zero EER")
    symmetric = equal_error_rate([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0])
    _assert(
        0.0 <= symmetric["equal_error_rate"] <= 1.0, "EER must lie within the unit interval"
    )
    _assert(
        "threshold" in symmetric, "EER must report the threshold at which the rates cross"
    )


def _self_test_threshold_candidates() -> None:
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    genuine = rng.normal(0.7, 0.05, 60)
    impostor = rng.normal(0.2, 0.05, 60)
    scores = [*genuine.tolist(), *impostor.tolist()]
    labels = [1] * 60 + [0] * 60

    result = calibrate(scores, labels, split=VALIDATION_SPLIT)
    _assert(result.status == CANDIDATES_STATUS, "stage 1 must never produce a frozen artifact")
    expected = {"balanced_accuracy", "f1", "eer", "target_fmr_0.001", "target_fmr_0.01", "target_fmr_0.05"}
    _assert(set(result.candidates) == expected, f"expected candidates {sorted(expected)}")
    # Calibration is confined to the validation split by construction.
    _assert_raises(
        CalibrationError,
        lambda: calibrate(scores, labels, split="test"),
        "calibration on a non-validation split must be refused",
    )


def _self_test_deterministic_selection() -> None:
    candidates = {
        "balanced_accuracy": {"threshold": 0.50},
        "f1": {"threshold": 0.40},
        "eer": {"threshold": 0.45},
    }
    dev_scores = [0.9, 0.8, 0.46, 0.44, 0.3, 0.1]
    dev_labels = [1, 1, 1, 0, 0, 0]

    first = select_final_threshold(candidates, dev_scores, dev_labels)
    second = select_final_threshold(candidates, dev_scores, dev_labels)
    _assert(
        first["selected_candidate"] == second["selected_candidate"],
        "selection must be reproducible for identical input",
    )
    _assert(first["selection_rule"] == SELECTION_RULE, "the published rule must be recorded")
    _assert(
        len(first["all_candidates_dev_metrics"]) == len(candidates),
        "every candidate's development metrics must be retained as evidence",
    )
    # A tie on balanced accuracy is broken by false match rate, then by name,
    # so the outcome never depends on dictionary ordering.
    tied = {"b_name": {"threshold": 0.45}, "a_name": {"threshold": 0.45}}
    tie_result = select_final_threshold(tied, dev_scores, dev_labels)
    _assert(tie_result["selected_candidate"] == "a_name", "a remaining tie is broken by name")


def _self_test_frozen_threshold_enforcement() -> None:
    _assert_raises(
        CalibrationError,
        lambda: require_frozen_threshold({"status": CANDIDATES_STATUS, "threshold": 0.4}),
        "a candidates artifact must never be usable for a final evaluation",
    )
    _assert_raises(
        CalibrationError,
        lambda: require_frozen_threshold({"threshold": 0.4}),
        "an artifact without a status must be refused",
    )
    _assert_raises(
        CalibrationError,
        lambda: require_frozen_threshold({"status": FROZEN_STATUS}),
        "a frozen artifact without a numeric threshold must be refused",
    )
    _assert_close(
        require_frozen_threshold({"status": FROZEN_STATUS, "threshold": 0.42}),
        0.42,
        "a properly frozen threshold must be accepted",
    )
    _assert_raises(
        CalibrationError,
        lambda: require_candidates({"status": FROZEN_STATUS, "candidates": {"a": {}}}),
        "an already-frozen artifact must never be re-selected from",
    )


def _self_test_failure_accounting() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        good_a, good_a_key = _write_synthetic_image(root, "good_a.png", 10)
        good_b, good_b_key = _write_synthetic_image(root, "good_b.png", 20)
        empty, empty_key = _write_synthetic_image(root, "empty.png", 30)
        crowd, crowd_key = _write_synthetic_image(root, "crowd.png", 40)

        # A stand-in detector told to find no face in one image and two in
        # another. Both are extraction failures, and the point of this test is
        # that neither is quietly dropped from the protocol total.
        detector = SyntheticDetector({empty_key: 0, crowd_key: 2})
        embedder = SyntheticEmbedder()

        # Five pairs: one that scores normally, and four that must fail because
        # one side of each cannot yield exactly one face.
        pairs = [
            Pair(good_a, good_b, True, "a", "b"),
            Pair(empty, good_b, True, "a", "b"),
            Pair(good_a, empty, False, "a", "b"),
            Pair(crowd, good_b, True, "a", "b"),
            Pair(good_a, crowd, False, "a", "b"),
        ]
        result = evaluate_pairs(pairs, detector=detector, embedder=embedder)

        _assert(result.total_pairs == 5, "the protocol total must be preserved")
        _assert(result.scored_pair_count == 1, "only the fully readable pair can be scored")
        _assert(result.failed_pairs == 4, "failed pairs stay inside the protocol total")
        _assert(
            result.failures
            == {
                "zero_faces_left": 1,
                "zero_faces_right": 1,
                "multiple_faces_left": 1,
                "multiple_faces_right": 1,
            },
            "the four extraction-failure categories must be counted separately",
        )
        result.validate_accounting()
        _assert_close(result.failure_rate, 4 / 5, "failure rate is a fraction of the protocol")
        _assert(good_a_key != good_b_key, "distinct fixtures must hash differently")


def _self_test_gallery_role_uniqueness() -> None:
    images = {
        "alpha": [Path("/tmp/alpha/alpha_0001.jpg"), Path("/tmp/alpha/alpha_0002.jpg")],
        "bravo": [Path("/tmp/bravo/bravo_0001.jpg"), Path("/tmp/bravo/bravo_0002.jpg")],
        "charlie": [Path("/tmp/charlie/charlie_0001.jpg")],
    }
    manifest = build_manifest(images)
    roles = [entry.role for entry in manifest.entries]
    _assert(roles.count("gallery") == 2, "each multi-image identity contributes one gallery entry")
    _assert(roles.count("duplicate_probe") == 2, "and exactly one duplicate probe")
    _assert(roles.count("unknown_probe") == 1, "single-image identities become unknown probes")
    paths = [entry.image_path for entry in manifest.entries]
    _assert(len(paths) == len(set(paths)), "no image may hold more than one role")
    # An identity excluded by calibration must not reappear in the gallery.
    filtered = build_manifest(images, excluded_images=[Path("/tmp/alpha/alpha_0002.jpg")])
    filtered_paths = {entry.image_path for entry in filtered.entries}
    _assert(
        Path("/tmp/alpha/alpha_0002.jpg") not in filtered_paths,
        "excluded calibration images must never enter the gallery experiment",
    )


def _self_test_opaque_id_stability() -> None:
    first = opaque_id("Example_Identity")
    _assert(first == opaque_id("Example_Identity"), "opaque IDs must be reproducible")
    _assert(first != opaque_id("Example_Identity2"), "distinct inputs must not collide")
    _assert(
        len(first) == OPAQUE_ID_HEX_LENGTH,
        f"opaque IDs are truncated to {OPAQUE_ID_HEX_LENGTH} hexadecimal characters",
    )
    _assert(
        "Example_Identity" not in first, "an opaque ID must not contain its own input"
    )
    with temporary_id_hmac_key("f" * 63 + "e"):
        _assert(
            opaque_id("Example_Identity") != first,
            "the secret key must participate in the digest",
        )
    _assert(opaque_ids_match(first, opaque_id("Example_Identity")), "constant-time equality holds")
    _assert(scrub_filename(Path("/private/root/Name_0001.jpg")) == "Name_0001.jpg", "filename only")


def _self_test_path_leak_detection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "clean.json").write_text('{"accuracy": 0.99}\n', encoding="utf-8")
        _assert(
            find_path_leaks(root, forbidden_substrings=["/Users/"]) == [],
            "a clean artifact must produce no findings",
        )
        (root / "leaky.json").write_text('{"root": "/Users/example/data"}\n', encoding="utf-8")
        findings = find_path_leaks(root, forbidden_substrings=["/Users/"])
        _assert(len(findings) == 1, "an absolute path in a published artifact must be reported")
        _assert("leaky.json" in findings[0], "the finding must name the offending file")
    _assert_raises(
        PrivacyLeakError,
        lambda: assert_no_leakage({"image_path": "x"}),
        "a key that names a path must be refused",
    )
    _assert_raises(
        PrivacyLeakError,
        lambda: assert_no_leakage({"root": "/private/location"}),
        "an absolute-looking value must be refused",
    )
    _assert_raises(
        PrivacyLeakError,
        lambda: assert_no_leakage({"vector": [0.1] * 64}),
        "a raw embedding vector must be refused",
    )
    assert_no_leakage({"identity_hash": "abc123", "accuracy": 0.99})


def _self_test_deterministic_gallery_sampling() -> None:
    # Twenty people with one photograph each, who therefore stand in for new
    # registrations, plus one with two photographs who can be enrolled and then
    # searched for. No real image is touched; only the sampling rule is tested.
    images = {
        f"identity_{index:02d}": [Path(f"/tmp/i{index}/identity_{index:02d}_0001.jpg")]
        for index in range(20)
    }
    images["anchor"] = [Path("/tmp/anchor/anchor_0001.jpg"), Path("/tmp/anchor/anchor_0002.jpg")]

    # Building twice with one seed must give the same manifest, and building
    # with another seed must give a different one. Together these show the
    # sampling is driven by the seed alone and is genuinely reproducible.
    first = build_manifest(images, seed=DEFAULT_RANDOM_SEED, max_unknown_identities=5)
    second = build_manifest(images, seed=DEFAULT_RANDOM_SEED, max_unknown_identities=5)
    _assert(
        [e.sample_id for e in first.entries] == [e.sample_id for e in second.entries],
        "the same seed must reproduce the same manifest exactly",
    )
    different = build_manifest(images, seed=DEFAULT_RANDOM_SEED + 1, max_unknown_identities=5)
    _assert(
        {e.sample_id for e in first.entries} != {e.sample_id for e in different.entries},
        "a different seed must sample a different unknown-probe set",
    )
    _assert(first.seed == DEFAULT_RANDOM_SEED, "the manifest must record the seed it used")


SELF_TESTS: Sequence[Tuple[str, Callable[[], None]]] = (
    ("cosine similarity", _self_test_cosine_similarity),
    ("L2 normalisation", _self_test_l2_normalisation),
    ("confusion-matrix accounting", _self_test_confusion_matrix_accounting),
    ("accuracy, precision, recall and F1", _self_test_rate_derivation),
    ("ROC-AUC", _self_test_roc_auc),
    ("equal error rate", _self_test_equal_error_rate),
    ("threshold candidate generation", _self_test_threshold_candidates),
    ("deterministic threshold selection", _self_test_deterministic_selection),
    ("rejection of non-frozen final thresholds", _self_test_frozen_threshold_enforcement),
    ("failure accounting", _self_test_failure_accounting),
    ("gallery role uniqueness", _self_test_gallery_role_uniqueness),
    ("opaque ID stability", _self_test_opaque_id_stability),
    ("path-leak detection", _self_test_path_leak_detection),
    ("deterministic gallery sampling", _self_test_deterministic_gallery_sampling),
)


def run_self_tests(verbose: bool = True) -> Tuple[int, int]:
    """Run every synthetic self-test. Returns (passed, failed).

    The whole run installs a fixed in-memory identifier key, so the self-tests
    stay runnable on a machine with no research configuration and never depend
    on the researcher's real key."""
    passed = 0
    failed = 0
    with temporary_id_hmac_key(SELF_TEST_ID_HMAC_KEY):
        for name, test in SELF_TESTS:
            try:
                test()
            except Exception as exc:  # noqa: BLE001 - any failure is reported, never swallowed
                failed += 1
                if verbose:
                    print(f"FAIL {name}: {redact_private_paths(str(exc))}")
            else:
                passed += 1
                if verbose:
                    print(f"PASS {name}")
    if verbose:
        print("")
        print(f"SELF-TEST RESULT: {'PASS' if failed == 0 else 'FAIL'}")
        print(f"Tests passed: {passed}")
        print(f"Tests failed: {failed}")
    return passed, failed



###############################################################################
# Results presentation
###############################################################################
#
# Presentation only. Nothing here computes, recalculates or rounds a scientific
# quantity into a stored artefact: every value is read back from the published
# JSON and CSV files and formatted for a reader who does not already know what
# FPIR, TPIR, enrolment or a cluster bootstrap are. The formal reports under
# results/aggregate/ keep their technical wording unchanged.

# Repeated wherever a referral count is shown. The programme produces review
# signals; it establishes none of the facts a reader might otherwise infer.
REFERRAL_DISCLAIMER = (
    "A referral is a model-generated review signal. It is not proof that two "
    "profiles belong to the same person, that a photograph was stolen or that "
    "fraud occurred."
)

# Printed once at start-up, so the scope of the artefact is stated before any
# result is shown.
PROGRAMME_INTRODUCTION = f"""{PROGRAMME_TITLE}

Purpose:
This programme evaluates whether pretrained face-comparison models can help
a human moderator review possible duplicate profiles and inconsistent profile
photographs.

Research objective:
Measure how detector and recogniser choice, enrolment images and threshold
calibration affect duplicate detection, false reviews, subgroup performance
and processing cost. Controlled comparisons test the contribution of each change.

The programme does not automatically identify fraud, ban users or prove that
two profiles belong to the same person.

It uses public academic benchmark datasets and pretrained models. No
face-recognition model is trained or fine-tuned by this project."""

# Internal status vocabulary is machine-readable and stays in the JSON exactly
# as it is. These are display strings only.
PLAIN_STATUS_WORDING = {
    "evaluated_non_commercial_academic_research":
        "Evaluation completed for non-commercial academic research.",
    "ml_review_tested": "Classifier evaluation completed.",
    "open_set_tested": "Held-out open-set evaluation completed.",
    "not_run_licensing_unresolved":
        "Not run: the optional comparison models were not available.",
    "not_run_models_unavailable":
        "Not run: the optional comparison model files were not found.",
}


def plain_status(status: Optional[str]) -> str:
    """Readable wording for an internal status value.

    An unrecognised status is shown as-is rather than hidden, so a new internal
    state cannot silently display as something it is not."""
    if not status:
        return "Status not recorded."
    return PLAIN_STATUS_WORDING.get(status, status)


def format_count_and_percentage(
    count: Optional[float],
    denominator: Optional[float],
    *,
    technical: str = "",
    noun: str = "",
) -> str:
    """One consistent way to state a result: count, denominator and percentage.

    A percentage on its own cannot be checked and hides whether it was measured
    over every intended item or only over those successfully processed, so the
    denominator is always shown beside it."""
    # A missing or not-a-number count is stated as unavailable rather than
    # printed as zero, which a reader would take for a measured result.
    if not isinstance(count, (int, float)) or count != count:
        return "not available"
    # Without a usable denominator only the count is shown. A percentage whose
    # base is unknown is exactly what this helper exists to prevent.
    if not isinstance(denominator, (int, float)) or denominator != denominator or not denominator:
        return f"{int(round(count)):,}" + (f" {noun}" if noun else "")
    share = 100.0 * float(count) / float(denominator)
    body = f"{int(round(count)):,} of {int(round(denominator)):,}"
    if noun:
        body += f" {noun}"
    body += f" ({share:.2f}%)"
    return f"{body} [{technical}]" if technical else body


def _percentage_of(value: Optional[float], *, technical: str = "") -> str:
    """A stored proportion rendered as a percentage, with its technical name."""
    if not isinstance(value, (int, float)) or value != value:
        return "not available"
    text = f"{float(value) * 100.0:.2f}%"
    return f"{text} [{technical}]" if technical else text


def plain_metric_description(key: str) -> str:
    """The interpretation of a metric, followed by its technical name."""
    return PLAIN_METRIC_DESCRIPTIONS.get(key, key)


PLAIN_METRIC_DESCRIPTIONS = {
    "fpir": "New profiles incorrectly sent for human review",
    "tpir_rank1": "Known duplicate test cases whose correct profile ranked first",
    "tpir_rank5": "Known duplicate test cases whose correct profile ranked in the top five",
    "end_to_end": "Known duplicate test cases correctly detected",
    "false_reviews_per_1000": "Unnecessary reviews per 1,000 new profiles",
    "mated_coverage": "Known-duplicate photographs successfully processed",
    "non_mated_coverage": "New-profile photographs successfully processed",
    "gallery_coverage": "Profiles successfully enrolled into the searchable gallery",
}

# The conditional and end-to-end denominators answer different questions, and a
# reader comparing them without knowing which is which would draw the wrong
# conclusion.
DENOMINATOR_NOTE = (
    "Conditional results use only the photographs the model processed "
    "successfully. End-to-end results use every photograph the experiment "
    "intended to process, including those that failed."
)



def wrap_plain(text: str, width: int = 78) -> str:
    """Wrap a paragraph to a terminal-friendly width.

    Presentation only; the wrapped text is never written into an artefact."""
    import textwrap

    return "\n".join(
        textwrap.fill(paragraph, width=width) if paragraph.strip() else ""
        for paragraph in text.split("\n")
    )


def render_plain_pipeline_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], *, indent: str = "  "
) -> str:
    """A fixed-width text table. Column widths follow the widest cell, so a
    long pipeline name cannot push a column out of alignment."""
    # Each column is made as wide as its widest cell, counting the header,
    # so a long pipeline name widens its own column instead of shifting the
    # ones beside it.
    columns = [list(headers)] + [list(row) for row in rows]
    widths = [
        max(len(str(column[index])) for column in columns)
        for index in range(len(headers))
    ]
    # Pad each cell to its column width. Trailing spaces are trimmed, so the
    # terminal output carries no invisible whitespace at the line ends.
    def line(cells: Sequence[str]) -> str:
        return indent + "  ".join(
            str(cell).ljust(widths[index]) for index, cell in enumerate(cells)
        ).rstrip()
    out = [line(headers), indent + "  ".join("-" * w for w in widths)]
    out += [line(row) for row in rows]
    return "\n".join(out)


def render_model_overview() -> str:
    """What each model does and who trained it. Nothing here is trained by this
    project except the logistic-regression review classifier."""
    return """MODELS USED

YuNet
Purpose: Locates faces in photographs.
Source: OpenCV Zoo.
Training: Pretrained externally; not trained by this project.

SFace
Purpose: Converts a detected face into a numerical representation used for
face comparison.
Source: OpenCV Zoo.
Training: Pretrained externally; not trained by this project.

SCRFD
Purpose: Higher-capacity face detector used in the comparison experiment.
Source: InsightFace.
Training: Pretrained externally; not trained by this project.

ArcFace
Purpose: Higher-capacity face-recognition model used in the comparison.
Source: InsightFace buffalo_l model pack.
Training: Pretrained externally; not trained by this project.

Logistic regression
Purpose: Combines search and image-quality measurements to decide whether a
case should be sent for human review.
Training: Fitted by this project using BFW development identities only."""


def render_dataset_overview() -> str:
    """Which benchmark is used where. These are public research benchmarks and
    the people in them are not users of any application."""
    return """DATASETS USED

LFW
Used for one-to-one face comparison and the LFW gallery experiment.

CPLFW
Used to test cross-pose generalisation.

BFW
Used for open-set duplicate-profile evaluation, subgroup reporting, classifier
evaluation and the pretrained pipeline comparison.

These are public academic benchmark datasets. The people photographed in them
are research subjects, not users of any deployed system."""


def render_glossary() -> str:
    """Offered at the end of every summary, so a reader meeting a term for the
    first time does not have to look elsewhere."""
    return """TERMS USED

Enrolment:
Creating a profile representation from one or more photographs.

Gallery:
The set of enrolled profile representations searched by the model.

Mated probe:
A test photograph whose correct profile is already enrolled.

Non-mated probe:
A test photograph belonging to a person who is not enrolled in the gallery.

False review:
A new or non-mated profile incorrectly sent for duplicate-profile review.

TPIR@1:
The percentage of successfully processed known duplicate cases whose correct
profile was ranked first and passed the threshold.

FPIR:
The percentage of successfully processed new profiles incorrectly sent for
review.

Conditional rate:
Calculated only from photographs successfully processed.

End-to-end rate:
Calculated from every intended photograph, including processing failures.

Confidence interval:
A range showing the uncertainty in a benchmark result."""




def render_ml_review_plain_summary(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """Summary of Experiment 7, including the negative finding."""
    payload = _load_optional(aggregate_root, "ml_review_test_metrics.json")
    if not payload:
        return missing_artefact_message("Experiment 7", "option 8")
    classifier = payload["classifier"]
    comparator = payload["comparator_three_image_open_set_calibrated"]["rates"]

    detection_gain = (
        (classifier.get("tpir_rank1", 0.0) - comparator.get("tpir_rank1", 0.0)) * 100.0
    )
    base_reviews = comparator.get("false_reviews_per_1000_non_mated", float("nan"))
    new_reviews = classifier.get("false_reviews_per_1000_non_mated", float("nan"))
    lines = [
        "EXPERIMENT 7 - LOGISTIC-REGRESSION REVIEW CLASSIFIER",
        "",
        "Question:",
        "Did the classifier reduce unnecessary reviews compared with the",
        "calibrated similarity threshold?",
        "",
        render_plain_pipeline_table(
            ["", "Similarity threshold", "Logistic classifier"],
            [
                ["Known duplicate-profile test cases correctly detected (TPIR@1)",
                 _percentage_of(comparator.get("tpir_rank1")),
                 _percentage_of(classifier.get("tpir_rank1"))],
                ["New profiles incorrectly referred for review (FPIR)",
                 _percentage_of(comparator.get("fpir")),
                 _percentage_of(classifier.get("fpir"))],
                ["False reviews per 1,000 new profiles",
                 f"{base_reviews:.1f}", f"{new_reviews:.1f}"],
            ],
        ),
        "",
        "Outcome:",
        "",
        wrap_plain(
            f"The classifier detected approximately {abs(detection_gain):.1f} percentage "
            f"points {'more' if detection_gain >= 0 else 'fewer'} known duplicate cases, "
            f"but it also created more unnecessary reviews."
        ),
        "",
        wrap_plain(
            f"The main hypothesis was not achieved because false reviews increased from "
            f"approximately {base_reviews:.0f} to {new_reviews:.0f} per 1,000 new profiles."
        ),
        "",
        wrap_plain(
            "This is a valid negative research finding. It does not indicate that the "
            "programme failed to run."
        ),
        "",
        "Success criteria:",
        "",
    ]
    # Display wording only. The machine-readable criterion keys and their
    # nested target/actual/outcome objects stay untouched in the JSON.
    for key, entry in sorted((payload.get("success_criteria") or {}).items()):
        if isinstance(entry, Mapping):
            outcome = entry.get("outcome")
        else:
            # A plain boolean, such as the declared-before-test flag.
            outcome = "achieved" if entry else "not_achieved"
        label = PLAIN_CRITERION_WORDING.get(key, key.replace("_", " ").capitalize())
        lines.append(f"  {label}: {str(outcome).replace('_', ' ')}")
    lines += ["", wrap_plain(DENOMINATOR_NOTE), "",
              wrap_plain(REFERRAL_DISCLAIMER)]
    return "\n".join(lines)


# Display wording for the pre-declared criteria. The stored keys are the
# machine-readable names inside the JSON artefact.
PLAIN_CRITERION_WORDING = {
    "criteria_declared_before_test":
        "Success criteria were declared before the held-out test",
    "end_to_end_detection_within_2pp_of_threshold_method":
        "End-to-end detection remained within the permitted difference",
    "fewer_false_reviews_than_threshold_method":
        "Fewer unnecessary reviews than the similarity method",
    "gallery_enrolment_coverage_at_least_90_percent":
        "At least 90% gallery enrolment coverage",
    "primary_fpir_at_or_below_1_percent":
        "New profiles wrongly reviewed stayed at or below 1%",
    "probe_extraction_coverage_at_least_90_percent":
        "At least 90% probe processing coverage",
    "tpir_rank1_at_least_90_percent": "At least 90% rank-one detection",
    "tpir_rank5_at_least_95_percent": "At least 95% top-five detection",
}


def render_pipeline_plain_summary(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """Summary of Experiment 8, with a comparative pipeline table."""
    payload = _load_optional(aggregate_root, "pipeline_comparison_metrics.json")
    if not payload:
        return missing_artefact_message("Experiment 8", "option 10")
    held_out = payload.get("held_out_metrics") or {}
    if payload.get("evaluated") != "yes" or not held_out:
        return "\n".join([
            "EXPERIMENT 8 - PRETRAINED PIPELINE COMPARISON",
            "",
            plain_status(payload.get("status")),
            "",
            wrap_plain(
                "The optional comparison models were not available, so no comparison "
                "figures can be shown. Every other experiment is unaffected."
            ),
        ])

    # Baseline pipeline first, so the columns match every figure and report.
    names = sorted(held_out, key=lambda n: "opencv" not in n)
    def cell(name: str, getter) -> str:
        try:
            return getter(held_out[name])
        except (KeyError, TypeError):
            return "not available"

    rows = [
        ["Known duplicate-profile test cases detected (TPIR@1)",
         *[cell(n, lambda m: _percentage_of(m["rates"]["tpir_rank1"])) for n in names]],
        ["End-to-end detection (all intended photographs)",
         *[cell(n, lambda m: _percentage_of(m["end_to_end_duplicate_detection_rate"]))
           for n in names]],
        ["New profiles incorrectly referred for review (FPIR)",
         *[cell(n, lambda m: _percentage_of(m["rates"]["fpir"])) for n in names]],
        ["False reviews per 1,000 new profiles",
         *[cell(n, lambda m: f"{m['rates']['false_reviews_per_1000_non_mated']:.1f}")
           for n in names]],
        ["Known-duplicate photographs processed (mated coverage)",
         *[cell(n, lambda m: _percentage_of(
             1.0 - m["coverage"]["mated_extraction_failure_rate"])) for n in names]],
        ["New-profile photographs processed (non-mated coverage)",
         *[cell(n, lambda m: _percentage_of(
             1.0 - m["coverage"]["non_mated_extraction_failure_rate"])) for n in names]],
        ["Mean complete processing time per image",
         *[cell(n, lambda m: f"{m['coverage']['complete_pipeline_latency_mean_ms']:.1f} ms")
           for n in names]],
    ]
    # Embedding width and weight-file size are pipeline properties, recorded
    # once at the top of the artefact rather than per held-out result.
    descriptors = {
        names[0]: payload.get("primary_pipeline") or {},
        names[-1]: payload.get("comparison_pipeline") or {},
    }
    sizes = payload.get("model_file_sizes") or {}
    size_groups = {names[0]: sizes.get("primary") or {},
                   names[-1]: sizes.get("comparison") or {}}

    def total_megabytes(name: str) -> str:
        group = size_groups.get(name) or {}
        total = sum(
            entry.get("megabytes", 0.0) for entry in group.values()
            if isinstance(entry, Mapping)
        )
        return f"{total:.1f} MB" if total else "not available"

    rows += [
        ["Numerical face representation size (embedding dimensions)",
         *[f"{descriptors.get(n, {}).get('embedding_dimensions', 'not available')} values"
           for n in names]],
        ["Model storage", *[total_megabytes(n) for n in names]],
    ]
    lines = [
        "EXPERIMENT 8 - PRETRAINED PIPELINE COMPARISON",
        "",
        "Dataset:",
        "BFW held-out test identities.",
        "",
        "No model was trained or fine-tuned.",
        "",
        render_plain_pipeline_table(
            ["Metric", *[pipeline_display_name(n) for n in names]], rows
        ),
        "",
        "Result:",
        "",
    ]
    # The conclusion is generated from the values rather than asserted, so it
    # cannot contradict the table above it.
    baseline, comparison = names[0], names[-1]
    better_detection = (held_out[comparison]["rates"]["tpir_rank1"]
                        > held_out[baseline]["rates"]["tpir_rank1"])
    fewer_reviews = (held_out[comparison]["rates"]["fpir"]
                     < held_out[baseline]["rates"]["fpir"])
    slower = (held_out[comparison]["coverage"]["complete_pipeline_latency_mean_ms"]
              > held_out[baseline]["coverage"]["complete_pipeline_latency_mean_ms"])
    verdict = (
        f"{pipeline_display_name(comparison)} detected "
        f"{'more' if better_detection else 'no more'} known duplicate cases and sent "
        f"{'fewer' if fewer_reviews else 'no fewer'} new profiles for unnecessary review."
    )
    lines += [
        wrap_plain(verdict),
        "",
        wrap_plain(
            "The improvement required larger model files and more processing time."
            if slower else
            "The comparison pipeline did not cost additional processing time here."
        ),
        "",
        wrap_plain(
            "Because the detector, alignment, preprocessing and recognition model all "
            "changed, the improvement must be described as a complete-pipeline result."
        ),
        "",
        "Technical pipeline identifiers:",
    ]
    for name in names:
        lines.append(f"  {pipeline_display_name(name)}: {name}")
    lines += ["", wrap_plain(DENOMINATOR_NOTE), "",
              wrap_plain(REFERRAL_DISCLAIMER)]
    return "\n".join(lines)


# --- Every experiment on one page ---------------------------------------------


def render_experiment_comparison_table(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """All nine experiments side by side: models, dataset, task and outcome.

    Read back from the published artefacts, so a row cannot disagree with the
    file it summarises. An experiment that has not been run yet is shown as
    such rather than omitted, since a missing row would read as a gap in the
    method instead of a gap in what has been executed."""
    root = Path(aggregate_root)
    sub = root / VERIFICATION_COMPARISON_DIRNAME
    load = lambda name, base=root: _load_optional(base, name)

    final = load("lfw_final_metrics.json")
    cplfw = load("cplfw_metrics.json")
    threshold = load("calibrated_threshold.json")
    gallery = load("duplicate_gallery_metrics_v2.json")
    open_set = load("bfw_open_set_test_metrics.json")
    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")
    verif_final = load("lfw_final_metrics.json", sub)
    verif_cplfw = load("cplfw_metrics.json", sub)
    arcface_review = load("ml_review_test_metrics.json", root / ARCFACE_REVIEW_DIRNAME)

    pending = "not run yet"

    def pct(value: Any, digits: int = 2) -> str:
        return f"{value * 100:.{digits}f}%" if isinstance(value, (int, float)) else pending

    rows: List[List[str]] = []

    # Experiments 1-2 produce the operating point every later 1:1 result uses.
    rows.append([
        "1-2", "YuNet + SFace", "LFW dev pairs", "1:1",
        f"threshold frozen at {threshold['threshold']:.6f}" if threshold else pending,
    ])
    rows.append([
        "3", "YuNet + SFace", "LFW pairs.txt", "1:1",
        f"{pct(final.get('accuracy'))} correct, {pct(final.get('failure_rate'))} not scored"
        if final else pending,
    ])
    rows.append([
        "4", "YuNet + SFace", "CPLFW", "1:1",
        f"{pct(cplfw.get('accuracy'))} correct, {pct(cplfw.get('failure_rate'))} not scored"
        if cplfw else pending,
    ])
    # Experiment 5 reuses the 1:1 threshold for gallery search deliberately, so
    # its false-review rate is the problem the later experiments address.
    rows.append([
        "5", "YuNet + SFace", "LFW gallery", "1:N",
        f"{pct(gallery.get('end_to_end_duplicate_detection_rate'))} detected, "
        f"{pct(gallery.get('false_duplicate_review_rate'))} false reviews"
        if gallery else pending,
    ])
    if open_set:
        primary = open_set["methods"][METHOD_B]["primary_operating_point"]
        rows.append([
            "6", "YuNet + SFace", "BFW held-out", "1:N",
            f"{pct(primary.get('tpir_rank1'))} TPIR@1, "
            f"{primary.get('false_reviews_per_1000_non_mated', float('nan')):.1f} reviews/1,000",
        ])
    else:
        rows.append(["6", "YuNet + SFace", "BFW held-out", "1:N", pending])
    if review:
        classifier = review["classifier"]
        rows.append([
            "7", "+ logistic regression", "BFW held-out", "1:N",
            f"{pct(classifier.get('tpir_rank1'))} TPIR@1, "
            f"{classifier.get('false_reviews_per_1000_non_mated', float('nan')):.1f} reviews/1,000",
        ])
    else:
        rows.append(["7", "+ logistic regression", "BFW held-out", "1:N", pending])
    arcface = None
    if pipeline and pipeline.get("evaluated") == "yes":
        arcface = next(
            (v for k, v in (pipeline.get("held_out_metrics") or {}).items()
             if "arcface" in k.lower()), None
        )
    if arcface:
        rates = arcface["rates"]
        rows.append([
            "8", "SCRFD + ArcFace", "BFW held-out", "1:N",
            f"{pct(rates.get('tpir_rank1'))} TPIR@1, "
            f"{rates.get('false_reviews_per_1000_non_mated', float('nan')):.1f} reviews/1,000",
        ])
    else:
        rows.append(["8", "SCRFD + ArcFace", "BFW held-out", "1:N", pending])
    # One row per dataset: LFW and CPLFW pose different problems, and merging
    # them would hide that the coverage difference runs in opposite directions.
    rows.append([
        "9", "SCRFD + ArcFace", "LFW pairs.txt", "1:1",
        f"{pct(verif_final.get('accuracy'))} correct, "
        f"{pct(verif_final.get('failure_rate'))} not scored"
        if verif_final else pending,
    ])
    rows.append([
        "10", "SCRFD + ArcFace", "CPLFW", "1:1",
        f"{pct(verif_cplfw.get('accuracy'))} correct, "
        f"{pct(verif_cplfw.get('failure_rate'))} not scored"
        if verif_cplfw else pending,
    ])
    if arcface_review:
        classifier = arcface_review["classifier"]
        rows.append([
            "11", "SCRFD + ArcFace + classifier", "BFW held-out", "1:N",
            f"{pct(classifier.get('tpir_rank1'))} TPIR@1, "
            f"{classifier.get('false_reviews_per_1000_non_mated', float('nan')):.1f} reviews/1,000",
        ])
    else:
        rows.append(
            ["11", "SCRFD + ArcFace + classifier", "BFW held-out", "1:N", pending]
        )
    # Experiment 12 varies one component at a time, so each crossing is its own
    # row: reading them against 6 and 8 is what separates the two contributions.
    for label, crossing in (("12a", "scrfd-sface"), ("12b", "yunet-arcface")):
        crossed = load("bfw_open_set_test_metrics.json",
                       root / MIXED_PIPELINE_DIRNAME / crossing)
        point = (((crossed or {}).get("methods") or {}).get(METHOD_B) or {}).get(
            "primary_operating_point"
        ) if crossed else None
        detector, embedder = crossing.split("-")
        name = f"{detector.upper() if detector == 'scrfd' else 'YuNet'} + " \
               f"{'SFace' if embedder == 'sface' else 'ArcFace'}"
        rows.append([
            label, name, "BFW held-out", "1:N",
            f"{pct(point.get('tpir_rank1'))} TPIR@1, "
            f"{point.get('false_reviews_per_1000_non_mated', float('nan')):.1f} reviews/1,000"
            if point else pending,
        ])

    return "\n".join([
        "EVERY EXPERIMENT AT A GLANCE",
        "",
        render_plain_pipeline_table(
            ["Exp", "Models", "Dataset", "Task", "Headline result"], rows
        ),
        "",
        wrap_plain(
            "Task 1:1 compares two photographs and reports a false match rate. Task 1:N "
            "searches one photograph against a whole gallery and reports a false positive "
            "identification rate. The two are different quantities and are never pooled: "
            "one comparison per decision against one per enrolled profile."
        ),
        "",
        wrap_plain(
            "Only the logistic regression in Experiment 7 is trained by this project. "
            "YuNet, SFace, SCRFD and ArcFace are pretrained and used as published."
        ),
    ])


def render_processing_coverage_explanation(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """Explain processing coverage using saved results without changing artefacts."""
    root = Path(aggregate_root)
    pair_specs = (
        ("LFW 1:1 (Exp 3)", "YuNet + SFace", "lfw_final_metrics.json"),
        ("LFW 1:1 (Exp 9)", "SCRFD + ArcFace", "verification_comparison/lfw_final_metrics.json"),
        ("CPLFW 1:1 (Exp 4)", "YuNet + SFace", "cplfw_metrics.json"),
        ("CPLFW 1:1 (Exp 10)", "SCRFD + ArcFace", "verification_comparison/cplfw_metrics.json"),
    )
    pairs = [_load_optional(root, filename) for _, _, filename in pair_specs]
    gallery = _load_optional(root, "duplicate_gallery_metrics_v2.json")
    bfw = _load_optional(root, "bfw_open_set_test_metrics.json") or {}
    comparison = _load_optional(root, "pipeline_comparison_metrics.json") or {}
    arcface = next((value for key, value in (comparison.get("held_out_metrics") or {}).items()
                    if "arcface" in key.lower()), None)
    search_specs = [
        ("BFW 1:N (Exp 6)", "YuNet + SFace", (bfw.get("methods") or {}).get(METHOD_B)),
        ("BFW 1:N (Exp 8)", "SCRFD + ArcFace", arcface),
    ]
    for experiment, name, crossing in (
        ("12a", "SCRFD + SFace", "scrfd-sface"),
        ("12b", "YuNet + ArcFace", "yunet-arcface"),
    ):
        saved = _load_optional(root, f"mixed_pipelines/{crossing}/bfw_open_set_test_metrics.json") or {}
        search_specs.append((f"BFW 1:N (Exp {experiment})", name,
                             (saved.get("methods") or {}).get(METHOD_B)))

    def count(value: Any) -> str:
        return f"{value:,}" if isinstance(value, int) else "not available"

    def grouped_failures(breakdown: Mapping[str, int]) -> Dict[str, int]:
        grouped: Dict[str, int] = {}
        for reason, number in breakdown.items():
            reason = re.sub(r"_(left|right|gallery)$", "", reason)
            label = {"zero_faces": "zero faces", "multiple_faces": "multiple faces",
                     "gallery_reference_unavailable": "profile missing"}.get(
                         reason, reason.replace("_", " "))
            grouped[label] = grouped.get(label, 0) + number
        return grouped

    def reasons(breakdown: Mapping[str, int]) -> str:
        return ", ".join(f"{label} {number:,}" for label, number in sorted(
            grouped_failures(breakdown).items(), key=lambda item: (-item[1], item[0])
        ) if number) or "none"

    rows: List[List[str]] = []
    for (label, name, _), payload in zip(pair_specs, pairs):
        if not payload:
            rows.append([label, name, *["not available"] * 5])
            continue
        total, scored = payload["total_pairs"], payload["scored_pairs"]
        matrix = payload["confusion_matrix"]
        correct = matrix["true_positive"] + matrix["true_negative"]
        rows.append([
            label, name, f"{total:,} pairs",
            f"{scored:,} ({_percentage_of(scored / total if total else None)})",
            f"{total - scored:,}: {reasons(payload['failure_breakdown'])}",
            f"{_percentage_of(payload.get('accuracy'))} correct",
            f"{_percentage_of(correct / total if total else None)} correct",
        ])
    if gallery:
        intended = gallery["duplicate_probe_count"] + gallery["unknown_probe_count"]
        failed = gallery["duplicate_probe_failures"] + gallery["unknown_probe_failures"]
        rows.append([
            "LFW gallery 1:N (Exp 5)", "YuNet + SFace",
            f"{gallery['intended_gallery_size']:,} profiles; {intended:,} search photos",
            f"{gallery['embedded_gallery_size']:,} profiles; {intended - failed:,} photos",
            f"{gallery['gallery_entry_failure_count']:,} profiles "
            f"({reasons(gallery['gallery_failure_breakdown'])}); {failed:,} search photos",
            f"{_percentage_of(gallery.get('conditional_duplicate_detection_rate'))} detected; "
            f"{_percentage_of(gallery.get('false_duplicate_review_rate'))} false reviews",
            f"{_percentage_of(gallery.get('end_to_end_duplicate_detection_rate'))} detected",
        ])
    else:
        rows.append(["LFW gallery 1:N (Exp 5)", "YuNet + SFace", *["not available"] * 5])
    for label, name, payload in search_specs:
        if not payload:
            rows.append([label, name, *["not available"] * 5])
            continue
        coverage = payload["coverage"]
        rates = payload.get("primary_operating_point") or payload.get("rates") or {}
        intended = coverage["intended_mated_probes"] + coverage["intended_non_mated_probes"]
        scored = coverage["scored_mated_probes"] + coverage["scored_non_mated_probes"]
        rows.append([
            label, name,
            f"{coverage['intended_gallery_identities']:,} profiles; {intended:,} search photos",
            f"{coverage['enrolled_gallery_identities']:,} profiles; {scored:,} photos",
            f"{intended - scored:,} photos: {reasons(coverage['probe_failure_breakdown'])}",
            f"TPIR@1 {_percentage_of(rates.get('tpir_rank1'))}; FPIR {_percentage_of(rates.get('fpir'))}",
            f"{_percentage_of(payload.get('end_to_end_duplicate_detection_rate'))} detected",
        ])

    lfw_failures = grouped_failures(pairs[1]["failure_breakdown"]) if pairs[1] else {}
    if lfw_failures and lfw_failures.get("multiple faces", 0) == sum(lfw_failures.values()):
        lfw_note = "On LFW, every SCRFD failure was a multiple-face detection."
    elif pairs[1]:
        lfw_note = f"On LFW, SCRFD failures were: {reasons(pairs[1]['failure_breakdown'])}."
    else:
        lfw_note = "On LFW, SCRFD failure details are not available."
    cplfw_failures = grouped_failures(pairs[2]["failure_breakdown"]) if pairs[2] else {}
    if cplfw_failures.get("zero faces", 0) > sum(cplfw_failures.values()) / 2:
        cplfw_note = ("On CPLFW, where many faces are turned sideways, most YuNet failures "
                      "were zero-face detections.")
    elif pairs[2]:
        cplfw_note = f"On CPLFW, YuNet failures were: {reasons(pairs[2]['failure_breakdown'])}."
    else:
        cplfw_note = "On CPLFW, YuNet failure details are not available."
    primary_coverage = (search_specs[0][2] or {}).get("coverage") or {}
    arcface_coverage = (arcface or {}).get("coverage") or {}
    primary_failures = primary_coverage.get("probe_failure_breakdown")
    arcface_failures = arcface_coverage.get("probe_failure_breakdown")
    missed_yunet = count(primary_failures.get("zero_faces", 0) if primary_failures is not None else None)
    missed_scrfd = count(arcface_failures.get("zero_faces", 0) if arcface_failures is not None else None)
    return "\n\n".join([
        section_heading("WHY SOME PHOTOGRAPHS ARE NOT PROCESSED"),
        wrap_plain(
            "A photograph is compared only when exactly one face is found in it. No face "
            "gives a zero-face failure. More than one face gives a multiple-face failure, "
            "because the programme cannot tell which face belongs to the profile. In "
            "gallery search, a known-duplicate photograph also fails when none of its "
            "profile's enrolment photographs could be processed. A failed photograph "
            "gets no decision, but it is still counted."
        ),
        wrap_plain(
            f"The failures are not the same across datasets. SCRFD runs at a "
            f"{ARCFACE_DETECTION_THRESHOLD:g} confidence threshold and YuNet at "
            f"{DETECTOR_SCORE_THRESHOLD:g}. {lfw_note} The extra detections could be "
            f"people in the background or false detections; nobody checked them by hand. "
            f"{cplfw_note} On BFW, where the images are small face crops, YuNet works at "
            f"each photograph's own size and missed {missed_yunet} faces, while SCRFD "
            f"resizes every image to {ARCFACE_DETECTION_INPUT_SIZE} by "
            f"{ARCFACE_DETECTION_INPUT_SIZE} pixels first and missed {missed_scrfd}. "
            "So these results come partly from the settings and the one-face rule, "
            "not only from the models."
        ),
        render_plain_pipeline_table(
            ["Dataset and experiment", "Pipeline", "Intended", "Processed",
             "Not processed and reason", "Result on processed", "End-to-end"], rows
        ),
        "\n".join([
            f"- BFW search photos are {count(primary_coverage.get('intended_mated_probes'))} "
            f"known duplicates and {count(primary_coverage.get('intended_non_mated_probes'))} "
            "new profiles, and failure types are counted across all of them together.",
            "- End-to-end detection applies to known duplicates only.",
            "- The LFW gallery reused the one-to-one threshold, which is why its "
            "false-review rate is so high.",
        ]),
        "These figures only show processing failures. No face-selection rule, threshold "
        "or image size was changed to reduce them.",
    ])


def render_model_comparison_table(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """The five models side by side: what each does and what it cost.

    The experiment table compares arrangements; this compares the parts they
    are built from. Measured columns are read from the artefacts, so a figure
    quoted here is one the programme actually recorded rather than one taken
    from the publishing project's own claims."""
    root = Path(aggregate_root)
    pipeline = _load_optional(root, "pipeline_comparison_metrics.json")
    held_out = (pipeline or {}).get("held_out_metrics") or {}
    sizes = (pipeline or {}).get("model_file_sizes") or {}
    review = _load_optional(root, "ml_review_model.json")

    def group_mb(group: str) -> str:
        entries = (sizes.get(group) or {}).values()
        total = sum(e.get("megabytes", 0.0) for e in entries if isinstance(e, Mapping))
        return f"{total:.1f} MB" if total else "not available"

    def latency(key_fragment: str, stage: str) -> str:
        for name, metrics in held_out.items():
            if key_fragment in name:
                value = (metrics.get("coverage") or {}).get(stage)
                if isinstance(value, (int, float)):
                    return f"{value:.1f} ms"
        return "not available"

    rows = [
        ["YuNet", "Finds the face", "OpenCV Zoo", "bounding box + 5 landmarks",
         "no", "1-7"],
        ["SFace", "Face to numbers", "OpenCV Zoo", "128 values",
         "no", "1-7"],
        ["SCRFD", "Finds the face", "InsightFace", "bounding box + 5 landmarks",
         "no", "8-11"],
        ["ArcFace", "Face to numbers", "InsightFace buffalo_l", "512 values",
         "no", "8-11"],
        ["Logistic regression", "Decides referral", "fitted here",
         f"{len(review['model']['feature_order'])} features" if review else "9 features",
         "yes", "7, 11"],
    ]
    measured = [
        ["Detection, mean per image", latency("opencv", "detection_latency_mean_ms"),
         latency("arcface", "detection_latency_mean_ms")],
        ["Embedding, mean per image", latency("opencv", "embedding_latency_mean_ms"),
         latency("arcface", "embedding_latency_mean_ms")],
        ["Complete pipeline, mean per image",
         latency("opencv", "complete_pipeline_latency_mean_ms"),
         latency("arcface", "complete_pipeline_latency_mean_ms")],
        ["Weight files on disk", group_mb("primary"), group_mb("comparison")],
    ]
    return "\n".join([
        "THE MODELS COMPARED",
        "",
        render_plain_pipeline_table(
            ["Model", "Role", "Source", "Output", "Trained here", "Used in"], rows
        ),
        "",
        "Measured cost, from the BFW held-out run:",
        "",
        render_plain_pipeline_table(
            ["", PIPELINE_DISPLAY_NAMES["opencv"], PIPELINE_DISPLAY_NAMES["insightface"]],
            measured,
        ),
        "",
        wrap_plain(
            "Only the logistic regression is fitted by this project, on development "
            "identities that share nobody with the held-out test set. The four face "
            "networks are pretrained and used exactly as published, which is what makes "
            "the arrangement rather than the models the contribution under test."
        ),
        "",
        wrap_plain(
            "A similarity threshold belongs to the model that produced it. SFace and "
            "ArcFace embed into spaces of different width, so a cutoff calibrated for "
            "one is never applied to the other."
        ),
    ])


def render_overall_conclusion(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """The five project-level findings, with the figures read from artefacts."""
    open_set = _load_optional(aggregate_root, "bfw_open_set_test_metrics.json")
    review = _load_optional(aggregate_root, "ml_review_test_metrics.json")
    pipeline = _load_optional(aggregate_root, "pipeline_comparison_metrics.json")

    def reviews_per_1000(source: Optional[Mapping[str, Any]], *keys: str) -> str:
        node: Any = source
        for key in keys:
            if not isinstance(node, Mapping):
                return "not available"
            node = node.get(key)
        return f"{node:.1f}" if isinstance(node, (int, float)) else "not available"

    control = reviews_per_1000(
        open_set, "methods", METHOD_A, "rates", "false_reviews_per_1000_non_mated")
    calibrated = reviews_per_1000(
        open_set, "methods", METHOD_B, "primary_operating_point",
        "false_reviews_per_1000_non_mated")
    classifier = reviews_per_1000(
        review, "classifier", "false_reviews_per_1000_non_mated")

    lines = [
        "OVERALL PROJECT CONCLUSION",
        "",
        wrap_plain(
            f"1. The transferred one-to-one threshold was not suitable for searching a "
            f"large profile gallery because it produced too many unnecessary reviews "
            f"({control} per 1,000 new profiles)."
        ),
        "",
        wrap_plain(
            f"2. Selecting a threshold specifically for open-set gallery search reduced "
            f"the false-review burden substantially, to {calibrated} per 1,000."
        ),
        "",
        wrap_plain(
            f"3. The logistic-regression classifier increased duplicate detection "
            f"slightly but did not reduce false reviews ({classifier} per 1,000). Its "
            f"main hypothesis was therefore not achieved."
        ),
        "",
    ]
    if pipeline and pipeline.get("evaluated") == "yes":
        held_out = pipeline.get("held_out_metrics") or {}
        arcface = next((v for k, v in held_out.items() if "arcface" in k.lower()), None)
        if arcface:
            lines += [wrap_plain(
                f"4. SCRFD + ArcFace produced the strongest benchmark results, with "
                f"{arcface['rates']['false_reviews_per_1000_non_mated']:.1f} false reviews "
                f"per 1,000 and "
                f"{_percentage_of(arcface['rates']['tpir_rank1'])} detection, but required "
                f"more processing time and larger model files."
            ), ""]
    else:
        lines += [wrap_plain(
            "4. The pretrained pipeline comparison has not been run in this checkout, so "
            "no comparison finding is reported."
        ), ""]
    lines += [wrap_plain(
        "5. Every result is a benchmark-based human-review signal. The project does not "
        "prove identity, photograph ownership, fraud or profile duplication."
    )]
    return "\n".join(lines)


# --- Previews shown before a long-running option starts -----------------------

EXPERIMENT_PREVIEWS = {
    "all": """Selected: Run all experiments, comparisons and reports

Purpose:
Run the software and input checks, programme self-tests, Experiments 1-12,
controlled comparison analyses and paired statistics in dependency order.
Then refresh the reports and figures and show every experiment at a glance.

Requires the configured LFW, raw CPLFW and BFW datasets, all four pretrained
models (YuNet, SFace, SCRFD and ArcFace), and their dependencies.

This can take a long time and updates the saved results. Each experiment runs
once; its action also displays its results. The batch stops at the first failed
step, so later experiments cannot use incomplete prerequisite results.
Human review moderation remains a separate interactive option (25).""",

    "full": """Selected: Experiments 1-5 - face verification and gallery evaluation

Purpose:
Evaluate LFW with its official ten folds. Separately choose a development
threshold, test its transfer when facial pose changes, and then show what
happens when that same one-to-one threshold is used to search one photograph
against many enrolled profiles.

Datasets:
LFW and CPLFW, using the official published pair protocols.

Models:
YuNet face detector + SFace face-recognition model.

This evaluation will:
1. Produce candidate thresholds from the LFW training pairs only.
2. Select and freeze one threshold using the LFW development pairs.
3. Evaluate LFW's ten folds, fitting each threshold on the other nine folds.
4. Apply the same frozen threshold to CPLFW without recalibrating it.
5. Search a 1:N profile gallery under that same one-to-one threshold.

No model will be trained or fine-tuned.""",

    "open-set": """Selected: Experiment 6 - BFW duplicate-profile evaluation

Purpose:
Test whether a face-comparison system can recognise known duplicate-profile
test cases while avoiding unnecessary human reviews of new profiles.

Dataset:
BFW - 20,000 facial images from 800 benchmark identities.

Models:
YuNet face detector + SFace face-recognition model.

This experiment will:
1. Create separate development and held-out identity groups.
2. Build three-image profile templates.
3. Choose the operating threshold using development identities only.
4. Test the frozen threshold on unseen identities.
5. Report detection, false reviews and processing failures.

No model will be trained or fine-tuned.""",

    "ml-review": """Selected: Experiment 7 - machine-learning review classifier

Purpose:
Test whether a logistic-regression classifier can reduce unnecessary human
reviews while retaining duplicate-profile detection.

Dataset:
BFW, using identity groups that share no person with one another.

The classifier uses similarity and image-quality measurements. It does not use
sex, ethnicity, identity names, image paths or face embeddings as predictor
variables.

This experiment will:
1. Build search features for each test photograph.
2. Fit the classifier on development identities only.
3. Freeze a referral probability using separate calibration identities.
4. Apply the frozen probability to unseen held-out identities.
5. Compare the outcome with the similarity threshold of Experiment 6.

No face-recognition model will be trained or fine-tuned.""",

    "pipeline-compare": """Selected: Experiment 8 - pretrained pipeline comparison

Purpose:
Compare the existing YuNet + SFace pipeline with the higher-capacity
SCRFD + ArcFace pipeline.

Both pipelines use:
- the same BFW identities;
- the same development and held-out partitions;
- separate thresholds selected using development data only;
- the same human-review policy.

This is a complete-pipeline comparison. Differences cannot be attributed only
to the recognition model because detection, alignment and preprocessing also
differ.

No model will be trained or fine-tuned.""",

    "extensions": """Selected: Experiments 7 and 8, then regenerate all figures

Purpose:
Run the logistic-regression review classifier and the pretrained pipeline
comparison, then rebuild every figure from the resulting artefacts.

This option runs Experiments 7 and 8 only. Experiment 6 must already have been
run, because both extensions reuse its frozen threshold and its canonical run.

No model will be trained or fine-tuned.""",

    "verification-compare": """Selected: Experiments 9 and 10 - both pipelines on 1:1 verification

Purpose:
Test whether the advantage SCRFD + ArcFace showed on gallery search also holds
for one-to-one verification, and in particular under the extreme pose
variation of CPLFW.

Datasets:
LFW pairs.txt and CPLFW pairs_CPLFW.txt, 6,000 pairs each, raw images.

Models:
InsightFace SCRFD detector + ArcFace buffalo_l recognition model.

This experiment will:
1. Produce candidate thresholds from the LFW training pairs only.
2. Select and freeze one threshold using the LFW development pairs.
3. Evaluate LFW's ten folds, fitting each threshold on the other nine folds.
4. Apply that same threshold to CPLFW without recalibrating it.

The baseline threshold is never reused: a similarity from a 512-dimensional
ArcFace vector does not mean what the same number means for SFace. Results are
written to a separate directory, so Experiments 1-5 are
untouched.

No model will be trained or fine-tuned.""",

    "mixed-pipelines": """Selected: Experiment 12 - the detectors and embedders crossed

Purpose:
Experiments 6 and 8 change the detector and the embedder together, so the gap
between them cannot say which component earned it. Running the two crossings
varies one component at a time and separates the two contributions.

Dataset:
BFW, the same identity-disjoint protocol, groups and seed as Experiment 6, so
all four cells are measured on the same held-out identities.

Models:
SCRFD detector with the SFace embedder, then the YuNet detector with the
ArcFace embedder. Both halves of each crossing are pretrained and used as
published; only the arrangement is new.

This experiment will:
1. Freeze an operating threshold for each crossing on the development
   identities alone, by the rule Experiment 6 uses.
2. Apply each frozen threshold unchanged to the held-out identities.
3. Compute cluster-bootstrap intervals over identities.
4. Print the two-by-two table against Experiments 6 and 8.

Only the three-image template method is run. The single-image control needs a
one-to-one threshold calibrated in each crossing's own embedding space, which
no crossing has.

Each crossing writes into its own directory and its own cache, so no existing
result is overwritten.
""",
    "arcface-review": """Selected: Experiment 11 - the review classifier on SCRFD + ArcFace

Purpose:
Experiment 7 fitted the classifier on YuNet + SFace and Experiment 8 compared
the pipelines without it, so the framework's most elaborate addition and its
strongest components were never combined. This tests whether the classifier
still helps once the models underneath it are better.

Dataset:
BFW, using the same identity groups and the same seed as Experiment 7, so the
two classifiers are directly comparable.

Models:
InsightFace SCRFD + ArcFace, plus a logistic-regression classifier fitted here.

This experiment will:
1. Score the BFW development identities with SCRFD + ArcFace.
2. Fit the classifier on the training group only.
3. Freeze a referral probability on the separate calibration group.
4. Apply that frozen probability to the held-out identities.

Neither face network is trained or fine-tuned. Results are written to a
separate directory and use a separate run cache, so Experiments 6 to 9 are
untouched.""",

    "comparison-diagnostics": """Selected: Additional comparison diagnostics

Purpose:
Compare separately calibrated one- and three-image enrolment, gallery sizes
25/50/100/200 with three fixed sampling seeds, classifier feature sets, and
error rates by image quality. Both available pipelines are evaluated.

Development identities determine thresholds and quality bins. Test identities
remain separate. These extensions are exploratory because the earlier test
results were already inspected. Existing policies are preserved.

First runs extract real BFW images and may take several minutes. Later runs
reuse verified private score caches. Outputs: comparison_diagnostics/.
No face-recognition model is trained or fine-tuned.""",
    "comparative-statistics": """Selected: Paired pipeline comparisons

Purpose:
Compare all four detector/recogniser combinations on the same identities,
including every intended probe and a common successfully processed subset.
Report paired confidence intervals and the detector-by-recogniser interaction.

Requires saved canonical runs and policies from Experiments 6, 8, 11 and 12.
No threshold is selected from test outcomes. No new face extraction runs.""",
    "review": """HUMAN REVIEW MODERATION

Purpose:
This interface displays cases created by Experiment 5 using the LFW
gallery method.

These cases are not confirmed duplicate profiles.

The Experiment 5 threshold produced a high false-review rate and is
included as a research baseline. The interface demonstrates the review
workflow, not a production moderation decision.""",
}


def render_experiment_preview(key: str) -> str:
    """The preview for one option, or an empty string when none is defined."""
    return EXPERIMENT_PREVIEWS.get(key, "")


def announce_stage(step: int, total: int, title: str, detail: str = "") -> None:
    """Display the stage title and progress within the current sequence."""
    print(section_heading(f"{step}. {title}"))
    print(f"Step {step} of {total}")
    if detail:
        print(f"  {detail}")


def section_heading(title: str) -> str:
    """Use a consistent separator for menu entries, stages and result sections."""
    rule = "# " + "=" * 77
    return f"{rule}\n# {title}\n{rule}"


def render_plain_section(body: str) -> str:
    """Display the summary under its own heading."""
    return f"{section_heading('Summary')}\n\n{body}"


def render_technical_section(body: str) -> str:
    """The technical half: thresholds, FPIR, TPIR, intervals, digests and the
    pipeline identifier, presented in full after the summary."""
    return f"{section_heading('TECHNICAL DETAILS')}\n\n{body}"


def render_reference_section() -> str:
    """Model provenance, dataset roles and the glossary, offered at the end of
    every summary so a reader meeting a term for the first time need not look
    elsewhere."""
    return "\n\n".join(
        [section_heading("REFERENCE INFORMATION"), render_model_overview(),
         render_dataset_overview(), render_glossary()]
    )


def missing_artefact_message(what: str, option: str) -> str:
    """A clear instruction rather than a traceback when an optional experiment
    has not been run yet."""
    return (
        f"This result is not available yet.\n"
        f"Run {option} before showing the {what} summary."
    )


# --- Summaries read from the published artefacts -----------------------------


def _load_optional(aggregate_root: Path, name: str) -> Optional[Dict[str, Any]]:
    """Read an artefact if it exists. A missing optional experiment is a normal
    state, not an error."""
    path = aggregate_root / name
    return read_json_artifact(path) if path.is_file() else None


def render_baseline_plain_summary(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """Summary of Experiments 1-5, with every count read from file."""
    lfw = _load_optional(aggregate_root, "lfw_final_metrics.json")
    cplfw = _load_optional(aggregate_root, "cplfw_metrics.json")
    gallery = _load_optional(aggregate_root, "duplicate_gallery_metrics_v2.json")
    if not lfw:
        return missing_artefact_message("Experiments 1-5", "option 4")

    lines = [
        "WHAT THIS PART OF THE PROJECT TESTED",
        "",
        "The first experiments tested the YuNet + SFace pipeline.",
        "",
        "LFW was used to choose and test a one-to-one face-comparison threshold.",
        "CPLFW tested how that threshold performed when facial pose changed.",
        "The final LFW gallery experiment tested what happened when the same",
        "one-to-one threshold was used to search one photograph against many",
        "profiles.",
        "",
        wrap_plain(DENOMINATOR_NOTE),
        "",
        "FINAL LFW FACE COMPARISON",
        "",
    ]
    scored, total = lfw["scored_pairs"], lfw["total_pairs"]
    # Correct decisions are recovered from the stored confusion matrix rather
    # than recomputed, so the printed count cannot drift from the artefact.
    matrix = lfw["confusion_matrix"]
    correct = matrix["true_positive"] + matrix["true_negative"]
    lines += [
        "Successfully processed:",
        format_count_and_percentage(scored, total, noun="image pairs") + ".",
        "",
        "Correct decisions among processed pairs:",
        format_count_and_percentage(correct, scored, noun="scored pairs") + ".",
        "",
        "Processing failures:",
        format_count_and_percentage(lfw["failed_pairs"], total, noun="pairs") + ".",
        "",
        "Meaning:",
        "The model performed well on successfully processed LFW pairs, but",
        "approximately one in ten pairs did not reach comparison.",
        "",
    ]
    if cplfw:
        scored_c, total_c = cplfw["scored_pairs"], cplfw["total_pairs"]
        matrix_c = cplfw["confusion_matrix"]
        correct_c = matrix_c["true_positive"] + matrix_c["true_negative"]
        lines += [
            "CPLFW CROSS-POSE TEST",
            "",
            "Successfully processed:",
            format_count_and_percentage(scored_c, total_c, noun="pairs") + ".",
            "",
            "Correct decisions among processed pairs:",
            format_count_and_percentage(correct_c, scored_c, noun="scored pairs") + ".",
            "",
            "Processing failures:",
            format_count_and_percentage(cplfw["failed_pairs"], total_c, noun="pairs") + ".",
            "",
            "Meaning:",
            "Accuracy remained relatively high among processed pairs, but pose",
            "variation caused a large number of face-extraction failures.",
            "",
        ]
    if gallery:
        intended_dup = gallery["duplicate_probe_count"]
        detected = round(gallery["end_to_end_duplicate_detection_rate"] * intended_dup)
        intended_new = gallery["unknown_probe_count"]
        scored_new = intended_new - gallery["unknown_probe_failures"]
        referred = round(gallery["false_duplicate_review_rate"] * scored_new)
        lines += [
            "DUPLICATE-PROFILE GALLERY TEST",
            "",
            "Known duplicate-profile test cases correctly detected:",
            format_count_and_percentage(
                detected, intended_dup, noun="intended duplicate cases"
            ) + " end-to-end.",
            "",
            "New profiles incorrectly referred for review:",
            format_count_and_percentage(
                referred, scored_new, noun="scored new profiles"
            ) + " conditional.",
            "",
            "New-profile photographs successfully processed:",
            format_count_and_percentage(
                scored_new, intended_new, noun="intended new profiles"
            ) + ".",
            "",
            "Meaning:",
            "The transferred one-to-one threshold detected many known duplicate",
            "cases, but it also referred far too many genuinely new profiles.",
            "This experiment demonstrates why gallery search needs its own",
            "threshold.",
            "",
        ]
    lines.append(wrap_plain(REFERRAL_DISCLAIMER))
    return "\n".join(lines)


def render_open_set_plain_summary(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """Summary of Experiment 6, comparing the transferred threshold with
    the gallery-calibrated one side by side."""
    payload = _load_optional(aggregate_root, "bfw_open_set_test_metrics.json")
    if not payload:
        return missing_artefact_message("Experiment 6", "option 6")
    proposed = payload["methods"][METHOD_B]
    control = payload["methods"][METHOD_A]["rates"]
    primary = proposed["primary_operating_point"]
    coverage = proposed["coverage"]

    lines = [
        "EXPERIMENT 6 - BFW DUPLICATE-PROFILE EVALUATION",
        "",
        "Dataset:",
        "BFW, using held-out identities not used for threshold selection.",
        "",
        "Model:",
        "YuNet face detector + SFace face-recognition model.",
        "",
        render_plain_pipeline_table(
            ["", "Transferred 1:1 threshold", "Gallery-calibrated threshold"],
            [
                ["New profiles incorrectly referred for review (FPIR)",
                 _percentage_of(control.get("fpir")),
                 _percentage_of(primary.get("fpir"))],
                ["Known duplicate-profile test cases correctly detected (TPIR@1)",
                 _percentage_of(control.get("tpir_rank1")),
                 _percentage_of(primary.get("tpir_rank1"))],
                ["False reviews per 1,000 new profiles",
                 f"{control.get('false_reviews_per_1000_non_mated', float('nan')):.1f}",
                 f"{primary.get('false_reviews_per_1000_non_mated', float('nan')):.1f}"],
            ],
        ),
        "",
        "Result:",
        "",
        wrap_plain(
            f"The gallery-calibrated method reduced unnecessary human reviews from "
            f"approximately {control.get('false_reviews_per_1000_non_mated', 0):.0f} per "
            f"1,000 new profiles to approximately "
            f"{primary.get('false_reviews_per_1000_non_mated', 0):.0f} per 1,000."
        ),
        "",
        wrap_plain(
            f"It retained approximately {primary.get('tpir_rank1', 0) * 100:.0f} of every "
            f"100 successfully processed known duplicate cases."
        ),
        "",
        "This is a referral system only. It does not prove that a real profile",
        "is fake or duplicated.",
        "",
        "Photographs successfully processed:",
        "",
        "Known-duplicate photographs:",
        format_count_and_percentage(
            coverage.get("scored_mated_probes"), coverage.get("intended_mated_probes"),
            noun="intended photographs",
        ) + ".",
        "",
        "New-profile photographs:",
        format_count_and_percentage(
            coverage.get("scored_non_mated_probes"),
            coverage.get("intended_non_mated_probes"), noun="intended photographs",
        ) + ".",
        "",
        wrap_plain(DENOMINATOR_NOTE),
        "",
        wrap_plain(REFERRAL_DISCLAIMER),
    ]
    return "\n".join(lines)




# =============================================================================
# 30. Interactive VS Code launcher
# =============================================================================
#
# Running this file with no arguments prints a menu rather than starting a
# multi-minute benchmark, so the VS Code play button is safe to press.

# Consecutive choices follow the evaluation workflow. Each experiment has
# a run action and a saved-results action, with previews for longer tasks.
MENU_TEXT = f"""
{PROGRAMME_TITLE}

Select a number. Run generates results; Show displays saved results.

SETUP AND VALIDATION
  1. Check the software environment
  2. Verify models and datasets
  3. Run programme self-tests

EXPERIMENTS AND RESULTS
     Experiments 1-5 — models: YuNet + SFace
  4. Run Experiments 1-5 — LFW, CPLFW and duplicate-profile gallery
  5. Show the saved results from Experiments 1-5

     YuNet + SFace on BFW
  6. Run Experiment 6 — calibrated gallery search
  7. Show the saved Experiment 6 results

     YuNet + SFace + logistic regression on BFW; requires Experiment 6
  8. Run Experiment 7 — review classifier
  9. Show the saved Experiment 7 results

     Both pipelines on the same BFW identities: YuNet + SFace / SCRFD + ArcFace
 10. Run Experiment 8 — pipeline comparison
 11. Show the saved Experiment 8 results

     SCRFD + ArcFace through the same one-to-one chain: LFW and CPLFW
 12. Run Experiments 9 and 10 — one-to-one verification comparison
 13. Show the saved results from Experiments 9 and 10

 14. Run Experiment 11 — SCRFD + ArcFace review classifier
 15. Show the saved Experiment 11 results

 16. Run Experiment 12 — crossed detectors and recognisers
 17. Show the saved Experiment 12 results

CONTROLLED COMPARISONS
 18. Run enrolment, gallery-size, classifier-feature and image-quality analyses
 19. Show the saved comparison analyses
 20. Compute paired pipeline differences from saved runs
 21. Show paired differences and confidence intervals

OVERVIEW AND REPORTS
 22. Show every experiment at a glance
 23. Run Experiments 7 and 8 and generate figures (requires Experiment 6)
 24. Generate reports and figures from saved results
100. Run all experiments, comparisons and reports (includes setup checks)

MODERATION
 25. Open human review moderation

 26. Exit (or q)
"""

MODES = (
    "menu", "check", "verify", "full", "summary", "review", "self-test",
    # Supplementary Experiment 6. Deliberately separate from "full", which
    # continues to mean face verification and gallery evaluation.
    "open-set", "open-set-summary",
    # Experiments 7 and 8. Separate from "full", which remains the five
    # baseline experiments only.
    "ml-review", "ml-review-summary",
    "pipeline-compare", "pipeline-compare-summary",
    # Experiment 9: the same comparison pipeline on the 1:1 protocols, which
    # tests whether its open-set advantage also holds for pair verification.
    "verification-compare", "verification-compare-summary",
    # Experiment 11: the review classifier fitted on the comparison pipeline.
    "arcface-review", "arcface-review-summary",
    # Experiment 12: the detector and the embedder crossed, so each component's
    # contribution can be read separately.
    "mixed-pipelines", "mixed-pipelines-summary",
    # A single table covering every experiment, for orientation.
    "experiment-table", "comparison-diagnostics", "comparison-diagnostics-summary",
    "comparative-statistics", "comparative-statistics-summary", "refresh-reports",
    "extensions", "all",
)


# --- Stage 0: environment and input verification -----------------------------


def action_check_environment() -> int:
    """Report the interpreter, platform and pinned dependency versions, then
    which storage variables are configured — never their values."""
    report = software_environment_report()
    ok = True
    try:
        check_dependency_contract(strict=True)
    except DependencyContractError as exc:
        ok = False
        dependency_error = str(exc)
    else:
        dependency_error = ""

    print(f"Python: {report['python_version']}")
    print(f"Platform: {report['platform']}")
    print(f"Processor: {report['processor']}")
    dependencies = report["dependencies"]
    assert isinstance(dependencies, dict)
    for package, info in dependencies.items():
        status = "OK" if info["installed"] == info["expected"] else "MISMATCH"
        print(f"  {package}: expected={info['expected']} installed={info['installed']} [{status}]")

    config = EnvironmentConfig.load()
    missing = config.missing_variables()
    print("")
    print("Research storage configuration (values deliberately not printed):")
    for variable in REQUIRED_ENVIRONMENT_VARIABLES:
        print(f"  {variable}: {'set' if variable not in missing else 'NOT SET'}")
    optional_values = {
        "FACE_CACHE_ROOT": config.cache_root,
        "FACE_BFW_ROOT": config.bfw_root,
        "FACE_BFW_METADATA_ROOT": config.bfw_metadata_root,
        "FACE_ARCFACE_MODEL_ROOT": config.arcface_model_root,
    }
    for variable in OPTIONAL_ENVIRONMENT_VARIABLES:
        state = "set" if optional_values.get(variable) is not None else "not set (optional)"
        print(f"  {variable}: {state}")
    # Reported as configured/not configured only. The key's length, encoding
    # and any digest of it stay unprinted: confirming a guess must be impossible.
    print(
        f"  {ID_HMAC_KEY_VARIABLE}: "
        f"{'set' if id_hmac_key_is_configured() else 'NOT SET (required for identifiers)'}"
    )
    if not id_hmac_key_is_configured():
        ok = False
        print("")
        print(
            f"FAILED: {ID_HMAC_KEY_VARIABLE} is not set. Generate one with: "
            'python -c "import secrets;print(secrets.token_urlsafe(32))"',
            file=sys.stderr,
        )
    if missing:
        ok = False
        print("")
        print(
            "FAILED: copy .env.example to .env and fill in the missing variable(s) above, "
            "or export them directly.",
            file=sys.stderr,
        )
    if dependency_error:
        print("")
        print(f"FAILED: {dependency_error}", file=sys.stderr)
    return 0 if ok else 1


def action_verify_inputs() -> int:
    """Hash-verify the two pinned models, then structurally verify the LFW and
    raw CPLFW protocols against the images they reference."""
    config = EnvironmentConfig.load()
    ok = True

    model_root = config.require_model_root()
    print("Models")
    for filename, expected in ((YUNET_FILENAME, YUNET_SHA256), (SFACE_FILENAME, SFACE_SHA256)):
        try:
            actual = verify_model_file(model_root / filename, expected)
            print(f"  OK   {filename}  sha256={actual}")
        except ModelUnavailableError as exc:
            ok = False
            print(f"  FAIL {filename}: {redact_private_paths(str(exc))}", file=sys.stderr)

    lfw_root = config.require_lfw_root()
    protocol_root = config.require_protocol_root()
    print("")
    print("LFW dataset and protocols")
    if not lfw_root.is_dir():
        ok = False
        print("  FAIL LFW dataset root does not exist (check FACE_DATA_ROOT)", file=sys.stderr)
    else:
        for filename in REQUIRED_LFW_PROTOCOLS:
            protocol_path = protocol_root / filename
            if not protocol_path.is_file():
                ok = False
                print(f"  FAIL missing protocol file: {filename}", file=sys.stderr)
                continue
            try:
                pairs = parse_lfw_pairs(protocol_path, lfw_root)
                same = sum(1 for pair in pairs if pair.same_identity)
                print(
                    f"  OK   {filename}: {len(pairs)} pairs "
                    f"({same} matched, {len(pairs) - same} mismatched)"
                )
            except ProtocolError as exc:
                ok = False
                print(f"  FAIL {filename}: {redact_private_paths(str(exc))}", file=sys.stderr)

    cplfw_root = config.require_cplfw_raw_root()
    print("")
    print("Raw CPLFW dataset and protocol")
    print(f"  dataset_image_variant: raw ({CPLFW_RAW_ARCHIVE_FILENAME})")
    if not cplfw_root.is_dir():
        ok = False
        print("  FAIL CPLFW dataset root does not exist (check FACE_CPLFW_RAW_ROOT)", file=sys.stderr)
    else:
        protocol_path = protocol_root / CPLFW_PROTOCOL
        if not protocol_path.is_file():
            ok = False
            print(f"  FAIL missing protocol file: {CPLFW_PROTOCOL}", file=sys.stderr)
        else:
            try:
                pairs = parse_cplfw_pairs(protocol_path, cplfw_root)
                same = sum(1 for pair in pairs if pair.same_identity)
                different = len(pairs) - same
                referenced = {p.left_path for p in pairs} | {p.right_path for p in pairs}
                print(f"  OK   total_pairs             : {len(pairs)}")
                print(f"  OK   same_identity_pairs     : {same}")
                print(f"  OK   different_identity_pairs: {different}")
                print(f"  OK   unique_images_referenced: {len(referenced)}")
                print("  OK   every protocol-referenced image resolved")
                print("  OK   no malformed rows, mismatched labels or duplicate pairs")
                # Parsing aborts on the first unresolved reference, so there is
                # no partial-success path and nothing is silently excluded.
                problems = []
                if len(pairs) != CPLFW_EXPECTED_PAIRS:
                    problems.append(f"expected {CPLFW_EXPECTED_PAIRS} pairs, parsed {len(pairs)}")
                if same != CPLFW_EXPECTED_PER_CLASS:
                    problems.append(
                        f"expected {CPLFW_EXPECTED_PER_CLASS} same-identity pairs, parsed {same}"
                    )
                if different != CPLFW_EXPECTED_PER_CLASS:
                    problems.append(
                        f"expected {CPLFW_EXPECTED_PER_CLASS} different-identity pairs, "
                        f"parsed {different}"
                    )
                for problem in problems:
                    ok = False
                    print(f"  FAIL {problem}", file=sys.stderr)
            except ProtocolError as exc:
                ok = False
                print(f"  FAIL {CPLFW_PROTOCOL}: {redact_private_paths(str(exc))}", file=sys.stderr)

    print("")
    print("Verification complete." if ok else "Verification FAILED — see the messages above.")
    return 0 if ok else 1


# --- Stages 1 to 5: the five experiments -------------------------------------


def experiment_calibrate(
    config: EnvironmentConfig,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    output_root: Path,
    *,
    description: Optional[PipelineDescription] = None,
) -> Path:
    """Experiment 1, stage 1: candidate thresholds from pairsDevTrain.txt only.
    Never reads the development or final protocol, and never selects a winner."""
    lfw_root = config.require_lfw_root()
    protocol_path = config.require_protocol_root() / LFW_CALIBRATION_PROTOCOL
    pairs = parse_lfw_pairs(protocol_path, lfw_root)

    result = evaluate_pairs(pairs, detector=detector, embedder=embedder)
    if not result.valid_scores:
        raise SystemExit(
            "No pairs were successfully scored during calibration; stopping rather than "
            "fabricating a threshold."
        )

    calibration = calibrate(result.valid_scores, result.valid_labels, split=VALIDATION_SPLIT)
    evaluated_images = {p.left_path for p in pairs} | {p.right_path for p in pairs}
    output_path = output_root / "calibrated_threshold.json"

    write_json_artifact(
        output_path,
        {
            "artifact_type": "calibrated_threshold",
            "dataset": "LFW",
            "protocol_file": LFW_CALIBRATION_PROTOCOL,
            "protocol_sha256": sha256_of_file(protocol_path),
            "evaluated_image_set_sha256": sha256_of_evaluated_image_set(
                evaluated_images, lfw_root
            ),
            "dataset_archive_md5": LFW_ARCHIVE_MD5,
            "split": calibration.split,
            "status": calibration.status,
            "candidates": {
                name: {"threshold": candidate.threshold, "metrics": candidate.metrics}
                for name, candidate in calibration.candidates.items()
            },
            "total_pairs": result.total_pairs,
            "scored_pairs": len(result.valid_scores),
            "failure_breakdown": dict(result.failures),
            **pipeline_identity_fields(detector, embedder, description),
            "software_environment": software_environment_report(),
        },
    )
    announce(
        f"Wrote {len(calibration.candidates)} candidate threshold(s) to "
        f"{project_relative(output_path)} (status=candidates; nothing is frozen yet)"
    )
    return output_path


def experiment_evaluate_lfw(
    config: EnvironmentConfig,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    *,
    split: str,
    threshold_artifact: Path,
    output_path: Path,
    description: Optional[PipelineDescription] = None,
) -> Dict[str, Any]:
    """split='dev' is selection stage 2: it scores every candidate on
    pairsDevTest.txt, selects one by SELECTION_RULE and rewrites the threshold
    artifact as frozen for CPLFW/gallery transfer. Final LFW uses the official
    ten folds, fitting each threshold on the other nine folds only."""
    if split == "final":
        return experiment_lfw_cross_validation(
            config, detector, embedder, output_path=output_path, description=description
        )
    if split != "dev":
        raise ProtocolError(f"Unknown LFW split: {split!r}")
    protocol_filename = (
        LFW_DEVELOPMENT_PROTOCOL if split == "dev" else LFW_FINAL_PROTOCOL
    )
    lfw_root = config.require_lfw_root()
    protocol_path = config.require_protocol_root() / protocol_filename
    pairs = parse_lfw_pairs(protocol_path, lfw_root)

    result = evaluate_pairs(pairs, detector=detector, embedder=embedder)
    if not result.valid_scores:
        raise SystemExit(f"No pairs were successfully scored on {protocol_filename}; stopping.")

    protocol_sha256 = sha256_of_file(protocol_path)
    evaluated_images = {p.left_path for p in pairs} | {p.right_path for p in pairs}
    # Recorded before any freeze rewrite, so a development artefact references
    # the candidates file it actually selected from.
    threshold_artifact_sha256 = sha256_of_file(threshold_artifact)
    threshold_payload = read_json_artifact(threshold_artifact)

    extra_fields: Dict[str, Any] = {}

    if split == "dev":
        candidates = require_candidates(
            threshold_payload, context=project_relative(threshold_artifact)
        )
        selection = select_final_threshold(candidates, result.valid_scores, result.valid_labels)
        threshold = selection["selected_threshold"]

        frozen_payload = dict(threshold_payload)
        frozen_payload["status"] = FROZEN_STATUS
        frozen_payload["threshold"] = threshold
        frozen_payload["operating_strategy"] = selection["selected_candidate"]
        frozen_payload["selection_rule"] = selection["selection_rule"]
        frozen_payload["selection_evidence"] = selection["all_candidates_dev_metrics"]
        frozen_payload["frozen_from_protocol"] = protocol_filename
        frozen_payload["frozen_from_protocol_sha256"] = protocol_sha256
        write_json_artifact(threshold_artifact, frozen_payload)
        announce(
            f"Selected and froze threshold={threshold:.6f} "
            f"(candidate={selection['selected_candidate']}) in "
            f"{project_relative(threshold_artifact)}, based on {protocol_filename}"
        )

        extra_fields = {
            "selected_candidate": selection["selected_candidate"],
            "selection_rule": selection["selection_rule"],
            "all_candidates_dev_metrics": selection["all_candidates_dev_metrics"],
        }
    else:
        threshold = require_frozen_threshold(
            threshold_payload, context=project_relative(threshold_artifact)
        )

    summary = summarize_metrics(result, threshold)

    write_json_artifact(
        output_path,
        {
            "artifact_type": "lfw_verification_metrics",
            "split": split,
            "protocol_file": protocol_filename,
            "protocol_sha256": protocol_sha256,
            "evaluated_image_set_sha256": sha256_of_evaluated_image_set(
                evaluated_images, lfw_root
            ),
            "dataset_archive_md5": LFW_ARCHIVE_MD5,
            "threshold_source": project_relative(threshold_artifact),
            "threshold_artifact_sha256": threshold_artifact_sha256,
            "threshold_status": FROZEN_STATUS,
            # Add the stage-specific evidence to the shared result metadata.
            # Only the selection stage contributes extra fields.
            **extra_fields,
            **summary,
            **pipeline_identity_fields(detector, embedder, description),
            "software_environment": software_environment_report(),
        },
    )
    announce(
        f"Wrote {split} LFW metrics to {project_relative(output_path)} "
        f"(accuracy={summary['accuracy']:.4f})"
    )
    return summary


def experiment_evaluate_cplfw(
    config: EnvironmentConfig,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    *,
    image_variant: str,
    threshold_artifact: Path,
    output_path: Path,
    description: Optional[PipelineDescription] = None,
) -> Dict[str, Any]:
    """Experiment 4: cross-pose generalisation using the exact LFW-frozen
    threshold. There is deliberately no CPLFW-specific calibration step."""
    provenance_fields = cplfw_provenance_fields(image_variant)

    cplfw_root = config.require_cplfw_raw_root()
    protocol_path = config.require_protocol_root() / CPLFW_PROTOCOL
    # The threshold is read back from the artefact LFW froze, and its digest is
    # recorded. Reusing that exact value unchanged is what makes this a test of
    # generalisation to a harder pose rather than a second calibration.
    threshold_artifact_sha256 = sha256_of_file(threshold_artifact)
    threshold_payload = read_json_artifact(threshold_artifact)
    threshold = require_frozen_threshold(
        threshold_payload, context=project_relative(threshold_artifact)
    )

    pairs = parse_cplfw_pairs(protocol_path, cplfw_root)
    # The set of distinct photographs actually compared. Its digest is
    # published, so a later reader can confirm which images produced the result.
    evaluated_images = {p.left_path for p in pairs} | {p.right_path for p in pairs}

    result = evaluate_pairs(pairs, detector=detector, embedder=embedder)
    summary = summarize_metrics(result, threshold)

    write_json_artifact(
        output_path,
        {
            "artifact_type": "cplfw_verification_metrics",
            **provenance_fields,
            "protocol_file": CPLFW_PROTOCOL,
            "protocol_sha256": sha256_of_file(protocol_path),
            "evaluated_image_set_sha256": sha256_of_evaluated_image_set(
                evaluated_images, cplfw_root
            ),
            "threshold_source": project_relative(threshold_artifact),
            "threshold_artifact_sha256": threshold_artifact_sha256,
            "threshold_status": threshold_payload.get("status"),
            "note": (
                "Frozen threshold calibrated on LFW pairsDevTrain.txt and selected on "
                "pairsDevTest.txt; not recalibrated for CPLFW. This measures cross-pose "
                "generalisation, not a separately tuned CPLFW-specific result."
            ),
            **summary,
            **pipeline_identity_fields(detector, embedder, description),
            "software_environment": software_environment_report(),
        },
    )
    announce(
        f"Wrote CPLFW metrics to {project_relative(output_path)} "
        f"(accuracy={summary['accuracy']:.4f})"
    )
    return summary


def experiment_duplicate_gallery(
    config: EnvironmentConfig,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    *,
    threshold_artifact: Path,
    manifest_path: Path,
    output_path: Path,
    review_db: Optional[Path] = None,
    seed: int = DEFAULT_RANDOM_SEED,
    description: Optional[PipelineDescription] = None,
) -> Dict[str, Any]:
    """Experiment 5: build a deterministic 1:N gallery from real LFW images and
    measure duplicate detection against the same frozen threshold. Identities
    used for calibration are excluded, so calibration data never leaks in."""
    lfw_root = config.require_lfw_root()
    identity_to_images = discover_identity_images(lfw_root)

    calibration_pairs = parse_lfw_pairs(
        config.require_protocol_root() / LFW_CALIBRATION_PROTOCOL, lfw_root
    )
    excluded_images: Set[Path] = set()
    for pair in calibration_pairs:
        excluded_images.add(pair.left_path)
        excluded_images.add(pair.right_path)

    manifest = build_manifest(identity_to_images, excluded_images=excluded_images, seed=seed)
    write_gallery_manifest(manifest, manifest_path)

    gallery_count = sum(1 for e in manifest.entries if e.role == "gallery")
    duplicate_count = sum(1 for e in manifest.entries if e.role == "duplicate_probe")
    unknown_count = sum(1 for e in manifest.entries if e.role == "unknown_probe")
    announce(
        f"Wrote the private gallery manifest to {project_relative(manifest_path)}: "
        f"{gallery_count} gallery, {duplicate_count} duplicate probes, "
        f"{unknown_count} unknown probes (contains real image paths — kept out of Git)"
    )

    manifest_sha256 = sha256_of_file(manifest_path)
    threshold_artifact_sha256 = sha256_of_file(threshold_artifact)
    threshold_payload = read_json_artifact(threshold_artifact)
    threshold = require_frozen_threshold(
        threshold_payload, context=project_relative(threshold_artifact)
    )

    result = evaluate_gallery(
        manifest, detector=detector, embedder=embedder, duplicate_review_threshold=threshold
    )
    summary = summarize_gallery_metrics(result)

    write_json_artifact(
        output_path,
        {
            "artifact_type": "duplicate_gallery_metrics_v2",
            # The scheme, never the key: this records which identifier family
            # the opaque values below belong to, so two runs under different
            # keys are not mistaken for comparable ones.
            "opaque_id_version": OPAQUE_ID_VERSION,
            "duplicate_review_threshold": threshold,
            "threshold_source": project_relative(threshold_artifact),
            "threshold_artifact_sha256": threshold_artifact_sha256,
            "threshold_strategy": threshold_payload.get("operating_strategy"),
            "manifest_sha256": manifest_sha256,
            "dataset_archive_md5": LFW_ARCHIVE_MD5,
            "seed": manifest.seed,
            "policy_note": POLICY_NOTE,
            **summary,
            **pipeline_identity_fields(detector, embedder, description),
            "software_environment": software_environment_report(),
        },
    )
    announce(
        f"Wrote duplicate gallery metrics to {project_relative(output_path)} "
        f"(duplicate_detection_rate={summary['duplicate_detection_rate']:.4f})"
    )

    if review_db is not None:
        flagged = populate_review_database(review_db, result, threshold)
        announce(f"Wrote {flagged} review case(s) to {project_relative(review_db)}")

    return summary


###############################################################################
# Experiment 9: 1:1 verification with the higher-capacity pipeline
###############################################################################
#
# Experiment 8 compared the two pipelines on open-set gallery search only, so
# its conclusion rested on a single task. This runs the same SCRFD + ArcFace
# pipeline through the 1:1 chain of Experiments 1-4, which tests whether the
# advantage it showed there also holds for pair verification and, in
# particular, under the extreme pose variation of CPLFW.
#
# The existing experiment functions are reused unchanged. They already take a
# detector, an embedder and an output location, so the only difference is which
# models are supplied and where the artefacts are written. Writing into a
# sub-directory keeps Experiments 1-5 untouched: the two chains
# cannot overwrite one another's threshold.

VERIFICATION_COMPARISON_DIRNAME = "verification_comparison"


def run_verification_comparison(
    output_root: Path = AGGREGATE_ROOT, *, cplfw_image_variant: str = "raw"
) -> Dict[str, Any]:
    """Calibrate, freeze and evaluate the comparison pipeline on LFW and CPLFW.

    The same three-stage separation the baseline obeys is enforced here too:
    candidates from pairsDevTrain only, selection and freezing on pairsDevTest,
    then ten-fold LFW evaluation and development-threshold transfer to CPLFW. A
    threshold calibrated for SFace is never reused, because a cosine similarity
    from a 512-dimensional ArcFace vector does not mean what the same number
    means in SFace's 128-dimensional space."""
    config = EnvironmentConfig.load()
    (detector, embedder), description = load_arcface_pipeline(config)

    # A separate directory, so the baseline threshold artefact and metrics are
    # never overwritten by this chain.
    sub_root = Path(output_root) / VERIFICATION_COMPARISON_DIRNAME
    sub_root.mkdir(parents=True, exist_ok=True)

    announce_stage(1, 4, "Producing candidate thresholds for SCRFD + ArcFace",
                   "LFW pairsDevTrain only; no threshold is selected yet.")
    threshold_artifact = experiment_calibrate(
        config, detector, embedder, sub_root, description=description
    )

    announce_stage(2, 4, "Selecting and freezing the threshold",
                   "LFW pairsDevTest, using the same deterministic rule as the "
                   "baseline pipeline.")
    experiment_evaluate_lfw(
        config, detector, embedder, split="dev",
        threshold_artifact=threshold_artifact,
        output_path=sub_root / "lfw_development_metrics.json",
        description=description,
    )

    announce_stage(3, 4, "Evaluating LFW with ten-fold cross-validation",
                   "Each threshold is fitted on the other nine folds only.")
    experiment_evaluate_lfw(
        config, detector, embedder, split="final",
        threshold_artifact=threshold_artifact,
        output_path=sub_root / "lfw_final_metrics.json",
        description=description,
    )

    announce_stage(4, 4, "Evaluating CPLFW under the same frozen threshold",
                   "Cross-pose generalisation; no CPLFW-specific calibration.")
    experiment_evaluate_cplfw(
        config, detector, embedder, image_variant=cplfw_image_variant,
        threshold_artifact=threshold_artifact,
        output_path=sub_root / "cplfw_metrics.json",
        description=description,
    )

    # Published artefacts are scanned like every other, so a private storage
    # path cannot reach a result through this chain either.
    leaks = find_path_leaks(sub_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise PrivacyLeakError(
            f"{len(leaks)} personal or absolute path(s) found in the verification "
            f"comparison artefacts."
        )
    announce("Privacy validation passed for every verification-comparison artefact")
    return {"output_root": str(project_relative(sub_root)),
            "pipeline": description.as_dict()}


def render_verification_comparison_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    """Both pipelines on the same 1:1 protocols, side by side.

    Extraction coverage is shown beside accuracy because the two answer
    different questions: whether a face was found at all, and whether the
    comparison was correct once it had been."""
    sub_root = Path(output_root) / VERIFICATION_COMPARISON_DIRNAME
    baseline_final = _load_optional(Path(output_root), "lfw_final_metrics.json")
    baseline_cplfw = _load_optional(Path(output_root), "cplfw_metrics.json")
    compare_final = _load_optional(sub_root, "lfw_final_metrics.json")
    compare_cplfw = _load_optional(sub_root, "cplfw_metrics.json")
    if not compare_final or not compare_cplfw:
        return missing_artefact_message(
            "1:1 pipeline comparison", "option 12 (--mode verification-compare)"
        )

    def row(label: str, getter) -> List[str]:
        cells = []
        for payload in (baseline_final, compare_final, baseline_cplfw, compare_cplfw):
            cells.append(getter(payload) if payload else "not available")
        return [label, *cells]

    lines = [
        "EXPERIMENT 9 - 1:1 VERIFICATION, BOTH PIPELINES",
        "",
        "Datasets:",
        "LFW pairs.txt and CPLFW pairs_CPLFW.txt, raw images, 6,000 pairs each.",
        "",
        "LFW uses official ten-fold cross-validation with training-only thresholds.",
        "CPLFW uses each pipeline's separate frozen LFW development threshold.",
        "No threshold is shared between pipelines.",
        "",
        render_plain_pipeline_table(
            ["Metric", "LFW: YuNet+SFace", "LFW: SCRFD+ArcFace",
             "CPLFW: YuNet+SFace", "CPLFW: SCRFD+ArcFace"],
            [
                row("Correct decisions among scored pairs",
                    lambda d: _percentage_of(d.get("accuracy"))),
                row("Pairs reaching comparison",
                    lambda d: _percentage_of(1.0 - d.get("failure_rate", float("nan")))),
                row("Pairs scored",
                    lambda d: f"{d.get('scored_pairs', 0):,} of {d.get('total_pairs', 0):,}"),
                row("Wrongly accepted (FMR)",
                    lambda d: _percentage_of(d.get("false_match_rate"))),
                row("Wrongly rejected (FNMR)",
                    lambda d: _percentage_of(d.get("false_non_match_rate"))),
                row("Threshold protocol",
                    lambda d: "10 training-only folds" if d.get("fold_results")
                    else "Frozen development threshold"),
                row("Frozen threshold (CPLFW)",
                    lambda d: "Per-fold thresholds" if d.get("fold_results")
                    else format_number(d.get("threshold"), 6)),
            ],
        ),
        "",
        wrap_plain(DENOMINATOR_NOTE),
        "",
        wrap_plain(
            "Coverage includes detection, alignment and embedding failures. Conditional "
            "accuracy describes the scored subset, which differs between pipelines; "
            "it cannot alone establish which complete pipeline works better."
        ),
    ]
    return "\n".join(lines)


def action_run_complete_evaluation(
    output_root: Path = AGGREGATE_ROOT,
    *,
    manifest_path: Path = DEFAULT_GALLERY_MANIFEST,
    review_db: Path = DEFAULT_REVIEW_DB,
    cplfw_image_variant: str = "raw",
) -> int:
    """The five experiments in their required order, stopping with the
    underlying error rather than fabricating a result if any input is missing."""
    config = EnvironmentConfig.load()
    missing = config.missing_variables()
    if missing:
        raise SystemExit(
            "Cannot run the evaluation: "
            + ", ".join(missing)
            + " not configured. Copy .env.example to .env and fill it in."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    announce_stage(1, 7, "Verifying the pinned models")
    detector, embedder = load_models(config.require_model_root())
    print(f"  OK   {YUNET_FILENAME} and {SFACE_FILENAME} match their pinned SHA-256 digests")

    print("")
    announce_stage(2, 7, "Experiment 1 — threshold candidates (pairsDevTrain.txt, validation only)")
    threshold_artifact = experiment_calibrate(config, detector, embedder, output_root)

    print("")
    announce_stage(3, 7, "Experiment 2 — development selection and freezing (pairsDevTest.txt)")
    experiment_evaluate_lfw(
        config,
        detector,
        embedder,
        split="dev",
        threshold_artifact=threshold_artifact,
        output_path=output_root / "lfw_development_metrics.json",
    )

    print("")
    announce_stage(4, 7, "Experiment 3 — official LFW ten-fold evaluation")
    experiment_evaluate_lfw(
        config,
        detector,
        embedder,
        split="final",
        threshold_artifact=threshold_artifact,
        output_path=output_root / "lfw_final_metrics.json",
    )

    print("")
    announce_stage(5, 7, "Experiment 4 — raw CPLFW cross-pose generalisation (same frozen threshold)")
    experiment_evaluate_cplfw(
        config,
        detector,
        embedder,
        image_variant=cplfw_image_variant,
        threshold_artifact=threshold_artifact,
        output_path=output_root / "cplfw_metrics.json",
    )

    print("")
    announce_stage(6, 7, "Experiment 5 — 1:N duplicate-profile gallery (real LFW images)")
    experiment_duplicate_gallery(
        config,
        detector,
        embedder,
        threshold_artifact=threshold_artifact,
        manifest_path=manifest_path,
        # Versioned filename: the corrected accounting changes what the
        # denominators mean, so it is written alongside the historical artefact
        # rather than over it.
        output_path=output_root / "duplicate_gallery_metrics_v2.json",
        review_db=review_db,
    )

    print("")
    announce_stage(7, 7, "Aggregate outputs and privacy validation")
    write_aggregate_reports(output_root, manifest_path, cplfw_image_variant)

    elapsed = time.perf_counter() - started
    print("")
    announce(
        f"Complete in {elapsed / 60:.1f} minutes. Aggregate results are in "
        f"{project_relative(output_root)}"
    )
    print("")
    print(render_results_summary(output_root))
    return 0


def action_show_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    try:
        # Summary first, followed by technical details and reference
        # material. The formal reports keep their own unchanged wording.
        print(render_plain_section(render_baseline_plain_summary(output_root)))
        print("")
        print(render_technical_section(render_results_summary(output_root)))
        print("")
        print(render_reference_section())
    except ArtifactError:
        print(
            "No results are available yet. Run option 4 (the complete five-experiment "
            "evaluation) first, or `python ACP_arden.py --mode full`.",
            file=sys.stderr,
        )
        return 1
    return 0


def action_self_test() -> int:
    _passed, failed = run_self_tests()
    return 0 if failed == 0 else 1


# --- Menu and command-line entry point ---------------------------------------


def _run_action(action: Callable[[], int]) -> int:
    """Run one menu action, reporting an expected failure as a redacted message
    rather than a traceback that could disclose a storage location."""
    try:
        return action()
    # Every failure listed here is one the programme anticipates: a missing
    # dataset, an unverified model, an unfrozen threshold. Each is reported as
    # a short redacted sentence, because a traceback would print the local
    # research storage path it failed on.
    except (
        ArtifactError,
        BfwDatasetError,
        CalibrationError,
        ConfigurationError,
        GalleryError,
        ImageLoadError,
        ModelUnavailableError,
        OpaqueIdentifierKeyError,
        OpenSetPolicyError,
        MlReviewError,
        OpenSetProtocolError,
        PipelineUnavailableError,
        PipelineComparisonError,
        MetricsError,
        PrivacyLeakError,
        ProtocolError,
        ReviewDatabaseVersionError,
        SystemExit,
    ) as exc:
        message = str(exc)
        if message:
            print(f"\nStopped: {redact_private_paths(message)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def action_run_open_set_evaluation(output_root: Path = AGGREGATE_ROOT) -> int:
    """Experiment 6. Requires the official BFW dataset; stops on the exact
    blocker rather than degrading to a partial or invented result."""
    run_open_set_experiment(output_root=output_root)
    print("")
    for line in report_optional_dataset_status():
        print(line)
    print("")
    print(render_plain_section(render_open_set_plain_summary(output_root)))
    print("")
    print(render_technical_section(render_open_set_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_show_open_set_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_plain_section(render_open_set_plain_summary(output_root)))
    print("")
    print(render_technical_section(render_open_set_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_run_ml_review(output_root: Path = AGGREGATE_ROOT) -> int:
    """Experiment 7. Requires BFW and a completed open-set run for the
    comparator threshold."""
    run_ml_review_experiment(output_root=output_root)
    written = generate_figures(aggregate_root=output_root)
    announce(f"Wrote {len(written)} figure(s) to {project_relative(FIGURES_ROOT)}")
    print("")
    print(render_plain_section(render_ml_review_plain_summary(output_root)))
    print("")
    print(render_technical_section(render_ml_review_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_show_ml_review_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_plain_section(render_ml_review_plain_summary(output_root)))
    print("")
    print(render_technical_section(render_ml_review_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_run_pipeline_comparison(output_root: Path = AGGREGATE_ROOT) -> int:
    result = run_pipeline_comparison(output_root=output_root)
    print("")
    print(render_plain_section(render_pipeline_plain_summary(output_root)))
    print("")
    print(render_technical_section(render_pipeline_comparison_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0 if result["evaluated"] == "yes" else 1


def action_show_pipeline_comparison_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_plain_section(render_pipeline_plain_summary(output_root)))
    print("")
    print(render_technical_section(render_pipeline_comparison_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_run_verification_comparison(output_root: Path = AGGREGATE_ROOT) -> int:
    """Experiment 9. Requires the optional comparison models, exactly as
    Experiment 8 does."""
    run_verification_comparison(output_root=output_root)
    print("")
    print(render_plain_section(render_verification_comparison_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_show_verification_comparison_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_plain_section(render_verification_comparison_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_run_arcface_review(output_root: Path = AGGREGATE_ROOT) -> int:
    """Experiment 11. Requires the optional comparison models."""
    run_arcface_review_experiment(output_root)
    print("")
    print(render_plain_section(render_arcface_review_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_show_arcface_review_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_plain_section(render_arcface_review_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_run_mixed_pipelines(output_root: Path = AGGREGATE_ROOT) -> int:
    """Experiment 12. Requires the optional comparison models."""
    run_mixed_pipeline_matrix(output_root)
    print("")
    print(render_plain_section(render_mixed_pipeline_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_show_mixed_pipeline_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_plain_section(render_mixed_pipeline_summary(output_root)))
    print("")
    print(render_reference_section())
    return 0


def action_show_experiment_table(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_experiment_comparison_table(output_root))
    print("")
    print(render_processing_coverage_explanation(output_root))
    print("")
    print(render_model_comparison_table(output_root))
    return 0


def action_run_extensions(output_root: Path = AGGREGATE_ROOT) -> int:
    """Both extension experiments. An unavailable optional pipeline must not
    prevent the classifier experiment from being reported."""
    # Experiment 7 first: it reuses the frozen threshold of Experiment 6 and
    # does not depend on the optional comparison models.
    status = run_ml_review_experiment(output_root=output_root)
    # Experiment 8 is optional. Its models are not redistributed, so an absent
    # weight file must leave Experiment 7 reported rather than abandoning the
    # whole run. The blocker is announced, not swallowed.
    try:
        run_pipeline_comparison(output_root=output_root)
    except (PipelineUnavailableError, ModelUnavailableError) as exc:
        announce(f"Pipeline comparison: NOT RUN — {redact_private_paths(str(exc))}")
    # Figures are rebuilt from whichever artefacts now exist, so a chart never
    # illustrates a result that was not produced in this run.
    written = generate_figures(aggregate_root=output_root)
    announce(f"Wrote {len(written)} figure(s) to {project_relative(FIGURES_ROOT)}")
    print("")
    print(render_plain_section(
        render_ml_review_plain_summary(output_root) + "\n\n"
        + render_pipeline_plain_summary(output_root)
    ))
    print("")
    print(render_technical_section(
        render_ml_review_summary(output_root) + "\n\n"
        + render_pipeline_comparison_summary(output_root)
    ))
    print("")
    print(render_reference_section())
    print("")
    print(render_experiment_comparison_table(output_root))
    print("")
    print(section_heading("OVERALL PROJECT CONCLUSION"))
    print("")
    print(render_overall_conclusion(output_root))
    return 0 if status else 1


def action_run_comparison_diagnostics(output_root: Path = AGGREGATE_ROOT) -> int:
    run_comparison_diagnostics(output_root)
    print(render_comparison_diagnostics_summary(output_root))
    return 0


def action_run_comparative_statistics(output_root: Path = AGGREGATE_ROOT) -> int:
    generate_comparative_statistics(output_root)
    text = render_paired_comparison_report(output_root)
    write_markdown_artifact(output_root / "COMPARATIVE_STATISTICS_REPORT.md", text)
    print(text)
    return 0


def action_refresh_reports(output_root: Path = AGGREGATE_ROOT) -> int:
    baseline = ("lfw_development_metrics.json", "lfw_final_metrics.json", "cplfw_metrics.json", "calibrated_threshold.json")
    if all((output_root / name).is_file() for name in baseline) and any(
        (output_root / name).is_file() for name in ("duplicate_gallery_metrics_v2.json", "duplicate_gallery_metrics.json")
    ):
        write_aggregate_reports(output_root, DEFAULT_GALLERY_MANIFEST, "raw")
    open_set = ("bfw_open_set_protocol_summary.json", "bfw_open_set_development_metrics.json", "bfw_open_set_test_metrics.json",
                "open_set_confidence_intervals.json", "bfw_subgroup_confidence_intervals.json")
    if all((output_root / name).is_file() for name in open_set):
        protocol, development, test, ci, subgroups = [read_json_artifact(output_root / name) for name in open_set]
        groups = subgroups["subgroups"]
        write_markdown_artifact(output_root / "OPEN_SET_EVALUATION_REPORT.md", render_open_set_report(
            protocol_summary=protocol, development=development, test=test, intervals=ci["intervals"],
            per_subgroup=groups, disparity=subgroup_disparity_summary(groups)))
    if (output_root / "comparative_statistics.json").is_file():
        write_markdown_artifact(output_root / "COMPARATIVE_STATISTICS_REPORT.md", render_paired_comparison_report(output_root))
    if (output_root / "comparison_diagnostics" / "comparison_diagnostics.json").is_file():
        write_markdown_artifact(output_root / "comparison_diagnostics" / "COMPARISON_DIAGNOSTICS_REPORT.md", render_comparison_diagnostics_summary(output_root))
    written = generate_figures(aggregate_root=output_root)
    announce(f"Refreshed reports and {len(written)} figures from saved results.")
    return 0


def action_run_all(output_root: Path = AGGREGATE_ROOT) -> int:
    """Run the complete workflow once, stopping if any prerequisite fails."""
    steps: List[Tuple[str, Callable[[], int]]] = [
        ("Check the software environment", action_check_environment),
        ("Verify models and datasets", action_verify_inputs),
        ("Run programme self-tests", action_self_test),
        ("Experiments 1-5 — LFW, CPLFW and duplicate-profile gallery",
         lambda: action_run_complete_evaluation(output_root)),
        ("Experiment 6 — calibrated gallery search",
         lambda: action_run_open_set_evaluation(output_root)),
        ("Experiment 7 — review classifier",
         lambda: action_run_ml_review(output_root)),
        ("Experiment 8 — pipeline comparison",
         lambda: action_run_pipeline_comparison(output_root)),
        ("Experiments 9 and 10 — one-to-one verification comparison",
         lambda: action_run_verification_comparison(output_root)),
        ("Experiment 11 — SCRFD + ArcFace review classifier",
         lambda: action_run_arcface_review(output_root)),
        ("Experiment 12 — crossed detectors and recognisers",
         lambda: action_run_mixed_pipelines(output_root)),
        ("Controlled comparison analyses",
         lambda: action_run_comparison_diagnostics(output_root)),
        ("Paired pipeline differences and confidence intervals",
         lambda: action_run_comparative_statistics(output_root)),
        ("Refresh reports and figures",
         lambda: action_refresh_reports(output_root)),
        ("Show every experiment at a glance",
         lambda: action_show_experiment_table(output_root)),
    ]
    for index, (title, action) in enumerate(steps, start=1):
        print("")
        announce_stage(index, len(steps), title)
        status = _run_action(action)
        if status != 0:
            announce(f"Run all stopped at step {index} of {len(steps)}: {title}.")
            return status
    announce("Run all complete: all experiments, comparisons and reports finished.")
    return 0


# Which preview belongs to which menu option. Options that only display saved
# results need no preview, because nothing long-running is about to start.
MENU_PREVIEW_KEYS = {
    "100": "all",
    "4": "full",
    "6": "open-set",
    "8": "ml-review",
    "10": "pipeline-compare",
    "12": "verification-compare",
    "14": "arcface-review",
    "16": "mixed-pipelines",
    "18": "comparison-diagnostics",
    "20": "comparative-statistics",
    "23": "extensions",
    "25": "review",
}


def _print_saved_summary(renderer: Callable[[Path], str], output_root: Path) -> int:
    print(renderer(output_root))
    return 0


def run_menu(output_root: Path = AGGREGATE_ROOT) -> int:
    """Interactive menu. Nothing long-running starts until an option is chosen."""
    actions: Dict[str, Callable[[], int]] = {
        "1": action_check_environment,
        "2": action_verify_inputs,
        "3": action_self_test,
        "4": lambda: action_run_complete_evaluation(output_root),
        "5": lambda: action_show_summary(output_root),
        "6": lambda: action_run_open_set_evaluation(output_root),
        "7": lambda: action_show_open_set_summary(output_root),
        "8": lambda: action_run_ml_review(output_root),
        "9": lambda: action_show_ml_review_summary(output_root),
        "10": lambda: action_run_pipeline_comparison(output_root),
        "11": lambda: action_show_pipeline_comparison_summary(output_root),
        "12": lambda: action_run_verification_comparison(output_root),
        "13": lambda: action_show_verification_comparison_summary(output_root),
        "14": lambda: action_run_arcface_review(output_root),
        "15": lambda: action_show_arcface_review_summary(output_root),
        "16": lambda: action_run_mixed_pipelines(output_root),
        "17": lambda: action_show_mixed_pipeline_summary(output_root),
        "18": lambda: action_run_comparison_diagnostics(output_root),
        "19": lambda: _print_saved_summary(render_comparison_diagnostics_summary, output_root),
        "20": lambda: action_run_comparative_statistics(output_root),
        "21": lambda: _print_saved_summary(render_paired_comparison_report, output_root),
        "22": lambda: action_show_experiment_table(output_root),
        "23": lambda: action_run_extensions(output_root),
        "24": lambda: action_refresh_reports(output_root),
        "25": lambda: launch_review_interface(DEFAULT_REVIEW_DB),
        "100": lambda: action_run_all(output_root),
    }
    # The scope of the artefact is stated before any option is offered.
    print("")
    print(PROGRAMME_INTRODUCTION)
    last_status = 0
    while True:
        print(re.sub(
            r"(?m)^ *(\d+\. .+)$",
            lambda match: "\n" + section_heading(match.group(1)),
            MENU_TEXT,
        ))
        try:
            choice = input("Select an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return last_status

        if choice in {"26", "q", "quit", "exit"}:
            return last_status
        action = actions.get(choice)
        if action is None:
            print(f"'{choice}' is not one of the options above.")
            continue
        preview = render_experiment_preview(MENU_PREVIEW_KEYS.get(choice, ""))
        if preview:
            print("")
            print(preview)
        print("")
        last_status = _run_action(action)
        print("")
        try:
            input("Press Enter to return to the menu...")
        except (EOFError, KeyboardInterrupt):
            print("")
            return last_status


# Every menu option also has a command-line mode, so the whole project can be
# run without the interactive launcher.
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ACP_arden.py",
        description=(
            f"{PROGRAMME_TITLE} (v{PROGRAMME_VERSION}). Run without arguments for an "
            f"interactive menu."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="menu",
        help=(
            "menu (default): interactive launcher. all: checks, Experiments 1-12, "
            "comparisons and reports. check: environment and dependencies. "
            "verify: models and benchmark datasets. full: the complete five-experiment "
            "evaluation. summary: the existing results. review: human review moderation "
            "interface. self-test: deterministic synthetic tests. open-set: the "
            "supplementary BFW open-set duplicate-profile experiment (Experiment 6). "
            "open-set-summary: the existing open-set results. ml-review: the "
            "machine-learning review classifier (Experiment 7). pipeline-compare: the "
            "pretrained pipeline comparison (Experiment 8). extensions: both, then figures."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=AGGREGATE_ROOT,
        help="Where aggregate results are written and read (default: results/aggregate).",
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=DEFAULT_REVIEW_DB,
        help="Moderation database (default: results/raw/review.sqlite; never committed).",
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAMME_NAME} {PROGRAMME_VERSION}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Streamlit re-runs this file on every interaction and appends its own
    # arguments, so unknown arguments are ignored rather than fatal.
    parser = build_argument_parser()
    args, _unknown = parser.parse_known_args(argv)

    if args.mode == "review" and running_under_streamlit():
        render_review_page(args.review_db)
        return 0

    if args.mode == "menu":
        return run_menu(args.results_root)
    if args.mode == "all":
        return _run_action(lambda: action_run_all(args.results_root))
    if args.mode == "check":
        return _run_action(action_check_environment)
    if args.mode == "verify":
        return _run_action(action_verify_inputs)
    if args.mode == "full":
        return _run_action(lambda: action_run_complete_evaluation(args.results_root))
    if args.mode == "summary":
        return _run_action(lambda: action_show_summary(args.results_root))
    if args.mode == "review":
        return launch_review_interface(args.review_db)
    if args.mode == "self-test":
        return action_self_test()
    if args.mode == "open-set":
        return _run_action(lambda: action_run_open_set_evaluation(args.results_root))
    if args.mode == "open-set-summary":
        return _run_action(lambda: action_show_open_set_summary(args.results_root))
    if args.mode == "ml-review":
        return _run_action(lambda: action_run_ml_review(args.results_root))
    if args.mode == "ml-review-summary":
        return _run_action(lambda: action_show_ml_review_summary(args.results_root))
    if args.mode == "pipeline-compare":
        return _run_action(lambda: action_run_pipeline_comparison(args.results_root))
    if args.mode == "pipeline-compare-summary":
        return _run_action(lambda: action_show_pipeline_comparison_summary(args.results_root))
    if args.mode == "verification-compare":
        return _run_action(lambda: action_run_verification_comparison(args.results_root))
    if args.mode == "verification-compare-summary":
        return _run_action(
            lambda: action_show_verification_comparison_summary(args.results_root)
        )
    if args.mode == "arcface-review":
        return _run_action(lambda: action_run_arcface_review(args.results_root))
    if args.mode == "arcface-review-summary":
        return _run_action(lambda: action_show_arcface_review_summary(args.results_root))
    if args.mode == "mixed-pipelines":
        return _run_action(lambda: action_run_mixed_pipelines(args.results_root))
    if args.mode == "mixed-pipelines-summary":
        return _run_action(lambda: action_show_mixed_pipeline_summary(args.results_root))
    if args.mode == "experiment-table":
        return _run_action(lambda: action_show_experiment_table(args.results_root))
    if args.mode == "comparison-diagnostics":
        return _run_action(lambda: action_run_comparison_diagnostics(args.results_root))
    if args.mode == "comparison-diagnostics-summary":
        return _run_action(lambda: _print_saved_summary(render_comparison_diagnostics_summary, args.results_root))
    if args.mode == "comparative-statistics":
        return _run_action(lambda: action_run_comparative_statistics(args.results_root))
    if args.mode == "comparative-statistics-summary":
        return _run_action(lambda: _print_saved_summary(render_paired_comparison_report, args.results_root))
    if args.mode == "refresh-reports":
        return _run_action(lambda: action_refresh_reports(args.results_root))
    if args.mode == "extensions":
        return _run_action(lambda: action_run_extensions(args.results_root))

    parser.error(f"Unhandled mode: {args.mode}")
    return 2


if __name__ == "__main__":
    # Streamlit re-executes this file for every interaction and owns the
    # process lifecycle itself. Raising SystemExit there aborts the script run
    # before Streamlit can mark it finished, which leaves the page hanging, so
    # the review path returns normally and only a plain interpreter run exits
    # with a status code.
    if running_under_streamlit():
        main()
    else:
        raise SystemExit(main())
