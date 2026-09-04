"""The backend registry: what engines exist, and how `--backend` resolves to one.

THE GOVERNING RULE. This skill exists because a competing tool
(`markitdown`'s audio route, `speech_recognition.recognize_google`) uploaded
a user's audio to a third party with nothing in the interface saying so. The
failure to avoid is not "uses an API" -- it is "uploaded the audio and the
user could not tell". So:

  - A network backend (`groq`, `openai`) is reachable ONLY by the caller
    naming it explicitly, e.g. `--backend groq`.
  - `resolve("auto")` NEVER returns a network backend -- not as a default,
    not as a fallback, not after a local backend fails to import. There is
    no code path here from "auto" to the network; `test_backends.py` asserts
    exactly that for every platform this module can be asked to pretend to
    be.

Importing this module must stay cheap and dependency-free: it only describes
backends, it does not load them. Each backend's own third-party import
happens inside `load()`, and only for the one backend actually selected --
installing the skill must not drag in mlx-whisper, faster-whisper AND
nemo_toolkit at once.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

__all__ = [
    "COST_PER_HOUR_USD",
    "REGISTRY",
    "BackendInfo",
    "MissingDependency",
    "UnknownBackend",
    "estimate_cost",
    "load",
    "resolve",
]


@dataclass(frozen=True)
class BackendInfo:
    """Everything about one backend except its code, which lives in `load()`."""

    name: str
    kind: Literal["local", "network"]
    pip_spec: str | None  # e.g. "mlx-whisper>=0.4.2"; None for the stdlib-urllib backends
    import_name: str | None  # module to probe for availability, e.g. "mlx_whisper"
    has_whisper_metrics: bool  # False only for parakeet -- see types.py's QUALITY METRICS note
    platforms: tuple[str, ...]  # "<sys.platform>-<platform.machine()>" pairs, or ("any",)
    default_model: str
    notes: str


REGISTRY: dict[str, BackendInfo] = {
    "mlx-whisper": BackendInfo(
        name="mlx-whisper",
        kind="local",
        pip_spec="mlx-whisper>=0.4.2",
        import_name="mlx_whisper",
        has_whisper_metrics=True,
        platforms=("darwin-arm64",),
        default_model="turbo",
        notes="Apple Silicon only -- MLX does not run on Intel Macs or non-Apple hardware.",
    ),
    "faster-whisper": BackendInfo(
        name="faster-whisper",
        kind="local",
        pip_spec="faster-whisper>=1.2",
        import_name="faster_whisper",
        has_whisper_metrics=True,
        platforms=("any",),
        default_model="turbo",
        notes="CTranslate2, no torch dependency. The cross-platform local default.",
    ),
    "parakeet": BackendInfo(
        name="parakeet",
        kind="local",
        pip_spec="parakeet-mlx",
        import_name="parakeet_mlx",
        has_whisper_metrics=False,
        platforms=("any",),
        default_model="mlx-community/parakeet-tdt-0.6b-v3",
        notes=(
            "UNTESTED on this machine. A CTC/TDT model, not Whisper: it returns none of "
            "avg_logprob/compression_ratio/no_speech_prob, so the hallucination guard's "
            "metric-based rules do not apply here -- only its metric-free repetition rule does."
        ),
    ),
    "groq": BackendInfo(
        name="groq",
        kind="network",
        pip_spec=None,  # stdlib urllib.request only -- no groq SDK dependency
        import_name=None,
        has_whisper_metrics=True,
        platforms=("any",),
        default_model="whisper-large-v3",
        notes=(
            "Opt-in only: reachable exclusively via explicit --backend groq, never from "
            "'auto'. Hosts Whisper, so verbose_json returns the full metric set. "
            "Reads GROQ_API_KEY from the environment; never accepts a key as a flag."
        ),
    ),
    "elevenlabs": BackendInfo(
        name="elevenlabs",
        kind="network",
        pip_spec=None,
        import_name=None,
        # Scribe returns a per-word logprob and none of Whisper's three segment
        # metrics, so the guard's metric rules cannot fire. Its backend-independent
        # rules (decoded_from_silence, repeated_token) still apply.
        has_whisper_metrics=False,
        platforms=("any",),
        default_model="scribe_v2",
        notes=(
            "The only backend here that can attribute speakers -- diarization for up to 32. "
            "Returns words rather than segments, so segmentation is ours. Labels are "
            "speaker_0/speaker_1, not names: a person still maps them, and notes_check "
            "rejects a raw label in a notes document. Pricing is not published in the API "
            "docs, so no cost estimate is offered for it."
        ),
    ),
    "openai": BackendInfo(
        name="openai",
        kind="network",
        pip_spec=None,  # stdlib urllib.request only -- no openai SDK dependency
        import_name=None,
        has_whisper_metrics=True,
        platforms=("any",),
        default_model="whisper-1",
        notes=(
            "Opt-in only: reachable exclusively via explicit --backend openai, never from "
            "'auto'. Reads OPENAI_API_KEY from the environment; never accepts a key as a flag."
        ),
    ),
}

# $/hour, derived from published per-minute or per-hour API pricing.
# Groq figures are from Groq's own pricing page. OpenAI's whisper-1 is
# $0.006/minute published, so $0.006 * 60 = $0.36/hour.
# USD per hour of audio, from each provider's own published pricing. A model
# missing here yields None from estimate_cost(), which callers must read as
# "cannot estimate" and never as "free".
COST_PER_HOUR_USD: dict[str, dict[str, float]] = {
    "groq": {"whisper-large-v3": 0.111, "whisper-large-v3-turbo": 0.04},
    "openai": {"whisper-1": 0.36},
    # Verified 2026-09-04 from elevenlabs.io/pricing/api. Flat across every
    # plan tier -- only the included hours differ, not the rate.
    "elevenlabs": {"scribe_v2": 0.22, "scribe_v2_realtime": 0.39},
}


class UnknownBackend(ValueError):
    """Raised by resolve() for a name that is not in REGISTRY."""


class MissingDependency(RuntimeError):
    """Raised by load() when a backend's third-party import is not installed."""


def _local_default(system: str, machine: str) -> BackendInfo:
    """auto -> mlx-whisper on Apple Silicon, else faster-whisper. Always local."""
    if system == "darwin" and machine in ("arm64", "aarch64"):
        return REGISTRY["mlx-whisper"]
    return REGISTRY["faster-whisper"]


def resolve(
    name: str = "auto",
    *,
    system: str | None = None,
    machine: str | None = None,
) -> BackendInfo:
    """Resolve a `--backend` argument to a `BackendInfo`.

    `system`/`machine` default to `sys.platform`/`platform.machine()` but are
    overridable so tests can force a non-Apple platform without mocking the
    interpreter itself.

    'auto' NEVER returns a network backend -- see the module docstring. Every
    other path here is a literal registry lookup: an explicit '--backend groq'
    still resolves to groq, because that is the caller opting in by name, not
    'auto' choosing it.
    """
    if name == "auto":
        resolved_system = system if system is not None else sys.platform
        resolved_machine = machine if machine is not None else platform.machine()
        return _local_default(resolved_system, resolved_machine)

    info = REGISTRY.get(name)
    if info is None:
        valid = ", ".join(sorted(REGISTRY))
        raise UnknownBackend(f"unknown backend {name!r}; valid backends are: {valid}")
    return info


def load(info: BackendInfo) -> ModuleType:
    """Import and return the tslib backend module implementing `info`.

    Raises MissingDependency, with the exact install command in the message,
    if the underlying third-party library is not importable. `pip_spec` is
    None for the urllib-only network backends, which have nothing to install.
    """
    module = importlib.import_module(f"tslib.backends.{info.name.replace('-', '_')}")

    if info.import_name is not None:
        try:
            importlib.import_module(info.import_name)
        except ImportError as exc:
            script = "scripts/transcribe.py"
            raise MissingDependency(
                f"backend {info.name!r} needs {info.pip_spec!r}, which is not installed.\n"
                f"Run it with:\n"
                f"  uv run --with '{info.pip_spec}' --script {script} --backend {info.name} ..."
            ) from exc

    return module


def estimate_cost(backend: str, model: str, seconds: float) -> float | None:
    """Dollar estimate for sending `seconds` of audio to a network backend.

    Returns None for a local backend or an unpriced model -- callers must
    treat None as "cannot estimate", not "free".
    """
    per_hour = COST_PER_HOUR_USD.get(backend, {}).get(model)
    if per_hour is None:
        return None
    return per_hour * (seconds / 3600.0)
