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
    "check_multilingual",
    "check_prompt",
    "check_task",
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

    # TRANSLATION. Whisper's second task is X->English and nothing else --
    # mlx_whisper/decoding.py says so in one line: 'whether to perform X->X
    # "transcribe" or X->English "translate"'. So this is a boolean, not a target
    # language, and it will stay one until a backend appears that can aim
    # somewhere other than English.
    can_translate: bool = False
    # Empty means every model this backend offers can translate. Non-empty is the
    # exception Groq forces: whisper-large-v3 translates and whisper-large-v3-turbo
    # does not, which their own comparison table says outright, so asking turbo to
    # translate has to be refused here rather than discovered as an API error.
    translate_models: tuple[str, ...] = ()
    # Substrings that DISQUALIFY a model from translating, for backends that take
    # an open-ended model name and so cannot be covered by an allow-list.
    # "turbo" is here because large-v3-turbo is a distillation that DROPPED the
    # translate task, and it does not error -- it silently transcribes. Measured
    # 2026-09-06 on German audio: --model turbo --task translate returned "Guten
    # Morgen..." under a header saying "Translated to English", while --model
    # large-v3 on the same file returned "Good morning...". Groq documents the
    # same thing for their hosted turbo, so it is the model, not the host.
    translate_excludes: tuple[str, ...] = ()

    # CODE-SWITCHING within one recording.
    #   "no"     detects the language once and applies it to the whole file
    #   "flag"   can re-detect per segment, when asked
    #   "always" does it unconditionally and cannot be turned off
    # This matters more than it looks: Whisper decides the language from the first
    # 30 seconds, so a call that opens in German and switches to Spanish is decoded
    # entirely as German, and the output does not say so.
    multilingual: Literal["no", "flag", "always"] = "no"

    # Why --prompt cannot be used with this backend, or None if it can. Gemini is
    # the only entry with one: its custom_vocabulary is documented "Incompatible
    # with speaker diarization and word-level timestamps", and this backend needs
    # word timestamps for the clock map, so the two can never coexist here.
    prompt_conflict: str | None = None
    # One extra line printed inside the egress disclosure, for a provider
    # whose terms the user is agreeing to and cannot see from here (Gemini
    # stores the upload for 48 hours). None for everything else.
    egress_note: str | None = None


REGISTRY: dict[str, BackendInfo] = {
    "mlx-whisper": BackendInfo(
        name="mlx-whisper",
        kind="local",
        pip_spec="mlx-whisper>=0.4.2",
        import_name="mlx_whisper",
        has_whisper_metrics=True,
        platforms=("darwin-arm64",),
        default_model="turbo",
        can_translate=True,
        translate_excludes=("turbo",),
        multilingual="no",  # one detect_language call, "using up to the first 30 seconds"
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
        can_translate=True,
        translate_excludes=("turbo",),
        # The ONLY backend here that can re-detect the language per segment.
        multilingual="flag",
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
            "Verified live 2026-09-06 on Apple Silicon via parakeet-mlx, including German. "
            "A CTC/TDT model, not Whisper: it returns none of "
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
        can_translate=True,
        # Groq's own comparison table marks translation "No" for turbo.
        translate_models=("whisper-large-v3",),
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
            "Diarization for up to 32 speakers, the most of any backend here. "
            "Returns words rather than segments, so segmentation is ours. Labels are "
            "speaker_0/speaker_1, not names: a person still maps them, and notes_check "
            "rejects a raw label in a notes document. Pricing is not published in the API "
            "docs, so no cost estimate is offered for it."
        ),
    ),
    "gemini": BackendInfo(
        name="gemini",
        kind="network",
        pip_spec=None,  # stdlib http.client only -- no google-genai SDK dependency
        import_name=None,
        # Gemini 3.5 Transcribe returns neither Whisper's three segment metrics
        # nor a per-word probability, so the guard's metric rules cannot fire.
        # Its backend-independent rules (decoded_from_silence, repeated_token)
        # still apply.
        has_whisper_metrics=False,
        platforms=("any",),
        default_model="gemini-3.5-transcribe",
        can_translate=False,
        prompt_conflict=(
            "gemini maps --prompt to custom_vocabulary, which the API refuses alongside word "
            "timestamps: 'custom_vocabulary is incompatible with timestamps' (HTTP 400, "
            "reproduced 2026-09-06). This backend always requests word timestamps, because "
            "without them there is no clock to map back onto your recording, so the two can "
            "never be combined. Fix known misrecognitions afterwards with --replace "
            "'wrong=right', or use a local backend, where --prompt works."
        ),
        # "Handles intra-sentence and inter-sentential code-switching without manual
        # configuration" -- unconditional, so there is nothing for a flag to switch.
        multilingual="always",
        notes=(
            "Opt-in only. The one backend here that uploads in TWO steps: the audio goes to "
            "the Files API first and the transcription request carries only the returned URI. "
            "Diarizes up to 8 speakers (3 or more is experimental) and returns word timestamps. "
            "Its `smart` mode -- filler removal, self-correction resolution, auto-formatting -- "
            "is deliberately NOT offered: the API refuses timestamps and diarization alongside "
            "it, so there would be no clock to map back. Reads GEMINI_API_KEY from the "
            "environment; never accepts a key as a flag."
        ),
        egress_note=(
            "Google's Files API keeps the upload for 48 hours before deleting it."
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
        can_translate=True,
        translate_models=("whisper-1",),
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
    # Gemini publishes a token rate, not an hourly one, so this is derived and
    # the derivation is written down rather than left as a magic number.
    # Verified 2026-09-06 from ai.google.dev/gemini-api/docs/pricing:
    #   25 audio tokens/second in, 175 text tokens/minute out;
    #   $2.00 per M input tokens, $12.00 per M output tokens.
    #   in  = 25 * 3600 = 90,000 tok/hr * $2/M  = $0.180
    #   out = 175 * 60  = 10,500 tok/hr * $12/M = $0.126
    # Both halves are counted because both appear on the invoice. There is
    # also a free tier, which this deliberately does not assume you are on.
    "gemini": {"gemini-3.5-transcribe": 0.306},
}


class UnknownBackend(ValueError):
    """Raised by resolve() for a name that is not in REGISTRY."""


class MissingDependency(RuntimeError):
    """Raised by load() when a backend's third-party import is not installed."""


class UnsupportedOption(ValueError):
    """Raised when a backend cannot honour --task or --multilingual.

    REFUSING IS THE POINT. Accepting the flag and quietly ignoring it would hand
    back a German transcript to someone who asked for English, or a
    single-language decode to someone who said the call was mixed -- in both
    cases a document that reads fine and is not what was asked for. That is the
    exact failure this skill exists to make visible, so an unhonourable option is
    an error, never a no-op.
    """


def check_task(info: BackendInfo, task: str, model: str) -> None:
    """Validate --task against a backend and its model. Raises UnsupportedOption."""
    if task == "transcribe":
        return
    if task != "translate":
        raise UnsupportedOption(f"unknown --task {task!r}; use 'transcribe' or 'translate'")

    able = sorted(name for name, entry in REGISTRY.items() if entry.can_translate)
    if not info.can_translate:
        raise UnsupportedOption(
            f"--backend {info.name} cannot translate; it only transcribes what was said. "
            f"Backends that translate to English: {', '.join(able)}. "
            f"For a mixed-language recording, {info.name} may still be the better transcriber -- "
            f"take the transcript in the spoken languages and write the notes in English."
        )
    excluded = next((bad for bad in info.translate_excludes if bad in model), None)
    if excluded is not None:
        raise UnsupportedOption(
            f"{info.name} model {model!r} cannot translate. large-v3-turbo is a distillation "
            f"that dropped the translate task, and it does not fail -- it silently returns the "
            f"original language under a header claiming English. Use --model large-v3."
        )
    if info.translate_models and model not in info.translate_models:
        raise UnsupportedOption(
            f"{info.name} model {model!r} cannot translate; use "
            f"--model {info.translate_models[0]}. "
            f"(Only {', '.join(info.translate_models)} supports the translations endpoint.)"
        )


def check_prompt(info: BackendInfo, prompt: str | None) -> None:
    """Refuse --prompt where the backend's API rejects it. Raises UnsupportedOption.

    Checked BEFORE the upload, not after: the failure is an HTTP 400 that arrives
    once the audio is already on the provider's servers and billed.
    """
    if prompt and info.prompt_conflict:
        raise UnsupportedOption(info.prompt_conflict)


def check_multilingual(info: BackendInfo, want: bool) -> str | None:
    """Validate --multilingual. Returns a note to print, or raises UnsupportedOption.

    A returned string is not a warning about the flag -- it is the thing the user
    needs to know either way, including when they did NOT pass it.
    """
    if info.multilingual == "always":
        return (
            f"{info.name} detects language per utterance and cannot be told not to, "
            f"so --multilingual is already in effect."
        )
    if not want:
        if info.multilingual == "flag":
            # This branch used to return None, which contradicted this
            # function's own docstring and left the WORST case unwarned: the
            # backend recommended for mixed recordings, being run without the
            # flag that makes it handle them. The failure is identical to the
            # "no" case below -- one detection, applied to the whole file.
            return (
                f"{info.name} detects the language once by default, from the first 30 seconds. "
                f"If this recording changes language part-way, pass --multilingual to re-detect "
                f"on every segment."
            )
        if info.multilingual == "no":
            return (
                f"{info.name} detects the language ONCE, from the first 30 seconds, and applies it "
                f"to the whole recording. If the audio changes language after that, it is not "
                f"decoded badly -- it is silently TRANSLATED into the first language, and the "
                f"transcript does not say so. Measured 2026-09-06: German after 30 s of English "
                f"came back as English, FLUENTLY and wrongly -- 'heute Nachmittag' (this "
                f"afternoon) was rendered 'to the next day'. You will not spot that by reading. "
                f"On a mixed "
                f"recording use faster-whisper, parakeet or gemini instead."
            )
        return None
    if info.multilingual == "flag":
        return None
    able = sorted(n for n, e in REGISTRY.items() if e.multilingual in ("flag", "always"))
    raise UnsupportedOption(
        f"--backend {info.name} cannot re-detect the language during a recording; it decides once "
        f"from the first 30 seconds. Backends that handle a language change mid-recording: "
        f"{', '.join(able)}."
    )


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
