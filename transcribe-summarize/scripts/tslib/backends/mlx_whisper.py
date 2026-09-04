"""Local transcription via mlx-whisper. Apple Silicon only (see REGISTRY).

The third-party import is deferred to inside `transcribe()`, not done at
module level, for two reasons: `--help` stays instant (the prior art in
`scripts/transcribe.py` does the same for the same reason), and this module
stays importable -- and therefore testable -- on a machine that has never
installed mlx-whisper. `tslib.backends.load()` relies on that: it imports
this module to get at `transcribe`, then separately probes `import_name` for
availability.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tslib.types import Result, Segment, Word, empty_result

# Friendly model names -> the MLX-community HF repos that ship the weights.
MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

# TURBO, NOT large-v3. Measured 2026-09-04 on a real laptop-mic recording of
# accented English: turbo scored 9/12 on planted hard items against large-v3's
# 8/12, in 13.8s against 21.5s. large-v3's miss was the serious kind -- the
# speaker corrected a figure mid-sentence and large-v3 dropped the correction
# entirely. One recording is not a benchmark, but it is the only real measurement
# this project has, and it does not support paying 56% more time for large-v3.
DEFAULT_MODEL = "turbo"


def _resolve_repo(model: str) -> str:
    """A friendly name maps to its repo; anything else is passed through as
    a literal `path_or_hf_repo`, so a user's own fine-tuned repo still works.
    """
    return MODEL_REPOS.get(model, model)


def _to_word(raw: dict) -> Word:
    return {
        "word": raw.get("word", ""),
        "start": raw.get("start", 0.0),
        "end": raw.get("end", 0.0),
        "probability": raw.get("probability"),
    }


def _to_segment(raw: dict, index: int) -> Segment:
    segment: Segment = {
        "id": raw.get("id", index),
        "start": raw["start"],
        "end": raw["end"],
        "text": raw.get("text", ""),
        "words": [_to_word(w) for w in raw.get("words", [])],
        "avg_logprob": raw.get("avg_logprob"),
        "compression_ratio": raw.get("compression_ratio"),
        "no_speech_prob": raw.get("no_speech_prob"),
    }
    return segment


def transcribe(
    wav: Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    prompt: str | None = None,
    progress: Callable[[Segment], None] | None = None,
    condition_on_previous_text: bool = False,
) -> Result:
    """Decode `wav` (16 kHz mono PCM) with mlx-whisper.

    `condition_on_previous_text` defaults to False: carrying decoded text
    between the ~30s decode windows invites the repetition loops documented
    in README.md, "Why this exists" (a single hallucinated word, repeated 55 times,
    over near-silent audio). It is exposed as a parameter, not hard-coded,
    because a caller who has already run the hallucination guard and wants
    long-range vocabulary continuity should be able to opt back in --
    silently defaulting it off is the safe choice, not the only one.

    NEVER pass `verbose` to mlx_whisper.transcribe: neither `True` nor
    `False` streams segments as they decode (verified against the installed
    library -- a verified bug in the prior art), so it cannot substitute for `progress`,
    and passing it at all would revive a documented dead end. Progress
    reporting is done here, after the fact, by iterating the segments
    mlx_whisper returns in one batch and invoking `progress` on each --
    which is honest about what the library actually provides: a single
    return, not a stream.
    """
    import mlx_whisper  # deferred: see module docstring

    raw = mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=_resolve_repo(model),
        language=language,
        initial_prompt=prompt,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=True,
    )

    result = empty_result("mlx-whisper", model)
    result["text"] = raw.get("text", "").strip()
    result["language"] = raw.get("language")

    segments: list[Segment] = []
    for i, raw_segment in enumerate(raw.get("segments", [])):
        segment = _to_segment(raw_segment, i)
        segments.append(segment)
        if progress is not None:
            progress(segment)
    result["segments"] = segments

    return result
