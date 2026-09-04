"""Local transcription via faster-whisper (CTranslate2). Cross-platform.

This is the local default on Linux, Windows, and Intel Macs -- everywhere
`mlx-whisper` cannot run (see `tslib.backends.resolve`). No torch dependency.

The third-party import is deferred to inside `transcribe()`, not done at
module level -- see `mlx_whisper.py`'s docstring for why; the same reasoning
applies here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tslib.types import Result, Segment, Word, empty_result

# TURBO, NOT large-v3. See mlx_whisper.py for the measurement. faster-whisper
# resolves "turbo" to mobiuslabsgmbh/faster-whisper-large-v3-turbo (verified in
# faster_whisper/utils.py's _MODELS map) -- note it is NOT a Systran repo, unlike
# every other short name, so it comes from a different publisher.
DEFAULT_MODEL = "turbo"


def _to_word(raw) -> Word:
    return {
        "word": raw.word,
        "start": raw.start,
        "end": raw.end,
        "probability": getattr(raw, "probability", None),
    }


def _to_segment(raw, index: int) -> Segment:
    return {
        "id": getattr(raw, "id", index),
        "start": raw.start,
        "end": raw.end,
        "text": raw.text,
        "words": [_to_word(w) for w in (raw.words or [])],
        # VERIFIED: faster-whisper's Segment dataclass carries these three
        # fields unchanged from the underlying Whisper decode, so the full
        # hallucination guard (see types.py) applies to this backend exactly
        # as it does to mlx-whisper.
        "avg_logprob": raw.avg_logprob,
        "compression_ratio": raw.compression_ratio,
        "no_speech_prob": raw.no_speech_prob,
    }


def transcribe(
    wav: Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    prompt: str | None = None,
    progress: Callable[[Segment], None] | None = None,
    condition_on_previous_text: bool = False,
) -> Result:
    """Decode `wav` (16 kHz mono PCM) with faster-whisper.

    `condition_on_previous_text` defaults to False for the same reason it
    does in `mlx_whisper.transcribe`: carrying text across decode windows
    invites repetition loops.

    `model.transcribe()` returns a lazy generator: nothing decodes until it
    is iterated. That laziness is exactly where `progress` belongs -- each
    `next()` on the generator is a segment finishing decode, so invoking
    `progress` inside the loop reports real progress rather than a replay
    of an already-complete result (which is all `mlx_whisper.transcribe`
    can offer, since it returns everything at once).
    """
    from faster_whisper import WhisperModel  # deferred: see module docstring

    engine = WhisperModel(model, device="auto", compute_type="auto")
    segment_iter, info = engine.transcribe(
        str(wav),
        language=language,
        initial_prompt=prompt,
        word_timestamps=True,
        condition_on_previous_text=condition_on_previous_text,
    )

    result = empty_result("faster-whisper", model)
    result["language"] = info.language

    segments: list[Segment] = []
    text_parts: list[str] = []
    for i, raw_segment in enumerate(segment_iter):
        segment = _to_segment(raw_segment, i)
        segments.append(segment)
        text_parts.append(segment["text"])
        if progress is not None:
            progress(segment)

    result["segments"] = segments
    result["text"] = "".join(text_parts).strip()

    return result
