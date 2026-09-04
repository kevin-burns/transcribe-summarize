"""Parakeet (NVIDIA ASR), local. Two runtimes, whichever is installed.

Parakeet ships behind more than one runtime and they are not interchangeable in
weight:

  parakeet-mlx   Apple Silicon only. Small dependency set, and the model
                 (mlx-community/parakeet-tdt-0.6b-v3) is what several macOS
                 dictation apps already download, so it is often on the machine
                 already. VERIFIED WORKING.
  nemo_toolkit   Cross-platform, but pulls torch and a multi-GB tree.
                 NOT VERIFIED -- written from the documented API, never run here.

This module picks whichever imports, preferring parakeet-mlx. There is no
behavioural difference in the Result it returns.

THE GUARD DOES NOT FULLY APPLY HERE, AND THAT IS NOT A BUG IN THIS FILE.
Parakeet is a CTC/TDT model, not a Whisper decoder. It returns none of
`avg_logprob`, `compression_ratio` or `no_speech_prob`, so those are None and the
metric rules cannot fire. `None` means "this backend cannot tell you", never
"this segment is fine". What still works:

  * `decoded_from_silence` -- evidence from ffmpeg's measured silence spans
    rather than from the decoder, so it is unaffected by the model choice. This
    is the strong rule, and it covers the case the metrics were bought for.
  * `repeated_token` -- the metric-free repetition heuristic.

parakeet-mlx does return a per-sentence `confidence`, which is recorded on each
segment as `confidence` for inspection. It is deliberately NOT thresholded: it is
not on the same scale as `avg_logprob`, and this project does not ship a
threshold it has not measured.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

from tslib.types import Result, Segment, empty_result

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
NEMO_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetError(RuntimeError):
    """Raised when no Parakeet runtime is available, or one fails."""


def _available_runtime() -> str:
    """Prefer parakeet-mlx: lighter, and its model is often already cached."""
    if importlib.util.find_spec("parakeet_mlx") is not None:
        return "parakeet-mlx"
    if importlib.util.find_spec("nemo") is not None:
        return "nemo"
    raise ParakeetError(
        "no Parakeet runtime found. Install one:\n"
        "  Apple Silicon : uv run --with parakeet-mlx --script scripts/transcribe.py --backend parakeet ...\n"
        "  anywhere      : uv run --with 'nemo_toolkit[asr]' --script scripts/transcribe.py --backend parakeet ..."
    )


def _emit(segments: list[Segment], progress: Callable[[Segment], None] | None) -> None:
    if progress is not None:
        for segment in segments:
            progress(segment)


def _transcribe_mlx(wav: Path, model: str) -> tuple[str, list[Segment]]:
    from parakeet_mlx import Beam, from_pretrained
    from parakeet_mlx.parakeet import DecodingConfig

    # BEAM, NOT GREEDY -- and this is a timestamp fix, not an accuracy tweak.
    # parakeet-mlx's maintainer attributes abnormal segment timestamps to greedy
    # TDT decoding (senstella/parakeet-mlx#43), and the same class of anomaly
    # shows up in a different TDT implementation on the same weights
    # (FluidInference/FluidAudio#128), so it is a decoder issue rather than a
    # port bug. Measured here on a 48.8s file whose speech ends at 40.79s:
    #
    #     greedy  final segment 36.64-48.48s   overrun +7.69s   3.2s
    #     beam    final segment 36.64-41.52s   overrun +0.73s   2.7s
    #
    # A tenfold reduction, and faster. That overrun is what broke the first
    # version of the silence guard, so fixing it at the decoder is worth more
    # than compensating for it downstream -- the guard still has to cope, because
    # +0.73s is not zero, but it no longer has to cope with eight seconds.
    aligned = from_pretrained(model).transcribe(
        str(wav), decoding_config=DecodingConfig(decoding=Beam())
    )
    segments: list[Segment] = []
    for index, sentence in enumerate(aligned.sentences):
        segments.append({
            "id": index,
            "start": float(sentence.start),
            "end": float(sentence.end),
            "text": sentence.text.strip(),
            "words": [
                {"word": token.text, "start": float(token.start), "end": float(token.end),
                 "probability": float(token.confidence) if token.confidence is not None else None}
                for token in (sentence.tokens or [])
            ],
            # Not a Whisper decoder: these three do not exist here. See the module
            # docstring -- None means "cannot tell you", not "fine".
            "avg_logprob": None,
            "compression_ratio": None,
            "no_speech_prob": None,
            # Recorded, never thresholded. Not on avg_logprob's scale.
            "confidence": float(sentence.confidence) if sentence.confidence is not None else None,
        })
    return aligned.text.strip(), segments


def _transcribe_nemo(wav: Path, model: str) -> tuple[str, list[Segment]]:
    """UNVERIFIED. Written from NeMo's documented API and never run here."""
    from nemo.collections.asr.models import ASRModel

    asr = ASRModel.from_pretrained(model_name=model)
    output = asr.transcribe([str(wav)], timestamps=True)[0]
    text = getattr(output, "text", str(output)).strip()

    raw_segments = (getattr(output, "timestamp", {}) or {}).get("segment", [])
    segments: list[Segment] = [
        {
            "id": index,
            "start": float(raw.get("start", 0.0)),
            "end": float(raw.get("end", 0.0)),
            "text": str(raw.get("segment", "")).strip(),
            "words": [],
            "avg_logprob": None,
            "compression_ratio": None,
            "no_speech_prob": None,
        }
        for index, raw in enumerate(raw_segments)
    ]
    if not segments and text:
        segments = [{
            "id": 0, "start": 0.0, "end": 0.0, "text": text, "words": [],
            "avg_logprob": None, "compression_ratio": None, "no_speech_prob": None,
        }]
    return text, segments


def transcribe(
    wav: Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,  # noqa: ARG001 -- Parakeet v3 is multilingual; no language arg
    prompt: str | None = None,  # noqa: ARG001 -- CTC/TDT models take no initial prompt
    progress: Callable[[Segment], None] | None = None,
) -> Result:
    runtime = _available_runtime()

    if runtime == "parakeet-mlx":
        text, segments = _transcribe_mlx(wav, model)
    else:
        # The MLX repo id is meaningless to NeMo; swap to its own unless overridden.
        text, segments = _transcribe_nemo(wav, NEMO_MODEL if model == DEFAULT_MODEL else model)

    _emit(segments, progress)
    result = empty_result("parakeet", model)
    result["text"] = text
    result["segments"] = segments
    result["language"] = language
    return result
