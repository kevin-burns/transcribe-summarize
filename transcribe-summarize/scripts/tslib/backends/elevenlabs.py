"""Network transcription via ElevenLabs Scribe. OPT-IN ONLY.

THE GOVERNING RULE, as for every network backend: reachable only when the caller
names `--backend elevenlabs` in that invocation. `resolve("auto")` never returns
it. Before a byte leaves the machine the caller has already said it out loud.

WHY THIS IS NOT IN `_openai_compatible`. Scribe differs on both axes that module
abstracts:

  auth      `xi-api-key: <key>`, not `Authorization: Bearer` -- so the shared
            header builder does not apply.
  response  WORDS, not segments. There is no `segments` array at all; the
            segmentation below is ours, built by grouping words on speaker
            change and pause.

Sharing a module across that would mean two branches in every function, which is
the shape `_openai_compatible` exists to avoid.

WHAT IT CAN DO THAT NOTHING ELSE HERE CAN: speaker diarization, up to 32
speakers. Every Whisper backend and Parakeet return no speaker field whatsoever.
That does not make attribution automatic -- Scribe returns `speaker_0`,
`speaker_1`, which is a decoder artefact, not a name. A person still maps labels
to people, and `notes_check.py` still rejects a raw label in a notes document.

QUALITY METRICS. Scribe returns a per-word `logprob` and none of Whisper's three
segment metrics. The mean word logprob is recorded as `confidence` and used only
to rank what a human should check -- never to suppress. It is a log probability,
so it looks like `avg_logprob`, but it is computed differently and this project
does not ship a threshold it has not measured. The guard therefore runs its
backend-independent rules here: `decoded_from_silence` and `repeated_token`.

Verified against the API reference on 2026-09-04: endpoint, `xi-api-key`,
`model_id=scribe_v2`, `diarize`, and the `words[]` shape with
`text`/`type`/`logprob`/`start`/`end`/`speaker_id`.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.parse
import uuid
from collections.abc import Callable
from pathlib import Path

from tslib.types import Result, Segment, Word, empty_result

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
ENV_VAR = "ELEVENLABS_API_KEY"
DEFAULT_MODEL = "scribe_v2"

# Documented limits: files up to 3 GB, audio up to 10 hours.
MAX_UPLOAD_BYTES = 3 * 1024 * 1024 * 1024
MAX_DURATION_SECONDS = 10 * 3600

# Start a new segment when the speaker changes, or after a pause this long.
# Scribe returns words, so segmentation is entirely ours.
SEGMENT_GAP = 1.0


class ElevenLabsError(RuntimeError):
    """Raised for a failed request or response. Never carries the API key."""


def _api_key() -> str:
    key = os.environ.get(ENV_VAR)
    if not key:
        raise ElevenLabsError(f"{ENV_VAR} is not set. Export it before selecting --backend elevenlabs.")
    return key


def _check_size(wav: Path) -> None:
    size = wav.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise ElevenLabsError(
            f"{wav.name} is {size / 1_000_000_000:.1f} GB, over Scribe's 3 GB upload cap."
        )


def _encode_multipart(fields: list[tuple[str, str]], filename: str, file_bytes: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def build_request(
    wav: Path, *, model: str, language: str | None, diarize: bool, key: str
) -> tuple[str, str, bytes, dict[str, str]]:
    """Build the request without sending it, so a test can inspect the headers."""
    _check_size(wav)
    fields: list[tuple[str, str]] = [("model_id", model), ("diarize", "true" if diarize else "false")]
    if language:
        fields.append(("language_code", language))

    body, content_type = _encode_multipart(fields, wav.name, wav.read_bytes())
    parts = urllib.parse.urlparse(ENDPOINT)
    if parts.scheme != "https":
        raise ElevenLabsError(f"endpoint is not https (scheme {parts.scheme!r}); refusing to send")
    return parts.netloc, parts.path, body, {"Content-Type": content_type, "xi-api-key": key}


def send(request: tuple[str, str, bytes, dict[str, str]], timeout: float = 600.0) -> dict:
    """POST over HTTPS. HTTPSConnection, not urlopen -- no scheme to subvert."""
    host, path, body, headers = request
    connection = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if response.status >= 400:
            detail = raw.decode("utf-8", errors="replace")[:800]
            # `from None`: chaining would attach the request, which holds the key.
            raise ElevenLabsError(f"ElevenLabs API returned HTTP {response.status}: {detail}") from None
        return json.loads(raw.decode("utf-8"))
    except (OSError, http.client.HTTPException) as exc:
        raise ElevenLabsError(f"could not reach ElevenLabs API: {exc}") from None
    finally:
        connection.close()


def _segments_from_words(raw_words: list[dict]) -> list[Segment]:
    """Group Scribe's word stream into segments on speaker change and pause.

    Scribe returns no segments, so this is the only place they come from. Words
    of `type` other than "word" (spacing, punctuation-only entries) carry no
    timing worth grouping on and are folded into the surrounding text.
    """
    segments: list[Segment] = []
    current: Segment | None = None

    for raw in raw_words:
        text = raw.get("text", "")
        if not text:
            continue
        if raw.get("type") not in (None, "word"):
            # Scribe emits explicit "spacing" entries between words. Fold them in,
            # but never doubling an existing space -- the word branch below adds
            # its own separator, and appending both produced "Right.  Morning".
            if current is not None and not current["text"].endswith(" "):
                current["text"] += text
            continue

        start = float(raw.get("start", 0.0))
        end = float(raw.get("end", start))
        speaker = raw.get("speaker_id")
        logprob = raw.get("logprob")

        word: Word = {"word": text, "start": start, "end": end, "probability": None}

        new_speaker = current is not None and current.get("speaker") != speaker
        long_pause = current is not None and (start - current["end"]) > SEGMENT_GAP
        if current is None or new_speaker or long_pause:
            current = {
                "id": len(segments),
                "start": start,
                "end": end,
                "text": text,
                "words": [word],
                "speaker": speaker,
                # Whisper's three: Scribe produces none of them. None means
                # "cannot tell you", so the metric rules correctly stay silent.
                "avg_logprob": None,
                "compression_ratio": None,
                "no_speech_prob": None,
                "confidence": None,
            }
            current["_logprobs"] = [logprob] if logprob is not None else []  # type: ignore[typeddict-unknown-key]
            segments.append(current)
        else:
            needs_space = not current["text"].endswith(" ") and not text.startswith((" ", ",", ".", "?", "!"))
            current["text"] += (" " if needs_space else "") + text
            current["end"] = end
            current["words"].append(word)
            if logprob is not None:
                current["_logprobs"].append(logprob)  # type: ignore[typeddict-item]

    for segment in segments:
        logprobs = segment.pop("_logprobs", [])  # type: ignore[typeddict-item]
        if logprobs:
            segment["confidence"] = round(sum(logprobs) / len(logprobs), 4)
        segment["text"] = segment["text"].strip()
    return segments


def transcribe(
    wav: Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    prompt: str | None = None,  # noqa: ARG001 -- Scribe takes keyterms, not a decoder prompt
    progress: Callable[[Segment], None] | None = None,
    diarize: bool = True,
) -> Result:
    payload = send(build_request(wav, model=model, language=language, diarize=diarize, key=_api_key()))

    segments = _segments_from_words(payload.get("words", []))
    if progress is not None:
        for segment in segments:
            progress(segment)

    result = empty_result("elevenlabs", model)
    result["text"] = payload.get("text", "").strip()
    result["language"] = payload.get("language_code")
    result["segments"] = segments
    return result
