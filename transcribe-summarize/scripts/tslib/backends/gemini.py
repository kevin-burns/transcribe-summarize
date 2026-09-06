"""Network transcription via Google's Gemini 3.5 Transcribe. OPT-IN ONLY.

THE GOVERNING RULE, as for every network backend: reachable only when the caller
names `--backend gemini` in that invocation. `resolve("auto")` never returns it.
Before a byte leaves the machine the caller has already said it out loud.

WHY THIS IS NOT IN `_openai_compatible`. Gemini differs on all three axes that
module abstracts:

  auth      `x-goog-api-key: <key>`, not `Authorization: Bearer`.
  transport TWO round trips. The audio is not posted to the transcription
            endpoint at all -- it goes to the Files API first, and the
            transcription request carries only the URI that came back.
  response  a generic `interactions` envelope with word-level *annotations*,
            not a `segments` array. Segmentation below is ours.

WHY THE GROUPER IS NOT SHARED WITH `elevenlabs.py`, which does something
similar: Scribe emits explicit "spacing" entries, per-word logprobs and
punctuation-only tokens; Gemini emits none of those and encodes its offsets as
protobuf duration strings. Merging them means a parameterised function with a
branch per provider at every step, which is the shape `_openai_compatible`
exists to avoid. Two small graspable groupers beat one branchy one.

WHY `smart` MODE IS NOT OFFERED, though the API has it and it is the headline
feature of the model. Verified in the API reference on 2026-09-06:

    "Smart transcription (`"smart"`) is incompatible with
     `timestamp_granularities` and `diarization_mode`."

It removes filler words, resolves spoken self-corrections ("Tuesday -- no,
Wednesday") and reflows the text into paragraphs and bullet lists. Three
consequences, each fatal here on its own: there is no word timing, so the clock
map has nothing to map back onto the user's original file and the `.srt` cannot
be built; there are no speaker labels; and the transcript is no longer a record
of what was said but a model's tidied version of it, which is the exact class of
silent alteration this skill exists to make visible. `mode` is therefore pinned
to verbatim and is not a flag. Cleaning up the prose is what the notes document
is for, downstream, where it is labelled as a summary.

QUALITY METRICS. Gemini returns none of Whisper's three, and no per-word
probability either. `has_whisper_metrics=False`, the metric rules correctly stay
silent, and the guard runs only its backend-independent rules
(`decoded_from_silence`, `repeated_token`).

Verified against ai.google.dev/gemini-api/docs/transcribe, .../docs/files and
.../docs/pricing on 2026-09-06: the `interactions` endpoint, `x-goog-api-key`,
`generation_config.transcription_config.{language_codes,custom_vocabulary,mode}`,
the resumable Files API upload headers, and the
`steps[].content[].annotations[]` shape with
`type`/`text`/`speaker`/`start_offset`/`end_offset`.

ONE THING THE REFERENCE GETS WRONG, measured against four live responses on
2026-09-06: it documents `"output_text": "transcribed text"` at the top level of
the reply. That key is **absent** -- `payload.get("output_text")` was None every
time. The transcript is in `steps[].content[].text`. See `full_text()`.

AND GEMINI DOES NOT TRANSLATE, probed three ways on German audio rather than
taken from the page: the shipped config, `language_codes: ["en-US"]`, and a plain
text instruction "give the result in English" alongside the audio. All three
returned the German. `language_codes` really is a hint about what is spoken.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.parse
import wave
from collections.abc import Callable
from pathlib import Path

from tslib.types import Result, Segment, Word, empty_result

API_HOST = "generativelanguage.googleapis.com"
ENDPOINT = f"https://{API_HOST}/v1beta/interactions"
UPLOAD_PATH = "/upload/v1beta/files"
ENV_VAR = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-3.5-transcribe"

# The Files API caps a single file at 2 GB (and a project at 20 GB, which we
# cannot see from here). Uploaded files are deleted after 48 hours -- stated in
# the disclosure block, because "it sits on Google's servers for two days" is
# part of what the user is agreeing to, not something to leave them to look up.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
FILE_RETENTION_HOURS = 48

# "Standard unary requests support audio files up to 1 hour. Audio processing is
# limited to 30 minutes when features like speaker diarization or word-level
# timestamps are enabled." This backend always asks for both -- without word
# timestamps there is no clock to map back -- so the real cap is 30 minutes, and
# quoting the 1 hour figure here would be quoting a limit we never operate under.
MAX_DURATION_SECONDS = 30 * 60

# Start a new segment on a speaker change, or after a pause this long. Gemini
# returns words, so segments are entirely ours. Same figure as the Scribe path.
SEGMENT_GAP = 1.0

MIME_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/m4a", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".opus": "audio/opus", ".aac": "audio/aac", ".aiff": "audio/aiff",
    ".webm": "audio/webm",
}


class GeminiError(RuntimeError):
    """Raised for a failed request or response. Never carries the API key."""


def _api_key() -> str:
    key = os.environ.get(ENV_VAR)
    if not key:
        raise GeminiError(f"{ENV_VAR} is not set. Export it before selecting --backend gemini.")
    return key


def _mime_type(path: Path) -> str:
    try:
        return MIME_TYPES[path.suffix.lower()]
    except KeyError:
        raise GeminiError(
            f"{path.name}: Gemini accepts {', '.join(sorted(MIME_TYPES))}, not {path.suffix!r}."
        ) from None


def wav_duration(path: Path) -> float | None:
    """Seconds, read from the WAV header. None if it is not a readable PCM WAV.

    stdlib `wave`, not ffprobe: this runs on the *prepared* file, which this
    tool wrote itself as 16 kHz mono PCM, so there is nothing to shell out for.
    None means "cannot tell", and the caller must not read that as "short".
    """
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (wave.Error, OSError, EOFError):
        return None


def check_limits(path: Path) -> None:
    """Refuse before uploading, not after. Raises GeminiError with the reason."""
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise GeminiError(
            f"{path.name} is {size / 1_000_000_000:.1f} GB, over the Files API's 2 GB cap."
        )

    duration = wav_duration(path)
    if duration is not None and duration > MAX_DURATION_SECONDS:
        raise GeminiError(
            f"{path.name} is {duration / 60:.1f} minutes after silence-trimming, over Gemini's "
            f"{MAX_DURATION_SECONDS // 60}-minute limit: word-level timestamps and speaker "
            f"diarization both cap a request there, and this backend always requests them. "
            f"Split the recording, or use a local backend, which has no duration limit at all."
        )


# --------------------------------------------------------------- Files API upload

def build_upload_start(path: Path, *, key: str) -> tuple[str, str, bytes, dict[str, str]]:
    """The resumable upload's opening request. Built separately so a test can
    read the headers without a network call."""
    mime = _mime_type(path)
    size = path.stat().st_size
    body = json.dumps({"file": {"display_name": path.name}}).encode()
    headers = {
        "x-goog-api-key": key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size),
        "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json",
    }
    return API_HOST, UPLOAD_PATH, body, headers


def parse_upload_url(raw: str) -> tuple[str, str]:
    """Split the server-supplied resumable upload URL, refusing anything odd.

    This URL is the one value in the whole exchange that comes from the network
    rather than from this file, so it is the one place a redirect could send the
    audio somewhere else. https and a googleapis.com host are required; nothing
    else is followed.
    """
    parts = urllib.parse.urlparse(raw)
    host = parts.hostname or ""
    if parts.scheme != "https":
        raise GeminiError(f"upload URL is not https (scheme {parts.scheme!r}); refusing to send audio")
    if host != API_HOST and not host.endswith(".googleapis.com"):
        raise GeminiError(f"upload URL points at {host!r}, not a googleapis.com host; refusing to send audio")
    return parts.netloc, parts.path + (f"?{parts.query}" if parts.query else "")


def upload(path: Path, *, key: str, timeout: float = 600.0) -> str:
    """Upload the audio and return its `files/...` URI."""
    host, upload_path, body, headers = build_upload_start(path, key=key)
    start = _send(host, "POST", upload_path, body, headers, timeout=timeout, want_json=False)

    location = start.get("X-Goog-Upload-URL") or start.get("Location")
    if not location:
        raise GeminiError("the Files API did not return an upload URL; nothing was sent")

    up_host, up_path = parse_upload_url(location)
    payload = _send(
        up_host, "POST", up_path, path.read_bytes(),
        {
            "Content-Length": str(path.stat().st_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        timeout=timeout,
    )
    uri = (payload.get("file") or {}).get("uri")
    if not uri:
        raise GeminiError(f"the Files API returned no file URI: {json.dumps(payload)[:400]}")
    return uri


# ------------------------------------------------------------------ transcription

def build_body(
    uri: str, *, mime: str, model: str, language: str | None, vocabulary: list[str] | None
) -> dict:
    """The transcription request body. Pure, so a test can assert its shape."""
    config: dict[str, object] = {
        # [] is not the same as omitting it: an empty list is documented as
        # "detect the language, and handle code-switching within one recording".
        "language_codes": [language] if language else [],
        # Pinned. See the module docstring on why smart mode is not offered.
        "mode": {
            "type": "verbatim",
            "diarization_mode": "speaker",
            "timestamp_granularities": ["word"],
        },
    }
    if vocabulary:
        # Documented cap is 1000 terms, "best results ... with up to 100". The
        # caller is responsible for not reaching here while timestamps are on --
        # see backends.check_prompt.
        config["custom_vocabulary"] = vocabulary[:1000]

    return {
        "model": model,
        "input": [{"type": "audio", "uri": uri, "mime_type": mime}],
        "generation_config": {"transcription_config": config},
    }


def _send(
    host: str, method: str, path: str, body: bytes, headers: dict[str, str],
    *, timeout: float, want_json: bool = True,
) -> dict:
    """POST over HTTPS. HTTPSConnection, not urlopen -- no scheme to subvert.

    Returns the decoded JSON body, or the response headers as a dict when
    `want_json` is False (the resumable upload's opening call answers in headers).
    """
    connection = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if response.status >= 400:
            detail = raw.decode("utf-8", errors="replace")[:800]
            # `from None`: chaining would attach the request, which holds the key.
            raise GeminiError(f"Gemini API returned HTTP {response.status}: {detail}") from None
        if not want_json:
            return dict(response.headers.items())
        return json.loads(raw.decode("utf-8"))
    except (OSError, http.client.HTTPException) as exc:
        raise GeminiError(f"could not reach the Gemini API: {exc}") from None
    finally:
        connection.close()


def parse_offset(value: object) -> float:
    """'1.250s' -> 1.25. Protobuf encodes a Duration as seconds with a trailing s."""
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if text.endswith("s"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        raise GeminiError(f"could not read a timestamp from {value!r}") from None


def full_text(payload: dict) -> str:
    """The model's own rendering of the transcript.

    NOT `output_text`, which the API reference shows as
    `"output_text": "transcribed text"` and which is **absent from every real
    response measured on 2026-09-06** -- `payload.get("output_text")` was None on
    all four calls made against the live endpoint. The text is in
    `steps[].content[].text`.

    This matters beyond tidiness: falling through to joining our own segments
    means the punctuation and spacing come from this file's word-grouper rather
    than from the model, so the transcript would differ from what Gemini actually
    produced in exactly the small ways nobody would think to check.
    """
    parts: list[str] = []
    for step in payload.get("steps") or []:
        for block in (step or {}).get("content") or []:
            text = (block or {}).get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    if parts:
        return " ".join(parts)
    # Kept as a fallback rather than removed: if the field ever appears, use it.
    return str(payload.get("output_text") or "").strip()


def collect_words(payload: dict) -> list[dict]:
    """Pull every `word_info` annotation out of the interactions envelope.

    The envelope nests steps -> content -> annotations, and a response may carry
    several content blocks. Anything whose `type` is not `word_info` is skipped
    rather than guessed at.
    """
    words: list[dict] = []
    for step in payload.get("steps") or []:
        for block in (step or {}).get("content") or []:
            for annotation in (block or {}).get("annotations") or []:
                if isinstance(annotation, dict) and annotation.get("type") == "word_info":
                    words.append(annotation)
    return words


def segments_from_words(raw_words: list[dict]) -> list[Segment]:
    """Group the word stream into segments on speaker change and long pause."""
    segments: list[Segment] = []
    current: Segment | None = None

    for raw in raw_words:
        text = str(raw.get("text") or "")
        if not text.strip():
            continue
        start = parse_offset(raw.get("start_offset"))
        end = parse_offset(raw.get("end_offset")) or start
        speaker = raw.get("speaker")

        word: Word = {"word": text, "start": start, "end": end, "probability": None}

        new_speaker = current is not None and current.get("speaker") != speaker
        long_pause = current is not None and (start - current["end"]) > SEGMENT_GAP
        if current is None or new_speaker or long_pause:
            current = {
                "id": len(segments),
                "start": start,
                "end": end,
                "text": text.strip(),
                "words": [word],
                "speaker": speaker,
                # Whisper's three, and a confidence. Gemini returns none of them.
                # None means "cannot tell you", so the metric rules stay silent.
                "avg_logprob": None,
                "compression_ratio": None,
                "no_speech_prob": None,
                "confidence": None,
            }
            segments.append(current)
        else:
            joiner = "" if text.startswith((" ", ",", ".", "?", "!", "'", ":", ";")) else " "
            current["text"] = (current["text"] + joiner + text.strip()).strip()
            current["end"] = end
            current["words"].append(word)

    return segments


def transcribe(
    wav: Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    prompt: str | None = None,
    progress: Callable[[Segment], None] | None = None,
) -> Result:
    """Upload, transcribe, and return segments on the trimmed clock.

    `prompt` maps to `custom_vocabulary`, the nearest thing Gemini offers -- but
    the API refuses it alongside word timestamps ("custom_vocabulary is
    incompatible with timestamps", HTTP 400, reproduced 2026-09-06) and this
    backend always asks for those. `backends.check_prompt` refuses the
    combination before anything is uploaded; the mapping is kept because it is
    correct for the day the incompatibility is lifted.
    """
    check_limits(wav)

    key = _api_key()
    uri = upload(wav, key=key)
    vocabulary = [term.strip() for term in (prompt or "").split(",") if term.strip()]

    body = build_body(uri, mime=_mime_type(wav), model=model, language=language, vocabulary=vocabulary)
    parts = urllib.parse.urlparse(ENDPOINT)
    payload = _send(
        parts.netloc, "POST", parts.path, json.dumps(body).encode(),
        {"x-goog-api-key": key, "Content-Type": "application/json"},
        timeout=600.0,
    )

    result = empty_result("gemini", model)
    result["segments"] = segments_from_words(collect_words(payload))
    result["text"] = full_text(payload) or " ".join(s["text"] for s in result["segments"]).strip()

    if progress is not None:
        for segment in result["segments"]:
            progress(segment)

    return result
