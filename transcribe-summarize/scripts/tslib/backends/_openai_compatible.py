"""One client for every OpenAI-compatible transcription endpoint.

WHY THIS MODULE EXISTS. Groq's transcription endpoint is literally
`https://api.groq.com/openai/v1/audio/transcriptions` -- it is the OpenAI API,
served by someone else. Written as two backends, `groq.py` and `openai.py` came
out with three byte-identical functions and a fourth differing only in the name
of an environment variable. That is the duplication smell in its clearest form:
a fix to the multipart encoding or the `verbose_json` parsing would have to be
made, correctly, twice.

So the client lives here once and a provider is a `Provider` value: an endpoint,
an env var, a default model, an error type, and optionally an upload cap. Adding
a third OpenAI-compatible host is a five-line constant, not another file of
copied parsing.

THE GOVERNING RULE, inherited by every provider that uses this module: nothing
here is reachable from `resolve("auto")`. A caller has already had to name the
backend out loud before a byte can leave the machine.

THE API KEY. Read from the environment only -- never a CLI flag (a flag lands in
shell history and process listings), never echoed, never placed in a dict that
could reach the JSON sidecar or a log line. It is used once, to build one
`Authorization` header. It must never appear in an exception message, and the
error paths below make that structural rather than careful: every message is
either text written before the key existed, or the HTTP response body verbatim.
Neither can contain it.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.parse
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tslib.types import Result, Segment, Word, empty_result


@dataclass(frozen=True)
class Provider:
    """Everything that differs between one OpenAI-compatible host and the next."""

    name: str
    label: str
    # The full URL, for display in the egress disclosure only. The transport
    # below never parses a scheme out of it -- see `host` and `path`.
    endpoint: str
    env_var: str
    default_model: str
    error: type[Exception]
    # None means the provider publishes no cap we guard against. It is NOT
    # "unlimited" -- an oversized upload simply fails at the far end instead.
    max_upload_bytes: int | None = None
    cap_note: str = ""

    @property
    def host(self) -> str:
        """Host only. Raises on a URL that is not https, at config time."""
        parts = urllib.parse.urlparse(self.endpoint)
        if parts.scheme != "https":
            raise ValueError(f"{self.name}: endpoint must be https, got {parts.scheme!r}")
        return parts.netloc

    @property
    def path(self) -> str:
        parts = urllib.parse.urlparse(self.endpoint)
        return parts.path or "/"


def api_key(provider: Provider) -> str:
    key = os.environ.get(provider.env_var)
    if not key:
        raise provider.error(
            f"{provider.env_var} is not set. Export it before selecting --backend {provider.name}."
        )
    return key


def check_size(provider: Provider, wav: Path) -> None:
    if provider.max_upload_bytes is None:
        return
    size = wav.stat().st_size
    if size > provider.max_upload_bytes:
        raise provider.error(
            f"{wav.name} is {size / 1_000_000:.1f} MB, over {provider.label}'s "
            f"{provider.max_upload_bytes // 1_000_000} MB {provider.cap_note}upload cap. "
            f"Shrink it first:\n"
            f"  ffmpeg -i {wav} -ar 16000 -ac 1 -map 0:a -c:a flac {wav.with_suffix('.flac')}"
        )


def encode_multipart(
    fields: list[tuple[str, str]],
    file_field: str,
    filename: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    """Hand-roll a multipart/form-data body -- the one part of this module that
    would normally come from a requests-style library.

    `fields` is a list, not a dict, because `timestamp_granularities[]` is sent
    as the same key twice (segment and word), which a dict cannot represent.
    """
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


@dataclass(frozen=True)
class PreparedRequest:
    """Everything needed to send, with nothing that can choose a scheme."""

    host: str
    path: str
    body: bytes
    headers: dict[str, str]


def build_request(
    provider: Provider,
    wav: Path,
    *,
    model: str,
    language: str | None,
    prompt: str | None,
    key: str,
) -> PreparedRequest:
    """Build the outgoing request without sending it.

    Split from `transcribe()` so a test can assert the key landed in the headers,
    and only there, without making a network call.
    """
    check_size(provider, wav)
    fields: list[tuple[str, str]] = [
        ("model", model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
        ("timestamp_granularities[]", "word"),
    ]
    if language:
        fields.append(("language", language))
    if prompt:
        fields.append(("prompt", prompt))

    body, content_type = encode_multipart(fields, "file", wav.name, wav.read_bytes())
    return PreparedRequest(
        host=provider.host,
        path=provider.path,
        body=body,
        headers={"Content-Type": content_type, "Authorization": f"Bearer {key}"},
    )


def _assign_words(segments: list[Segment], words: list[dict]) -> None:
    """`verbose_json` returns word timestamps at the top level, not nested in
    each segment. Assign each word to the segment whose window contains it.
    """
    for raw_word in words:
        start = raw_word.get("start", 0.0)
        word: Word = {
            "word": raw_word.get("word", ""),
            "start": start,
            "end": raw_word.get("end", 0.0),
            "probability": raw_word.get("probability"),
        }
        for segment in segments:
            if segment["start"] <= start < segment["end"]:
                segment.setdefault("words", []).append(word)
                break


def _to_segment(raw: dict, index: int) -> Segment:
    return {
        "id": raw.get("id", index),
        "start": raw["start"],
        "end": raw["end"],
        "text": raw.get("text", ""),
        "words": [],
        # These hosts serve Whisper itself, so verbose_json carries all three
        # decoder metrics and the full hallucination guard applies. Verified
        # against a live OpenAI response, and asserted on every live run by
        # tests/test_openai_live.py -- if a provider stops returning them, the
        # guard silently weakens, and that test is what catches it.
        "avg_logprob": raw.get("avg_logprob"),
        "compression_ratio": raw.get("compression_ratio"),
        "no_speech_prob": raw.get("no_speech_prob"),
    }


def send(provider: Provider, request: PreparedRequest, timeout: float = 300.0) -> dict:
    """POST over HTTPS and return the decoded JSON body.

    WHY `http.client.HTTPSConnection` AND NOT `urllib.request.urlopen`. urlopen
    dispatches on the URL's scheme, so it will happily open `file://` and hand
    back the contents of a local path -- which is how an API client becomes a
    file reader the moment somebody makes the endpoint configurable (a
    self-hosted or Azure-style OpenAI-compatible host is an obvious future ask).
    Validating the scheme before calling urlopen would work, but it leaves the
    dangerous capability in place behind a check somebody can move or forget.

    `HTTPSConnection` cannot speak any other scheme. There is no URL to
    mis-parse and no plaintext fallback, so the Authorization header cannot be
    sent in clear either. The guarantee is structural rather than asserted.

    `from None` on every raise: chaining would attach the original exception,
    which holds the request and therefore the Authorization header.
    """
    connection = http.client.HTTPSConnection(request.host, timeout=timeout)
    try:
        connection.request("POST", request.path, body=request.body, headers=request.headers)
        response = connection.getresponse()
        raw = response.read()
        if response.status >= 400:
            detail = raw.decode("utf-8", errors="replace")[:800]
            raise provider.error(f"{provider.label} API returned HTTP {response.status}: {detail}") from None
        return json.loads(raw.decode("utf-8"))
    except (OSError, http.client.HTTPException) as exc:
        raise provider.error(f"could not reach {provider.label} API: {exc}") from None
    finally:
        connection.close()


def transcribe(
    provider: Provider,
    wav: Path,
    *,
    model: str,
    language: str | None = None,
    prompt: str | None = None,
    progress: Callable[[Segment], None] | None = None,
) -> Result:
    """Send `wav` and return the shared Result shape."""
    request = build_request(provider, wav, model=model, language=language, prompt=prompt, key=api_key(provider))
    payload = send(provider, request)

    result = empty_result(provider.name, model)
    result["text"] = payload.get("text", "").strip()
    result["language"] = payload.get("language")

    segments: list[Segment] = [_to_segment(raw, i) for i, raw in enumerate(payload.get("segments", []))]
    _assign_words(segments, payload.get("words", []))
    if progress is not None:
        for segment in segments:
            progress(segment)
    result["segments"] = segments
    return result
