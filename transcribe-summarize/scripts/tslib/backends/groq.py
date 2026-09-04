"""Network transcription via Groq's hosted Whisper endpoint. OPT-IN ONLY.

Groq serves the OpenAI transcription API -- the endpoint below is literally
`/openai/v1/audio/transcriptions` -- so the client lives in
`_openai_compatible` and this module is the provider's configuration. See that
module for the key handling and the multipart details.

THE GOVERNING RULE: reachable exclusively by a caller naming `--backend groq` in
that invocation. `resolve("auto")` never returns it. Before a byte leaves the
machine, the caller has already had to say "groq" out loud.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tslib.backends import _openai_compatible as oai
from tslib.types import Result, Segment

ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
ENV_VAR = "GROQ_API_KEY"
DEFAULT_MODEL = "whisper-large-v3"

# Groq's free-tier upload cap. The dev tier raises it to 100 MB, but 25 MB is
# the figure to guard against: refusing a file the free tier would reject is
# recoverable, letting one through that fails after the upload is not.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class GroqError(RuntimeError):
    """Raised for a failed request or response. Never carries the API key."""


PROVIDER = oai.Provider(
    name="groq",
    label="Groq",
    endpoint=ENDPOINT,
    env_var=ENV_VAR,
    default_model=DEFAULT_MODEL,
    error=GroqError,
    max_upload_bytes=MAX_UPLOAD_BYTES,
    cap_note="free-tier ",
)


def transcribe(
    wav: Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    prompt: str | None = None,
    progress: Callable[[Segment], None] | None = None,
) -> Result:
    return oai.transcribe(PROVIDER, wav, model=model, language=language, prompt=prompt, progress=progress)
