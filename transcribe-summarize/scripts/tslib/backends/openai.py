"""Network transcription via OpenAI's Whisper endpoint. OPT-IN ONLY.

The client lives in `_openai_compatible`; this module is the provider's
configuration. See that module for the key handling and the multipart details.

THE GOVERNING RULE: reachable exclusively by a caller naming `--backend openai`
in that invocation. `resolve("auto")` never returns it.

No upload cap is declared here. OpenAI publishes one, but this project has not
verified the current figure, and a guard built on a guessed number either
refuses valid files or gives false assurance. Absent a verified value the honest
behaviour is to let the API reject the request and report what it said.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tslib.backends import _openai_compatible as oai
from tslib.types import Result, Segment

ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
ENV_VAR = "OPENAI_API_KEY"
DEFAULT_MODEL = "whisper-1"


class OpenAIError(RuntimeError):
    """Raised for a failed request or response. Never carries the API key."""


PROVIDER = oai.Provider(
    name="openai",
    label="OpenAI",
    endpoint=ENDPOINT,
    env_var=ENV_VAR,
    default_model=DEFAULT_MODEL,
    error=OpenAIError,
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
