"""A LIVE test against the real OpenAI transcription API. Opt-in, and billed.

WHY IT IS DOUBLE-GATED. Every other test here runs offline. This one uploads
audio to a third party and costs money, so a key being present in the
environment is deliberately NOT enough to run it -- a developer who exports
OPENAI_API_KEY for unrelated work must not be billed for typing `pytest`. Both
of these must hold:

    OPENAI_API_KEY   set
    TS_LIVE_API=1    set explicitly, for this run

CI sets neither, so it always skips there.

    TS_LIVE_API=1 uv run --with pytest python -m pytest tests/test_openai_live.py -v

WHAT IT IS ACTUALLY FOR. The offline tests prove the request is built correctly
and that a decoy key reaches the header and never an error message. They cannot
prove the shape of what comes BACK. This does -- and specifically it asserts that
`avg_logprob`, `compression_ratio` and `no_speech_prob` are populated, because
the hallucination guard's strong rules are worthless on this backend if the API
stops returning them. That is a claim about somebody else's service, so it can
only be checked by asking the service.

Audio is synthesised locally with macOS `say`. Nothing real, nothing anyone's.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tslib import audio  # noqa: E402
from tslib.backends import REGISTRY, estimate_cost, load  # noqa: E402

SPOKEN = (
    "Good morning everyone. The migration to the new platform will start in March "
    "and finish by June. We agreed the budget stays at forty thousand euros."
)

live = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("TS_LIVE_API") == "1"),
    reason="live API test: needs OPENAI_API_KEY and TS_LIVE_API=1 (it uploads audio and is billed)",
)
needs_say = pytest.mark.skipif(shutil.which("say") is None, reason="needs macOS `say` to synthesise speech")


@pytest.fixture(scope="module")
def spoken_wav(tmp_path_factory) -> Path:
    """A few seconds of synthetic speech, 16 kHz mono -- what a backend receives."""
    tmp = tmp_path_factory.mktemp("live")
    aiff = tmp / "speech.aiff"
    subprocess.run(["say", "-o", str(aiff), SPOKEN], check=True)
    wav = tmp / "speech.wav"
    audio.decode_to_wav(aiff, wav)
    return wav


@live
@needs_say
def test_openai_returns_the_transcript_and_the_guard_metrics(spoken_wav):
    duration = audio.probe_duration(spoken_wav)
    cost = estimate_cost("openai", REGISTRY["openai"].default_model, duration)
    # Printed so a live run always says what it just spent, rather than only what it asserted.
    print(f"\nsending {duration:.1f}s to OpenAI, estimated ${cost:.4f}")

    module = load(REGISTRY["openai"])
    result = module.transcribe(spoken_wav, model=REGISTRY["openai"].default_model, language="en", prompt=None)

    assert result["backend"] == "openai"
    assert result["segments"], "the API returned no segments"

    text = result["text"].lower()
    for word in ("migration", "march", "june", "budget"):
        assert word in text, f"expected {word!r} in the transcript, got: {result['text']!r}"

    # THE ASSERTION THIS TEST EXISTS FOR. If OpenAI ever stops returning these,
    # the guard silently degrades to its repetition rule on this backend and the
    # only warning would be a transcript that looks fine.
    first = result["segments"][0]
    for metric in ("avg_logprob", "compression_ratio", "no_speech_prob"):
        assert first.get(metric) is not None, (
            f"{metric} came back None -- the quality guard's strong rules do not work "
            f"on this backend any more. See references/backends.md."
        )


@live
@needs_say
def test_a_live_run_never_puts_the_key_in_an_error(spoken_wav, monkeypatch):
    """A wrong key must fail loudly and say nothing about the key's value."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-decoy-not-a-real-key")
    module = load(REGISTRY["openai"])
    with pytest.raises(Exception) as excinfo:  # noqa: B017 -- any failure is acceptable; the text is the point
        module.transcribe(spoken_wav, model=REGISTRY["openai"].default_model, language="en", prompt=None)
    rendered = f"{excinfo.value!r} {excinfo.value!s}"
    assert "decoy-not-a-real-key" not in rendered, "the key leaked into an error message"
