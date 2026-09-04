"""A LIVE test against ElevenLabs Scribe. Opt-in, and billed.

DOUBLE-GATED for the same reason as the OpenAI live test: a key sitting in the
environment for unrelated work must not mean `pytest` spends money. Both of:

    ELEVENLABS_API_KEY   set
    TS_LIVE_API=1        set explicitly, for this run

    TS_LIVE_API=1 uv run --with pytest python -m pytest tests/test_elevenlabs_live.py -v -s

WHAT ONLY A LIVE RUN CAN ESTABLISH. The offline tests prove the request is shaped
right and that the key reaches `xi-api-key` and nowhere else. They cannot prove
what comes back. Two claims here depend on the provider and are asserted on every
live run, so that if either stops being true it is caught rather than assumed:

  * diarization actually returns distinct `speaker_id` values for two speakers,
    which is the one capability no other backend in this skill has;
  * the response really is words-with-no-segments, which is why segmentation is
    built locally -- if Scribe ever starts returning segments, that code should
    be revisited rather than silently ignored.

Audio is synthesised locally with two macOS `say` voices. Roughly 13 s, so about
$0.0008 at the published $0.22/hour.
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
from tslib.backends import REGISTRY, elevenlabs, estimate_cost, load  # noqa: E402

FIRST = "Right, morning everyone. The migration to the new platform starts in March."
SECOND = "Agreed. The budget stays at forty thousand euros and Northwind handles the rollout."

live = pytest.mark.skipif(
    not (os.environ.get("ELEVENLABS_API_KEY") and os.environ.get("TS_LIVE_API") == "1"),
    reason="live API test: needs ELEVENLABS_API_KEY and TS_LIVE_API=1 (it uploads audio and is billed)",
)
needs_say = pytest.mark.skipif(shutil.which("say") is None, reason="needs macOS `say`")


@pytest.fixture(scope="module")
def two_speaker_wav(tmp_path_factory) -> Path:
    """Two distinct voices, so diarization has something real to separate."""
    tmp = tmp_path_factory.mktemp("el")
    a, b = tmp / "a.aiff", tmp / "b.aiff"
    # BOTH voices pinned. An earlier version left the first one to `say`'s
    # default -- and the default on the machine this was written on IS Daniel, so
    # both clips were the same voice and Scribe correctly reported one speaker.
    # The test failed, the API was right, and the fixture was the bug. A fixture
    # that depends on an unpinned system default is flaky by construction.
    subprocess.run(["say", "-v", "Samantha", "-o", str(a), FIRST], check=True)
    subprocess.run(["say", "-v", "Daniel", "-o", str(b), SECOND], check=True)

    # Guard the guard: if these ever render identically the diarization assertion
    # below is meaningless, and it should say so rather than look like an API fault.
    assert a.read_bytes() != b.read_bytes(), (
        "the two voice clips are byte-identical, so there is nothing for diarization "
        "to separate -- check that both `say` voices are installed"
    )
    combined = tmp / "two.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(a), "-i", str(b),
         "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]", "-map", "[out]",
         "-ar", "16000", "-ac", "1", str(combined)],
        check=True,
    )
    return combined


@live
@needs_say
def test_scribe_transcribes_and_separates_the_two_speakers(two_speaker_wav):
    duration = audio.probe_duration(two_speaker_wav)
    cost = estimate_cost("elevenlabs", REGISTRY["elevenlabs"].default_model, duration)
    print(f"\nsending {duration:.1f}s to ElevenLabs Scribe, estimated ${cost:.4f}")

    module = load(REGISTRY["elevenlabs"])
    result = module.transcribe(
        two_speaker_wav, model=REGISTRY["elevenlabs"].default_model, language="en", prompt=None
    )

    assert result["backend"] == "elevenlabs"
    assert result["segments"], "no segments were built from the word stream"

    text = result["text"].lower()
    for word in ("migration", "march", "budget", "rollout"):
        assert word in text, f"expected {word!r} in: {result['text']!r}"

    # THE CAPABILITY THAT JUSTIFIES THIS BACKEND EXISTING.
    speakers = {s.get("speaker") for s in result["segments"] if s.get("speaker")}
    assert len(speakers) >= 2, (
        f"diarization returned {len(speakers)} speaker(s) for two distinct voices: {speakers}. "
        "If this stops holding, elevenlabs is just a more expensive transcriber."
    )


@live
@needs_say
def test_whisper_metrics_are_absent_and_confidence_is_present(two_speaker_wav):
    """Pins the metric situation. If Scribe ever returns Whisper's three, the
    registry's has_whisper_metrics=False is wrong and the guard is weaker than
    it needs to be."""
    module = load(REGISTRY["elevenlabs"])
    result = module.transcribe(two_speaker_wav, model="scribe_v2", language="en", prompt=None)
    first = result["segments"][0]

    for metric in ("avg_logprob", "compression_ratio", "no_speech_prob"):
        assert first[metric] is None, f"{metric} is now populated — revisit has_whisper_metrics"
    assert first.get("confidence") is not None, "per-word logprob was expected and is missing"
    assert "_logprobs" not in first, "internal scratch field leaked into the result"


@live
@needs_say
def test_a_wrong_key_fails_without_naming_its_value(two_speaker_wav, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "decoy-not-a-real-key")
    module = load(REGISTRY["elevenlabs"])
    with pytest.raises(elevenlabs.ElevenLabsError) as excinfo:
        module.transcribe(two_speaker_wav, model="scribe_v2", language="en", prompt=None)
    rendered = f"{excinfo.value!r} {excinfo.value!s}"
    assert "decoy-not-a-real-key" not in rendered, "the key leaked into an error message"
