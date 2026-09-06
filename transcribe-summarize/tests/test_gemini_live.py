"""A LIVE test against Gemini 3.5 Transcribe. Opt-in, and billed.

DOUBLE-GATED for the same reason as the OpenAI and ElevenLabs live tests: a key
sitting in the environment for unrelated work must not mean `pytest` spends
money. Both of:

    GEMINI_API_KEY   set
    TS_LIVE_API=1    set explicitly, for this run

    TS_LIVE_API=1 uv run --with pytest python -m pytest tests/test_gemini_live.py -v -s

WHAT ONLY A LIVE RUN CAN ESTABLISH. The offline tests prove the request is shaped
right, that the key reaches `x-goog-api-key` and nowhere else, and that a
non-Google upload URL is refused. They cannot prove what comes back. Four claims
here are taken from Google's published reference rather than measured, and each
is asserted on every live run so that if one stops being true it is caught rather
than assumed:

  * the two-step upload works at all -- the Files API returns an upload URL in
    `X-Goog-Upload-URL` and then a `files/...` URI;
  * word timestamps really do arrive as `word_info` annotations with protobuf
    duration strings, which is the whole basis for mapping back to the original
    clock;
  * diarization really does return distinct `speaker` values for two voices;
  * Whisper's three metrics really are absent, so `has_whisper_metrics=False` is
    measured rather than inherited from the docs.

Audio is synthesised locally with two macOS `say` voices. Roughly 13 s, so about
$0.001 at the derived $0.306/hour.

A CAUTION EARNED ON THE SCRIBE PATH: the first live diarization run there
reported ONE speaker and looked like an API failure. It was the fixture -- both
clips used macOS `say`'s default voice and were byte-identical. If this reports
one speaker, check the fixture before blaming the vendor.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tslib import audio  # noqa: E402
from tslib.backends import REGISTRY, estimate_cost, gemini  # noqa: E402

LIVE = os.environ.get("TS_LIVE_API") == "1" and bool(os.environ.get("GEMINI_API_KEY"))

pytestmark = pytest.mark.skipif(
    not LIVE, reason="set TS_LIVE_API=1 and GEMINI_API_KEY to run the billed live test"
)

# Two clearly different voices. Named explicitly, never the default -- see the
# caution in the module docstring.
VOICES = ("Daniel", "Karen")
LINES = (
    "Right. Morning everyone. The migration finished on Thursday.",
    "Good morning. Did the Terragrunt units all apply cleanly?",
)


@pytest.fixture(scope="module")
def two_speaker_wav(tmp_path_factory) -> Path:
    if not shutil.which("say"):
        pytest.skip("macOS `say` is needed to synthesise the two-speaker fixture")

    out = tmp_path_factory.mktemp("gemini-live")
    clips = []
    for index, (voice, line) in enumerate(zip(VOICES, LINES, strict=True)):
        aiff = out / f"{index}.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), line], check=True)
        clips.append(aiff)

    # Two byte-identical clips would silently defeat the diarization assertion.
    assert clips[0].read_bytes() != clips[1].read_bytes(), (
        f"the {VOICES[0]} and {VOICES[1]} clips are identical -- both voices are "
        "probably not installed, so this fixture cannot test diarization"
    )

    joined = out / "two-speakers.wav"
    concat = out / "concat.txt"
    concat.write_text("".join(f"file '{clip}'\n" for clip in clips))
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(joined)],
        check=True,
    )
    return joined


def test_the_two_step_upload_returns_a_file_uri(two_speaker_wav):
    uri = gemini.upload(two_speaker_wav, key=os.environ["GEMINI_API_KEY"])
    assert uri.startswith("files/") or "/files/" in uri, f"unexpected file URI {uri!r}"


def test_gemini_returns_word_timestamps_and_separates_the_two_speakers(two_speaker_wav):
    result = gemini.transcribe(two_speaker_wav)

    assert result["text"].strip(), "no text came back"
    assert result["segments"], "no segments were built"

    # Word timestamps are the basis of the clock map. Without them the .srt is
    # wrong by exactly the amount of silence trimmed, silently.
    words = [word for segment in result["segments"] for word in segment["words"]]
    assert words, "no word-level annotations came back; the clock map has nothing to map"
    assert any(word["end"] > word["start"] for word in words), "every word has zero duration"

    speakers = {segment.get("speaker") for segment in result["segments"]} - {None}

    # Pin the LITERAL format, not just the count. Gemini's labels are `spk:0`
    # with a COLON -- measured 2026-09-06, and not what the API reference led us
    # to write. `notes_check.py`'s decoder-artefact rule exists to stop a raw
    # label reaching a notes document, and a rule written against a guessed
    # format is a rule that silently does not fire. It missed this one once.
    print(f"\nGemini speaker labels: {sorted(speakers)}")
    for label in speakers:
        assert re.fullmatch(r"spk:\d+", label), (
            f"Gemini's label format changed to {label!r}; notes_check.py's "
            "decoder-artefact rule must be re-checked against it"
        )

    assert len(speakers) >= 2, (
        f"diarization returned {speakers}; expected at least two distinct labels. "
        "Check the fixture really holds two voices before blaming the API."
    )

    print(f"\ntranscript: {result['text']}")
    print(f"speakers  : {sorted(s for s in speakers if s)}")


def test_whisper_metrics_really_are_absent(two_speaker_wav):
    """`has_whisper_metrics=False` decides whether the guard's strong rules can
    fire. Asserting it against the live response keeps it measured."""
    result = gemini.transcribe(two_speaker_wav)
    for segment in result["segments"]:
        for key in ("avg_logprob", "compression_ratio", "no_speech_prob"):
            assert segment[key] is None, f"{key} arrived as {segment[key]!r} -- the registry is now wrong"

    assert REGISTRY["gemini"].has_whisper_metrics is False


def test_a_wrong_key_fails_without_naming_its_value(two_speaker_wav, monkeypatch):
    decoy = "gemini-decoy-key-not-a-real-credential"
    monkeypatch.setenv("GEMINI_API_KEY", decoy)
    with pytest.raises(gemini.GeminiError) as excinfo:
        gemini.transcribe(two_speaker_wav)
    assert decoy not in str(excinfo.value), "the error message carries the key"


def test_the_cost_estimate_matches_the_duration_actually_sent(two_speaker_wav):
    seconds = audio.probe_duration(two_speaker_wav)
    estimate = estimate_cost("gemini", REGISTRY["gemini"].default_model, seconds)
    assert estimate is not None and estimate > 0
    print(f"\n{seconds:.1f}s of audio -> ~${estimate:.5f}")
