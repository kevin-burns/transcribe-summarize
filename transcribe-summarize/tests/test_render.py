"""Tests for the document writers and the clock rewrite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tslib.audio import ClockMap  # noqa: E402
from tslib.render import (  # noqa: E402
    clock_hms,
    kept,
    map_to_original,
    srt_timestamp,
    write_json,
    write_markdown_transcript,
    write_srt,
)


def _result(segments):
    return {"text": "", "segments": segments, "language": "en", "backend": "test", "model": "m"}


def _seg(i, start, end, text, **extra):
    seg = {"id": i, "start": start, "end": end, "text": text}
    seg.update(extra)
    return seg


def test_srt_timestamp_format():
    assert srt_timestamp(0) == "00:00:00,000"
    assert srt_timestamp(3661.5) == "01:01:01,500"


def test_clock_hms_keeps_the_hour_column():
    assert clock_hms(0) == "00:00:00"
    assert clock_hms(3725) == "01:02:05"


def test_map_to_original_rewrites_segments_and_words():
    result = _result([
        _seg(0, 0.0, 5.0, "hello", words=[{"word": "hello", "start": 0.0, "end": 1.0}]),
    ])
    map_to_original(result, ClockMap([(40.0, 100.0)]))
    assert result["segments"][0]["start"] == 40.0
    assert result["segments"][0]["end"] == 45.0
    assert result["segments"][0]["words"][0]["start"] == 40.0


def test_map_to_original_is_a_no_op_for_an_identity_clock():
    result = _result([_seg(0, 1.0, 2.0, "hi")])
    map_to_original(result, ClockMap([]))
    assert result["segments"][0]["start"] == 1.0


def test_kept_excludes_suppressed_segments():
    segments = [_seg(0, 0, 1, "real"), _seg(1, 1, 2, "invented", suppressed=True)]
    assert [s["text"] for s in kept(segments)] == ["real"]


def test_srt_skips_suppressed_and_renumbers(tmp_path):
    """A suppressed segment must not leave a gap in the subtitle numbering."""
    segments = [
        _seg(0, 0.0, 1.0, "one"),
        _seg(1, 1.0, 2.0, "thank you thank you", suppressed=True, suppressed_reason="repetition"),
        _seg(2, 2.0, 3.0, "three"),
    ]
    dest = write_srt(segments, tmp_path / "a.srt")
    body = dest.read_text()
    assert "thank you" not in body
    assert body.startswith("1\n")
    assert "\n2\n" in body
    assert "3\n00:" not in body


def test_json_keeps_suppressed_segments_as_the_audit_trail(tmp_path):
    """The guard's rejects stay inspectable -- a gap in the transcript is answerable."""
    segments = [
        _seg(0, 0.0, 1.0, "real"),
        _seg(1, 1.0, 19.0, "no " * 55, suppressed=True, suppressed_reason="repetition",
             compression_ratio=17.38, no_speech_prob=0.708),
    ]
    dest = write_json(_result(segments), tmp_path / "a.json")
    loaded = json.loads(dest.read_text())
    assert len(loaded["segments"]) == 2
    rejected = loaded["segments"][1]
    assert rejected["suppressed"] is True
    assert rejected["suppressed_reason"] == "repetition"
    assert rejected["compression_ratio"] == 17.38


def test_markdown_transcript_groups_on_pauses_and_anchors_the_original_clock(tmp_path):
    segments = [
        _seg(0, 40.0, 42.0, "First thought."),
        _seg(1, 42.1, 44.0, "Still the same thought."),
        _seg(2, 60.0, 62.0, "A new one after a long pause."),
    ]
    meta = {"source_name": "planning-call", "original_duration": 120.0}
    dest = write_markdown_transcript(_result(segments), tmp_path / "a.md", meta)
    body = dest.read_text()

    assert "# Transcript — planning-call" in body
    assert "**[00:00:40]** First thought. Still the same thought." in body
    assert "**[00:01:00]** A new one after a long pause." in body
    assert "- Duration: 00:02:00" in body


def test_markdown_transcript_notes_what_the_guard_removed(tmp_path):
    segments = [
        _seg(0, 0.0, 1.0, "real"),
        _seg(1, 1.0, 2.0, "invented", suppressed=True, suppressed_reason="silence"),
    ]
    dest = write_markdown_transcript(
        _result(segments), tmp_path / "a.md", {"original_duration": 10.0}
    )
    body = dest.read_text()
    assert "invented" not in body
    assert "1 segment(s) were suppressed" in body
    assert ".json" in body


def test_markdown_transcript_carries_provenance(tmp_path):
    """The TRANSCRIPT says where it came from. The notes document must not -- that
    is a different file, written elsewhere, under references/notes-register.md."""
    dest = write_markdown_transcript(
        _result([_seg(0, 0.0, 1.0, "hello")]),
        tmp_path / "a.md",
        {"original_duration": 10.0, "guard_line": "guard: 0 of 1 segments suppressed"},
    )
    body = dest.read_text()
    assert "Engine: test / m" in body
    assert "Quality guard:" in body


def test_markdown_transcript_survives_everything_being_suppressed(tmp_path):
    segments = [_seg(0, 0.0, 1.0, "invented", suppressed=True, suppressed_reason="silence")]
    dest = write_markdown_transcript(
        _result(segments), tmp_path / "a.md", {"original_duration": 10.0}
    )
    assert "No speech was retained" in dest.read_text()
