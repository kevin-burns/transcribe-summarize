"""Tests for reformatting a transcript the meeting platform already produced."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import normalise_transcript as nt  # noqa: E402

TEAMS_VTT = """WEBVTT

0:00:03.120 --> 0:00:07.480
<v A. Okonkwo>Right, morning everyone. First item is the</v>

0:00:07.480 --> 0:00:12.010
<v A. Okonkwo>migration. We're moving eighteen units off the old root config.</v>

0:00:12.400 --> 0:00:15.220
<v M. Reyes>Sorry, is that all of them or just the EKS ones?</v>
"""

SRT = """1
00:00:01,000 --> 00:00:04,000
Right, morning everyone.

2
00:00:04,000 --> 00:00:08,000
The budget stays at 40,000 euros.
"""


def test_a_sentence_split_across_two_cues_is_rejoined():
    """THE BUG THIS CAUGHT. The gap must be measured from the previous cue's END.
    Measuring from its start measures the cue's duration, which splits exactly the
    wrapped sentences this script exists to reassemble."""
    cues = nt.dedupe(nt.parse_cues(TEAMS_VTT))
    paragraphs = nt.to_paragraphs(cues, keep_speakers=True)
    assert len(paragraphs) == 2, f"expected 2 paragraphs, got {len(paragraphs)}"
    assert "First item is the migration." in paragraphs[0][2]


def test_a_speaker_change_always_starts_a_new_paragraph():
    paragraphs = nt.to_paragraphs(nt.dedupe(nt.parse_cues(TEAMS_VTT)), keep_speakers=True)
    assert paragraphs[0][1] == "A. Okonkwo"
    assert paragraphs[1][1] == "M. Reyes"


def test_voice_tags_are_stripped_from_the_text():
    for _, _, text in nt.to_paragraphs(nt.dedupe(nt.parse_cues(TEAMS_VTT)), keep_speakers=True):
        assert "<v" not in text and "</v>" not in text


def test_srt_without_speakers_parses_too():
    cues = nt.parse_cues(SRT)
    assert len(cues) == 2
    assert all(cue.speaker is None for cue in cues)
    assert "40,000" in cues[1].text


def test_srt_cue_numbers_are_not_treated_as_text():
    assert not any(cue.text.strip() in {"1", "2"} for cue in nt.parse_cues(SRT))


def test_rolling_caption_repeats_are_collapsed():
    """Teams re-emits a growing cue as someone speaks; keep only the longest."""
    rolling = """WEBVTT

0:00:01.000 --> 0:00:02.000
<v A>The budget</v>

0:00:02.000 --> 0:00:03.000
<v A>The budget stays at</v>

0:00:03.000 --> 0:00:04.000
<v A>The budget stays at forty thousand.</v>
"""
    cues = nt.dedupe(nt.parse_cues(rolling))
    assert len(cues) == 1
    assert cues[0].text == "The budget stays at forty thousand."


def test_no_speakers_drops_attribution_but_keeps_the_words():
    paragraphs = nt.to_paragraphs(nt.dedupe(nt.parse_cues(TEAMS_VTT)), keep_speakers=False)
    assert all(speaker is None for _, speaker, _ in paragraphs)
    assert any("EKS" in text for _, _, text in paragraphs)


def test_speaker_prefix_style_transcripts_are_recognised():
    """Some exports use 'Name: text' rather than a voice tag."""
    plain = """WEBVTT

0:00:01.000 --> 0:00:04.000
A. Okonkwo: Right, morning everyone.
"""
    cues = nt.parse_cues(plain)
    assert cues[0].speaker == "A. Okonkwo"
    assert cues[0].text == "Right, morning everyone."


def test_html_entities_are_unescaped():
    entity = """WEBVTT

0:00:01.000 --> 0:00:04.000
Ops &amp; platform, that&#39;s the scope.
"""
    assert nt.parse_cues(entity)[0].text == "Ops & platform, that's the scope."


def test_timestamps_without_an_hour_field_parse():
    assert nt.parse_timestamp("07:30.500") == 450.5
    assert nt.parse_timestamp("1:02:03,250") == 3723.25


def test_the_output_says_nothing_was_re_transcribed():
    """The header must not imply the tool decoded audio it never saw."""
    paragraphs = nt.to_paragraphs(nt.dedupe(nt.parse_cues(TEAMS_VTT)), keep_speakers=True)
    rendered = nt.render(paragraphs, "meeting.vtt", ["A. Okonkwo"])
    assert "Nothing was re-transcribed" in rendered
    assert rendered.startswith("# Transcript — meeting.vtt")
