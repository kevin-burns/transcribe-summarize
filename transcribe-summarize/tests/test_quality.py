"""Tests for tslib.quality -- the hallucination guard.

The compression_ratio/no_speech_prob values used below (17.38, 0.923) are the
real measured numbers, not invented -- see
scripts/tslib/quality.py's module docstring for the recording they came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tslib import quality  # noqa: E402
from tslib.types import Result, Segment  # noqa: E402


def _segment(
    text: str,
    *,
    compression_ratio: float | None = None,
    no_speech_prob: float | None = None,
    avg_logprob: float | None = None,
) -> Segment:
    return {
        "id": 0,
        "start": 0.0,
        "end": 1.0,
        "text": text,
        "words": [],
        "compression_ratio": compression_ratio,
        "no_speech_prob": no_speech_prob,
        "avg_logprob": avg_logprob,
    }


def _result(*segments: Segment) -> Result:
    return {
        "text": " ".join(s["text"] for s in segments),
        "segments": list(segments),
        "language": "en",
        "backend": "test",
        "model": "test",
    }


def test_repetition_uses_the_measured_value() -> None:
    # The real word-loop segment: compression_ratio 17.38, against Whisper's
    # own reject threshold of 2.4.
    seg = _segment("go go go go go go go go go go go", compression_ratio=17.38)
    assert quality.inspect(seg) == "repetition"


def test_silence_requires_both_halves_and_they_are_the_measured_values() -> None:
    seg = _segment("Thank you.", no_speech_prob=0.923, avg_logprob=-1.4)
    assert quality.inspect(seg) == "silence"


def test_high_no_speech_alone_is_not_silence() -> None:
    # Confident-but-quiet speech: no_speech_prob is high but avg_logprob is
    # not low, so the two-part rule must not fire on no_speech_prob alone.
    seg = _segment("okay", no_speech_prob=0.923, avg_logprob=-0.2)
    assert quality.inspect(seg) is None


def test_clean_segment_passes() -> None:
    seg = _segment("The migration finished around noon.", compression_ratio=1.4, no_speech_prob=0.02, avg_logprob=-0.3)
    assert quality.inspect(seg) is None


def test_repeated_token_is_the_parakeet_path() -> None:
    # No Whisper metrics at all -- this is what a CTC/TDT backend like
    # Parakeet returns, and repeated_token is the only rule that can catch its
    # version of the 55x word loop.
    seg = _segment("okay okay okay okay okay okay okay okay")
    assert quality.inspect(seg) == "repeated_token"


def test_ordinary_prose_with_no_metrics_passes() -> None:
    seg = _segment("We agreed to reconvene next Tuesday at ten.")
    assert quality.inspect(seg) is None


def test_short_segment_is_exempt_from_repeated_token() -> None:
    # A real "yeah yeah yeah" is not a hallucination -- fewer than 4 tokens
    # is exempt regardless of how dominant the repeated token is.
    seg = _segment("yeah yeah yeah")
    assert quality.inspect(seg) is None


def test_drop_mode_keeps_segment_but_excludes_its_text() -> None:
    bad = _segment("go go go go go go go go go go go", compression_ratio=17.38)
    good = _segment("Let's begin.", compression_ratio=1.2, no_speech_prob=0.01, avg_logprob=-0.2)
    result = _result(good, bad)

    report = quality.apply_guard(result, mode="drop")

    assert len(result["segments"]) == 2
    assert result["segments"][1]["suppressed"] is True
    assert result["segments"][1]["suppressed_reason"] == "repetition"
    assert "go go go" not in result["text"]
    assert result["text"] == "Let's begin."
    assert report.checked == 2
    assert report.suppressed == 1
    assert report.reasons == {"repetition": 1}


def test_off_mode_suppresses_nothing() -> None:
    bad = _segment("go go go go go go go go go go go", compression_ratio=17.38)
    result = _result(bad)

    report = quality.apply_guard(result, mode="off")

    assert result["segments"][0]["suppressed"] is False
    assert "go go go" in result["text"]
    assert report.suppressed == 0
    assert report.reasons == {}


def test_metrics_available_false_when_every_segment_has_none() -> None:
    result = _result(_segment("okay okay okay okay okay"), _segment("ordinary prose here today"))
    report = quality.apply_guard(result, mode="drop")
    assert report.metrics_available is False


def test_metrics_available_true_when_at_least_one_segment_has_a_metric() -> None:
    result = _result(_segment("fine", compression_ratio=1.1))
    report = quality.apply_guard(result, mode="drop")
    assert report.metrics_available is True


def test_summary_line_format() -> None:
    report = quality.GuardReport(checked=41, suppressed=2, reasons={"repetition": 1, "silence": 1})
    assert report.summary_line() == "guard: 2 of 41 segments suppressed (repetition 1, silence 1)"


def test_summary_line_when_nothing_suppressed() -> None:
    report = quality.GuardReport(checked=10, suppressed=0, reasons={})
    assert report.summary_line() == "guard: 0 of 10 segments suppressed"


# --------------------------------------------------------------------------- silence overlap
#
# These cover the rule the metrics could NOT catch. Measured on a synthetic call
# with a 35 s silent head: mlx-whisper turbo invented "Thank you." at 0.0 s with
# compression_ratio 0.56 and no_speech_prob 0.000 -- values every metric rule
# reads as a healthy segment.

CONFIDENT_HALLUCINATION = {
    "start": 0.0, "end": 1.4, "text": " Thank you.",
    "compression_ratio": 0.56, "no_speech_prob": 0.0, "avg_logprob": -0.4,
}


def test_the_metric_rules_alone_do_not_catch_a_confident_hallucination():
    """The gap this rule exists to close. If this ever starts failing, the metric
    thresholds changed and the silence rule may no longer be load-bearing."""
    assert quality.inspect(dict(CONFIDENT_HALLUCINATION)) is None


def test_segment_inside_measured_silence_is_suppressed():
    result = {
        "text": "", "language": "en", "backend": "t", "model": "m",
        "segments": [
            dict(CONFIDENT_HALLUCINATION),
            {"start": 35.0, "end": 38.0, "text": "Good morning everyone.",
             "compression_ratio": 1.27, "no_speech_prob": 0.0, "avg_logprob": -0.3},
        ],
    }
    report = quality.apply_guard(result, silences=[(0.0, 34.7)])
    assert result["segments"][0]["suppressed"] is True
    assert result["segments"][0]["suppressed_reason"] == "decoded_from_silence"
    assert result["segments"][1]["suppressed"] is False
    assert result["text"] == "Good morning everyone."
    assert "decoded_from_silence 1" in report.summary_line()


def test_a_segment_that_merely_starts_inside_a_silence_is_speech_beginning_late():
    starts_late = {"start": 33.0, "end": 38.0, "text": "Good morning everyone and welcome."}
    assert not quality.in_measured_silence(starts_late, [(0.0, 34.7)])
    fully_inside = {"start": 10.0, "end": 12.0, "text": "Thank you."}
    assert quality.in_measured_silence(fully_inside, [(0.0, 34.7)])


def test_a_segment_whose_end_overruns_into_trailing_silence_is_kept():
    """THE FALSE POSITIVE THAT CHANGED THIS RULE. Real numbers from a Parakeet run.

    Parakeet stretched its final segment to 48.48s on a file whose speech stopped
    at 40.79s. Under the original midpoint test the midpoint (42.56s) fell inside
    the trailing silence and REAL SPEECH was suppressed. Containment keeps it: the
    segment starts 4s before the silence does, so it cannot have been decoded
    from it.
    """
    overrunning = {
        "start": 36.64, "end": 48.48,
        "text": "The budget stays at 40,000 euros, and Northwind will handle the rollout.",
    }
    silences = [(0.0, 30.008), (40.794, 48.814)]
    assert not quality.in_measured_silence(overrunning, silences)


def test_a_hallucination_in_the_silent_head_is_still_caught_with_the_same_spans():
    """Same file, same silences: the invention must still be found."""
    invented = {"start": 0.0, "end": 1.4, "text": " Thank you."}
    silences = [(0.0, 30.008), (40.794, 48.814)]
    assert quality.in_measured_silence(invented, silences)


def test_boundary_jitter_within_tolerance_still_counts_as_contained():
    """ffmpeg detects at frame resolution; a decoder's times are its own. A few
    hundred ms of disagreement must not let an invention through."""
    invented = {"start": -0.05, "end": 30.2, "text": "you " * 30}
    assert quality.in_measured_silence(invented, [(0.0, 30.008)])


def test_a_segment_spanning_two_silences_and_the_speech_between_is_kept():
    spanning = {"start": 29.0, "end": 41.5, "text": "Real speech across the whole middle."}
    assert not quality.in_measured_silence(spanning, [(0.0, 30.008), (40.794, 48.814)])


def test_silence_overlap_works_with_no_metrics_at_all():
    """Evidence from ffmpeg, not the decoder -- so it reaches Parakeet too."""
    segment = {"start": 5.0, "end": 6.0, "text": "Thank you.",
               "compression_ratio": None, "no_speech_prob": None, "avg_logprob": None}
    assert quality.in_measured_silence(segment, [(0.0, 34.7)])


def test_no_silences_means_the_rule_never_fires():
    assert not quality.in_measured_silence(dict(CONFIDENT_HALLUCINATION), [])


# ------------------------------------------------------------ review candidates
#
# The guard suppresses; this only points. It exists because measured failures on
# real audio -- large-v3 dropping a spoken self-correction, every backend hearing
# "board pack" as "board packs up" -- are fluent and confident, so no threshold
# reaches them. Directing a human's attention is the only thing that does.


def _seg(start, text, **kw):
    return {"start": start, "end": start + 3.0, "text": text, **kw}


def test_segments_with_figures_are_surfaced_for_checking():
    segments = [
        _seg(0.0, "Right, morning everyone, let's get started."),
        _seg(10.0, "We came in at 42,300 euros against a forecast of 38,000."),
    ]
    picked = quality.review_candidates(segments)
    assert any("42,300" in s["text"] for s, _ in picked)
    assert all("contains figures" in reason for _, reason in picked)


def test_low_confidence_segments_rank_above_merely_numeric_ones():
    segments = [
        _seg(0.0, "The budget stays at 40,000 euros.", avg_logprob=-0.20),
        _seg(10.0, "Their contact is Priya Raghunathan.", avg_logprob=-1.10),
    ]
    picked = quality.review_candidates(segments)
    assert picked, "a low-confidence segment must be surfaced"
    assert "Raghunathan" in picked[0][0]["text"], "least confident should come first"
    assert "low confidence" in picked[0][1]


def test_suppressed_segments_are_not_offered_for_review():
    """They are already gone from the document; pointing at them wastes attention."""
    segments = [_seg(0.0, "Thank you.", suppressed=True, suppressed_reason="decoded_from_silence",
                     avg_logprob=-2.0)]
    assert quality.review_candidates(segments) == []


def test_clean_prose_with_no_figures_is_not_flagged():
    segments = [_seg(0.0, "Right, morning everyone, let's get started.", avg_logprob=-0.15)]
    assert quality.review_candidates(segments) == []


def test_the_list_stays_short_enough_to_act_on():
    segments = [_seg(float(i), f"Item {i} costs {i}00 euros.", avg_logprob=-0.9) for i in range(40)]
    assert len(quality.review_candidates(segments)) <= 8


def test_it_works_on_a_backend_with_no_metrics_at_all():
    """Parakeet returns no metrics; figures alone must still raise a flag."""
    segments = [_seg(0.0, "We came in at 42,300 euros.",
                     avg_logprob=None, compression_ratio=None, no_speech_prob=None)]
    picked = quality.review_candidates(segments)
    assert len(picked) == 1
    assert "contains figures" in picked[0][1]
