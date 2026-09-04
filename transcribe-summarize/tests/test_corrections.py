"""Tests for tslib.corrections -- the --replace pass.

Covers bugs 3 and 4 of the prior art as explicit regressions, not just as a
side effect of other cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tslib import corrections  # noqa: E402
from tslib.types import Result, Segment  # noqa: E402


def _segment(text: str, words: list[str] | None = None) -> Segment:
    return {
        "id": 0,
        "start": 0.0,
        "end": 1.0,
        "text": text,
        "words": [{"word": w, "start": 0.0, "end": 0.0, "probability": 1.0} for w in (words or [])],
    }


def _result(*segments: Segment) -> Result:
    return {
        "text": " ".join(s["text"] for s in segments),
        "segments": list(segments),
        "language": "en",
        "backend": "test",
        "model": "test",
    }


def test_bug3_regression_hit_counted_once_not_three_times() -> None:
    # One occurrence in the text, plus a matching word entry -- the old
    # apply_replacements() summed text + segment + word and reported 3.
    result = _result(_segment("we spoke to north wind about billing", words=["north wind"]))
    rules = corrections.build(["north wind=Northwind"])

    hits = corrections.apply(result, rules)

    assert hits == 1


def test_bug4_regression_chained_rules_do_not_recompound() -> None:
    result = _result(_segment("a b"))
    rules = corrections.build(["a=b", "b=c"])

    corrections.apply(result, rules)

    assert result["segments"][0]["text"] == "b c"


def test_case_insensitive_replacement_uses_given_casing() -> None:
    result = _result(_segment("GROG Whisper hosts it"))
    rules = corrections.build(["grog=Groq"])

    corrections.apply(result, rules)

    assert result["segments"][0]["text"] == "Groq Whisper hosts it"


def test_longer_term_wins_over_shorter_overlapping_one() -> None:
    result = _result(_segment("I love New York City"))
    # Built in the "short rule first" order deliberately -- build() must sort
    # by length, not trust the order the caller passed them in.
    rules = corrections.build(["New York=Big Apple", "New York City=NYC"])

    corrections.apply(result, rules)

    assert result["segments"][0]["text"] == "I love NYC"


def test_term_ending_in_punctuation_still_matches() -> None:
    result = _result(_segment("we moved to cloud. held. the line"))
    rules = corrections.build(["cloud.=Cloud."])

    corrections.apply(result, rules)

    assert result["segments"][0]["text"] == "we moved to Cloud. held. the line"


def test_malformed_pair_raises_replacement_error() -> None:
    try:
        corrections.build(["no-equals-sign"])
    except corrections.ReplacementError as exc:
        assert "no-equals-sign" in str(exc)
    else:
        raise AssertionError("expected ReplacementError")


def test_empty_left_side_raises_replacement_error() -> None:
    try:
        corrections.build(["=right"])
    except corrections.ReplacementError as exc:
        assert "=right" in str(exc)
    else:
        raise AssertionError("expected ReplacementError")


def test_result_text_rebuilt_from_joined_segments() -> None:
    result = _result(_segment("we spoke to north wind"), _segment("about the north wind issue"))
    rules = corrections.build(["north wind=Northwind"])

    corrections.apply(result, rules)

    assert result["text"] == "we spoke to Northwind about the Northwind issue"
