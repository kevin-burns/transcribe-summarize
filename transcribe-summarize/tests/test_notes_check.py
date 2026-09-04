"""Tests for scripts/notes_check.py.

Imports the checker directly (via sys.path insertion) for the rule-level assertions, and shells
out to it for the process-level ones (exit codes, --json shape) so those exercise the real CLI
rather than a proxy for it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

sys.path.insert(0, str(SCRIPTS_DIR))
import notes_check  # noqa: E402

ALL_RULE_IDS = {
    "recording-leak",
    "machine-origin",
    "decoder-artefact",
    "inference-heading",
    "inference-phrasing",
}


def _findings_for(fixture_name: str) -> list[notes_check.Finding]:
    path = FIXTURES_DIR / fixture_name
    return notes_check.check_text(str(path), path.read_text(encoding="utf-8"))


def test_clean_fixture_has_zero_findings():
    findings = _findings_for("notes_clean.md")
    assert findings == []


def test_polluted_fixture_trips_every_rule():
    findings = _findings_for("notes_polluted.md")
    rules_hit = {f.rule for f in findings}
    assert rules_hit == ALL_RULE_IDS


def test_approved_subtitle_does_not_trip_recording_leak():
    findings = notes_check.check_text("inline", notes_check.APPROVED_SUBTITLE)
    assert findings == []


def test_approved_closing_does_not_trip_recording_leak():
    findings = notes_check.check_text("inline", notes_check.APPROVED_CLOSING)
    assert findings == []


def test_near_miss_boilerplate_is_still_flagged():
    # Reintroduces the word "recording" that the approved sentence was written to avoid. An
    # allowlist that let this through would be matching loosely, which is the hole this guards.
    findings = notes_check.check_text("inline", "Not a verbatim record of the recording.")
    assert any(f.rule == "recording-leak" for f in findings)


def test_audio_word_boundary_does_not_fire_inside_longer_words():
    findings = notes_check.check_text("inline", "The audiological assessment was routine.")
    assert findings == []


def test_leading_timecode_trips_decoder_artefact():
    findings = notes_check.check_text("inline", "00:14:32 the meeting opened.")
    assert any(f.rule == "decoder-artefact" for f in findings)


def test_inline_ratio_does_not_trip_decoder_artefact():
    findings = notes_check.check_text("inline", "the budget was 3:1 in favour.")
    assert findings == []


def test_next_steps_heading_trips_inference_heading():
    findings = notes_check.check_text("inline", "## Next steps")
    assert any(f.rule == "inference-heading" for f in findings)


def test_next_steps_phrase_in_body_text_does_not_trip():
    # This is a heading-only rule: the same words in ordinary prose are not a section promoting
    # analysis, so they must not be flagged.
    findings = notes_check.check_text("inline", "We discussed the next steps informally before the call ended.")
    assert findings == []


def test_exit_code_zero_on_clean_file():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "notes_check.py"), str(FIXTURES_DIR / "notes_clean.md")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_exit_code_one_on_polluted_file():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "notes_check.py"), str(FIXTURES_DIR / "notes_polluted.md")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1


def test_json_output_shape():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "notes_check.py"),
            "--json",
            str(FIXTURES_DIR / "notes_clean.md"),
            str(FIXTURES_DIR / "notes_polluted.md"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(proc.stdout)
    assert set(payload) == {"files", "total"}
    assert payload["total"] > 0
    assert len(payload["files"]) == 2

    for file_entry in payload["files"]:
        assert {"path", "findings", "checked"} <= set(file_entry)
        assert file_entry["checked"] is True
        assert isinstance(file_entry["findings"], list)
        for finding in file_entry["findings"]:
            assert {"line", "rule", "match", "message"} <= set(finding)

    clean_entry, polluted_entry = payload["files"]
    assert clean_entry["findings"] == []
    assert len(polluted_entry["findings"]) > 0


def test_unreadable_file_is_not_reported_as_clean():
    # An unsupported extension can never be "checked": the point is it must not silently count
    # as passing just because it produced no findings.
    missing = FIXTURES_DIR / "does_not_exist.docx"
    findings, checked, error = notes_check.process_file(missing)
    assert checked is False
    assert findings == []
    assert error is not None


def test_strict_flag_fails_on_unreadable_file_even_without_findings():
    missing = FIXTURES_DIR / "does_not_exist.docx"

    lenient = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "notes_check.py"), str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert lenient.returncode == 0

    strict = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "notes_check.py"), "--strict", str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode == 1


# --------------------------------------------------------------- wrapped boilerplate
#
# Found by an end-to-end smoke test, not by these unit tests: a document that was
# clean as Markdown FAILED its own check once rendered to PDF. `pdftotext` wraps
# to the rendered column, so the approved closing came back split in two and the
# line-exact allowlist stopped matching, reporting "record" as a leak. Both the
# README and SKILL.md tell people to run the checker on the PDF, so this was a
# false positive on the documented workflow -- and a checker that cries wolf on
# its own boilerplate gets ignored.


def test_boilerplate_split_across_lines_by_pdf_wrapping_is_still_exempt():
    """Exactly how pdftotext returned the approved closing from a real PDF."""
    wrapped = (
        "Written up from the meeting by an attendee. A summary, not a verbatim record. "
        "No part to be read\nas a quotation.\n"
    )
    assert notes_check.check_text("notes.pdf", wrapped) == []


def test_the_subtitle_survives_wrapping_too():
    wrapped = "Notes taken by the attendee. Not a\nverbatim record.\n"
    assert notes_check.check_text("notes.pdf", wrapped) == []


def test_containment_did_not_turn_the_allowlist_into_a_hole():
    """The near miss must still fail. Containment exempts fragments OF the
    approved text; it must not exempt text that merely contains a fragment."""
    findings = notes_check.check_text("notes.md", "Not a verbatim record of the recording.\n")
    assert any(f.rule == "recording-leak" for f in findings)


def test_a_sentence_that_merely_starts_like_the_boilerplate_is_not_exempt():
    findings = notes_check.check_text(
        "notes.md", "Notes taken by the attendee. The recording was reviewed afterwards.\n"
    )
    assert any(f.rule == "recording-leak" for f in findings)
