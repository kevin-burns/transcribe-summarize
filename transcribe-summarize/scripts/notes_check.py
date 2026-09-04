#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Check a meeting-notes document against references/notes-register.md.

The register (see that file) says a notes document must read as a protocol write-up: a factual
summary in the register of someone who was in the room, never a document that discloses it came
from a recording or a transcription pipeline, and never one that adds analysis the meeting itself
did not state. This script is the automated half of enforcing that -- a regex sweep, not a full
reader, so it catches vocabulary and structure, not judgement calls.

Five rule groups, matched case-insensitively and word-bounded so normal prose is not caught inside
longer words ("audio" must not fire on "audiological"):

    recording-leak      -- says a recording/transcript/audio file was involved
    machine-origin      -- names the transcription engine or discloses machine authorship
    decoder-artefact     -- timecodes, "Speaker N" labels, bare [N] segment markers
    inference-heading    -- a *heading* that promotes analysis into its own section
    inference-phrasing   -- inline hedge-and-infer phrasing ("this suggests", "it appears", ...)

The two approved boilerplate sentences are exempt, by exact string match only (after collapsing
whitespace) -- not a loose contains-check. A near-miss variant of either sentence is not exempt
and must still be flagged; that asymmetry is deliberate, see references/notes-register.md.

    ./notes_check.py notes.md
    ./notes_check.py notes.md analysis.md --json
    ./notes_check.py notes.md --strict          # also fail if a file could not be read at all
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Approved boilerplate (quoted exactly from references/notes-register.md).
# ---------------------------------------------------------------------------

APPROVED_SUBTITLE = "Notes taken by the attendee. Not a verbatim record."
APPROVED_CLOSING = (
    "Written up from the meeting by an attendee. A summary, not a verbatim record. "
    "No part to be read as a quotation."
)


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip the ends.

    This is the ONLY normalisation the allowlist gets. Anything beyond whitespace -- stripping
    markdown emphasis, trimming trailing punctuation, fuzzy matching -- would turn the allowlist
    into a hole: a document could smuggle a leak past it by wrapping the leak in something that
    "looks close enough" to the approved sentence. Exact match, whitespace aside, is the point.
    """
    return " ".join(text.split())


# Membership test used before rules 1-3 run on a line. Rules 4 and 5 still run on an allowlisted
# line too (the two approved sentences never contain a heading or inference phrase, so this never
# matters in practice, but the register is silent on making 4/5 an exception, so this doesn't
# invent one).
_ALLOWLIST = {_normalize_ws(APPROVED_SUBTITLE), _normalize_ws(APPROVED_CLOSING)}


def _is_approved_boilerplate(line: str) -> bool:
    """Is this line wholly part of the approved boilerplate?

    Exact match is not enough. A PDF is checked by extracting its text, and
    `pdftotext` hard-wraps to the rendered column, so the approved closing comes
    back split across two lines:

        "Written up from the meeting by an attendee. A summary, not a verbatim
         record. No part to be read"
        "as a quotation."

    Neither fragment equals the approved sentence, so a line-exact allowlist
    reported `record` as a leak on a document that was clean in Markdown. That is
    a false positive on the exact workflow the docs tell people to run, and a
    checker that cries wolf on its own boilerplate gets ignored.

    So a line is exempt when its normalised text is a contiguous substring of an
    approved sentence. Containment, not equality, survives re-wrapping -- and it
    stays strict, because a near miss like "Not a verbatim record of the
    recording." is not a substring of anything approved and is still flagged.
    """
    normalised = _normalize_ws(line)
    if not normalised:
        return False
    return any(normalised in approved for approved in _ALLOWLIST)

# ---------------------------------------------------------------------------
# Rule 1: recording-leak -- says a recording/transcript/audio artefact exists.
# ---------------------------------------------------------------------------

_RECORDING_LEAK: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brecord(?:ed|ing)?\b", re.I), "state what was said; do not disclose that a recording exists"),
    (re.compile(r"\btranscri(?:pt|ption|bed)\b", re.I), "do not disclose that a transcript exists"),
    (re.compile(r"\baudio\b", re.I), "do not reference the audio source"),
    (re.compile(r"\bsubtitle\b", re.I), "do not reference subtitle/caption files"),
    (re.compile(r"\.srt\b", re.I), "do not reference subtitle file formats"),
    (re.compile(r"\.mp3\b", re.I), "do not reference audio file formats"),
    (re.compile(r"\.m4a\b", re.I), "do not reference audio file formats"),
    (re.compile(r"\bplayback\b", re.I), "do not reference playback of a recording"),
    (re.compile(r"\blisten(?:ed)?\s+(?:back|again)\b", re.I), "do not reference listening back to a recording"),
]

# ---------------------------------------------------------------------------
# Rule 2: machine-origin -- names the engine or discloses machine authorship.
# ---------------------------------------------------------------------------

_MACHINE_ORIGIN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bAI[- ]generated\b", re.I), "do not disclose machine origin"),
    (re.compile(r"\bauto-generated\b", re.I), "do not disclose machine origin"),
    (re.compile(r"\bmachine-generated\b", re.I), "do not disclose machine origin"),
    (re.compile(r"\bwhisper\b", re.I), "do not name the transcription engine"),
    (re.compile(r"\bparakeet\b", re.I), "do not name the transcription engine"),
    (re.compile(r"\bspeech-to-text\b", re.I), "do not disclose machine origin"),
    (re.compile(r"\bASR\b", re.I), "do not disclose machine origin"),
    (re.compile(r"\bdiariz\w*\b", re.I), "do not disclose speaker diarization as a process"),
    (re.compile(r"\bgenerated (?:from|by)\b", re.I), "do not disclose machine origin"),
    (re.compile(r"\bconfidence score\b", re.I), "do not disclose decoder confidence scores"),
]

# ---------------------------------------------------------------------------
# Rule 3: decoder-artefact -- leftovers from a decoder's own output format.
# ---------------------------------------------------------------------------

# A bare timecode leading a line, e.g. "00:14:32 ..." or "1:05 ...". Anchored to line start so
# an ordinary ratio like "the budget was 3:1" is never touched -- it is never at column 0 in a
# way that matches this shape, and even when it is, \d{1,2}:\d{2} requires two digits after the
# colon, which "3:1" does not have.
_DECODER_LINE_START = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\b")

_DECODER_INLINE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bSpeaker\s+\d+\b", re.I), "do not use decoder speaker labels; name the person or role"),
    (re.compile(r"\[\d+\]"), "do not use bare segment markers"),
]

# ---------------------------------------------------------------------------
# Rule 4: inference-heading -- a HEADING that promotes analysis into a section.
# ---------------------------------------------------------------------------

# Only ever checked against heading text (see _scan_heading), which is why a "## Next steps"
# heading trips this and the words "next steps" inside an ordinary sentence of body text do not.
_INFERENCE_HEADING = re.compile(
    r"\b(?:next\s+steps|action\s+items|takeaways|recommendations|analysis|"
    r"not\s+(?:covered|discussed|mentioned)|conclusions?|implications)\b",
    re.I,
)
_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# ---------------------------------------------------------------------------
# Rule 5: inference-phrasing -- inline hedge-and-infer language, anywhere in body text.
# ---------------------------------------------------------------------------

_INFERENCE_PHRASING: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bthis suggests\b", re.I), "state what was said, not what it suggests"),
    (re.compile(r"\bit appears\b", re.I), "state what was said; do not characterise how it appears"),
    (re.compile(r"\bimplies\b", re.I), "state what was said; do not draw out an implication"),
    (re.compile(r"\bpresumably\b", re.I), "state what was said; do not presume the rest"),
    (re.compile(r"\blikely means\b", re.I), "state what was said, not what it likely means"),
    (re.compile(r"\breading between\b", re.I), "state what was said; do not read between lines"),
    (re.compile(r"\bseems to indicate\b", re.I), "state what was said, not what it seems to indicate"),
]


@dataclass
class Finding:
    """One rule violation: where it is, which rule, what matched, and how to fix it."""

    path: str
    line: int
    rule: str
    match: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"line": self.line, "rule": self.rule, "match": self.match, "message": self.message}


def _scan_patterns(
    path: str, lineno: int, line: str, patterns: list[tuple[re.Pattern[str], str]], rule_id: str
) -> list[Finding]:
    findings = []
    for pattern, message in patterns:
        for m in pattern.finditer(line):
            findings.append(Finding(path, lineno, rule_id, m.group(0), message))
    return findings


def _scan_decoder_artefact(path: str, lineno: int, line: str) -> list[Finding]:
    findings = []
    start_match = _DECODER_LINE_START.match(line)
    if start_match:
        findings.append(
            Finding(
                path,
                lineno,
                "decoder-artefact",
                start_match.group(0).strip(),
                "do not lead a line with a raw timecode; write it as prose",
            )
        )
    findings.extend(_scan_patterns(path, lineno, line, _DECODER_INLINE, "decoder-artefact"))
    return findings


def _scan_heading(path: str, lineno: int, line: str) -> list[Finding]:
    heading_match = _HEADING_LINE.match(line)
    if not heading_match:
        return []
    heading_text = heading_match.group(2)
    return [
        Finding(path, lineno, "inference-heading", m.group(0), "do not add this section; keep only what was said")
        for m in _INFERENCE_HEADING.finditer(heading_text)
    ]


def check_text(path: str, text: str) -> list[Finding]:
    """Run all five rule groups over one document's text and return every finding, in order."""
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        allowlisted = _is_approved_boilerplate(line)
        if not allowlisted:
            findings.extend(_scan_patterns(path, lineno, line, _RECORDING_LEAK, "recording-leak"))
            findings.extend(_scan_patterns(path, lineno, line, _MACHINE_ORIGIN, "machine-origin"))
            findings.extend(_scan_decoder_artefact(path, lineno, line))
        findings.extend(_scan_heading(path, lineno, line))
        findings.extend(_scan_patterns(path, lineno, line, _INFERENCE_PHRASING, "inference-phrasing"))
    return findings


# ---------------------------------------------------------------------------
# File extraction. A file that cannot be read must be reported as unreadable, never as clean.
# ---------------------------------------------------------------------------


def _extract_pdf_text(path: Path) -> tuple[str | None, str | None]:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        return None, "no PDF text extractor available on this machine (install poppler's pdftotext)"
    proc = subprocess.run([pdftotext, str(path), "-"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None, f"pdftotext failed ({proc.returncode}): {proc.stderr.strip()}"
    return proc.stdout, None


def extract_text(path: Path) -> tuple[str | None, str | None]:
    """Return (text, error). text is None exactly when the file could not be checked at all."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace"), None
        except OSError as exc:
            return None, f"could not read file: {exc}"
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    return None, f"unsupported file type: {suffix or '(no extension)'}"


def process_file(path: Path) -> tuple[list[Finding], bool, str | None]:
    text, error = extract_text(path)
    if text is None:
        return [], False, error
    return check_text(str(path), text), True, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_human(file_results: list[dict[str, object]], total_findings: int) -> None:
    for result in file_results:
        path = result["path"]
        if not result["checked"]:
            print(f"{path}: SKIPPED -- {result['error']}")
            continue
        findings: list[Finding] = result["findings"]  # type: ignore[assignment]
        if not findings:
            print(f"{path}: clean")
            continue
        for finding in sorted(findings, key=lambda f: f.line):
            print(f"{finding.path}:{finding.line}: [{finding.rule}] {finding.match!r} -- {finding.message}")
    print()
    print(f"{total_findings} finding(s) across {len(file_results)} file(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="notes_check.py",
        description="Check a meeting-notes document against references/notes-register.md.",
    )
    parser.add_argument("files", metavar="FILE", nargs="+", help="Markdown, text, or PDF file(s) to check.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    parser.add_argument(
        "--strict", action="store_true", help="Also fail (exit 1) if any file could not be checked at all."
    )
    args = parser.parse_args(argv)

    file_results: list[dict[str, object]] = []
    total_findings = 0
    any_unreadable = False

    for raw_path in args.files:
        path = Path(raw_path)
        findings, checked, error = process_file(path)
        total_findings += len(findings)
        any_unreadable = any_unreadable or not checked
        file_results.append({"path": str(path), "findings": findings, "checked": checked, "error": error})

    if args.json:
        payload = {
            "files": [
                {
                    "path": r["path"],
                    "checked": r["checked"],
                    "findings": [f.to_dict() for f in r["findings"]],  # type: ignore[union-attr]
                }
                for r in file_results
            ],
            "total": total_findings,
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_human(file_results, total_findings)

    # Findings always fail the run -- that is the point of a CI check. --strict adds a second,
    # independent failure mode: a file this tool never managed to read must not pass silently.
    if total_findings > 0:
        return 1
    if args.strict and any_unreadable:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
