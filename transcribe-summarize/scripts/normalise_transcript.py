#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Turn a transcript you already have into the shape the rest of this skill uses.

THE CASE THIS EXISTS FOR. Most meetings are already transcribed by the platform.
Teams and Zoom hand you a `.vtt` (or an `.srt`), and it is a mess to read: one
cue every few seconds, sentences chopped across cues, a speaker tag repeated on
every line, and timecodes everywhere. There is no audio to transcribe and no
reason to re-transcribe anything -- but the summarising half of this skill is
exactly as useful on that text as on a transcript it produced itself.

    ./normalise_transcript.py meeting.vtt              # -> meeting.md
    ./normalise_transcript.py meeting.vtt --no-speakers

WHAT IT DOES, AND DELIBERATELY DOES NOT DO. It joins cues back into paragraphs,
drops the timecode furniture, collapses the rolling duplicates Teams emits, and
keeps one `[hh:mm:ss]` anchor per paragraph. It does NOT rewrite anyone's words,
fix grammar, or remove filler -- this is a format conversion, and the output is
still a transcript, not a summary. Deciding what was said is the summarising
step, and it has its own rules in references/notes-register.md.

SPEAKERS ARE KEPT HERE AND FORBIDDEN LATER, which is not a contradiction. A
transcript is a working artefact and may carry provenance; the notes document is
a record and may not. `<v A. Okonkwo>` becomes a readable attribution here, and
`notes_check.py` will still reject a raw `Speaker 2` label in the notes.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# A new paragraph when the speaker changes, or when the gap between cues is
# longer than this. Platform cues are a few seconds each, so a real pause is the
# only structural signal in the file.
PARAGRAPH_GAP = 2.5

_TIMECODE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
)
_VOICE = re.compile(r"<v\s+([^>]+)>(.*?)</v>", re.S)
_TAG = re.compile(r"<[^>]+>")
_SPEAKER_PREFIX = re.compile(r"^\s*([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})\s*:\s+")


@dataclass
class Cue:
    start: float
    end: float
    speaker: str | None
    text: str


def parse_timestamp(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        parts = ["0", *parts]
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def parse_cues(text: str) -> list[Cue]:
    """Read WebVTT or SRT. Both are 'timecode line, then text lines, then blank'."""
    cues: list[Cue] = []
    start: float | None = None
    end: float = 0.0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal start, buffer
        if start is not None and buffer:
            raw = " ".join(buffer).strip()
            speaker = None
            voices = _VOICE.findall(raw)
            if voices:
                speaker = voices[0][0].strip()
                raw = " ".join(part[1] for part in voices)
            raw = html.unescape(_TAG.sub("", raw)).strip()
            if speaker is None and (match := _SPEAKER_PREFIX.match(raw)):
                speaker = match.group(1).strip()
                raw = raw[match.end():]
            if raw:
                cues.append(Cue(start, end, speaker, raw))
        start, buffer = None, []

    for line in text.splitlines():
        stripped = line.strip()
        if (match := _TIMECODE.search(stripped)) is not None:
            flush()
            start = parse_timestamp(match.group("start"))
            end = parse_timestamp(match.group("end"))
            continue
        if not stripped:
            flush()
            continue
        # Cue numbers (SRT) and the WEBVTT header carry nothing.
        if stripped.isdigit() or stripped.upper().startswith("WEBVTT") or stripped.startswith("NOTE "):
            continue
        buffer.append(stripped)
    flush()
    return cues


def dedupe(cues: list[Cue]) -> list[Cue]:
    """Drop rolling-caption repeats.

    Teams re-emits a growing cue as someone speaks, so the same words arrive
    several times with the tail extended. Keep the longest of a run where each
    cue starts with the previous one's text.
    """
    kept: list[Cue] = []
    for cue in cues:
        if kept and cue.speaker == kept[-1].speaker:
            previous = kept[-1].text
            if cue.text.startswith(previous) or previous.startswith(cue.text):
                if len(cue.text) > len(previous):
                    kept[-1] = cue
                continue
        kept.append(cue)
    return kept


def to_paragraphs(cues: list[Cue], keep_speakers: bool) -> list[tuple[float, str | None, str]]:
    groups: list[list[Cue]] = []
    for cue in cues:
        if not groups:
            groups.append([cue])
            continue
        previous = groups[-1][-1]
        speaker_changed = keep_speakers and cue.speaker != previous.speaker
        # Gap from the previous cue's END. Measuring from its start measures the
        # cue's duration instead, which splits every sentence a platform wrapped
        # across two cues -- the exact mess this script exists to undo.
        if speaker_changed or (cue.start - previous.end) > PARAGRAPH_GAP:
            groups.append([cue])
        else:
            groups[-1].append(cue)

    paragraphs = []
    for group in groups:
        text = " ".join(cue.text for cue in group)
        text = re.sub(r"\s+", " ", text).strip()
        paragraphs.append((group[0].start, group[0].speaker if keep_speakers else None, text))
    return paragraphs


def render(paragraphs, source_name: str, speakers: list[str]) -> str:
    lines = [f"# Transcript — {source_name}", ""]
    if paragraphs:
        lines.append(f"- Covers: {clock(paragraphs[0][0])} to {clock(paragraphs[-1][0])}")
    lines.append("- Source: an existing transcript, reformatted. Nothing was re-transcribed.")
    if speakers:
        lines.append(f"- Speakers named in the source: {', '.join(speakers)}")
    lines.append("")
    for start, speaker, text in paragraphs:
        who = f"**{speaker}:** " if speaker else ""
        lines.append(f"**[{clock(start)}]** {who}{text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reformat an existing .vtt/.srt transcript into readable Markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Format conversion only -- no words are changed, nothing is summarised.",
    )
    parser.add_argument("transcript", type=Path, help="a .vtt, .srt or similar cue file")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: <name>.md)")
    parser.add_argument("--no-speakers", action="store_true", help="drop speaker attribution")
    args = parser.parse_args()

    if not args.transcript.is_file():
        print(f"error: no such file: {args.transcript}", file=sys.stderr)
        return 1

    cues = dedupe(parse_cues(args.transcript.read_text(encoding="utf-8", errors="replace")))
    if not cues:
        print(
            f"error: no cues found in {args.transcript.name}.\n"
            "Expected WebVTT or SRT (timecode line, text, blank line).",
            file=sys.stderr,
        )
        return 1

    keep_speakers = not args.no_speakers
    seen: list[str] = []
    for cue in cues:
        if cue.speaker and cue.speaker not in seen:
            seen.append(cue.speaker)

    paragraphs = to_paragraphs(cues, keep_speakers)
    out = args.out or args.transcript.with_suffix(".md")
    out.write_text(render(paragraphs, args.transcript.name, seen if keep_speakers else []), encoding="utf-8")

    print(f"{len(cues)} cues -> {len(paragraphs)} paragraphs", file=sys.stderr)
    print(f"wrote {out.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
