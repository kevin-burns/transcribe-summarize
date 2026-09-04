"""Writing the outputs, and putting every timestamp back on the original clock.

Nothing here is written until `map_to_original()` has run. A backend reports
timestamps against the trimmed audio; a user compares them against the file on
their own disk. Those are different clocks, and the difference is exactly as long
as the silence that was cut out.

Two documents, two different rules about provenance:

  the TRANSCRIPT (.md)  is a working artefact. It says which engine produced it,
                        how long the audio was and what the guard suppressed,
                        because that is what makes it auditable.
  the NOTES (.notes.md) is a record. It says none of that. See
                        references/notes-register.md -- and note that this module
                        never writes the notes; it only writes the transcript.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .audio import ClockMap
from .types import Result, Segment

# Start a new paragraph when the speaker pauses this long. Whisper segments are
# breath-length, so one segment per line reads like a chat log rather than a
# transcript; grouping on a real pause is closer to where a person would break.
PARAGRAPH_GAP = 2.0
PARAGRAPH_MAX_SECONDS = 45.0


def map_to_original(result: Result, clock: ClockMap) -> None:
    """Rewrite every timestamp in place, trimmed clock -> original recording."""
    if clock.identity:
        return
    for segment in result["segments"]:
        segment["start"] = clock.to_original(segment["start"])
        segment["end"] = clock.to_original(segment["end"])
        for word in segment.get("words") or []:
            if "start" in word:
                word["start"] = clock.to_original(word["start"])
            if "end" in word:
                word["end"] = clock.to_original(word["end"])


def clock_hms(seconds: float) -> str:
    """[hh:mm:ss] for the reader. Hours are kept even at zero so the column lines up."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def srt_timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000.0))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def kept(segments: Iterable[Segment]) -> list[Segment]:
    """Segments the guard did not suppress. The .json keeps the rest; documents do not."""
    return [s for s in segments if not s.get("suppressed")]


def write_srt(segments: Iterable[Segment], dest: Path) -> Path:
    blocks = []
    for i, seg in enumerate(kept(segments), start=1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        blocks.append(f"{i}\n{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}\n{text}\n")
    dest.write_text("\n".join(blocks), encoding="utf-8")
    return dest


def _paragraphs(segments: list[Segment]) -> list[tuple[float, str]]:
    """Group segments into (start_time, text) paragraphs on pauses."""
    groups: list[tuple[float, list[str]]] = []
    last_end: float | None = None
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        gap = seg["start"] - last_end if last_end is not None else None
        too_long = groups and (seg["end"] - groups[-1][0]) > PARAGRAPH_MAX_SECONDS
        if not groups or (gap is not None and gap > PARAGRAPH_GAP) or too_long:
            groups.append((seg["start"], [text]))
        else:
            groups[-1][1].append(text)
        last_end = seg["end"]
    return [(start, " ".join(parts)) for start, parts in groups]


def write_markdown_transcript(result: Result, dest: Path, meta: dict[str, Any]) -> Path:
    """The readable transcript, with original-clock anchors and a provenance header.

    Provenance belongs here. This document exists to be checked against the audio,
    so hiding which engine produced it would only make it harder to trust.
    """
    segments = kept(result["segments"])
    lines: list[str] = [f"# Transcript — {meta.get('source_name', dest.stem)}", ""]

    facts = [
        f"- Duration: {clock_hms(meta['original_duration'])}",
        f"- Engine: {result.get('backend', 'unknown')} / {result.get('model', 'unknown')}",
    ]
    if result.get("language"):
        facts.append(f"- Language: {result['language']}")
    if meta.get("guard_line"):
        facts.append(f"- Quality guard: {meta['guard_line']}")
    if meta.get("prepared_duration") and meta.get("trimmed"):
        facts.append(
            f"- Audio prepared: normalised and silence-trimmed to "
            f"{clock_hms(meta['prepared_duration'])} before decoding; "
            "timestamps below are on the original recording's clock"
        )
    lines.extend(facts)
    lines.append("")

    if not segments:
        lines.append("_No speech was retained after the quality guard ran._")
    for start, text in _paragraphs(segments):
        lines.append(f"**[{clock_hms(start)}]** {text}")
        lines.append("")

    review = meta.get("review") or []
    if review:
        lines.append("---")
        lines.append("")
        lines.append("## Worth checking")
        lines.append("")
        lines.append(
            "_This transcript is a draft. Machine transcription is confidently wrong "
            "sometimes — a figure misheard, a speaker's correction dropped — and no "
            "threshold catches that, because the output reads fluently either way. "
            "The timestamps below are where a listen-back is most likely to pay off; "
            "they are ranked hints, not errors._"
        )
        lines.append("")
        for segment, reason in review:
            lines.append(f"- **[{clock_hms(segment['start'])}]** {reason} — {segment.get('text', '').strip()}")
        lines.append("")

    suppressed = [s for s in result["segments"] if s.get("suppressed")]
    if suppressed:
        lines.append("---")
        lines.append("")
        lines.append(
            f"_{len(suppressed)} segment(s) were suppressed by the quality guard and are not shown "
            "above. They are kept, with the metrics that rejected them, in the `.json` sidecar._"
        )
        lines.append("")

    dest.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return dest


def write_json(result: Result, dest: Path) -> Path:
    """Everything, including what the guard rejected and why. The audit trail."""
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def write_manifest(dest: Path, manifest: dict[str, Any]) -> Path:
    """How this run was produced. Never contains an API key -- see tslib.backends."""
    dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest
