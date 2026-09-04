#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Transcribe audio locally, on macOS, Windows or Linux.

Local is the default and the point. A network backend exists, but it is reachable
only by naming it in the invocation -- never as a default, never as a fallback
after a local backend fails, and never without saying what is about to be sent
first. The failure this tool exists to avoid is not "used an API"; it is
"uploaded the audio and the user could not tell".

    ./transcribe.py meeting.m4a
    ./transcribe.py meeting.m4a --backend faster-whisper --model turbo
    ./transcribe.py meeting.m4a --replace 'north wind=Northwind' --replace 'EKS=EKS'
    ./transcribe.py meeting.m4a --backend groq --dry-run     # says what it would send

NO DEPENDENCIES ARE DECLARED ABOVE, on purpose. The core -- audio preparation,
the quality guard, corrections and document rendering -- is stdlib. A backend
brings its own library and only when selected, so installing this skill does not
drag in every engine. Run it with `uv run --with '<spec>' --script ...`; if the
selected backend's library is missing, the error names the exact command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tslib import audio, backends, corrections, quality, render  # noqa: E402
from tslib.types import Result  # noqa: E402


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def note(msg: str = "") -> None:
    print(msg, file=sys.stderr)


def disclose_egress(
    info: backends.BackendInfo, src: Path, model: str, duration: float, *, prepared_first: bool = True
) -> str:
    """State exactly what is about to leave the machine, before any of it does.

    Duration comes from ffprobe on the original file, not from the decoder's last
    segment -- those differed by 15 s on a recording with trailing silence, and
    this is the number the provider bills against.
    """
    endpoint = getattr(backends.load(info), "ENDPOINT", f"the {info.name} API")
    host = endpoint.split("/")[2] if "//" in endpoint else endpoint
    size_mb = src.stat().st_size / 1_048_576
    cost = backends.estimate_cost(info.name, model, duration)

    lines = [
        "",
        "  ── this will send your audio over the network ──────────────",
        f"  provider : {info.name}",
        f"  endpoint : {host}",
        f"  file     : {src.name}  ({size_mb:.1f} MB)",
        f"  duration : {render.clock_hms(duration)}",
        f"  model    : {model}",
        f"  cost     : {f'up to ~${cost:.3f}' if cost is not None else 'unknown for this model'}",
    ]
    if prepared_first:
        # The figures above are the WHOLE file, because this block has to print
        # before any work happens. Silence trimming runs first and only the kept
        # audio is uploaded, so the real duration and bill are usually lower --
        # measured at 47s -> 12s on a call with a silent head. Say so rather than
        # quoting a number the invoice will not match.
        lines.append("  ────────────────────────────────────────────────────────────")
        lines.append("  silence is trimmed before sending, so the amount actually")
        lines.append("  uploaded and billed is usually less than shown above.")
    lines += ["  ────────────────────────────────────────────────────────────", ""]
    return "\n".join(lines)


def confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        # Non-interactive and unconfirmed is a refusal, not an assumption. A skill
        # running unattended must not upload because nobody was there to say no.
        return False
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio locally. Network backends are opt-in and disclosed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Writes <name>.md beside the audio, with .srt and .json in <name>-artifacts/.\n"
            "Backends: " + ", ".join(sorted(backends.REGISTRY)) + "  (default: auto)"
        ),
    )
    parser.add_argument("audio", type=Path, help="audio file (mp3, m4a, wav, ...)")
    parser.add_argument(
        "--backend", default="auto",
        help="auto (local, platform-appropriate), or name one. groq/openai send audio over the network.",
    )
    parser.add_argument("--model", default=None, help="engine-specific model name (default: the backend's)")
    parser.add_argument("--lang", default="en", help="ISO language code, or 'auto' to detect (default: en)")
    parser.add_argument("--prompt", default=None, help="names/jargon to bias the decoder toward")
    parser.add_argument(
        "--replace", action="append", default=[], metavar="WRONG=RIGHT",
        help="fix a known misrecognition after decoding. Repeatable.",
    )
    parser.add_argument("--outdir", type=Path, default=None, help="output directory (default: beside the audio)")

    prep = parser.add_argument_group("audio preparation (on by default -- it beat model choice, measured)")
    prep.add_argument("--no-normalise", action="store_true", help="skip loudness normalisation")
    prep.add_argument("--no-trim", action="store_true", help="skip silence trimming")
    prep.add_argument("--silence-threshold", type=float, default=audio.DEFAULT_SILENCE_DB, metavar="DB")
    prep.add_argument("--min-silence", type=float, default=audio.DEFAULT_MIN_SILENCE, metavar="SECONDS")
    prep.add_argument("--keep-intermediate", action="store_true", help="keep the prepared wav for inspection")

    guard = parser.add_argument_group("quality guard (catches Whisper's silence hallucinations)")
    guard.add_argument("--guard", choices=["drop", "mark", "off"], default="drop")

    parser.add_argument("--yes", action="store_true", help="skip the network confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="say what would happen; decode nothing, send nothing")
    parser.add_argument("--quiet", action="store_true", help="do not stream segments as they decode")
    parser.add_argument("--json", action="store_true", help="print a machine-readable summary to stdout")
    return parser.parse_args(argv)


def check_prerequisites(args: argparse.Namespace) -> None:
    if not args.audio.is_file():
        die(f"no such file: {args.audio}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        die(
            "ffmpeg and ffprobe are required.\n"
            "  macOS   brew install ffmpeg\n"
            "  Windows winget install Gyan.FFmpeg\n"
            "  Linux   apt install ffmpeg"
        )


def _dry_run_summary(info: backends.BackendInfo, model: str, duration: float, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"dry_run": True, "backend": info.name, "model": model,
                          "duration": round(duration, 2), "sent": False}, indent=2))


def egress_gate(args: argparse.Namespace, info: backends.BackendInfo, model: str, duration: float) -> bool:
    """Decide whether to proceed. Returns False when the caller should stop.

    Kept as its own function precisely BECAUSE it is the privacy boundary: the
    one place that can let audio leave the machine should be small enough to
    read in full, and testable without running a decode.
    """
    if info.kind == "network":
        note(disclose_egress(info, args.audio, model, duration,
                             prepared_first=not (args.no_trim and args.no_normalise)))
        if args.dry_run:
            note("dry run: nothing was sent.")
            _dry_run_summary(info, model, duration, args.json)
            return False
        if not args.yes and not confirm("  send it? [y/N] "):
            die("cancelled -- nothing was sent")
    elif args.dry_run:
        note(f"dry run: would decode locally with {info.name} / {model}, nothing sent.")
        _dry_run_summary(info, model, duration, args.json)
        return False
    return True


def write_outputs(
    args: argparse.Namespace,
    info: backends.BackendInfo,
    model: str,
    result: Result,
    prepared: audio.PreparedAudio,
    report: quality.GuardReport,
    hits: int,
    elapsed: float,
) -> dict[str, Path]:
    """Write the four output files and return where each one landed.

    Documents go in the output directory; machine artefacts go one level down in
    `<stem>-artifacts/`, so the readable deliverables are not buried among
    sidecars. The run manifest deliberately records the guard's tally and the
    trim offsets: a suppressed segment and a shifted timestamp both need to be
    answerable later without re-running anything. It never contains an API key.
    """
    outdir = args.outdir or args.audio.parent
    stem = args.audio.stem
    artifacts = outdir / f"{stem}-artifacts"
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    paths = {
        "transcript": outdir / f"{stem}.md",
        "srt": artifacts / f"{stem}.srt",
        "json": artifacts / f"{stem}.json",
        "manifest": artifacts / f"{stem}.run.json",
    }

    render.write_markdown_transcript(result, paths["transcript"], {
        "source_name": args.audio.name,
        "original_duration": prepared.original_duration,
        "prepared_duration": prepared.prepared_duration,
        "trimmed": prepared.trimmed,
        "guard_line": report.summary_line(),
        "review": quality.review_candidates(result["segments"]),
    })
    render.write_srt(result["segments"], paths["srt"])
    render.write_json(result, paths["json"])
    render.write_manifest(paths["manifest"], {
        "source": str(args.audio.resolve()),
        "backend": info.name,
        "model": model,
        "kind": info.kind,
        "language": result.get("language"),
        "has_whisper_metrics": info.has_whisper_metrics,
        "decode_seconds": round(elapsed, 2),
        "replacements_applied": hits,
        "guard": {
            "mode": args.guard,
            "checked": report.checked,
            "suppressed": report.suppressed,
            "reasons": report.reasons,
            "metrics_available": report.metrics_available,
        },
        "audio": prepared.as_json(),
    })
    return paths


def main() -> int:
    args = parse_args()
    check_prerequisites(args)

    try:
        info = backends.resolve(args.backend)
    except backends.UnknownBackend as exc:
        die(str(exc))

    model = args.model or info.default_model

    try:
        # Before the egress gate, not after: a file with no audio track would
        # otherwise be uploaded and billed, or fail locally with a raw ffmpeg error.
        audio.assert_has_audio(args.audio)
        duration = audio.probe_duration(args.audio)
    except audio.AudioError as exc:
        die(str(exc))

    if not egress_gate(args, info, model, duration):
        return 0

    try:
        module = backends.load(info)
    except backends.MissingDependency as exc:
        die(str(exc))

    if not info.has_whisper_metrics and args.guard != "off":
        note(
            f"note: {info.name} does not return Whisper's quality metrics, so the guard runs with\n"
            "      its repetition rule only. Silence hallucinations that are not repetitive will\n"
            "      not be caught. See references/backends.md."
        )

    note(f"engine : {info.name} / {model}")
    note(f"audio  : {args.audio}  ({render.clock_hms(duration)})")

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            prepared = audio.prepare(
                args.audio, work,
                do_normalise=not args.no_normalise,
                do_trim=not args.no_trim,
                threshold_db=args.silence_threshold,
                min_silence=args.min_silence,
            )
        except audio.AudioError as exc:
            die(str(exc))

        if prepared.trimmed:
            saved = prepared.original_duration - prepared.prepared_duration
            note(
                f"prep   : normalised and trimmed {render.clock_hms(saved)} of silence "
                f"({render.clock_hms(prepared.prepared_duration)} to decode)"
            )
        elif not args.no_trim:
            note("prep   : normalised; no silence long enough to trim")

        if args.keep_intermediate:
            kept_dir = (args.outdir or args.audio.parent) / f"{args.audio.stem}-artifacts"
            kept_dir.mkdir(parents=True, exist_ok=True)
            kept_wav = kept_dir / f"{args.audio.stem}.prepared.wav"
            shutil.copy2(prepared.wav, kept_wav)
            note(f"prep   : kept the prepared audio at {kept_wav}")

        count = 0

        def progress(segment) -> None:
            nonlocal count
            count += 1
            if not args.quiet:
                stamp = render.clock_hms(prepared.clock.to_original(segment["start"]))
                print(f"  [{stamp}] {segment.get('text', '').strip()}", file=sys.stderr)

        try:
            result: Result = module.transcribe(
                prepared.wav,
                model=model,
                language=None if args.lang == "auto" else args.lang,
                prompt=args.prompt,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001 -- backend libraries raise their own types
            die(f"{info.name} failed to decode: {exc}")

        elapsed = time.monotonic() - started

    # ----------------------------------------------------- corrections, then guard
    hits = 0
    if args.replace:
        try:
            hits = corrections.apply(result, corrections.build(args.replace))
        except corrections.ReplacementError as exc:
            die(str(exc))
        note(f"fixed  : {hits} correction(s) applied from --replace")

    # The clock rewrite comes BEFORE the guard, not after. Two reasons: the .srt
    # would otherwise drift against the user's own file by exactly the length of
    # the silence removed, and the guard's silence-overlap rule compares segment
    # times against ffmpeg's measured silence spans, which are on the original clock.
    render.map_to_original(result, prepared.clock)

    report = quality.apply_guard(result, mode=args.guard, silences=prepared.silences)
    note(report.summary_line())

    paths = write_outputs(args, info, model, result, prepared, report, hits, elapsed)

    note(f"\ndecoded {render.clock_hms(prepared.prepared_duration)} in {elapsed:.0f}s")
    for path in paths.values():
        note(f"  {path}")

    if args.json:
        print(json.dumps({
            **{key: str(path.resolve()) for key, path in paths.items()},
            "backend": info.name,
            "model": model,
            "duration": round(prepared.original_duration, 2),
            "suppressed": report.suppressed,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
