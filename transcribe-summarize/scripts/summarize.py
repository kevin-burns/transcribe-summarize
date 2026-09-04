#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Write meeting notes from a transcript -- the out-of-session fallback.

THIS IS NOT THE DEFAULT PATH. Inside Claude Code, the model reads the transcript
and writes the notes itself, following references/notes-register.md. That costs
no extra dependency and sends nothing anywhere new. This script exists for the
case where the pipeline runs outside a Claude Code session and there is no model
in the loop.

It therefore SENDS THE WHOLE TRANSCRIPT to a third party, which is a second,
separate egress from the transcription step and is disclosed as such. It never
runs unless you invoke it, and it never runs without saying what it will send.

THE RULES LIVE IN ONE FILE. references/notes-register.md is loaded verbatim and
becomes the system prompt. The same file is what SKILL.md tells Claude to read.
One source, so the in-session path and this one cannot drift apart -- which they
would within a month if the rules were paraphrased into a prompt string here.

    ./summarize.py transcript.md --backend groq          # writes transcript.notes.md
    ./summarize.py transcript.md --dry-run               # says what it would send
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import urllib.parse
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notes_check import check_text  # noqa: E402

REGISTER = Path(__file__).resolve().parents[1] / "references" / "notes-register.md"

PROVIDERS = {
    "groq": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
}

INSTRUCTION = """\
Write up the following meeting as notes, obeying every rule in the register above.

Produce Markdown with this header, then the notes under short topic headings, then
the approved closing line:

    # {title}
    **{date}**{attendees}

    Notes taken by the attendee. Not a verbatim record.

State only what was said. Do not add analysis, next steps, takeaways, or anything
about what was not covered. Do not mention a recording, a transcript, audio, or
that any of this was machine-produced.

{attribution}
"""

# Nothing in this pipeline can identify a speaker: no engine used here returns a
# speaker field. So attribution is supplied by a person or it is not made at all.
# Told to write up a conversation with no names, a model will happily invent
# "Sarah from finance" -- which in a document meant to carry weight is the worst
# failure available. These two paragraphs are the guard against it.
NAMED_ATTENDEES = """\
Attribute a statement to one of the named attendees ONLY where the meeting itself made
clear who was speaking. Where it is ambiguous, record what was said without a name.
Never guess which attendee said something."""

NO_ATTENDEES = """\
No attendees were supplied, and nothing in the source identifies a speaker. Write the
notes WITHOUT attribution -- impersonal throughout ("it was stated that...", "the
question was raised..."). Do not invent a name, a role, or a "Speaker 1" style label
for anyone."""


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_register() -> str:
    if not REGISTER.is_file():
        die(
            f"the notes register is missing: {REGISTER}\n"
            "It is the source of the content rules; refusing to guess them."
        )
    return REGISTER.read_text(encoding="utf-8")


def compose_instruction(title: str, date: str, attendees: list[str]) -> str:
    """Fill the header contract. Attribution guidance flips on whether names were given."""
    return INSTRUCTION.format(
        title=title,
        date=date,
        attendees=" — " + ", ".join(attendees) if attendees else "",
        attribution=NAMED_ATTENDEES if attendees else NO_ATTENDEES,
    )


def build_request(
    provider: str, model: str, register: str, transcript: str, instruction: str
) -> urllib.request.Request:
    """Build the call. Split out from send() so a test can inspect it without a network call."""
    config = PROVIDERS[provider]
    key = os.environ.get(config["env"])
    if not key:
        # Named, never echoed. The variable's name is safe to print; its value is not.
        die(f"{config['env']} is not set. Export it in your shell; it is never taken as a flag.")

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": register},
            {"role": "user", "content": f"{instruction}\n\n---\n\n{transcript}"},
        ],
    }
    # Host and path, not a URL: see tslib/backends/_openai_compatible.send for
    # why the transport is HTTPSConnection rather than urlopen.
    parts = urllib.parse.urlparse(config["endpoint"])
    if parts.scheme != "https":
        die(f"endpoint is not https (scheme {parts.scheme!r}); refusing to send")
    return (
        parts.netloc,
        parts.path,
        json.dumps(payload).encode("utf-8"),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )


def send(request: tuple[str, str, bytes, dict[str, str]], timeout: float = 180.0) -> str:
    """POST over HTTPS. `HTTPSConnection`, not `urlopen`, so no scheme exists to
    get wrong -- urlopen would open `file://` and there would be no plaintext
    fallback to leak the Authorization header. See
    tslib/backends/_openai_compatible.send for the full reasoning.
    """
    host, path, body, headers = request
    connection = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if response.status >= 400:
            # The provider's own message, never the request -- it carries the key.
            die(f"the provider returned HTTP {response.status}: {raw.decode('utf-8', 'replace')[:400]}")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, http.client.HTTPException) as exc:
        die(f"could not reach the provider: {exc}")
    finally:
        connection.close()
    return payload["choices"][0]["message"]["content"].strip()


def disclosure(provider: str, model: str, transcript: str, src: Path) -> str:
    config = PROVIDERS[provider]
    host = config["endpoint"].split("/")[2]
    words = len(transcript.split())
    return "\n".join([
        "",
        "  ── this will send the transcript text over the network ─────",
        f"  provider   : {provider}",
        f"  endpoint   : {host}",
        f"  file       : {src.name}",
        f"  sending    : ~{words:,} words of transcript",
        f"  model      : {model}",
        "  ────────────────────────────────────────────────────────────",
        "",
    ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write meeting notes from a transcript, using a network model. Opt-in, disclosed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Inside Claude Code, do not use this -- the model writes the notes directly. See SKILL.md.",
    )
    parser.add_argument("transcript", type=Path, help="the .md transcript to summarise")
    parser.add_argument("--backend", choices=sorted(PROVIDERS), default="groq")
    parser.add_argument("--model", default=None, help="override the provider's default chat model")
    parser.add_argument("--out", type=Path, default=None, help="output path (default: <transcript>.notes.md)")

    meeting = parser.add_argument_group("meeting facts (not inferable from audio -- supply them)")
    meeting.add_argument("--title", default=None, help="what the meeting was (default: the file's name)")
    meeting.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help="the date of the MEETING, not of processing. Defaults to today, which is often wrong.",
    )
    meeting.add_argument(
        "--attendee", action="append", default=[], metavar="NAME",
        help="who was present. Repeatable. No engine here can identify a speaker, so without "
             "these the notes are written with no attribution at all.",
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="say what would be sent; send nothing")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if not args.transcript.is_file():
        die(f"no such file: {args.transcript}")

    transcript = args.transcript.read_text(encoding="utf-8")
    register = load_register()
    model = args.model or PROVIDERS[args.backend]["model"]

    print(disclosure(args.backend, model, transcript, args.transcript), file=sys.stderr)
    if args.dry_run:
        print("dry run: nothing was sent.", file=sys.stderr)
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            die("cancelled -- not a terminal, and --yes was not given. Nothing was sent.")
        if input("  send it? [y/N] ").strip().lower() not in {"y", "yes"}:
            die("cancelled -- nothing was sent")

    title = args.title or args.transcript.stem.replace("-", " ").replace("_", " ").strip().capitalize()
    if args.date:
        meeting_date = args.date
    else:
        meeting_date = date.today().isoformat()
        print(
            f"  note: no --date given, using today ({meeting_date}). If the meeting was not today,\n"
            "        rerun with --date; a wrong date on a factual record is worse than none.",
            file=sys.stderr,
        )
    if not args.attendee:
        print(
            "  note: no --attendee given. Nothing here can identify a speaker, so the notes will\n"
            "        be written with no attribution rather than a guessed one.",
            file=sys.stderr,
        )

    instruction = compose_instruction(title, meeting_date, args.attendee)
    notes = send(build_request(args.backend, model, register, transcript, instruction))

    out = args.out or args.transcript.with_suffix(".notes.md")
    out.write_text(notes.rstrip() + "\n", encoding="utf-8")
    print(f"\nwrote {out.resolve()}", file=sys.stderr)

    # Generating notes and checking them is one action, not two. A model asked to
    # summarise will add analysis and cite its own source unprompted however well
    # the register is written, so the check runs here rather than on trust.
    findings = check_text(str(out), notes)
    if findings:
        print(f"\n{len(findings)} register violation(s) -- fix before this is used as a record:", file=sys.stderr)
        for finding in findings:
            print(f"  {out.name}:{finding.line}  [{finding.rule}]  {finding.matched}", file=sys.stderr)
        return 1
    print("register check: clean", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
