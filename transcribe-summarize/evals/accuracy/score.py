#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Score a transcript against the written-down accuracy checks.

WHY THIS EXISTS. README.md carried a table of per-backend `N / 12` accuracy
scores whose checklist was never written down, so no score in it could be
recomputed, confirmed or refuted. That is the defect claude-skills-luwe was filed
for. The checks now live in `checks.json` next to this file and the score is a
function of them.

The set turned out to be FOURTEEN, not twelve. The old figure was a sentence
describing the checks in prose -- "technical proper nouns, an invented surname,
four figures and a date, two spoken self-corrections, and one near-homophone
pair" -- which does not add up to twelve however it is grouped. Twelve is not
preserved here, because writing the set down and then trimming it to match a
number nobody derived would defeat the point.

The audio and the transcripts live OUTSIDE this repository and stay there. This
script takes a path.

    ./score.py /path/to/transcript.md
    ./score.py /path/to/eval-dir --json      # every *.md below it, one row each
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CHECKS = Path(__file__).with_name("checks.json")


def load_checks() -> list[dict]:
    return json.loads(CHECKS.read_text())["checks"]


def transcript_body(path: Path) -> str:
    """The spoken text only.

    The 'Worth checking' section quotes segments back verbatim, so scoring the
    whole file would let a term count twice -- once where it was said and once
    where the tool echoed it. A backend that got a term wrong would still fail,
    but one that got it right would be scored against a doubled sample. Cut it.
    """
    text = path.read_text()
    marker = "\n## Worth checking"
    return text[: text.index(marker)] if marker in text else text


def score(body: str, checks: list[dict]) -> dict:
    results = []
    for check in checks:
        found = re.search(check["expect"], body, re.I) is not None
        rejected = None
        if check.get("reject"):
            rejected = re.search(check["reject"], body, re.I) is not None
        # A reject hit is a FAILURE even if expect also matched: producing both
        # "board pack" and "board packs up" means the wrong one is in there.
        passed = found and not rejected
        results.append({
            "id": check["id"], "kind": check["kind"], "passed": passed,
            "found": found, "wrong_form_present": rejected,
        })
    return {
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "failed": [r["id"] for r in results if not r["passed"]],
        "checks": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path, help="a transcript .md, or a directory to walk")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    checks = load_checks()
    if args.target.is_dir():
        # Skip the notes documents: they are a summary and are not supposed to
        # contain every figure. Scoring one would measure the wrong artefact.
        paths = sorted(p for p in args.target.rglob("*.md") if not p.name.endswith(".notes.md"))
    else:
        paths = [args.target]
    if not paths:
        print(f"error: no transcript found at {args.target}", file=sys.stderr)
        return 1

    rows = []
    for path in paths:
        result = score(transcript_body(path), checks)
        result["transcript"] = str(path)
        rows.append(result)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    width = max(len(str(r["transcript"])) for r in rows)
    for row in rows:
        failed = ", ".join(row["failed"]) or "-"
        print(f"{row['transcript']:<{width}}  {row['passed']:>2}/{row['total']}  {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
