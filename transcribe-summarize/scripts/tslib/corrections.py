"""The --replace pass: deterministic post-decode corrections.

A deliberate alternative to letting the decoder carry vocabulary across
windows (which invites hallucinated repetition, see quality.py): the decode
stays independent per window, and known terms are corrected afterwards.

Fixes two verified bugs in the prior art (scripts/transcribe.py, see
the prior art):

BUG 3 -- hit count inflated ~3x. The old apply_replacements() summed matches
across result['text'], EACH segment's text, AND EACH word -- one real
correction counted three times. Fixed here by counting matches ONLY over
segment text, once each; result['text'] is then rebuilt by joining segments
rather than substituted (and counted) separately.

BUG 4 -- chained rules were not isolated. Applying rule 1 then rule 2 as two
separate passes let rule 2 re-match text rule 1 had just produced (--replace
'a=b' --replace 'b=c' turned "a" into "c"). Fixed here by compiling every pair
into ONE alternation and running ONE `re.sub` pass per string: Python's `re.sub`
finds all matches against the *original* string before substituting, so a
replaced span is never re-offered to another alternative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Result


class ReplacementError(ValueError):
    """A malformed --replace pair ('wrong=right' expected)."""


@dataclass(frozen=True)
class Rules:
    """One compiled alternation covering every pair, plus its replacement map.

    `pattern` has one named group per pair ("g0", "g1", ...), longest search
    term first (see build()). `replacements` maps each group name to the
    right-hand side, verbatim -- matching is case-insensitive but the
    replacement casing is exactly what the user gave.
    """

    pattern: re.Pattern[str]
    replacements: dict[str, str]


def build(pairs: list[str]) -> Rules:
    """Compile 'wrong=right' pairs into one word-bounded, case-insensitive Rules."""
    entries: list[tuple[str, str]] = []
    for pair in pairs:
        wrong, sep, right = pair.partition("=")
        if not sep or not wrong.strip():
            raise ReplacementError(f"--replace expects 'wrong=right', got: {pair!r}")
        entries.append((wrong.strip(), right.strip()))

    if not entries:
        # No rules at all: a pattern that can never match, rather than special-
        # casing an empty alternation (which is not valid regex syntax).
        return Rules(pattern=re.compile(r"(?!)"), replacements={})

    # Longer terms first: Python's re picks the first alternative that matches
    # at a given position, not the longest one (unlike POSIX) -- so "New York
    # City" must be OFFERED before "New York" or it can never win.
    entries.sort(key=lambda entry: len(entry[0]), reverse=True)

    parts = []
    replacements: dict[str, str] = {}
    for i, (wrong, right) in enumerate(entries):
        name = f"g{i}"
        # \b only means anything next to a word character. Anchoring a phrase
        # that starts or ends on punctuation ("cloud. held.") would never
        # match if both boundaries were unconditional.
        lead = r"\b" if wrong[:1].isalnum() else ""
        trail = r"\b" if wrong[-1:].isalnum() else ""
        parts.append(f"(?P<{name}>{lead}{re.escape(wrong)}{trail})")
        replacements[name] = right

    pattern = re.compile("|".join(parts), re.IGNORECASE)
    return Rules(pattern=pattern, replacements=replacements)


def apply(result: Result, rules: Rules) -> int:
    """Rewrite matched terms across `result`'s segments. Returns the hit count.

    Hits are counted once, over segment text only (BUG 3). Word entries get
    the same substitution so timestamps stay aligned with the corrected text,
    but those substitutions are not counted. result['text'] is rebuilt from
    the corrected segments rather than substituted on its own.
    """

    def _sub(match: re.Match[str]) -> str:
        return rules.replacements[match.lastgroup]

    hits = 0
    segments = result.get("segments", [])
    for segment in segments:
        text, count = rules.pattern.subn(_sub, segment.get("text", ""))
        segment["text"] = text
        hits += count

        for word in segment.get("words", []):
            word["word"] = rules.pattern.sub(_sub, word.get("word", ""))

    result["text"] = " ".join(seg.get("text", "").strip() for seg in segments).strip()
    return hits
