"""The hallucination guard.

Measured on a real recording (see README.md, "Why this exists"): ~40s of joining silence
at -69 dB, against -30 dB for speech, made `mlx-whisper` large-v3 emit an
18-second segment of ONE WORD REPEATED 55 TIMES (`compression_ratio` 17.38,
against Whisper's own reject threshold of 2.4) and a spurious "Thank you."
(`no_speech_prob` 0.923). Neither was filtered before this module existed --
both were written to the transcript as if real.

Whisper's own thresholds (`compression_ratio_threshold=2.4`,
`no_speech_threshold=0.6`, `logprob_threshold=-1.0` in `mlx_whisper/transcribe.py`
and upstream `openai-whisper`) are decode-time hints the library does not
enforce on its own output -- they steer resampling, but a segment that fails
them is still written. This module is the enforcement Whisper itself skips.

METRICS. `compression_ratio`, `no_speech_prob` and `avg_logprob` are Whisper
decoder metrics -- present on Whisper-family backends, absent (`None`) on
Parakeet (CTC/TDT, no decoder logprobs). `None` means "this backend cannot tell
you", never "this segment is fine": rules 1-3 below each check `is not None`
before comparing, and a backend with no metrics at all falls through to rule 4,
the metric-free repetition fallback -- the only rule that can catch Parakeet's
version of the same 55x-word-loop failure.
"""

from __future__ import annotations

import string
from collections import Counter
from dataclasses import dataclass, field

from .types import GuardMode, Result, Segment, segment_metrics

# Whisper's own reject threshold (mlx_whisper/transcribe.py,
# compression_ratio_threshold). The 55x word loop measured 17.38 -- 7x over.
DEFAULT_COMPRESSION_RATIO = 2.4

# Whisper's own no_speech_threshold.
DEFAULT_NO_SPEECH = 0.6

# Whisper's own logprob_threshold.
DEFAULT_LOGPROB = -1.0

# Metric-free fallback (rule 4, "repeated_token"): a token repeated more than
# this many times, and dominating the segment, is treated as a hallucinated
# loop even with no decoder metrics at all. 6 leaves room for a real repeated
# word ("no no no no no no", 6 reps) while still catching the 55x case this
# guard exists for.
DEFAULT_MAX_REPEATS = 6

# Canonical rule order, used to render GuardReport.summary_line deterministically
# regardless of which segments were checked in which order.
_REASON_ORDER = ("decoded_from_silence", "repetition", "silence", "low_confidence", "repeated_token")


@dataclass(frozen=True)
class Thresholds:
    """The guard's four tunables. Defaults are Whisper's own, not chosen by taste."""

    compression_ratio: float = DEFAULT_COMPRESSION_RATIO
    no_speech: float = DEFAULT_NO_SPEECH
    logprob: float = DEFAULT_LOGPROB
    max_repeats: int = DEFAULT_MAX_REPEATS


@dataclass
class GuardReport:
    checked: int
    suppressed: int
    reasons: dict[str, int] = field(default_factory=dict)
    # True iff at least one checked segment carried a real (non-None)
    # compression_ratio or no_speech_prob -- i.e. this was a Whisper-family
    # backend, not one (Parakeet) that only ever reaches rule 4.
    metrics_available: bool = False

    def summary_line(self) -> str:
        if self.suppressed == 0:
            return f"guard: 0 of {self.checked} segments suppressed"
        # Ordered by _REASON_ORDER, then anything not listed there, so a rule added
        # without updating that tuple still shows up. It went missing once already.
        listed = [f"{label} {self.reasons[label]}" for label in _REASON_ORDER if self.reasons.get(label)]
        extra = [f"{label} {count}" for label, count in sorted(self.reasons.items())
                 if label not in _REASON_ORDER and count]
        parts = listed + extra
        return f"guard: {self.suppressed} of {self.checked} segments suppressed ({', '.join(parts)})"


def _repeated_token_reason(text: str, thresholds: Thresholds) -> str | None:
    """Rule 4: metric-free fallback, the only rule Parakeet segments can hit.

    Strips surrounding punctuation per token so "okay," and "okay" count as the
    same token. Segments under 4 tokens are exempt -- a legitimate "yeah yeah
    yeah" is not a hallucination.
    """
    tokens = [tok.strip(string.punctuation) for tok in text.lower().split()]
    if len(tokens) < 4:
        return None

    counts = Counter(tok for tok in tokens if len(tok) >= 2)
    if not counts:
        return None

    _token, count = counts.most_common(1)[0]
    if count > thresholds.max_repeats and count > len(tokens) / 2:
        return "repeated_token"
    return None


def inspect(segment: Segment, thresholds: Thresholds | None = None) -> str | None:
    """Return a reason label if `segment` should be suppressed, else None.

    Rules run in order, first match wins. A metric that is None is a missing
    reading, never a passing one -- rules 1-3 gate on `is not None` before
    comparing, so a backend that returns no metrics falls through to rule 4
    rather than being waved through by default.
    """
    thresholds = thresholds if thresholds is not None else Thresholds()
    metrics = segment_metrics(segment)
    compression_ratio = metrics["compression_ratio"]
    no_speech_prob = metrics["no_speech_prob"]
    avg_logprob = metrics["avg_logprob"]

    # 1. repetition -- the 17.38-vs-2.4 case this guard was built for.
    if compression_ratio is not None and compression_ratio > thresholds.compression_ratio:
        return "repetition"

    # 2. silence -- Whisper's own two-part rule: high no_speech_prob ALONE is
    # not enough (confident quiet speech can trip it), so avg_logprob must also
    # be low before this counts as silence.
    if (
        no_speech_prob is not None
        and no_speech_prob > thresholds.no_speech
        and avg_logprob is not None
        and avg_logprob < thresholds.logprob
    ):
        return "silence"

    # 3. low_confidence -- a backend with partial metrics (avg_logprob but no
    # compression_ratio). Once compression_ratio is present, rule 1 is the
    # authority on that segment; this rule only fires when it's absent.
    if avg_logprob is not None and avg_logprob < thresholds.logprob and compression_ratio is None:
        return "low_confidence"

    # 4. repeated_token -- metric-free fallback, reached by every backend,
    # including one (Parakeet) that never populates the three metrics above.
    return _repeated_token_reason(segment.get("text", ""), thresholds)


# How far a segment may spill out of a silence span and still count as decoded
# from it. This absorbs ordinary boundary jitter between ffmpeg's frame-level
# detection and a decoder's segment times. Nothing more.
#
# WHAT THIS NUMBER IS NOT: it is not what protects against a badly overrunning
# segment. A segment that runs seconds past the end of speech fails the
# containment test at ANY tolerance, because it starts before the silence does.
# Containment does that work; the tolerance only stops a few hundred
# milliseconds of disagreement from letting a genuine invention through. Worth
# separating, because conflating the two would be a wrong claim.
#
# 0.35 s is chosen empirically and sits inside the range other tools use for the
# comparable job of padding around VAD speech boundaries -- verified in source:
#   faster-whisper / Silero  speech_pad_ms = 400   (faster_whisper/vad.py:46)
#   whisper.cpp native VAD   speech_pad_ms = 30    (src/whisper.cpp:4557)
# No published figure exists for this specific use, so this is an analogy, not a
# citation. If it ever matters, measure it on a real corpus.
SILENCE_TOLERANCE = 0.35


def in_measured_silence(segment: Segment, silences: list[tuple[float, float]]) -> bool:
    """Is this segment CONTAINED in a stretch ffmpeg measured as silent?

    THE RULE THE METRICS CANNOT REPLACE. Measured: mlx-whisper turbo invented
    "Thank you." over a silent head and reported compression_ratio 0.56 and
    no_speech_prob 0.000 for it -- a confident, wrong answer that every metric
    rule reads as healthy. Audio level does not care how confident a decoder was.
    Because this is evidence from ffmpeg rather than from the decoder, it works
    identically on every backend, including the ones (Parakeet) that return no
    Whisper metrics at all and would otherwise have only a repetition heuristic.

    CONTAINMENT, NOT MIDPOINT. The first version of this rule asked whether the
    segment's midpoint fell inside a silence, which assumes segment boundaries
    are tight. Whisper's are. Parakeet's are not: on a real run its final segment
    spanned 36.64-48.48s on a file whose speech stopped at 40.79s, so the
    midpoint landed in the trailing silence and REAL SPEECH WAS SUPPRESSED. A
    guard that deletes genuine content is worse than one that misses an
    invention, so the test is now containment: the segment must lie wholly inside
    a single silent span, within a small tolerance.

    That is also the honest definition. Text decoded from silence is decoded from
    silence for its whole duration; a segment that overlaps real audio could have
    come from that audio, and the guard has no business guessing otherwise.
    """
    if not silences:
        return False
    start = segment.get("start", 0.0)
    end = segment.get("end", start)
    if end < start:
        return False
    return any(
        span_start - SILENCE_TOLERANCE <= start and end <= span_end + SILENCE_TOLERANCE
        for span_start, span_end in silences
    )


def apply_guard(
    result: Result,
    mode: GuardMode = "drop",
    thresholds: Thresholds | None = None,
    silences: list[tuple[float, float]] | None = None,
) -> GuardReport:
    """Annotate/filter result['segments'] in place and return the report.

    Segments are never deleted -- the JSON sidecar is the audit trail, and
    document writers filter on the `suppressed` flag rather than relying on
    the list already being clean. `result['text']` is always rebuilt from the
    non-suppressed segments, in every mode, so it never drifts from the flags.
    """
    thresholds = thresholds if thresholds is not None else Thresholds()
    segments = result.get("segments", [])

    reasons: dict[str, int] = {}
    metrics_available = False
    suppressed_count = 0

    for segment in segments:
        metrics = segment_metrics(segment)
        if metrics["compression_ratio"] is not None or metrics["no_speech_prob"] is not None:
            metrics_available = True

        if mode == "off":
            segment["suppressed"] = False
            segment.pop("suppressed_reason", None)
            continue

        # The silence-overlap rule is checked FIRST and separately from inspect():
        # it is evidence from ffmpeg rather than from the decoder, so it outranks
        # a decoder that is confidently wrong. inspect() stays metric-only and
        # independently testable.
        reason = "decoded_from_silence" if in_measured_silence(segment, silences or []) else None
        if reason is None:
            reason = inspect(segment, thresholds)
        if reason is None:
            segment["suppressed"] = False
            segment.pop("suppressed_reason", None)
            continue

        segment["suppressed"] = True
        segment["suppressed_reason"] = reason
        suppressed_count += 1
        reasons[reason] = reasons.get(reason, 0) + 1
        if mode == "mark":
            text = segment.get("text", "")
            if not text.startswith("[?] "):
                segment["text"] = f"[?] {text}"

    result["text"] = " ".join(seg["text"].strip() for seg in segments if not seg.get("suppressed")).strip()

    return GuardReport(
        checked=len(segments),
        suppressed=suppressed_count,
        reasons=reasons,
        metrics_available=metrics_available,
    )


def review_candidates(segments: list[Segment], limit: int = 8) -> list[tuple[Segment, str]]:
    """Segments a human should listen back to, worst first, with the reason.

    THE GUARD SUPPRESSES; THIS ONE ONLY POINTS. A segment can be well under every
    rejection threshold and still be wrong -- measured: `large-v3` dropped a
    spoken self-correction, and every backend heard "board pack" as "board packs
    up". No threshold catches those, because the decoder was confident and the
    output is fluent.

    So the answer is not a better threshold, it is directing a person's attention.
    Transcription output is a draft: this ranks the places most worth checking so
    review is a few timestamps rather than re-listening to the whole recording.

    Ranking uses whatever the backend provides:
      * avg_logprob, lowest first -- the decoder's own least-confident moments
      * compression_ratio approaching (but under) the rejection threshold
      * digits, which are cheap to mis-hear and expensive to get wrong
      * on metric-free backends, digits and length alone

    Returns at most `limit` entries so the list stays short enough to act on.
    """
    scored: list[tuple[float, Segment, str]] = []
    for segment in segments:
        if segment.get("suppressed"):
            continue
        text = segment.get("text", "").strip()
        if not text:
            continue

        reasons: list[str] = []
        # Rank ascending: lower is more suspect. 0.0 is a neutral starting point.
        rank = 0.0

        avg_logprob = segment.get("avg_logprob")
        if avg_logprob is not None:
            rank += avg_logprob
            if avg_logprob < -0.6:
                reasons.append(f"low confidence ({avg_logprob:.2f})")

        compression_ratio = segment.get("compression_ratio")
        if compression_ratio is not None and compression_ratio > 1.8:
            rank -= 0.3
            reasons.append(f"repetitive ({compression_ratio:.2f})")

        if any(character.isdigit() for character in text):
            rank -= 0.25
            reasons.append("contains figures")

        if not reasons:
            continue
        scored.append((rank, segment, ", ".join(reasons)))

    scored.sort(key=lambda row: row[0])
    return [(segment, reason) for _, segment, reason in scored[:limit]]
