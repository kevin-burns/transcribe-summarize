"""Audio preparation, and the two clocks that come with it.

Cleaning the audio beats reaching for a bigger model. Measured on a real call:
normalising and silence-trimming took invented segments from 2 to 0, and fixed a
word that both large-v3 and turbo had decoded wrong on the raw audio. See
README.md, "Why this exists".

That cleaning used to be a manual step in an audio editor. Everything here does
it with ffmpeg instead: `loudnorm`, `silencedetect`, `atrim` and `concat` are
unconditional libavfilter built-ins, present in any standard ffmpeg build on
macOS, Windows and Linux. No second tool, no per-platform branch.

THE TWO CLOCKS. Trimming shortens the audio, so a timestamp the decoder reports
is measured against the *trimmed* file. Left alone, the .srt would drift against
the user's own recording. `silenceremove` would do the cut in one filter but
reports nothing about what it removed, so it cannot be undone. We detect the
silence, cut it ourselves, and keep an offset table -- `ClockMap` below maps
every timestamp back to the original file before anything is written.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16_000

# Defaults for silence detection. -40 dB sits well above the -69 dB measured on
# the joining silence that produced the 55x word loop, and well below the -30 dB
# measured for speech on the same file, so it separates the two cases that
# actually occurred rather than a threshold picked by taste.
DEFAULT_SILENCE_DB = -40.0
DEFAULT_MIN_SILENCE = 1.0

# Keep a little audio either side of a cut. Speech onsets are quiet, and slicing
# flush against the detector's boundary clips the first phoneme of a word.
DEFAULT_PAD = 0.25

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


class AudioError(RuntimeError):
    """ffmpeg/ffprobe was missing, or refused the file."""


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise AudioError(
            f"{name} not found on PATH. Install ffmpeg: "
            "macOS `brew install ffmpeg`, Windows `winget install Gyan.FFmpeg`, "
            "Debian/Ubuntu `apt install ffmpeg`."
        )
    return found


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    # check=False is explicit: the return code is inspected by every caller, and a
    # non-zero exit is reported with ffmpeg's own stderr rather than raised bare.
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe_duration(src: Path) -> float:
    """Duration of the file itself, in seconds.

    Bug 5 of the prior art: the old script used `segments[-1]["end"]`,
    which is where the last speech stopped, not where the file stops. On a
    recording with trailing silence the two were 15 s apart -- and that number is
    what an API backend bills against, so it has to be the real one.
    """
    cmd = [
        _tool("ffprobe"), "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(src),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioError(f"ffprobe could not read {src.name}:\n{result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AudioError(f"ffprobe returned no duration for {src.name}") from exc


def audio_stream_count(src: Path) -> int:
    """How many audio streams the file has. Zero is a real and common case.

    A Teams or Zoom download is an .mp4, and video containers are accepted
    precisely so those work without the user extracting anything first. But a
    screen recording with the microphone muted is also an .mp4, and it decodes to
    nothing.

    Checked BEFORE the egress gate, because the two failure modes are both bad:
    locally it produced a raw "Output file does not contain any stream" from
    ffmpeg, and on a network backend it would have uploaded the file and billed
    for it.
    """
    cmd = [
        _tool("ffprobe"), "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(src),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioError(f"ffprobe could not read {src.name}:\n{result.stderr.strip()}")
    return len([line for line in result.stdout.splitlines() if line.strip()])


# Text transcripts people hand over instead of audio. Naming them explicitly is
# the difference between a useful error and a baffling one: told a .vtt "carries
# only video", a user reasonably concludes the tool is broken.
TRANSCRIPT_SUFFIXES = {".vtt", ".srt", ".txt", ".md", ".json", ".csv", ".docx", ".rtf", ".html"}


def assert_has_audio(src: Path) -> None:
    """Fail early and specifically when there is nothing to transcribe."""
    if src.suffix.lower() in TRANSCRIPT_SUFFIXES:
        raise AudioError(
            f"{src.name} looks like a text transcript, not a recording.\n"
            "This tool transcribes audio; it has nothing to decode here.\n"
            "If you already have a transcript and want it written up as notes, that is the\n"
            "summarising half of this skill -- see SKILL.md, 'Starting from a transcript you\n"
            "already have'. No audio needed."
        )

    if audio_stream_count(src) == 0:
        raise AudioError(
            f"{src.name} has no audio track.\n"
            "Video containers are supported, but this file carries only video "
            "(a muted screen recording, or a video-only export).\n"
            "Check with:  ffprobe -select_streams a -show_entries stream=codec_name "
            f"-of csv=p=0 {src}"
        )


def decode_to_wav(src: Path, dest: Path) -> None:
    """Decode anything ffmpeg reads into 16 kHz mono PCM -- Whisper's native format.

    Doing the resample up front rather than leaving it to the library keeps it
    identical across input formats and backends, and gives a clear error on a
    corrupt file instead of a decoder traceback.
    """
    cmd = [
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(dest),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg could not decode {src.name}:\n{result.stderr.strip()}")


def normalise(src: Path, dest: Path) -> None:
    """Single-pass EBU R128 loudness normalisation.

    Single pass, not two: the measured win came from lifting quiet windows out of
    the range where the decoder invents speech (-69 dB to around -21 dB), and a
    second analysis pass doubles the ffmpeg time to refine a target that is
    already far past the point that mattered.
    """
    cmd = [
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(dest),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg loudnorm failed on {src.name}:\n{result.stderr.strip()}")


def detect_silence(src: Path, threshold_db: float, min_silence: float) -> list[tuple[float, float]]:
    """Return [(start, end), ...] for every silent stretch, in seconds.

    `silencedetect` passes the audio through untouched and reports on stderr, so
    this is a measurement pass with no output file.

    A file that ends silent gets a `silence_start` with no matching
    `silence_end`; that stretch is closed at the file duration by the caller,
    which is why the open interval is returned as-is here.
    """
    cmd = [
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "info",
        "-i", str(src),
        "-af", f"silencedetect=n={threshold_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg silencedetect failed on {src.name}:\n{result.stderr.strip()}")

    starts: list[float] = []
    ends: list[float] = []
    for line in result.stderr.splitlines():
        if (m := _SILENCE_START.search(line)) is not None:
            starts.append(max(0.0, float(m.group(1))))
        if (m := _SILENCE_END.search(line)) is not None:
            ends.append(float(m.group(1)))

    spans: list[tuple[float, float]] = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else float("inf")
        spans.append((start, end))
    return spans


def keep_ranges(
    silences: list[tuple[float, float]],
    duration: float,
    pad: float = DEFAULT_PAD,
) -> list[tuple[float, float]]:
    """Invert silent spans into the spans worth decoding, padded and merged.

    Padding can push two keep-ranges into each other; they are merged rather than
    emitted as adjacent slices, because every extra slice is another `atrim` in
    the filter graph and another row in the offset table for no gain.
    """
    closed = [(s, min(e, duration)) for s, e in silences if s < duration]
    closed.sort()

    ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in closed:
        if start > cursor:
            ranges.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        ranges.append((cursor, duration))

    padded = [(max(0.0, s - pad), min(duration, e + pad)) for s, e in ranges]

    merged: list[tuple[float, float]] = []
    for start, end in padded:
        if end - start <= 0:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def trim(src: Path, dest: Path, ranges: list[tuple[float, float]]) -> None:
    """Concatenate the keep-ranges into one file.

    The filter is `concat` with `v=0:a=1`. There is no `aconcat` filter -- reaching
    for that name fails with "Unknown filter", verified against ffmpeg 9.0.1.
    """
    if not ranges:
        raise AudioError("nothing to keep: every part of the audio was detected as silence")

    parts = [
        f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}]"
        for i, (start, end) in enumerate(ranges)
    ]
    labels = "".join(f"[a{i}]" for i in range(len(ranges)))
    graph = ";".join(parts) + f";{labels}concat=n={len(ranges)}:v=0:a=1[out]"

    cmd = [
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-filter_complex", graph, "-map", "[out]",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(dest),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg trim failed on {src.name}:\n{result.stderr.strip()}")


@dataclass(frozen=True)
class ClockMap:
    """Maps a timestamp on the trimmed audio back to the original recording.

    `ranges` are the kept spans in original-file time, in order. Their lengths
    laid end to end are the trimmed timeline, so a trimmed timestamp is located
    by which cumulative span it falls in and how far into it.

    An identity map (nothing was trimmed) is `ClockMap([])` and returns its input.
    """

    ranges: list[tuple[float, float]]
    _starts: list[float] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        cumulative: list[float] = []
        total = 0.0
        for start, end in self.ranges:
            cumulative.append(total)
            total += end - start
        object.__setattr__(self, "_starts", cumulative)

    @property
    def identity(self) -> bool:
        return not self.ranges

    @property
    def trimmed_duration(self) -> float:
        return sum(end - start for start, end in self.ranges)

    def to_original(self, t: float) -> float:
        """Trimmed-clock seconds -> original-clock seconds."""
        if self.identity:
            return t
        if t <= 0:
            return self.ranges[0][0]
        # bisect_right - 1 gives the last span starting at or before t.
        i = max(0, bisect_right(self._starts, t) - 1)
        start, end = self.ranges[i]
        offset = t - self._starts[i]
        # Past the end of this span means past the end of the audio: clamp to the
        # span's own end rather than running into the next span's original time,
        # which would report a timestamp inside a stretch that was cut out.
        return min(start + offset, end)

    def as_json(self) -> list[dict[str, float]]:
        """The offset table, for the run manifest -- a trim stays inspectable."""
        return [
            {"trimmed_start": round(self._starts[i], 3),
             "original_start": round(start, 3),
             "length": round(end - start, 3)}
            for i, (start, end) in enumerate(self.ranges)
        ]


@dataclass
class PreparedAudio:
    """The wav a backend will decode, plus what it took to get there."""

    wav: Path
    original_duration: float
    clock: ClockMap
    normalised: bool
    trimmed: bool
    silences: list[tuple[float, float]]

    @property
    def prepared_duration(self) -> float:
        return self.clock.trimmed_duration if self.trimmed else self.original_duration

    def as_json(self) -> dict:
        return {
            "original_duration": round(self.original_duration, 3),
            "prepared_duration": round(self.prepared_duration, 3),
            "normalised": self.normalised,
            "trimmed": self.trimmed,
            "silences_removed": len(self.silences) if self.trimmed else 0,
            "clock_map": self.clock.as_json(),
        }


def prepare(
    src: Path,
    workdir: Path,
    *,
    do_normalise: bool = True,
    do_trim: bool = True,
    threshold_db: float = DEFAULT_SILENCE_DB,
    min_silence: float = DEFAULT_MIN_SILENCE,
    pad: float = DEFAULT_PAD,
) -> PreparedAudio:
    """Decode, normalise, silence-trim -- and record how to undo the trim.

    Returns the file to hand to a backend together with a ClockMap. Callers must
    put every timestamp through that map before writing it anywhere a human will
    compare against their own recording.
    """
    duration = probe_duration(src)

    decoded = workdir / "decoded.wav"
    decode_to_wav(src, decoded)

    # DETECT SILENCE ON THE RAW DECODE, BEFORE NORMALISING. The order matters and
    # getting it wrong is invisible on synthetic audio.
    #
    # `loudnorm` compresses dynamic range: it lifts quiet passages towards the
    # target loudness. Measured on a real MacBook-microphone recording, room tone
    # went from -56.1 dB to -14.8 dB -- a 41 dB lift, far above any sane silence
    # threshold. silencedetect found 11 silent spans on the raw audio and ZERO on
    # the normalised audio, so trimming never fired AND the quality guard was
    # handed an empty list of silences, disabling its strongest rule.
    #
    # It never showed on the synthetic fixture, whose silence sat near -69 dB and
    # stayed under -40 dB even after normalisation. Real rooms are not that quiet.
    #
    # Silence spans are timestamps, so they remain valid across normalisation:
    # measure on the raw decode, then apply to whichever file the backend gets.
    #
    # Detected even when not trimming -- one ffmpeg pass with no output file, and
    # the spans are evidence in their own right: a segment the decoder placed
    # inside a measured silence is an invention whatever its metrics say. Measured
    # case: turbo invented "Thank you." over a silent head and reported
    # no_speech_prob 0.000 for it, so the metric rules could not see it.
    silences = detect_silence(decoded, threshold_db, min_silence)

    current = decoded
    if do_normalise:
        normalised = workdir / "normalised.wav"
        normalise(current, normalised)
        current = normalised

    if not do_trim:
        return PreparedAudio(current, duration, ClockMap([]), do_normalise, False, silences)

    ranges = keep_ranges(silences, duration, pad)

    # One range covering everything means the detector found nothing worth
    # cutting. Skipping the filter pass keeps the map an identity rather than a
    # single-span map that says the same thing more expensively.
    if len(ranges) == 1 and ranges[0][0] <= 0.0 and ranges[0][1] >= duration:
        return PreparedAudio(current, duration, ClockMap([]), do_normalise, False, silences)

    trimmed = workdir / "trimmed.wav"
    trim(current, trimmed, ranges)
    return PreparedAudio(trimmed, duration, ClockMap(ranges), do_normalise, True, silences)
