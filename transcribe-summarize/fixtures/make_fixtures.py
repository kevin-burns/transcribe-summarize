#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Generate the audio fixtures deterministically. Stdlib only, nothing committed.

This directory is in a PUBLIC repository and `.gitignore` blocks audio, so the
generator is the thing that gets committed and the .wav files are made locally.
Fixtures are synthesised, never taken from a real recording.

The case worth having is the one that produced the whole quality guard: a long
near-silent head against speech-level audio. Measured on the real recording, the
joining silence sat at -69 dB and speech at -30 dB, and Whisper filled the gap
with one word repeated 55 times. The levels here reproduce that separation.

    ./make_fixtures.py            # writes into this directory
    ./make_fixtures.py --outdir /tmp/fx

What these DO test: audio preparation, silence detection, the trim, and the clock
map -- all of which are level-based and need no real speech. What they do NOT
test is decoding accuracy; a sine tone is not speech, and no synthetic file will
tell you whether a model heard a word correctly. Decoder behaviour is covered by
recorded segment metrics in tests/, not by audio.
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

RATE = 16_000


def db_to_amplitude(db: float) -> float:
    """dBFS -> linear amplitude. -69 dB is not silence; it is quiet enough to hallucinate over."""
    return 10.0 ** (db / 20.0)


def tone(seconds: float, db: float, freq: float = 220.0, phase: float = 0.0) -> tuple[bytes, float]:
    """A steady tone at a given level, returning the frames and the ending phase.

    Phase is carried between calls so concatenated sections do not click; a
    discontinuity reads as a transient and can trip silencedetect on its own.
    """
    amplitude = db_to_amplitude(db)
    step = 2 * math.pi * freq / RATE
    frames = bytearray()
    for _ in range(int(seconds * RATE)):
        frames += struct.pack("<h", int(amplitude * 32767 * math.sin(phase)))
        phase += step
    return bytes(frames), phase


def write(path: Path, sections: list[tuple[float, float]]) -> None:
    """sections: [(seconds, dBFS), ...]"""
    frames = bytearray()
    phase = 0.0
    for seconds, db in sections:
        chunk, phase = tone(seconds, db, phase=phase)
        frames += chunk
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(bytes(frames))
    seconds = len(frames) / 2 / RATE
    print(f"  {path}  ({seconds:.1f}s)")


# TWO ACOUSTIC REGIMES, AND A FIXTURE SET THAT ONLY MODELS ONE CERTIFIES A
# PIPELINE THAT IS BROKEN FOR THE OTHER.
#
#   CONFERENCE (-69 dB "silence").  A Zoom or Teams recording has its quiet
#   passages gated and coded away, so silence is nearly digital. This is the
#   original field case: a call with ~40 s of joining silence at -69 dB, where
#   large-v3 invented one word repeated 55 times.
#
#   ROOM (-52 to -56 dB "silence").  A laptop or handheld microphone in a real
#   room records actual room tone. Measured on a real MacBook recording: -56.1 dB.
#
# The difference is not cosmetic. `loudnorm` lifts room tone over a -40 dB
# silence threshold (-56.1 dB -> -14.8 dB, measured) but cannot lift -69 dB that
# far. So a bug where silence is detected AFTER normalising is INVISIBLE on the
# conference fixtures and total on the room ones: 11 detected spans became 0,
# trimming never fired, and the quality guard lost its strongest rule.
#
# That bug shipped past a full synthetic fixture suite and was caught only when a
# real room recording arrived. Both regimes stay in this file for that reason.

CONFERENCE_SILENCE = -69.0
ROOM_SILENCE = -54.0
SPEECH = -30.0

FIXTURES: dict[str, list[tuple[float, float]]] = {
    # --- conference regime: gated, near-digital silence -----------------------
    # The headline case: 40 s of joining silence, then speech-level audio.
    "silence-head.wav": [(40.0, CONFERENCE_SILENCE), (60.0, SPEECH)],
    # Trailing silence -- the shape that made the old script misreport duration by 15 s.
    "silence-tail.wav": [(20.0, SPEECH), (15.0, CONFERENCE_SILENCE)],
    # A quiet stretch in the middle: the interior-cut case for the clock map.
    "silence-middle.wav": [(15.0, SPEECH), (20.0, CONFERENCE_SILENCE), (15.0, SPEECH)],
    # No silence at all: preparation must decide not to trim rather than trim nothing.
    "no-silence.wav": [(30.0, SPEECH)],

    # --- room regime: real room tone, the case that caught the ordering bug ----
    "room-head.wav": [(20.0, ROOM_SILENCE), (40.0, SPEECH)],
    # Several natural inter-sentence pauses, as a real meeting recording has.
    "room-pauses.wav": [
        (8.0, ROOM_SILENCE), (12.0, SPEECH), (4.0, ROOM_SILENCE), (15.0, SPEECH),
        (3.0, ROOM_SILENCE), (10.0, SPEECH), (6.0, ROOM_SILENCE),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"writing fixtures to {args.outdir.resolve()}")
    for name, sections in FIXTURES.items():
        write(args.outdir / name, sections)
    print("\nthese are gitignored by design -- regenerate rather than commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
