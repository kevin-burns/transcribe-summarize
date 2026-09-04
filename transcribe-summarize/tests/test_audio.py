"""Tests for audio preparation and the two clocks.

The pure logic (keep_ranges, ClockMap) runs everywhere. The ffmpeg round-trip is
skipped where ffmpeg is absent rather than failing, but it is the test that
actually proves a trim can be undone, so it should run locally.
"""

from __future__ import annotations

import math
import shutil
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tslib.audio import (  # noqa: E402
    DEFAULT_MIN_SILENCE,
    DEFAULT_SILENCE_DB,
    ClockMap,
    decode_to_wav,
    detect_silence,
    keep_ranges,
    normalise,
    prepare,
    probe_duration,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


# --------------------------------------------------------------------------- keep_ranges


def test_keep_ranges_inverts_a_leading_silence():
    # The shape that started all of this: ~40 s of joining silence at the head.
    ranges = keep_ranges([(0.0, 40.0)], duration=100.0, pad=0.0)
    assert ranges == [(40.0, 100.0)]


def test_keep_ranges_handles_silence_at_both_ends():
    ranges = keep_ranges([(0.0, 10.0), (90.0, 100.0)], duration=100.0, pad=0.0)
    assert ranges == [(10.0, 90.0)]


def test_keep_ranges_with_no_silence_keeps_everything():
    assert keep_ranges([], duration=100.0, pad=0.0) == [(0.0, 100.0)]


def test_keep_ranges_pads_outward_without_leaving_the_file():
    ranges = keep_ranges([(10.0, 20.0)], duration=30.0, pad=0.25)
    assert ranges[0] == (0.0, 10.25)          # clamped at the start of the file
    assert ranges[1] == (19.75, 30.0)         # clamped at the end


def test_keep_ranges_merges_spans_that_padding_pushed_together():
    # A 0.3 s gap with 0.25 s of padding either side overlaps: one range, not two.
    ranges = keep_ranges([(10.0, 10.3)], duration=20.0, pad=0.25)
    assert ranges == [(0.0, 20.0)]


def test_keep_ranges_ignores_silence_past_the_end_of_the_file():
    # An unterminated silence_start is reported as (start, inf); it must be closed
    # at the duration, not produce an infinite range.
    ranges = keep_ranges([(80.0, math.inf)], duration=100.0, pad=0.0)
    assert ranges == [(0.0, 80.0)]


# --------------------------------------------------------------------------- ClockMap


def test_clockmap_identity_returns_its_input():
    clock = ClockMap([])
    assert clock.identity is True
    assert clock.to_original(12.5) == 12.5


def test_clockmap_maps_across_a_single_removed_head():
    # 40 s cut from the front: trimmed t=0 is original t=40.
    clock = ClockMap([(40.0, 100.0)])
    assert clock.to_original(0.0) == pytest.approx(40.0)
    assert clock.to_original(10.0) == pytest.approx(50.0)
    assert clock.to_original(60.0) == pytest.approx(100.0)


def test_clockmap_maps_across_an_interior_cut():
    # Kept 0-10 and 30-40. Trimmed is 20 s long; trimmed 10 s is original 30 s.
    clock = ClockMap([(0.0, 10.0), (30.0, 40.0)])
    assert clock.to_original(0.0) == pytest.approx(0.0)
    assert clock.to_original(9.999) == pytest.approx(9.999)
    assert clock.to_original(10.0) == pytest.approx(30.0)
    assert clock.to_original(15.0) == pytest.approx(35.0)
    assert clock.trimmed_duration == pytest.approx(20.0)


def test_clockmap_boundary_lands_on_the_next_span_not_inside_the_cut():
    """The boundary is the case that silently corrupts an .srt if it is wrong.

    At exactly the seam, the answer must be the start of the next kept span. Any
    other answer places a subtitle inside audio that was removed.
    """
    clock = ClockMap([(0.0, 10.0), (30.0, 40.0)])
    assert clock.to_original(10.0) == pytest.approx(30.0)


def test_clockmap_clamps_past_the_end_instead_of_running_into_a_cut():
    clock = ClockMap([(0.0, 10.0), (30.0, 40.0)])
    assert clock.to_original(999.0) == pytest.approx(40.0)


def test_clockmap_negative_input_clamps_to_the_first_kept_moment():
    clock = ClockMap([(40.0, 100.0)])
    assert clock.to_original(-5.0) == pytest.approx(40.0)


def test_clockmap_is_monotonic_over_a_dense_sweep():
    # A map that ever goes backwards produces subtitles out of order.
    clock = ClockMap([(0.0, 10.0), (30.0, 40.0), (55.0, 60.0)])
    previous = -1.0
    for i in range(0, 2500):
        value = clock.to_original(i / 100.0)
        assert value >= previous
        previous = value


def test_clockmap_json_offsets_are_inspectable():
    rows = ClockMap([(0.0, 10.0), (30.0, 40.0)]).as_json()
    assert rows[0] == {"trimmed_start": 0.0, "original_start": 0.0, "length": 10.0}
    assert rows[1] == {"trimmed_start": 10.0, "original_start": 30.0, "length": 10.0}


# --------------------------------------------------------------------------- ffmpeg round trip


def _write_wav(path: Path, spec: list[tuple[float, float]], rate: int = 16_000) -> None:
    """Write a mono 16-bit wav from [(seconds, amplitude_0_to_1), ...].

    A 440 Hz tone rather than noise: silencedetect measures level, and a tone
    gives a level that is stable enough to assert against.
    """
    frames = bytearray()
    phase = 0.0
    step = 2 * math.pi * 440.0 / rate
    for seconds, amplitude in spec:
        for _ in range(int(seconds * rate)):
            frames += struct.pack("<h", int(amplitude * 32767 * math.sin(phase)))
            phase += step
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


@needs_ffmpeg
def test_probe_duration_reads_the_file_not_the_last_sound(tmp_path):
    """Bug 5: duration was taken from the last segment's end, missing trailing silence."""
    src = tmp_path / "tail.wav"
    _write_wav(src, [(2.0, 0.5), (3.0, 0.0)])   # 2 s of tone, then 3 s of true silence
    assert probe_duration(src) == pytest.approx(5.0, abs=0.05)


@needs_ffmpeg
def test_detect_silence_finds_a_quiet_head(tmp_path):
    src = tmp_path / "head.wav"
    _write_wav(src, [(3.0, 0.0), (3.0, 0.5)])
    spans = detect_silence(src, threshold_db=-40.0, min_silence=1.0)
    assert spans, "expected the 3 s silent head to be detected"
    start, end = spans[0]
    assert start == pytest.approx(0.0, abs=0.2)
    assert end == pytest.approx(3.0, abs=0.3)


@needs_ffmpeg
def test_prepare_trims_the_head_and_the_clock_still_points_at_the_original(tmp_path):
    """The end-to-end claim: audio gets shorter, timestamps do not move.

    3 s silent, 3 s tone, 3 s silent. After trimming, a moment 0.5 s into the
    trimmed audio is the tone -- which lives around 3.5 s in the ORIGINAL file.
    """
    src = tmp_path / "sandwich.wav"
    _write_wav(src, [(3.0, 0.0), (3.0, 0.5), (3.0, 0.0)])
    work = tmp_path / "work"
    work.mkdir()

    prepared = prepare(src, work, do_normalise=False, do_trim=True, pad=0.1)

    assert prepared.trimmed is True
    assert prepared.original_duration == pytest.approx(9.0, abs=0.1)
    assert prepared.prepared_duration < prepared.original_duration
    assert prepared.wav.exists()
    # 0.5 s into the kept audio is inside the tone, which starts at 3 s originally.
    mapped = prepared.clock.to_original(0.5)
    assert 2.5 <= mapped <= 4.5, f"expected the tone's original position, got {mapped}"


@needs_ffmpeg
def test_prepare_with_trim_disabled_leaves_an_identity_clock(tmp_path):
    src = tmp_path / "plain.wav"
    _write_wav(src, [(2.0, 0.5)])
    work = tmp_path / "work"
    work.mkdir()
    prepared = prepare(src, work, do_normalise=False, do_trim=False)
    assert prepared.trimmed is False
    assert prepared.clock.identity is True
    assert prepared.clock.to_original(1.0) == 1.0


@needs_ffmpeg
def test_prepare_on_audio_with_no_silence_does_not_trim(tmp_path):
    src = tmp_path / "solid.wav"
    _write_wav(src, [(4.0, 0.5)])
    work = tmp_path / "work"
    work.mkdir()
    prepared = prepare(src, work, do_normalise=False, do_trim=True)
    assert prepared.trimmed is False
    assert prepared.clock.identity is True


# ------------------------------------------------- normalisation vs silence detection
#
# Found on a real MacBook-microphone recording, invisible on synthetic fixtures.
# `loudnorm` compresses dynamic range and lifts quiet passages: measured room
# tone went from -56.1 dB to -14.8 dB, a 41 dB lift. Detecting silence AFTER
# normalising therefore found 11 spans on the raw audio and ZERO on the
# normalised audio -- so trimming never fired, and the quality guard was handed
# an empty silence list, disabling its strongest rule.
#
# The synthetic fixtures never caught it because their silence sits near -69 dB
# and stays under -40 dB even after the lift. Real rooms are not that quiet.


@needs_ffmpeg
def test_silences_are_measured_on_the_raw_decode_not_the_normalised_audio(tmp_path):
    """THE INVARIANT, asserted directly rather than through a proxy.

    A first version of this test only checked that *something* was trimmed. That
    was too weak: at -55 dB loudnorm lifted one span over the threshold but not
    the other, so the buggy order still found one span and still trimmed, and the
    test passed under the bug it was written to catch. Comparing against the raw
    decode's own spans fails whenever the order is wrong, at any level.
    """
    src = tmp_path / "room.wav"
    _write_wav(src, [(4.0, 10 ** (-55.0 / 20.0)), (4.0, 10 ** (-26.0 / 20.0)),
                     (4.0, 10 ** (-55.0 / 20.0)), (4.0, 10 ** (-26.0 / 20.0))])
    work = tmp_path / "w"
    work.mkdir()

    prepared = prepare(src, work, do_normalise=True, do_trim=True)

    reference = detect_silence(work / "decoded.wav", DEFAULT_SILENCE_DB, DEFAULT_MIN_SILENCE)
    assert len(prepared.silences) == len(reference), (
        f"got {len(prepared.silences)} spans, raw audio has {len(reference)} -- "
        "silence is being detected after loudnorm, which lifts room tone above the threshold"
    )
    assert prepared.trimmed is True


@needs_ffmpeg
def test_loudnorm_really_does_lift_room_tone_over_the_threshold(tmp_path):
    """The mechanism itself, so the reason for the ordering is documented in a
    test and not only in a comment. Measured on a real recording: -56.1 dB room
    tone became -14.8 dB after loudnorm."""
    src = tmp_path / "room.wav"
    _write_wav(src, [(4.0, 10 ** (-52.0 / 20.0)), (4.0, 10 ** (-26.0 / 20.0))])
    work = tmp_path / "w"
    work.mkdir()

    decoded = work / "decoded.wav"
    decode_to_wav(src, decoded)
    normalised = work / "normalised.wav"
    normalise(decoded, normalised)

    on_raw = detect_silence(decoded, DEFAULT_SILENCE_DB, DEFAULT_MIN_SILENCE)
    on_normalised = detect_silence(normalised, DEFAULT_SILENCE_DB, DEFAULT_MIN_SILENCE)
    assert on_raw, "the quiet head should be detectable before normalisation"
    assert len(on_normalised) < len(on_raw), (
        "normalisation is expected to hide silence; if this ever stops being true, "
        "the ordering constraint in prepare() can be revisited"
    )


@needs_ffmpeg
def test_the_guard_gets_real_silence_spans_even_when_normalising(tmp_path):
    src = tmp_path / "room.wav"
    _write_wav(src, [(4.0, 10 ** (-55.0 / 20.0)), (4.0, 10 ** (-26.0 / 20.0))])
    work = tmp_path / "w"
    work.mkdir()
    prepared = prepare(src, work, do_normalise=True, do_trim=False)
    assert prepared.silences, "silences must be measured even when not trimming"
    assert prepared.silences[0][0] < 1.0, "the quiet head should be found"


@needs_ffmpeg
def test_normalisation_does_not_shift_the_detected_spans(tmp_path):
    """The spans are timestamps, so they must be identical either way -- that is
    what makes it safe to measure on the raw decode and apply after normalising."""
    src = tmp_path / "room.wav"
    _write_wav(src, [(4.0, 10 ** (-55.0 / 20.0)), (4.0, 10 ** (-26.0 / 20.0))])
    work_a, work_b = tmp_path / "a", tmp_path / "b"
    work_a.mkdir()
    work_b.mkdir()

    plain = prepare(src, work_a, do_normalise=False, do_trim=True)
    loud = prepare(src, work_b, do_normalise=True, do_trim=True)

    assert len(plain.silences) == len(loud.silences)
    for (a_start, a_end), (b_start, b_end) in zip(plain.silences, loud.silences, strict=True):
        assert abs(a_start - b_start) < 0.1
        assert abs(a_end - b_end) < 0.1
