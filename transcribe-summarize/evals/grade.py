#!/usr/bin/env python3
"""Grade transcribe.py's privacy guarantee: 'auto' never resolves to the network.

Usage: uv run python grade.py     # run from evals/
Exit 0 if all cases pass (skips allowed), 1 if any case fails.

Offline by construction: every case below drives transcribe.py with --dry-run
(it returns before loading a backend or sending anything) or with argv that
fails before either. No network call is made; if one were attempted, this
machine's lack of network access would surface it as a failure -- that is
the point, not an accident.

Two groups of assertions live here, not in eval.json:

  - The backend-registry matrix (cases 9-13 in the task this suite was
    written against) takes no argv, so it is asserted directly against
    tslib.backends in check_registry() below, imported with no subprocess.

  - A malformed `--replace` pair (case 7) is asserted directly against
    tslib.corrections in check_replace_validation(), NOT through the CLI.
    transcribe.py only calls corrections.build() AFTER a successful decode
    (scripts/transcribe.py, the "corrections, then guard" section) -- every
    --dry-run path returns before that point by design (dry-run must decode
    nothing), and running past --dry-run needs a real local backend, which
    this suite cannot install (no ASR library, no network, no model
    download). So the CLI genuinely cannot be driven into this failure
    offline; the library call it would eventually make is verified instead.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "scripts" / "transcribe.py"
TSLIB = HERE.parent / "scripts"

sys.path.insert(0, str(TSLIB))

from tslib import backends, corrections  # noqa: E402


def _load_json(path):
    """Read and parse a JSON file, closing the handle. json.load(open(...)) leaks it."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _make_silent_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    """A tiny mono silent wav via the stdlib wave module. Never committed, never fetched."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def run(argv: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(TOOL), *argv], capture_output=True, text=True, check=False
    )
    return proc.returncode, proc.stdout, proc.stderr


def check_case(case: dict, audio_path: Path, missing_path: Path) -> tuple[str, list[tuple[str, bool]]]:
    """Returns ('pass'|'fail'|'skip', checks). checks is empty for a skip."""
    if case.get("needs_audio") and not _ffmpeg_available():
        return "skip", []

    if case.get("needs_audio"):
        target = audio_path
    elif case.get("use_missing_path"):
        target = missing_path
    else:
        target = audio_path

    rc, out, err = run([str(target), *case["argv"]])
    e = case["expect"]
    checks: list[tuple[str, bool]] = []

    if "exit" in e:
        checks.append((f"exit=={e['exit']}", rc == e["exit"]))
    if e.get("exit_nonzero"):
        checks.append(("exit!=0", rc != 0))

    for needle in e.get("stderr_contains", []):
        checks.append((f"stderr contains {needle!r}", needle in err))
    for needle in e.get("stderr_not_contains", []):
        checks.append((f"stderr NOT contains {needle!r}", needle not in err))

    if e.get("stdout_is_json"):
        try:
            parsed = json.loads(out)
            checks.append(("stdout parses as JSON", True))
        except json.JSONDecodeError:
            parsed = {}
            checks.append(("stdout parses as JSON", False))
        for key, want in e.get("json_fields", {}).items():
            checks.append((f"json[{key!r}]=={want!r}", parsed.get(key) == want))

    status = "pass" if all(p for _, p in checks) else "fail"
    return status, checks


def check_registry() -> list[tuple[str, bool]]:
    """Cases 9-13: the backend registry, asserted directly -- no argv, no subprocess."""
    results: list[tuple[str, bool]] = []

    # 9 + resolve() call itself: 'auto' is local on every platform pair it can be asked
    # to pretend to be -- the module docstring's central claim.
    platform_pairs = [
        ("darwin", "arm64"),
        ("darwin", "x86_64"),
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("win32", "AMD64"),
        ("win32", "ARM64"),
    ]
    for system, machine in platform_pairs:
        info = backends.resolve("auto", system=system, machine=machine)
        results.append((f"resolve(auto, {system}/{machine}).kind == 'local'", info.kind == "local"))

    # 10 + 11: the two concrete choices auto makes.
    darwin_arm = backends.resolve("auto", system="darwin", machine="arm64")
    results.append(("resolve(auto, darwin/arm64).name == 'mlx-whisper'", darwin_arm.name == "mlx-whisper"))
    linux_x86 = backends.resolve("auto", system="linux", machine="x86_64")
    results.append(("resolve(auto, linux/x86_64).name == 'faster-whisper'", linux_x86.name == "faster-whisper"))

    # 12: pin WHICH backends lack Whisper's metrics, rather than assuming a
    # single exception. The guard's strong rules cannot fire on these, so a new
    # metric-free backend added silently would weaken the guard for it with
    # nothing failing. This assertion caught exactly that when elevenlabs landed.
    metric_free = {name for name, info in backends.REGISTRY.items() if not info.has_whisper_metrics}
    results.append((
        f"metric-free backends are exactly {{parakeet, elevenlabs}} (got {sorted(metric_free)})",
        metric_free == {"parakeet", "elevenlabs"},
    ))
    for name in ("mlx-whisper", "faster-whisper", "groq", "openai"):
        results.append((
            f"REGISTRY[{name!r}].has_whisper_metrics is True",
            backends.REGISTRY[name].has_whisper_metrics is True,
        ))

    # 13: cost estimate rounds correctly for a priced network model, and is None
    # (not 0, not free) for a local backend -- callers must treat None as
    # "cannot estimate".
    groq_cost = backends.estimate_cost("groq", "whisper-large-v3", 3600)
    results.append(("estimate_cost(groq, whisper-large-v3, 3600h) rounds to 0.111",
                    groq_cost is not None and round(groq_cost, 3) == 0.111))
    local_cost = backends.estimate_cost("mlx-whisper", "large-v3", 3600)
    results.append(("estimate_cost(mlx-whisper, ...) is None", local_cost is None))

    return results


def check_replace_validation() -> list[tuple[str, bool]]:
    """Case 7: a malformed --replace pair, asserted against tslib.corrections directly.

    See the module docstring for why this cannot be driven through the CLI
    offline: transcribe.py validates --replace pairs only after a successful
    decode, and every --dry-run path returns before that point.
    """
    try:
        corrections.build(["bogus-no-equals-sign"])
        raised, message = False, ""
    except corrections.ReplacementError as exc:
        raised, message = True, str(exc)

    return [
        ("corrections.build(['bogus-no-equals-sign']) raises ReplacementError", raised),
        ("error names the offending pair", "bogus-no-equals-sign" in message),
    ]


def main() -> int:
    spec = _load_json(HERE / "eval.json")
    ok_all = True
    ffmpeg_ok = _ffmpeg_available()
    if not ffmpeg_ok:
        print("note: ffmpeg/ffprobe not found on PATH -- audio-dependent cases will be SKIPPED\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        audio_path = tmp_dir / "silent.wav"
        missing_path = tmp_dir / "does-not-exist.wav"
        if ffmpeg_ok:
            _make_silent_wav(audio_path)

        skipped = 0
        for case in spec["cases"]:
            status, checks = check_case(case, audio_path, missing_path)
            if status == "skip":
                skipped += 1
                print(f"[SKIP] {case['id']}  (needs ffmpeg/ffprobe on PATH, none found)")
                continue
            ok = status == "pass"
            ok_all &= ok
            print(f"[{'PASS' if ok else 'FAIL'}] {case['id']}")
            for label, p in checks:
                print(f"        {'ok' if p else 'XX'}  {label}")

    print("\n-- backend registry (cases 9-13, no argv) --")
    registry_checks = check_registry()
    reg_ok = all(p for _, p in registry_checks)
    ok_all &= reg_ok
    print(f"[{'PASS' if reg_ok else 'FAIL'}] backend-registry")
    for label, p in registry_checks:
        print(f"        {'ok' if p else 'XX'}  {label}")

    print("\n-- --replace validation (case 7, direct library call -- see module docstring) --")
    replace_checks = check_replace_validation()
    replace_ok = all(p for _, p in replace_checks)
    ok_all &= replace_ok
    print(f"[{'PASS' if replace_ok else 'FAIL'}] replace-validation")
    for label, p in replace_checks:
        print(f"        {'ok' if p else 'XX'}  {label}")

    print(f"\n{skipped} case(s) skipped" if skipped else "")
    print("\n" + ("ALL PASS" if ok_all else "FAILURES PRESENT"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
