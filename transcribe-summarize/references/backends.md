# Backends — what each one can and cannot do

Read this before choosing a backend, and before believing a transcript the guard
declared clean.

## The matrix

| backend | platform | install | quality guard | network |
|---|---|---|---|---|
| `mlx-whisper` | **Apple Silicon only** | `mlx-whisper>=0.4.2` | **full** | no |
| `faster-whisper` | macOS / Windows / Linux | `faster-whisper>=1.2` | **full** | no |
| `parakeet` | **macOS verified**, others untested | `parakeet-mlx` (Apple Silicon) or `nemo_toolkit[asr]` | **partial — see below** | no |
| `groq` | any | none — stdlib `urllib` | **full** | **yes** |
| `openai` | any | none — stdlib `urllib` | **full** | **yes** |
| `elevenlabs` (Scribe) | any | none — stdlib `http.client` | **partial** | **yes** |

`--backend auto` resolves to `mlx-whisper` on Apple Silicon and `faster-whisper`
everywhere else. **It never resolves to a network backend** — not as a default,
not as a fallback after a local backend fails. There is no code path from `auto`
to `groq` or `openai`; `resolve("auto")` can only return one of two local
entries. A test asserts it across twelve platform combinations.

## Default models

Both Whisper backends default to **`turbo`** (large-v3-turbo), changed from
`large-v3` on 2026-09-04. On the only real recording this project has measured —
accented English, laptop microphone — turbo scored 9/12 against large-v3's 8/12
in 13.8 s against 21.5 s, and large-v3 dropped a spoken self-correction that
every other backend kept. `--model large-v3` remains available.

`faster-whisper` resolves `turbo` to `mobiuslabsgmbh/faster-whisper-large-v3-turbo`,
which is **not** a Systran repository unlike its other short names (verified in
`faster_whisper/utils.py`'s `_MODELS` map). Worth knowing if you audit where
weights come from.

## Why the guard does not transfer everywhere

The guard's strong rules read three numbers the **Whisper decoder** produces per
segment: `compression_ratio`, `no_speech_prob` and `avg_logprob`. They are what
caught an 18-second fabricated segment (one word repeated 55 times,
`compression_ratio` 17.38 against Whisper's own reject threshold of 2.4) and a
spurious "Thank you." at `no_speech_prob` 0.923.

- **`faster-whisper` returns all three**, on the same scale. Verified in its
  `Segment` dataclass. This is why it is the cross-platform default rather than
  Parakeet: the guard is the *same code*, not an equivalent-in-spirit port.
- **Groq and OpenAI return all three** in `verbose_json`. Groq's is verified from
  its documentation; OpenAI's is verified from an actual API response on
  2026-09-04 — `tests/test_openai_live.py` asserts it on every live run, so if
  the provider ever stops returning them the guard's degradation is caught
  rather than silently accepted.
- **Parakeet returns none of them.** It is a CTC/TDT model, not a Whisper
  decoder, and there is nothing on its output to threshold. Those fields are
  `None`, and `None` means "this backend cannot tell you" — never "this segment
  is fine". The guard falls back to its one metric-free rule: a token repeated
  past a threshold and dominating the segment.

**So on Parakeet, a hallucination that is not repetitive will not be caught.**
The tool says so at run time, every run, rather than leaving you to infer it from
a clean-looking transcript.

## Parakeet runtimes: one solved, one still open

`parakeet.py` supports two runtimes and prefers whichever imports:

| runtime | platform | weight | status |
|---|---|---|---|
| `parakeet-mlx` 0.5.2 | Apple Silicon only | small deps; model 2.3 GB | **verified working** |
| `nemo_toolkit[asr]` | cross-platform | torch + multi-GB tree | **not verified** |

On Apple Silicon this is solved: `parakeet-mlx` is light, and its model
(`mlx-community/parakeet-tdt-0.6b-v3`) is the one several macOS dictation apps
already download, so it is frequently on the machine before this skill asks.

**The non-Apple path is still the open question**, and NeMo is probably the wrong
answer to it. Verified 2026-09-04 against PyPI and HuggingFace:

| | `nemo_toolkit[asr]` | `sherpa-onnx` 1.13.7 |
|---|---|---|
| dependencies | torch + a large tree | **one** (`sherpa-onnx-core`) |
| largest wheel | multi-GB install | **11.4 MB** |
| prebuilt wheels | — | macOS arm64 + x86_64, Windows win32 + amd64, manylinux x86_64 + aarch64 |

The model exists as a community ONNX export at
`istupakov/parakeet-tdt-0.6b-v3-onnx`, alongside NVIDIA's own
`nvidia/parakeet-tdt-0.6b-v3`. One small dependency with prebuilt wheels on all
three platforms fits this project's stdlib-first rule far better than dragging in
torch. Neither has been run here.

**Not carried over from the source that suggested this:** the throughput and WER
figures that circulate for these models ("3,300x real-time", "6.3% WER") come
from vendor and SEO blog posts, not from a benchmark run here or a paper. They
are deliberately absent. This project's own history includes a plausible claim
("large-v3 beats turbo on accented English") that did not survive being checked.

## Parakeet: now actually run

Verified 2026-09-04 via `parakeet-mlx` 0.5.2 on Apple Silicon, model
`mlx-community/parakeet-tdt-0.6b-v3`. It transcribed correctly and was the
fastest local backend measured (see the benchmark in README.md).

What came back, as opposed to what the docs implied:

- **`AlignedSentence` carries `text`, `tokens`, `start`, `end`, `duration` and
  `confidence`.** No `avg_logprob`, no `compression_ratio`, no `no_speech_prob` —
  exactly as expected for a CTC/TDT model, and the reason those fields are `None`.
- **`confidence` (0–1) is recorded on each segment but never thresholded.** It is
  not on `avg_logprob`'s scale, and this project does not ship a threshold it has
  not measured.
- **It segments much more finely**: four or five sentence-level segments where
  Whisper produced one or two.

### The timestamp problem, and the fix

Parakeet's segment boundaries were far looser than Whisper's — and it turned out
to be the decoder, not the model.

Measured on a 48.8 s file whose speech ends at 40.79 s:

| decoding | final segment | overrun | time |
|---|---|---|---|
| greedy (was the default) | 36.64–**48.48 s** | **+7.69 s** | 3.2 s |
| beam (now the default) | 36.64–**41.52 s** | **+0.73 s** | 2.7 s |

A tenfold reduction, and faster. Whisper's final segment on the same file ended
at 40.74 s, so beam-decoded Parakeet is now in the same league.

This is not a guess. `parakeet-mlx`'s maintainer attributes abnormal segment
timestamps to greedy TDT decoding and recommends beam search
([senstella/parakeet-mlx#43](https://github.com/senstella/parakeet-mlx/issues/43)),
and the same class of anomaly appears in a different TDT implementation on the
same NVIDIA weights
([FluidInference/FluidAudio#128](https://github.com/FluidInference/FluidAudio/issues/128)) —
so it is a decoder issue, not a quirk of the MLX port. NVIDIA's own NeMo
timestamp documentation and the TDT paper (arXiv:2304.06795) say nothing about
boundary accuracy either way.

The greedy overrun is what broke the first version of the silence guard, which
asked whether a segment's *midpoint* fell inside a silent span. It did, and real
speech was suppressed. Two changes came out of that, and they are independent:
beam decoding fixes the cause, and containment fixes the rule.

### Measured on a real recording

On the same 2m19s laptop-mic recording used for the other backends, scoring 12
planted hard items: **Scribe 10/12** — behind OpenAI's 11 and ahead of every
local backend (mlx-turbo 9, faster-whisper 9, mlx-large 8, parakeet 8). Decode
20 s, cost $0.0073 for the 120 s actually sent after trimming.

It recovered "Terragrunt", which only `large-v3` and OpenAI also managed, and it
captured both halves of a spoken self-correction that `large-v3` lost. Segment
count was 10 against 15–25 elsewhere — longer, more prose-like spans.

### Where this sits against how other tools do it

Worth knowing, because it is not the common pattern. WhisperX and
`faster-whisper`'s `vad_filter` both run VAD **before** decoding and gate the
audio fed to the model; `whisper.cpp` treats VAD as optional rather than
corrective. None of them reconciles decoder segments against independently
measured silence *after* the fact.

This skill does both, and on the default path the first one is what matters:
**silence-trimming before decoding is a VAD-first approach**, and it is why the
default path produced zero invented segments in every test. The containment rule
is the safety net for when trimming is off, when a quiet stretch falls below the
detector's threshold, or when a backend has no VAD gating of its own — which is
exactly Parakeet's situation.

No published containment or overlap threshold exists to calibrate against. The
nearest prior art is an unshipped proposal in
[chidiwilliams/buzz#1570](https://github.com/chidiwilliams/buzz/issues/1570)
suggesting an *overlap* test with no number attached.

### The non-Apple runtime is still open

The cross-platform NeMo path (`nemo_toolkit[asr]`) is **not** verified.
`scripts/tslib/backends/parakeet.py` supports both runtimes and prefers
`parakeet-mlx` when importable. Note that the beam-decoding fix above is applied
on the MLX path only; whether NeMo's own decoder shows the same overrun is
untested.

## ElevenLabs Scribe: the only backend that knows who spoke

Verified against the API reference 2026-09-04: `POST https://api.elevenlabs.io/v1/speech-to-text`,
auth header `xi-api-key` (**not** bearer), `model_id=scribe_v2`, files to 3 GB, audio to 10 hours.

Two things make it unlike everything else here:

**It returns words, not segments.** There is no `segments` array at all — just
`words[]` with `text`, `type`, `logprob`, `start`, `end`, `speaker_id`. The
segmentation is ours, grouping on speaker change and a 1 s pause. Every other
backend hands us segments.

**It diarizes, up to 32 speakers.** No Whisper backend and not Parakeet returns
any speaker field. This is the one place attribution is available at all.

**Verified live on 2026-09-04**, not merely documented: two distinct voices came
back as `speaker_0` and `speaker_1`; the same audio with `diarize=false` returned
no speaker ids at all. `num_speakers=2` was not needed. Also confirmed against
the live response: Whisper's three metrics really are absent and the per-word
`logprob` really is present, so `has_whisper_metrics=False` is measured rather
than assumed.

A caution from getting this wrong once: the first live run reported ONE speaker
and looked like an API failure. It was the test fixture — both voice clips were
rendered with macOS `say`'s default voice, which on that machine is Daniel, so
they were byte-identical. Scribe was right. If a diarization test ever reports
one speaker, check the fixture actually contains two before blaming the vendor.

That does **not** make attribution automatic. Scribe returns `speaker_0`,
`speaker_1` — positional labels, not names. A person still maps labels to people,
exactly as before, and `notes_check.py` still rejects a raw label in a notes
document. What changes is that the mapping is now *possible from the transcript*
rather than needing to be reconstructed from memory.

**Quality metrics: partial.** Scribe gives a per-word `logprob` and none of
Whisper's three segment metrics, so those stay `None` and the metric rules do not
fire. The mean word logprob is recorded as `confidence` and used only to rank
what a human should check — never to suppress. It is a log probability and so
*looks* like `avg_logprob`, but it is computed differently, and this project does
not threshold a number it has not calibrated. The backend-independent rules
(`decoded_from_silence`, `repeated_token`) apply here as they do to Parakeet.

**$0.22 per hour**, verified from elevenlabs.io/pricing/api on 2026-09-04 and
flat across every plan tier — only the included hours differ, not the rate.
Realtime is $0.39. That puts Scribe between Groq and OpenAI, and it is the only
one of the four that diarizes.

Diarization is on by default in the backend and is not currently exposed as a CLI
flag; if you need it off, that is a small addition to `transcribe.py`.

## Network backends: cost and limits

Per hour of audio, from the providers' own documentation:

| backend | model | $/hour |
|---|---|---|
| groq | `whisper-large-v3` | 0.111 |
| groq | `whisper-large-v3-turbo` | 0.04 |
| elevenlabs | `scribe_v2` | 0.22 |
| openai | `whisper-1` | 0.36 |

Groq caps uploads at **25 MB** (free tier) and 100 MB (dev tier). The tool
refuses an oversized file with the shrink command rather than chunking silently:

```
ffmpeg -i <in> -ar 16000 -ac 1 -map 0:a -c:a flac <out>.flac
```

**Trimming happens before uploading, so it cuts the bill too.** Measured on a
47-second call with a silent head: 12 seconds were actually sent, about a
quarter of the file. The disclosure block prints the *whole* file's duration,
because it has to print before any work happens — it says "up to" and notes
that the real figure is lower.

Before anything is sent, the tool prints the provider, endpoint host, file size,
**duration measured with `ffprobe` on the original file**, model and estimated
cost, then waits for confirmation. `--dry-run` prints that block and sends
nothing. In a non-interactive session an unconfirmed upload is refused, not
assumed — a skill running unattended must not upload because nobody was there to
say no.

API keys are read from `GROQ_API_KEY` / `OPENAI_API_KEY` only. Never a flag,
never printed, never written to the run manifest.

## No backend removes filler — not even the hosted ones

A common assumption is that the cloud models clean up speech and the local ones do not. Measured
2026-09-04 on audio containing three "um", three "uh", one "er", plus "I mean" and "you know":

| backend | filler kept | hedges kept |
|---|---|---|
| mlx-whisper turbo (local) | 5 | 2 |
| faster-whisper (local) | 5 | 2 |
| parakeet (local) | 5 | 2 |
| **openai whisper-1 (hosted)** | **5** | **2** |

Identical output, to the token. Whisper is trained to transcribe verbatim, and Parakeet behaves
the same way. **Paying for a hosted model does not buy you disfluency removal.**

That is not a defect. A transcript that silently drops words is worse than one that keeps them —
you cannot tell what was removed. Cleanup belongs in a later pass that knows it is editing.

In this skill that pass is the notes step, and it is where filler comes out. Products that appear
to transcribe cleanly are doing the same thing: ASR first, then a second model that rewrites.
**If you skip the notes step you have a verbatim transcript, and nothing has cleaned it.**

## Preparation is shared, and it matters more than the backend

Every backend receives the same prepared audio: decoded to 16 kHz mono, loudness
normalised, silence-trimmed. That step is not an optimisation. Measured on a real
recording, cleaning the audio took invented segments from 2 to 0 and fixed a word
that both `large-v3` and `turbo` had decoded wrong on the raw file — a bigger
model did not fix what cleaning did.

`loudnorm`, `silencedetect`, `atrim` and `concat` are unconditional libavfilter
built-ins, so this runs identically on all three platforms with no extra install.
