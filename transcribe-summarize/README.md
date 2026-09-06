# transcribe-summarize

![transcribe-summarize](images/banner.webp)

<!-- banner.png is the same image for anywhere WebP is not supported; images/README.md explains the set -->

Transcribe audio on your own machine, on macOS, Windows or Linux. Optionally turn
the result into meeting notes and a PDF.

Nothing leaves your computer unless you name a network backend in the command,
and when you do, the tool tells you what it is about to send and what it will
cost before it sends it.

## What it does

| in | what happens | out |
|---|---|---|
| **audio** — `.mp3`, `.m4a`, `.wav`, `.flac`, `.opus` | normalise, trim silence, decode, filter inventions | transcript `.md` + `.srt` + `.json` |
| **video** — `.mp4`, `.mov`, `.mkv`, `.webm` | ffmpeg takes the audio track; the video is ignored | same as above |
| **an existing transcript** — `.vtt`, `.srt` from Teams or Zoom | cues rejoined, timecodes dropped, speakers attributed | readable `.md`, no transcription at all |
| any of the above | you review, then Claude writes it up | meeting notes `.md` + `.pdf` |

A Teams or Zoom download is an `.mp4` and works as-is — you do not extract the
audio first. And if the platform already transcribed the meeting, skip the audio
entirely: `normalise_transcript.py` turns the `.vtt` into prose and the notes step
is identical.

```bash
# transcribing needs a backend, and the backend comes from --with
uv run --with 'mlx-whisper>=0.4.2' --script scripts/transcribe.py meeting.mp4

# everything else is stdlib and runs directly
./scripts/normalise_transcript.py meeting.vtt          # an existing transcript
./scripts/notes_check.py notes.md                      # check the notes register
./scripts/render_pdf.py notes.md                       # notes -> PDF
```

**Only `transcribe.py` needs `--with`.** It declares no dependencies of its own, so
installing this skill does not drag in every engine — the backend you pick brings
its own library and only when you pick it. Run it without one and it exits 1 with
the exact command to use. The `--with` spec per backend is in the table below.

A file with **no** audio track — a muted screen recording — is refused before any
upload is offered, so it is never sent or billed.

**Sovereign by default.** The local path never touches the network. Whisper runs
on your machine — `mlx-whisper` on Apple Silicon, `faster-whisper` elsewhere —
and `--backend auto` has no code path to a hosted API, asserted across twelve
platform combinations. Hosted backends exist and are reachable only by naming one
in that invocation, after a disclosure of what will be sent and what it costs.

## What it does NOT do

Worth being explicit, because several of these are things people assume:

- **It does not verify anything.** Output is a draft. Machine transcription is
  confidently wrong in ways no threshold catches — measured here, `large-v3`
  silently dropped a speaker correcting a figure, and *every* backend heard
  "board pack" as "board packs up". A human has to read it.
- **It does not remove filler by transcribing.** Whisper is trained to
  transcribe verbatim and Parakeet behaves the same way, so a hosted Whisper
  buys you nothing here that a local one does not. Gemini is the one backend here that
  advertises otherwise — its `smart` mode is documented to strip filler and
  rewrite spoken self-corrections — and **this tool deliberately does not offer
  that mode**, because Google's API refuses word timestamps and speaker labels
  alongside it, which would leave no clock to map back onto your file and no
  `.srt` at all. Filler comes out at the **notes** step instead, so a raw
  transcript stays verbatim and uncleaned.
- **It does not identify who is speaking**, except on `--backend elevenlabs` (up
  to 32 speakers) or `--backend gemini` (up to 8), and even there you get
  `speaker_0` or `spk:0`, not names. Attendees are something you tell it.
- **It does not translate**, do real-time streaming, or handle multi-track audio.
- **It does not edit your words.** The transcript is verbatim; judgement happens
  in the notes, under a fixed register.
- **It does not make a recording lawful.** Keeping provenance out of a document
  changes what the document says, not what happened. Consent is yours to obtain.
- **The quality guard is not equally strong everywhere.** Parakeet and ElevenLabs
  return no Whisper metrics, so only the silence and repetition rules apply.

## Why this exists

The obvious tools either only run on one platform or quietly upload your audio.
The specific failure worth avoiding is not "uses an API" — it is *uploaded the
audio and you could not tell*.

Beyond that, two findings from a day of measurement against real recordings shape
the whole design:

**Whisper invents speech over silence, and a bigger model does not save you.** On
a call with about 40 seconds of joining silence at −69 dB, `large-v3` emitted an
18-second segment consisting of one word repeated 55 times, plus a spurious
"Thank you." Both were written to the output as if real. Both were detectable
from numbers Whisper already returns:

| metric | on the hallucination | Whisper's own threshold |
|---|---|---|
| `compression_ratio` | 17.38 | 2.4 |
| `no_speech_prob` | 0.708 and 0.923 | 0.6 |

This tool filters them out and keeps them in the JSON sidecar with the reason, so
a gap in the transcript is always answerable.

**Cleaning the audio beat choosing a model.** Same file, same model, same flags —
normalising and silence-trimming took invented segments from 2 to 0 and fixed a
word that both `large-v3` and `turbo` had decoded wrong on the raw audio. It costs
nothing in throughput. So it is on by default, done with ffmpeg, on every
platform.

## Install

Three ways in, depending on what you want.

**As a standalone plugin** — this repo on its own:

```bash
claude plugin marketplace add kevin-burns/transcribe-summarize
claude plugin install transcribe-summarize@transcribe-summarize
```

**As part of the collection** — all the skills, one plugin:

```bash
claude plugin marketplace add kevin-burns/claude-skills
claude plugin install claude-skills@kevin-burns
```

**By hand**, if you would rather not use a marketplace: copy the `transcribe-summarize/`
directory into `~/.claude/skills/`. Everything the skill needs is inside it.

### Check it actually loaded

Do not take the install command's word for it. This repo has shipped plugin manifests that
failed **silently** — registering cleanly and listing zero plugins. Verify:

```bash
claude plugin details transcribe-summarize@transcribe-summarize
```

You want `Skills (1)  transcribe-summarize` under **Component inventory**. Zero means the
manifest registered and the skill did not load.

**A trap worth knowing:** `claude --safe-mode` disables plugin-provided skills, so a skill
installed this way will not appear in a safe-mode session. That is not a broken install.

### Prerequisite

`ffmpeg` and `ffprobe` on PATH:

```bash
brew install ffmpeg                  # macOS
winget install Gyan.FFmpeg           # Windows
sudo apt install ffmpeg              # Debian/Ubuntu
```

Everything else is Python standard library. Each transcription backend brings its
own dependency, and only when you choose it, so installing this does not drag in
every engine.

## Use

```bash
# Apple Silicon
uv run --with 'mlx-whisper>=0.4.2' --script scripts/transcribe.py meeting.m4a

# Windows / Linux
uv run --with 'faster-whisper>=1.2' --script scripts/transcribe.py meeting.m4a \
    --backend faster-whisper

# A network backend, when you want one. Nothing to install -- these are stdlib
# http.client -- but you must name the backend, and you will be asked first.
uv run --script scripts/transcribe.py meeting.m4a --backend gemini --dry-run
uv run --script scripts/transcribe.py meeting.m4a --backend gemini
```

You get:

```
meeting.md                      the transcript, with [hh:mm:ss] anchors
meeting-artifacts/
    meeting.srt                 subtitles
    meeting.json                every segment, including the suppressed ones
    meeting.run.json            backend, model, trim offsets, guard tally
```

Timestamps are on **your** file's clock. Trimming shortens the audio while it is
being decoded; the offsets are mapped back before anything is written, so the
`.srt` still lines up with the recording on your disk.

Useful flags:

```
--prompt 'Terragrunt, EKS'        bias the decoder toward known jargon
--replace 'north wind=Northwind'      fix a known misrecognition, repeatable
--guard mark                      keep suspect segments, flagged, instead of dropping
--no-trim / --no-normalise        turn off audio preparation
--keep-intermediate               keep the prepared wav to listen to
--dry-run                         say what would happen; decode nothing
--task translate                  output English instead of the spoken language
--multilingual                    re-detect the language on every segment
```

### Another language, and more than one of them

**`--task translate` gives you English.** Whisper has a second task built in and
it is `X -> English` — there is no other target, which is why this is a task and
not a `--target-lang`. Verified in `mlx_whisper/decoding.py`, whose own comment
reads *"whether to perform X->X `transcribe` or X->English `translate`"*.

```bash
uv run --with 'mlx-whisper>=0.4.2' --script scripts/transcribe.py call.m4a \
    --backend mlx-whisper --model large-v3 --task translate --lang auto
```

Three things to know, all of them measured on 2026-09-06:

- **`turbo` cannot translate, and does not say so.** `large-v3-turbo` is a
  distillation that dropped the task. Asked to translate German it returned the
  German, silently. Since turbo is the default, `--task translate` **refuses**
  unless you pass `--model large-v3`. Groq documents the same for their hosted
  turbo, so it is the model and not the host.
- **Only Whisper translates.** `parakeet`, `elevenlabs` and `gemini` refuse
  rather than transcribing and letting you find out later — and Gemini's refusal
  is measured, not assumed: asked three different ways to produce English from
  German audio, it returned German every time. `groq` and `openai` translate
  through a different endpoint, `/audio/translations`, English-only by their own
  documentation; the OpenAI route is verified live, Groq's is not, for want of a
  key on this machine.
- **A translated transcript says so, at the top**, because it reads exactly like
  a verbatim one: *"**Translated to English.** These are not the words that were
  spoken; the audio is in de."*

### A meeting where people speak different languages

This is the common case — you speak English, a colleague answers in German — and
it is the one where the default backend on a Mac gets it wrong.

**Whisper decides the language once, from the first 30 seconds, and applies it to
the whole file.** Measured on a 43 s clip: 29.5 s of English, then German.

| backend | the German half came back as |
|---|---|
| `mlx-whisper` (Apple Silicon **default**) | **"the budget is located at 42.000€ against a prognosis of 38.000€ … I'm going to send the show today to the next day"** |
| `faster-whisper` | "das Budget lag bei 42.000 Euro gegenüber einer Prognose von 38.000 …" |
| `parakeet` | "das Budget lag by 42.000 gegenüber einer Prognose von 38.000 …" |
| `gemini` | "das Budget lag bei 42.000 Euro gegenüber einer Prognose von 38.000 …" |

`mlx-whisper` did not decode the German badly. It **translated it into English**,
fluently, with nothing in the output saying so — and the last sentence, *"Ich
schicke die Aufstellung heute Nachmittag herum"* ("I'll circulate the breakdown
this afternoon"), became **"I'm going to send the show today to the next day"**.
Confident, readable, and wrong. That is the failure this whole tool exists to
surface, so the caveat now prints on every run of a backend that cannot do better,
whether or not you passed a flag.

**So: on a mixed-language recording, do not use `mlx-whisper`.** Use
`faster-whisper`, `parakeet` or `gemini`.

**Where the boundary actually is.** On a 25 s version of the same conversation —
English for 11 s, then German — **every** backend got it right, `mlx-whisper`
included, because the whole clip fits inside one 30-second decode window. The
failure needs a later window to be a different language than the first, which is
what any real meeting looks like and what a short test clip does not. Two earlier
tests here proved nothing for exactly that reason, and are written up in
`references/backends.md` rather than quietly discarded.

**`--multilingual` re-runs detection on every segment** and works on
`faster-whisper` only; `gemini` does it unconditionally. Everything else refuses
the flag rather than accepting and ignoring it. Honest caveat: on both mixed clips
tested here `faster-whisper large-v3` was **already correct without it**, so the
flag has not yet been shown to rescue a case that would otherwise fail — it is
there because the engine's own detection is documented to be per-file by default,
and one recording is not a proof.

### Which language do I get?

**The transcript is always in the language that was spoken.** There is no default
to English and no setting that changes it — German audio produces a German
transcript on every backend here, verified on all five.

**`--task translate` produces English, and only English.** Not "a target
language" — Whisper has exactly one translation direction. There is no way to ask
any backend in this skill for a German transcript of English speech.

**So if your colleague wants it in German, that is the notes step, not the
transcript.** The transcript stays verbatim in whatever was said; the summary is
written by a model that will write it in any language you ask for. That split is
deliberate — translating a record changes what the record *is*, and a
record-of-what-was-said should not quietly become a translation of it.

## Backends

| backend | platform | quality guard | network |
|---|---|---|---|
| `mlx-whisper` | Apple Silicon only | full | no |
| `faster-whisper` | mac / Windows / Linux | full | no |
| `parakeet` | mac / Windows / Linux | **repetition rule only** | no |
| `groq` | any | full | **yes** |
| `openai` | any | full | **yes** |
| `elevenlabs` (Scribe) | any | partial | **yes** |
| `gemini` (3.5 Transcribe) | any | partial | **yes** |

`--backend auto` picks `mlx-whisper` on Apple Silicon and `faster-whisper`
elsewhere. **It can never pick a network backend** — not as a default, not as a
fallback when a local backend fails. There is no code path from `auto` to one.

The guard's strong rules read Whisper decoder metrics. `faster-whisper`, Groq and
OpenAI all return them, so the guard is literally the same code. **Parakeet
returns none of them** and falls back to a repetition heuristic; the tool prints
that fact on every run rather than letting you infer it from a clean-looking
transcript. **Parakeet runs on Apple Silicon only today**, via `parakeet-mlx`; the
cross-platform path is not written yet, and the plan is `sherpa-onnx` (one
dependency, an 11 MB wheel) rather than `nemo_toolkit[asr]` (torch, multi-GB) —
see `references/backends.md`. It does handle other languages: German audio came
back as German, verified 2026-09-06.

## What these backends actually did

Re-measured **2026-09-06** on an Apple M1 Pro / 16 GB / macOS 26.6.2 / ffmpeg 9.0.1,
against a **2 min 19 s recording made with a laptop's built-in microphone** — one
speaker, a real room, nothing done to it before the tool saw it. Model weights
were already cached. Silence-trimming removed 18.9 s of 138.7 s (13.6%) before
decoding, identically for every backend.

**Accuracy is now a score you can recompute.** The fourteen checks live in
`evals/accuracy/checks.json` and `evals/accuracy/score.py` scores a transcript
against them. The previous version of this table said `N / 12` against a
checklist that was never written down, so no figure in it could be confirmed or
refuted; that is what these files exist to prevent. The set turned out to be
fourteen, not twelve, and it is not trimmed to match a number nobody derived.

Local backends ran three times, network backends twice.

| backend | model | accuracy | decode, median (range) | what it got wrong |
|---|---|---|---|---|
| `openai` | whisper-1 | **13 / 14** | 10.1 s (9.6–10.6) | board pack |
| `gemini` | gemini-3.5-transcribe | 12 / 14 | 11.5 s (11.4–11.6) | Terragrunt, board pack |
| `mlx-whisper` | **turbo** (default) | 12 / 14 | 14.5 s (13.4–15.5) | Terragrunt, board pack |
| `faster-whisper` | **turbo** (default) | 12 / 14 | 24.2 s (24.1–24.6) | Terragrunt, board pack |
| `parakeet` | parakeet-tdt-0.6b-v3 | 11 / 14 | 13.3 s (13.1–18.9) | Terragrunt, Raghunathan, board pack |
| `mlx-whisper` | large-v3 | 11 / 14 | 22.0 s (21.1–23.2) | AKS, 99.95% vs 99.5%, board pack |
| `faster-whisper` | large-v3 | 11 / 14 | 69.5 s (62.6–78.2) | Terragrunt, AKS, board pack |
| `elevenlabs` | scribe_v2 | **8 / 14** and 12 / 14 | 9.5 s (9.5–9.6) | varies — see below |

**Read the decode column as an order of magnitude, not a benchmark.** The range is
there because the same model on the same file moves: `faster-whisper large-v3`
spanned 62.6–78.2 s across three runs, a 25% spread. Two rows separated by less
than a few seconds are not separated.

**ElevenLabs returned a different transcript on each of two identical runs**, and
the scores were 8 and 12. The worse run dropped a budget figure the better one
kept. Every other backend here scored the same on every run. That is worth more
than the score itself: a backend whose output is not stable is one you cannot
diff against yesterday's, and nothing in the response says which run you got.

**Every backend fails "board pack", every time.** All eight rows, both the local
and the hosted ones, render it as "board packs up". It is grammatical, it is
fluent, and it is wrong — which is exactly why the check is kept in the set and
why the transcript ships with a "Worth checking" section instead of a claim to be
correct.

### What the numbers actually say

**The old default was the worst Whisper result here, so the default changed.**
`large-v3` scored 11/14 against turbo's 12/14 while taking 52% longer (22.0 s
against 14.5 s), and its failure was the serious kind: the speaker corrects
himself mid-sentence — "oh, actually, correction, that was 99.95%" — and
`large-v3` renders the figure as "99.5%, not Not 99.5%". It keeps the word
"correction" and loses the number the correction was about, which is worse than
dropping the sentence, because what survives reads like a correction that was
captured. In a document meant to record what was said, that is the worst
available failure.

**Both Whisper backends therefore default to `turbo`.** `--model large-v3` is
still there. Note that turbo **cannot translate** — see `--task` below — so
`--task translate` requires `--model large-v3`.

One recording is not a benchmark, and this does not establish that turbo beats
large-v3 in general. It does mean the common claim that large-v3 is worth its
extra time on accented English is **not supported by the only real measurement
this project has**. If large-v3 is better on your audio, measure it and use it.

(`faster-whisper` resolves `turbo` to `mobiuslabsgmbh/faster-whisper-large-v3-turbo`
rather than a Systran repo — a different publisher from its other short names.
It ran at 24.2 s here. An earlier run of the same configuration recorded 50.7 s,
which is the clearest single illustration of why the decode column carries a
range: nothing about the model changed between them.)

**Nothing recovered "board pack".** All eight rows heard "board packs up". Some
errors are in the audio, not the model.

**Terragrunt split the field three to five.** `openai`, `mlx-whisper large-v3`
and `elevenlabs` got it; `gemini`, both turbo builds, `faster-whisper large-v3`
and `parakeet` produced "terror grunt" — two ordinary words, which is why no
confidence threshold flags it.

**Two backends lost the 99.95% distinction**: `mlx-whisper large-v3` and
`elevenlabs`. Every other row kept it. All eight kept the *word* "correction",
which is the trap — a transcript can preserve the fact that a correction happened
and still get the corrected value wrong.

**Scribe produced the fewest, longest segments** — 10, against 12 for Gemini, 15
for OpenAI and 20–25 for the rest. That reads better as prose and matters if
anything downstream assumes a segment is a fixed unit. It is also the backend
whose output was not reproducible between runs, so treat the segment count as
descriptive of one run.

Its diarization returned one speaker here, correctly: this is a single-speaker
recording. Two voices were separated in a dedicated live test.

**Proper nouns are what `--prompt` and `--replace` exist for**, and both work:

```bash
--prompt 'Terragrunt, EKS, AKS, Cloudflare Access'    # recovered Terragrunt on turbo
--replace 'terror grunt=Terragrunt'                   # deterministic, reported "2 correction(s)"
```

**Speed ordering held from the synthetic run**: `faster-whisper` on CPU was
roughly 4× slower than `mlx-whisper` on the Apple GPU. On a CUDA machine that
ordering would likely differ, and was not measured here.

### One boundary difference worth knowing about

Whisper's segment end times hug the speech. Parakeet's do not: on this file its
final segment ran to **48.48 s** on audio whose speech stopped at **40.79 s** —
nearly eight seconds of overrun to near end-of-file.

That matters because the silence guard compares segments against measured silent
spans. An earlier version of the rule asked whether a segment's *midpoint* fell
inside a silence, which is fine for Whisper and wrong for Parakeet: the
overrunning segment's midpoint landed in the trailing silence and **real speech
was suppressed**. A guard that deletes genuine content is worse than one that
misses an invention.

Two independent fixes came out of it:

- **The rule now requires containment** — the segment must lie wholly inside one
  silent span. A segment that starts before the silence cannot have been decoded
  from it.
- **Parakeet now decodes with beam search rather than greedy.** Its maintainer
  attributes the timestamp anomaly to greedy TDT decoding
  ([parakeet-mlx#43](https://github.com/senstella/parakeet-mlx/issues/43)), and
  the same behaviour appears in another TDT implementation on the same weights
  ([FluidAudio#128](https://github.com/FluidInference/FluidAudio/issues/128)).
  Measured here: the overrun fell from **+7.69 s to +0.73 s**, and decoding got
  slightly *faster*. Details in `references/backends.md`.

## Cost, if you use a network backend

Per hour of audio, from each provider's published pricing (verified 2026-09-04):

| backend | model | $/hour | diarizes |
|---|---|---|---|
| `groq` | whisper-large-v3-turbo | **0.04** | no |
| `groq` | whisper-large-v3 | 0.111 | no |
| `elevenlabs` | scribe_v2 | 0.22 | **yes, up to 32** |
| `gemini` | gemini-3.5-transcribe | 0.306 | **yes, up to 8** |
| `openai` | whisper-1 | 0.36 | no |

Silence-trimming happens *before* upload, so you are billed for the trimmed
duration — measured at 12 s sent from a 47 s file. The disclosure block prints
"up to" for that reason.

A model with no published rate yields no estimate, and the disclosure says
"unknown for this model". That means *cannot estimate*, never *free*.

## Sending audio to an API

Only when you ask for it by name. First you get this, and a prompt:

```
  ── this will send your audio over the network ──────────────
  provider : groq
  endpoint : api.groq.com
  file     : meeting.m4a  (3.1 MB)
  duration : 00:41:07
  model    : whisper-large-v3
  cost     : ~$0.076
  ────────────────────────────────────────────────────────────
```

Silence is trimmed *before* uploading, so it cuts the bill as well as the
hallucinations — on a 47-second call with a silent head, 12 seconds were
actually sent. The block says "up to" because it has to print before any work
happens. In a non-interactive session an unconfirmed upload is refused rather
than assumed.

Keys are read from the environment, one per backend, and nowhere else — never a
flag, never printed, never written into the run manifest:

| backend | environment variable | endpoint |
|---|---|---|
| `groq` | `GROQ_API_KEY` | `api.groq.com` |
| `openai` | `OPENAI_API_KEY` | `api.openai.com` |
| `elevenlabs` | `ELEVENLABS_API_KEY` | `api.elevenlabs.io` |
| `gemini` | `GEMINI_API_KEY` | `generativelanguage.googleapis.com` |

### `--backend gemini` has two properties the others do not

Verified against Google's API reference and a live run on **2026-09-06**:

- **It uploads twice.** Gemini has no single-shot transcription endpoint. The
  audio goes to Google's Files API first, and the transcription request then
  carries only the URI that came back. **Google keeps that upload for 48 hours**
  before deleting it — the disclosure block says so, because you cannot see it
  from the command line otherwise. The second leg posts to a URL Google chooses
  and returns in a header; that URL is checked for `https` and a
  `googleapis.com` host before any audio follows it.
- **Its ceiling is 30 minutes, not the hour Google's page leads with.** The hour
  applies only when word timestamps and diarization are off, and this backend
  always asks for both — without word offsets there is no clock to map back onto
  your file, and the `.srt` could not be built. A longer recording is refused
  before the upload starts, not after.

**Gemini's `smart` mode is deliberately not offered.** It is the model's headline
feature — it strips filler words, resolves spoken self-corrections and reflows
the text — and Google's own reference says why it cannot be used here: *"Smart
transcription (`"smart"`) is incompatible with `timestamp_granularities` and
`diarization_mode`."* No word timing, no speaker labels, and a transcript that is
a model's tidied version of what was said rather than what was said. Cleaning up
the prose belongs at the **notes** step, one stage later, where it is labelled a
summary and checked. See `references/backends.md`.

## What has actually been run

Every claim on this page came from one of three places, and they are not worth the same.
This table says which, per path, as of **2026-09-06**. It exists because three separate
things in this repository were written from a vendor's documentation and turned out to be
wrong the first time a real response came back.

**live** — exercised end to end against the real engine or API, and asserted.
**offline** — request shape, parsing and refusals covered by tests, response never seen.
**unrun** — implemented, reviewed, never executed. Believe accordingly.

| path | status | what was run |
|---|---|---|
| `mlx-whisper` transcribe | **live** | 3 runs on the 2 min 19 s recording, scored 12/14 and 11/14 |
| `mlx-whisper` `--task translate` | **live** | German clip → English; and turbo correctly refused |
| `faster-whisper` transcribe | **live** | 3 runs, both models |
| `faster-whisper` `--task translate` | **live** | German clip → English, `large-v3` |
| `faster-whisper` `--multilingual` | **live** | 35 s German→Spanish clip, both with and without the flag |
| `parakeet` | **live** | 3 runs, `parakeet-mlx` on Apple Silicon |
| `parakeet` on German | **live** | returned German, confirming the multilingual claim |
| `parakeet` on a non-Apple runtime | **unrun** | needs `sherpa-onnx`; not built yet |
| mixed-language recording | **live** | 43 s English→German through all five backends |
| `openai` transcribe | **live** | 2 runs, plus a dedicated live test |
| `openai` `--task translate` | **unrun** | endpoint verified from OpenAI's docs only |
| `elevenlabs` transcribe | **live** | 2 runs, plus a live test |
| `elevenlabs` diarization | **live** | two synthesised voices → `speaker_0` / `speaker_1`, format asserted |
| `gemini` transcribe | **live** | 2 runs, plus 5 live tests |
| `gemini` diarization | **live** | two voices → `spk:0` / `spk:1`, format asserted |
| `gemini` cannot translate | **live** | probed three ways on German audio; all returned German |
| `groq` (any path) | **unrun** | no API key on the machine this was built on |

**The two `unrun` API rows are the honest risk here.** Both are shaped from published
documentation and covered by offline tests, which is exactly the state Gemini was in the
morning of 2026-09-06 — when its first live response returned `spk:0` and corrected two
documents that had confidently said `spk_1`. A request that is built correctly is not a
response you have seen.

Three findings that only a live run produced, kept here as the argument for the column:

- **`large-v3-turbo` cannot translate and does not fail.** It returned German under a
  header saying "Translated to English". Turbo is the default model, so the broken path
  was the common one. Every unit test passed.
- **Gemini's speaker labels use a colon.** Written as `spk_1` from the API reference; they
  are `spk:0`. The checker built from the reference still missed the real string.
- **ElevenLabs is not reproducible.** Two identical runs returned different transcripts,
  scoring 8/14 and 12/14; the worse one dropped a budget figure. Nothing in either response
  indicates which you got.
- **Gemini does not translate**, despite being a Gemini model and despite "smart
  transcription" sounding like it might. Probed three ways on German audio — the shipped
  config, `language_codes: ["en-US"]`, and a plain instruction *"give the result in
  English"* sent alongside the audio. All three returned the German unchanged.
- **The Apple Silicon default silently translates a mixed-language call.** 29.5 s of
  English then German: `mlx-whisper` returned the German half in English, rendering "Ich
  schicke die Aufstellung heute Nachmittag herum" as "I'm going to send the show today to
  the next day". Every other backend returned German. Two shorter versions of the same test
  passed on every backend, because the switch fell inside the first decode window.
- **`--prompt` was an unconditional HTTP 400 on Gemini.** `custom_vocabulary` cannot be
  combined with the word timestamps this backend always requests. The failure arrived after
  the upload, so the audio had already been sent and billed.
- **Gemini's documented `output_text` field does not exist.** The API reference shows
  `"output_text": "transcribed text"` at the top level of the reply; it was absent from all
  four live responses. The transcript is in `steps[].content[].text`. Reading the documented
  field would have silently fallen back to text this repository reassembled itself.

## It produces a draft, not a verified record

Machine transcription is confidently wrong in ways no threshold catches.
Measured here: `large-v3` silently dropped a speaker correcting a figure, and
**every** backend heard "board pack" as "board packs up". Both read fluently. The
quality guard catches text invented over silence; it cannot catch a plausible
mis-hearing.

So every transcript ends with a **"Worth checking"** section — ranked timestamps
where a listen-back is most likely to pay off, chosen from the decoder's own
least-confident moments and anything containing figures:

```
## Worth checking

- **[00:01:27]** contains figures — our uptime figure last quarter was 99.5%...
- **[00:01:37]** contains figures — correction, that was 99.95%, not 99.5%...
```

Those are hints, not errors. Names, figures and dates are what to verify: first
to degrade, most expensive to get wrong.

The `.md` is the source of truth. Correct it by hand (or re-run with `--replace`),
then rebuild the PDF without re-transcribing:

```bash
./scripts/notes_check.py notes.md      # run this again after every edit
./scripts/render_pdf.py  notes.md
```

Re-run the checker after editing — hand-editing is exactly when a provenance leak
or an inferred "next steps" line gets introduced.

## Meeting notes

The notes document is a **factual summary and nothing else**: what was said,
under topic headings. No analysis, no implications, no "next steps", no "what the
call did not cover". It reads as a write-up by someone who was in the room, and
it never mentions a recording, a transcript, or that a machine was involved.

**It cannot tell you who was speaking.** No Whisper backend returns a speaker
field — not mlx-whisper, faster-whisper, Groq or OpenAI. Two network backends do
diarize: `elevenlabs` (up to 32 speakers) and `gemini` (up to 8). Neither solves
this, because both return positional labels rather than names — `speaker_0` and
`spk:0`, the formats measured from live responses on 2026-09-04 and 2026-09-06 —
and this register forbids a raw decoder label outright. What they buy you is that
the mapping is possible *from the transcript* instead of from memory. The
attendees, the meeting date and the title are still supplied by you:

```bash
./scripts/summarize.py transcript.md --backend groq \
    --title 'Q3 platform planning' --date 2026-09-01 \
    --attendee 'A. Okonkwo' --attendee 'M. Reyes'
```

Without `--attendee` the notes are written with no attribution at all rather than
a guessed one. Without `--date` it warns you that it is stamping today.

Those rules live in one file, `references/notes-register.md`, and they are
enforced rather than hoped for:

```bash
./scripts/notes_check.py notes.md      # exit 1 and file:line for every violation
./scripts/render_pdf.py notes.md       # headless Chrome/Chromium/Edge
```

Run the checker on the PDF too. A leak that only exists in the rendered file
still counts.

Guidance and analysis are welcome — *after* the notes and outside the document.

**A limit, stated plainly:** keeping provenance out of a document changes what the
document says, not what happened. Recording a call without consent is a criminal
offence in some jurisdictions — in Germany, StGB §201 — and a carefully worded
write-up is not a defence to it. Getting consent is your responsibility. This tool
does not provide legal cover and nothing here should be read as suggesting it
does.

## Security posture

Audited with `bandit`: **0 high, 0 medium, 6 low** across the shipped code.
Nothing is suppressed with `#nosec` — the medium findings were fixed, not
annotated.

- **`B310` (`urlopen`) — removed.** `urlopen` dispatches on the URL's scheme, so
  it will open `file://` and hand back a local file. That turns an API client
  into a file reader the moment someone makes the endpoint configurable, which
  a self-hosted or Azure-style OpenAI-compatible host would prompt. Validating
  the scheme first would work but leaves the capability in place behind a check
  someone can move. Both network paths now use `http.client.HTTPSConnection`,
  which speaks no other scheme: there is no URL to mis-parse and no plaintext
  fallback that could send the `Authorization` header in clear. A test asserts,
  on the AST, that `urlopen` has not come back.
- **`B404` / `B603` (6 low) — shelling out to ffmpeg and a browser.** There is no
  `shell=True`, no `os.system`, no `os.popen`; every call is list-form argv, so
  shell injection is not reachable. The only user values interpolated into an
  ffmpeg filter string are `--silence-threshold` and `--min-silence`, both
  `type=float` at the argparse boundary, plus trim offsets formatted `:.6f` from
  parsed output. A string cannot get through. These findings are informational —
  bandit flags the import and the safe call form to prompt exactly this audit.

Key handling: read from the environment only, never a CLI flag (a flag lands in
shell history and process listings), never echoed, never written to the run
manifest or any log. Errors are raised `from None`, so the chained exception —
which holds the request and its `Authorization` header — is not attached. Tests
assert a decoy key reaches the header and appears in no error message.

`ruff` runs with `S` (bandit's ruleset) disabled repo-wide, because 206 of the
findings in this skill's tests are `B101` (`assert`), the correct construct
there. Bandit is run separately over `scripts/`, where it earns its keep.

## Development

```bash
uv run --with pytest python -m pytest tests/ -q
cd evals && uv run python grade.py
python3 fixtures/make_fixtures.py       # synthesised audio, never committed
```

Fixtures are generated, not committed: this repository is public, and audio — or
anything derived from a real recording — does not belong in it.
