# Backends — what each one can and cannot do

Read this before choosing a backend, and before believing a transcript the guard
declared clean.

## The matrix

| backend | platform | install | quality guard | network |
|---|---|---|---|---|
| `mlx-whisper` | **Apple Silicon only** | `mlx-whisper>=0.4.2` | **full** | no |
| `faster-whisper` | macOS / Windows / Linux | `faster-whisper>=1.2` | **full** | no |
| `parakeet` | **Apple Silicon only today**; cross-platform is planned on `sherpa-onnx` | `parakeet-mlx` | **partial — see below** | no |
| `groq` | any | none — stdlib `urllib` | **full** | **yes** |
| `openai` | any | none — stdlib `urllib` | **full** | **yes** |
| `elevenlabs` (Scribe) | any | none — stdlib `http.client` | **partial** | **yes** |
| `gemini` (3.5 Transcribe) | any | none — stdlib `http.client` | **partial** | **yes** |

`--backend auto` resolves to `mlx-whisper` on Apple Silicon and `faster-whisper`
everywhere else. **It never resolves to a network backend** — not as a default,
not as a fallback after a local backend fails. There is no code path from `auto`
to any of them; `resolve("auto")` can only return one of two local
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

**It is genuinely multilingual, verified rather than assumed.** `parakeet-tdt-0.6b-v3`
is documented as covering 25 European languages, and German audio came back as
German on 2026-09-06 — with one truncation, "verschoben" rendered "verschob". On
the 43 s English-then-German clip it kept the German as German, which the Apple
Silicon Whisper default did not.

**The non-Apple path is still the open question**, and NeMo is probably the wrong
answer to it. Verified 2026-09-04 against PyPI and HuggingFace:

| | `nemo_toolkit[asr]` | `sherpa-onnx` 1.13.7 |
|---|---|---|
| dependencies | torch + a large tree | **one** (`sherpa-onnx-core`) |
| wheel size | multi-GB install | **2.1 MB** arm64 mac, **4.4 MB** linux x86_64 |
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

The cross-platform path is **not built**, and NeMo is not the plan: `sherpa-onnx`
is, for the reasons in the table above (one dependency against torch; a 2.1 MB wheel on
Apple Silicon and 4.4 MB on Linux x86_64, against a multi-GB tree). Tracked as `claude-skills-rq95`. Nothing here
should be read as saying `nemo_toolkit[asr]` is coming.

The old NeMo path (`nemo_toolkit[asr]`) is **not** verified and probably never
will be.
`scripts/tslib/backends/parakeet.py` supports both runtimes and prefers
`parakeet-mlx` when importable. Note that the beam-decoding fix above is applied
on the MLX path only; whether NeMo's own decoder shows the same overrun is
untested.

## ElevenLabs Scribe: words, not segments, and 32 speakers

Verified against the API reference 2026-09-04: `POST https://api.elevenlabs.io/v1/speech-to-text`,
auth header `xi-api-key` (**not** bearer), `model_id=scribe_v2`, files to 3 GB, audio to 10 hours.

Two things make it unlike everything else here:

**It returns words, not segments.** There is no `segments` array at all — just
`words[]` with `text`, `type`, `logprob`, `start`, `end`, `speaker_id`. The
segmentation is ours, grouping on speaker change and a 1 s pause. Every other
backend hands us segments.

**It diarizes, up to 32 speakers.** No Whisper backend and not Parakeet returns
any speaker field. Gemini also diarizes, up to 8 (3 or more is described as
experimental), so Scribe is still the one to reach for when a meeting has more
people in it than a workshop.

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
`speaker_1` — positional labels, not names. **Re-measured 2026-09-06 and now
pinned by a live assertion** (`re.fullmatch(r"speaker_\d+")`), because the format
had only ever been a sentence in this file: the live test asserted that *two*
labels came back, never what they looked like. Gemini's turn out to use a colon
(`spk:0`), so the two are not interchangeable and `notes_check.py` has to match
both. A person still maps labels to people,
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
Realtime is $0.39. That puts Scribe between Groq and OpenAI. Gemini, added
2026-09-06 at a derived $0.306, sits between them and also diarizes.

Diarization is on by default in the backend and is not currently exposed as a CLI
flag; if you need it off, that is a small addition to `transcribe.py`.

## Gemini 3.5 Transcribe: two round trips, and a mode deliberately left out

Verified against ai.google.dev/gemini-api/docs/transcribe, .../docs/files and
.../docs/pricing on 2026-09-06: `POST https://generativelanguage.googleapis.com/v1beta/interactions`,
auth header `x-goog-api-key` (**not** bearer), `model=gemini-3.5-transcribe`.

**It uploads in two steps, which nothing else here does.** The audio never goes
to the transcription endpoint. It goes to the Files API first — a resumable
upload whose second leg posts to a URL *the server chooses* and returns in an
`X-Goog-Upload-URL` header — and the transcription request then carries only the
`files/...` URI that came back. That server-chosen URL is the one value in the
whole exchange this repository did not write, so it is checked before any audio
follows it: https, and a `googleapis.com` host, or the backend refuses. A test
covers the redirect case and the `generativelanguage.googleapis.com.evil.example`
suffix trick.

**Google keeps the upload for 48 hours.** That is stated in the egress
disclosure block, because it is part of what the user is agreeing to and is not
visible from the command line otherwise.

**It returns words, not segments** — `steps[].content[].annotations[]` entries of
`type: word_info`, carrying `text`, `speaker`, `start_offset` and `end_offset`.
The offsets are protobuf durations (`"0.100s"`), so they are parsed rather than
read as floats. Segmentation is ours, grouping on speaker change and a 1 s
pause — the same rule as the Scribe path, but a separate implementation, because
Scribe's spacing entries, logprobs and punctuation tokens have no equivalent here
and merging them means a branch per provider at every step.

**It diarizes up to 8 speakers** (3 or more described as experimental), against
Scribe's 32. Same caveat as Scribe: `spk:0` is a positional label, not a name.

### Why `smart` mode is not offered

It is the model's headline feature — it removes filler words, resolves spoken
self-corrections ("let's meet Tuesday — no, Wednesday"), and reflows the text
into paragraphs, bullet lists and formatted numbers. It is not exposed here, and
the reason is in Google's own reference:

> Mode compatibility: Smart transcription (`"smart"`) cannot be combined with
> `timestamp_granularities` or `diarization_mode`.

So it returns no word timing and no speaker labels. Three consequences, each
disqualifying on its own:

- **No clock.** Every timestamp this tool writes is mapped back from the trimmed
  audio onto the user's original file. With no word offsets there is nothing to
  map, and the `.srt` cannot be built at all.
- **No speakers**, losing the one thing this backend has over the Whisper family.
- **It is not a transcript.** It is a model's tidied version of one. This skill
  exists because what a decoder silently adds or drops is the thing you cannot
  see from reading the output — and a mode whose job is to change the words is
  the largest possible instance of that.

Tidying the prose is what the notes document is for, one step downstream, where
it is labelled a summary and checked by `notes_check.py`. `mode` is pinned to
verbatim in `build_body()` and a test asserts the string `smart` appears nowhere
in a built request.

**Quality metrics: partial.** Gemini returns none of Whisper's three and no
per-word probability either, so `has_whisper_metrics=False` and the metric rules
stay silent. `confidence` is `None`, not a substituted zero. The
backend-independent rules (`decoded_from_silence`, `repeated_token`) apply, as
they do to Parakeet and Scribe.

**Limits.** The Files API caps a file at 2 GB and a project at 20 GB. A unary
request takes an hour of audio — but **30 minutes** once word timestamps or
diarization are enabled, and this backend always asks for both, so 30 minutes is
the cap it actually operates under and the one it enforces. The check runs on the
prepared file before the upload starts, reading the duration from the WAV header
with stdlib `wave` rather than shelling out to ffprobe.

**`--prompt` cannot be used with this backend.** It maps to `custom_vocabulary`,
and the parameter table says of it: *"Incompatible with speaker diarization and
word-level timestamps."* The Limitations section is blunter — *"the API rejects
requests that specify `custom_vocabulary` alongside either feature."* Reproduced:

    HTTP 400 {"error":{"message":"custom_vocabulary is incompatible with
    timestamps.","code":"invalid_request"}}

This backend always requests word timestamps, because without them there is no
clock to map onto the original recording, so the combination can never work here.
It is refused before the upload rather than after — the 400 arrives once the audio
is already on Google's servers. Use `--replace 'wrong=right'` afterwards, or a
local backend, where `--prompt` biases the decoder as normal.

**How this was missed for a day, and the fix.** Both earlier passes over this page
were `WebFetch`, which returns a small model's *summary* of a page rather than the
page. That summary carried "up to 1,000 terms, best results with up to 100" and
dropped the incompatibility sentence sitting in the same paragraph. Reading the
page with `defuddle parse <url> --md` — 738 lines — surfaced it, along with two
other corrections below. **For a vendor's API reference, snapshot the page; do not
fetch a summary of it.**

**Two more corrections from that snapshot:**

- **Diarization goes to 8 speakers, not 3.** *"Up to 8 speakers are supported
  (attribution for 3 or more speakers is experimental)."* The "three" figure came
  from the launch blog post, which was the looser source.
- **The page gets its own label format wrong.** It says diarization "tags each
  segment with a speaker identifier like `spk_1` or `spk_2`". The live API returns
  `spk:0` and `spk:1` — a colon, zero-indexed. Measured twice. The documentation
  and the API disagree, and the API wins.

**The reference is wrong about one field.** It documents `"output_text": "transcribed
text"` at the top level of the reply. That key was **absent from all four live responses**
measured on 2026-09-06. The transcript is in `steps[].content[].text`, and `full_text()`
reads it there; reading the documented field would have fallen through to a transcript this
repository reassembled from its own word-grouping, differing from Gemini's own punctuation
in exactly the small ways nobody checks.

**The token rate in the cost table checks out against a real invoice line.** A 6.6 s clip
reported `input_tokens_by_modality: [{"modality": "audio", "tokens": 166}]` — 25.2 tokens
per second, against the 25/second the pricing page states and the $0.306/hour derivation
uses.

**Verified live on 2026-09-06**, not merely documented. `tests/test_gemini_live.py`
is double-gated (`TS_LIVE_API=1` *and* `GEMINI_API_KEY`) and all five checks pass:
the two-step upload returns a `files/...` URI, word timestamps arrive as
`word_info` annotations with protobuf duration strings, two synthesised voices
came back as two distinct speakers, Whisper's three metrics really are absent,
and a wrong key fails without the key appearing in the error.

**And the live run corrected the docs, which is the whole reason for running it.**
The speaker labels are `spk:0` and `spk:1` — a **colon**, not the underscore this
file and `notes_check.py` had both been written to expect. The decoder-artefact
rule in `notes_check.py` had just been widened from `\bSpeaker\s+\d+\b` to catch
Scribe's `speaker_0`, using `[\s_-]*` — and that widened rule still let `spk:0`
straight through, because it was written from the API reference rather than a
real reply. Both are now pinned to measured strings in
`tests/test_notes_check.py`. Same lesson Scribe taught about spacing entries: the
response shape is not knowable from the docs.

**One accuracy note from the same run**, on 7.4 s of synthesised speech:
"Terragrunt" came back as "peregrine". That is a single data point on synthetic
audio and is not comparable to the measured table in `README.md`, but it is the
same failure mode most backends show on that word. Three of the eight rows in
README's measured table DO get it — `openai`, `mlx-whisper large-v3` and
`elevenlabs` — so "nothing gets Terragrunt" would be wrong; five of eight is the
honest figure.

## Language: translating, and recordings that change language

| backend | `--task translate` | `--prompt` | mixed-language recording, **measured** |
|---|---|---|---|
| `mlx-whisper` | yes, **`--model large-v3` only** | yes | **NO — silently translates into the opening language** |
| `faster-whisper` | yes, **`--model large-v3` only** | yes | **yes**, and correct even without `--multilingual` |
| `parakeet` | no | yes | **yes** |
| `groq` | yes, `whisper-large-v3` only | yes | untested |
| `openai` | yes, `whisper-1` | yes | untested |
| `elevenlabs` | no | yes | untested |
| `gemini` | no | **NO — API 400 with word timestamps** | **yes**, code-switching is unconditional |

The last column is what a 43 s English-then-German clip actually returned on
2026-09-06, not what a vendor claims. `mlx-whisper` is the Apple Silicon default
and is the one that fails, which is the single most useful line in this file for
anyone whose meetings are not monolingual.

**There is no target language, only English.** `mlx_whisper/decoding.py` states it
in one line: *"whether to perform X->X `transcribe` or X->English `translate`"*.
Both hosted routes agree -- OpenAI: *"This endpoint supports translation into
English only."*; Groq: *"The translations endpoint only supports 'en' as a
parameter option."* All three verified 2026-09-06.

**Gemini genuinely cannot translate, and this was probed rather than read.** It is a
Gemini model whose headline feature is called "smart transcription", so the natural
assumption is that asking for English would work. Three attempts on German audio on
2026-09-06 — the shipped verbatim config, `language_codes: ["en-US"]`, and a plain text
instruction *"Transcribe this audio and give the result in English"* passed alongside the
audio with no `transcription_config` at all — returned *"Guten Morgen. Die Migration ist am
Donnerstag fertig geworden."* every time. `language_codes` is a hint about what is spoken,
exactly as documented.

**turbo cannot translate and fails silently.** Measured here, not read off a page:
`--model turbo --task translate` on German audio returned the German, under a
transcript header that said "Translated to English". `--model large-v3` on the
same file returned "Good morning. The migration is finished on Thursday." No
exception, no warning. Groq's own comparison table marks translation "No" for
whisper-large-v3-turbo, so it is a property of the distillation rather than of
any one host, and the registry refuses it on every Whisper backend.

**The single-detection trap, measured twice, and the second one is the case that
matters.** A 43 s clip: 29.5 s of English, then German — so the German falls in a
later decode window than the one the language was detected from.

| backend | the German half |
|---|---|
| `mlx-whisper` | "the budget is located at 42.000€ against a prognosis of 38.000€ … I'm going to send the show today to the next day" |
| `faster-whisper` large-v3 | "das Budget lag bei 42.000 Euro gegenüber einer Prognose von 38.000 …" |
| `faster-whisper` + `--multilingual` | identical to the line above |
| `parakeet` | "das Budget lag by 42.000 gegenüber einer Prognose von 38.000 …" |
| `gemini` | "das Budget lag bei 42.000 Euro gegenüber einer Prognose von 38.000 …" |

**mlx-whisper is the Apple Silicon default and it is the one that fails.** It did
not garble the German, it translated it — badly, and in the way that is hardest to
catch. *"Ich schicke die Aufstellung heute Nachmittag herum"* is "I'll circulate
the breakdown this afternoon". It came back as *"I'm going to send the show today
to the next day"*:

| German | means | rendered as |
|---|---|---|
| `Nachmittag` | afternoon — literally *nach Mittag*, after midday | "to the next day" |
| `Aufstellung` | a breakdown, an itemised list | "the show" (cf. *Aufführung*) |

Both are lexically adjacent to something real, and both survive as fluent English.
The first one moves a stated commitment by up to a day. **An earlier version of
this file called it "not a translation of anything", which was wrong and made the
finding sound milder than it is:** a reader told the output is garbled expects to
notice it. Nobody notices a smooth mistranslation, and no confidence metric flags
one, because the decoder was not uncertain.

**Two earlier attempts at this test proved nothing, and are kept here because the
reason is the useful part.** A 13 s German→Spanish clip and a 25 s
English→German clip both came back correct on every backend, `mlx-whisper`
included. Both switches fell inside the **first 30-second window**, which is the
one the detection reads, so there was nothing for a per-file decision to get
wrong. A test clip short enough to be convenient is short enough to be a single
window. Make the second language start after 30 seconds or the test is measuring
nothing.

**A defect in this project's own scorer, found 2026-09-06 by an independent
fact-check rather than by the suite.** `evals/accuracy/checks.json` matched the
budget figures as digits only (`42[,.]?300`), so ElevenLabs -- which wrote
"forty-two thousand three hundred euros" on one run and "€42,300" on the next
from the same audio -- scored a MISS for a figure it had transcribed perfectly.
The check was measuring whether a number had been **digitised**, not whether it
had **survived**. That put "ElevenLabs scored 8 and 12 and dropped a budget
figure" into the README, which was wrong twice over: nothing was dropped, and its
real score is 13/14 on both runs. Both forms now count, and every backend scores
identically across its own runs.

The transferable part: a check that requires a particular *form* will fail
correct content, and it fails silently in the direction that looks like a finding
-- a false defect attributed to a vendor, which is harder to doubt than a false
pass.

**`--multilingual` has not yet been shown to rescue a case that would otherwise
fail.** On both mixed clips `faster-whisper large-v3` was already correct without
it. The flag is exposed because the engine documents its detection as per-file by
default and per-segment when asked, and one recording is not a proof either way.
Do not claim more for it than that.

**An earlier single-detection measurement**, kept for the second language pair. A
35 s clip, 23 s of German then Spanish:

    without --multilingual   [00:30] und der Prognose war 38.000.
                                     Wir müssen ihn vor Freitag korrigieren.
    with    --multilingual   [00:30] y el pronóstico era de 38.000.
                                     Necesitamos corregirlo antes del viernes.

The Spanish was **translated into German** and nothing said so. faster-whisper's
own docstring is the fix -- *"multilingual: Perform language detection on every
segment"* -- and it is the only local engine that has it. On a 13 s clip the flag
made no difference at all, because the switch fell inside a single 30 s decode
window; it only matters once the recording is long enough for the second window
to be a different language, which is exactly when a real call would.

**Refusing, not ignoring.** A backend that cannot honour `--task` or
`--multilingual` raises `UnsupportedOption` and the run stops. Accepting the flag
and quietly doing something else would hand back a German transcript to someone
who asked for English, in a document that reads perfectly well -- the failure
class this whole skill exists to make visible.

## Network backends: cost and limits

Per hour of audio, from the providers' own documentation:

| backend | model | $/hour |
|---|---|---|
| groq | `whisper-large-v3` | 0.111 |
| groq | `whisper-large-v3-turbo` | 0.04 |
| elevenlabs | `scribe_v2` | 0.22 |
| gemini | `gemini-3.5-transcribe` | 0.306 (derived) |
| openai | `whisper-1` | 0.36 |

Gemini publishes a token rate rather than an hourly one, so 0.306 is derived:
25 audio tokens/second in at $2.00/M is $0.180/hour, and 175 text tokens/minute
out at $12.00/M is $0.126/hour. Both halves appear on the invoice, so both are
counted — quoting the input half alone would understate every estimate by 41%.
There is also a free tier, which the estimate deliberately does not assume.

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

API keys are read from the environment, one variable per backend, and nowhere
else. Never a flag, never printed, never written to the run manifest.

| backend | environment variable |
|---|---|
| `groq` | `GROQ_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `elevenlabs` | `ELEVENLABS_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |

This list is checked against the registry rather than maintained by hand — the
`--backend` help text in `transcribe.py` derives its network list the same way,
after it spent a day saying "groq/openai" while ElevenLabs was already shipping.

## Transcription does not remove filler — with one exception, which is not wired up

A common assumption is that the cloud models clean up speech and the local ones do not. They do
not: Whisper is trained to transcribe verbatim, and Parakeet behaves the same way. **Paying for a
hosted Whisper does not buy you disfluency removal.**

The one real exception is **Gemini's `smart` mode**, which is documented to strip filler words and
resolve spoken self-corrections. It is deliberately not offered here — see the Gemini section
above for why — so nothing in this tool removes filler at the transcription step.

That is not a defect. A transcript that silently drops words is worse than one that keeps them:
you cannot tell what was removed. Cleanup belongs in a later pass that knows it is editing.

In this skill that pass is the notes step, and it is where filler comes out. Products that appear
to transcribe cleanly are doing the same thing: ASR first, then a second model that rewrites.
**If you skip the notes step you have a verbatim transcript, and nothing has cleaned it.**

> **A table of filler counts used to sit here and was removed on 2026-09-06.** It claimed the test
> audio contained three "um", three "uh" and one "er" plus "I mean" and "you know", and that four
> backends each kept 5 and 2 — "identical output, to the token". Every eval run in this project
> used one recording, named in every `run.json`, and it contains **none** of those tokens: zero
> hits for `um`, `uh`, `er`, "I mean", "you know", "like" and "sort of", across the transcripts and
> across the `.json` sidecars that retain suppressed segments, in every backend. Four backends
> returning no filler from audio with no filler demonstrates nothing, so the table was evidence for
> a claim it could not support. The claim above is kept because it is true on other grounds; the
> numbers are gone because they had no source. Re-measuring on audio that actually contains filler
> is `claude-skills-luwe`.

## Preparation is shared, and it matters more than the backend

Every backend receives the same prepared audio: decoded to 16 kHz mono, loudness
normalised, silence-trimmed. That step is not an optimisation. Measured on a real
recording, cleaning the audio took invented segments from 2 to 0 and fixed a word
that both `large-v3` and `turbo` had decoded wrong on the raw file — a bigger
model did not fix what cleaning did.

`loudnorm`, `silencedetect`, `atrim` and `concat` are unconditional libavfilter
built-ins, so this runs identically on all three platforms with no extra install.
