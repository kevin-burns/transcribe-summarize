---
name: transcribe-summarize
description: Transcribe audio locally on macOS, Windows or Linux, and optionally write it up as meeting notes. Use whenever the user has an audio or video file and wants what was said - "transcribe this", "what was said on the call", "turn this recording into notes", "write up this meeting", "subtitle this", "get me an .srt", a voice memo, a dictation, an interview, a lecture, a podcast. Runs on-device by default with Whisper (mlx-whisper on Apple Silicon, faster-whisper everywhere else, Parakeet when asked for); Groq, OpenAI and ElevenLabs Scribe are available but only when named explicitly, and never without first stating what will be sent and what it will cost. Normalises and silence-trims the audio with ffmpeg before decoding, which measurably beats reaching for a bigger model, and filters the fabricated segments Whisper invents over silence. Also produces a factual meeting-notes document and a PDF. Not for writing prose or summarising text you already have.
license: MIT
---

# transcribe-summarize

Audio in, a transcript out, and optionally a set of meeting notes. The default
path runs entirely on the user's machine.

**Two things make this different from calling Whisper yourself**, and both came
out of a day of measurement rather than design:

1. **Whisper invents speech over silence, and a bigger model does not help.** On
   a real call with 40 s of joining silence, `large-v3` produced an 18-second
   segment of one word repeated 55 times. This skill filters those out and keeps
   them, with the metrics that rejected them, in a sidecar so the decision is
   auditable.
2. **Cleaning the audio beats choosing a model.** Normalising and silence-trimming
   took invented segments from 2 to 0 and fixed a word both `large-v3` and `turbo`
   had wrong. It is on by default.

## Setup

`ffmpeg` and `ffprobe` must be on PATH — macOS `brew install ffmpeg`, Windows
`winget install Gyan.FFmpeg`, Linux `apt install ffmpeg`.

Everything else is stdlib. The **backend** brings its own library, and only the
one selected, so pick it with `uv run --with`:

| backend | platform | `--with` spec |
|---|---|---|
| `mlx-whisper` — default on Apple Silicon | macOS arm64 only | `'mlx-whisper>=0.4.2'` |
| `faster-whisper` — default elsewhere | mac / Windows / Linux | `'faster-whisper>=1.2'` |
| `parakeet` — opt-in | Apple Silicon verified | `'parakeet-mlx'` |
| `groq`, `openai`, `elevenlabs` — opt-in, **network** | any | nothing to install |

If the library is missing, the error prints the exact command. Don't guess it.

## Run it

Declare the helper at the top of every command block — shell state does not
persist between tool calls:

```bash
ts() {
  UV="$(command -v uv || ls "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" \
        /opt/homebrew/bin/uv /usr/local/bin/uv 2>/dev/null | head -1)"
  "$UV" run --with "$1" --script "$HOME/.claude/skills/transcribe-summarize/scripts/transcribe.py" "${@:2}"
}

# Apple Silicon
ts 'mlx-whisper>=0.4.2' meeting.m4a

# Windows / Linux
ts 'faster-whisper>=1.2' meeting.m4a --backend faster-whisper

# Bias the decoder toward known jargon, and fix a name it always gets wrong
ts 'mlx-whisper>=0.4.2' meeting.m4a --prompt 'Terragrunt, EKS' --replace 'north wind=Northwind'
```

Use the absolute path. A relative one only resolves if the cwd happens to be the
skill directory.

## Output contract

```
<outdir>/
  <stem>.md                    the transcript, with [hh:mm:ss] anchors
  <stem>-artifacts/
      <stem>.srt               subtitles
      <stem>.json              every segment, including the suppressed ones
      <stem>.run.json          backend, model, trim offsets, guard tally
```

`--json` prints those paths to stdout as JSON. Report absolute paths to the user;
a file they have to hunt for is not a deliverable.

**Timestamps are on the original recording's clock.** Trimming shortens the audio
during decoding, and the offsets are mapped back before anything is written, so
the `.srt` lines up with the user's own file.

## The network rule — state it, then obey it

A network backend is reachable **only** when the user asks for it in that
invocation. Never as a fallback, never after a local backend fails, never because
it would be faster. `--backend auto` cannot reach one; there is no code path.

Before anything is sent, the tool prints provider, endpoint, file size, duration
and estimated cost, then waits. Use `--dry-run` to show the user that block
before committing. In a non-interactive session an unconfirmed upload is refused.

Keys come from `GROQ_API_KEY` / `OPENAI_API_KEY`. Never accept one as a flag,
never echo one, never write one into a file.

## The quality guard

On by default. Suppressed segments stay in the `.json` with the reason and the
metrics; they are dropped from the `.md` and `.srt`. `--guard mark` keeps them
inline flagged, `--guard off` disables it.

**It does not transfer to every backend.** The strong rules read Whisper decoder
metrics. `faster-whisper`, Groq and OpenAI return all three, so the guard is the
same code. **Parakeet returns none of them** and falls back to a repetition rule
only — the tool says so at run time. See `references/backends.md`.

If the user asks why a stretch of transcript is missing, read the `.json`: the
answer is there with the numbers that caused it.

## Starting from a transcript you already have

**Most meetings are already transcribed by the platform, and there is often no
audio at all.** Teams and Zoom hand out a `.vtt` (sometimes `.srt`), and it is
unreadable as prose: a cue every few seconds, sentences chopped across cues, the
speaker's name repeated on every line, timecodes throughout.

That is a supported entry point. Nothing is re-transcribed — the audio path is
skipped entirely:

```bash
"$HOME/.claude/skills/transcribe-summarize/scripts/normalise_transcript.py" meeting.vtt
# -> meeting.md, cues rejoined into paragraphs, speakers attributed, one anchor per paragraph
```

Then carry on exactly as if the transcript had been decoded here: read it, ask
for the meeting facts, and write the notes under `references/notes-register.md`.

`--no-speakers` drops attribution if the source labels are wrong or unwanted.

**It is a format conversion and nothing more.** No words are changed, no filler
removed, no grammar fixed. Deciding what was said is the summarising step and it
has its own rules. If a user hands you a messy transcript and asks for it "tidied
up", normalise the format here and do the judgement in the notes — do not quietly
rewrite someone's words in a document that is supposed to record them.

Speakers are kept in the transcript and forbidden in the notes, which is not a
contradiction: a transcript is a working artefact and may carry provenance; the
notes are a record and may not. `notes_check.py` will still reject a raw
`Speaker 2` label in the notes.

If someone passes a `.vtt` to `transcribe.py` it stops and points here rather
than trying to decode it.

## Video files

`.mp4` is what a Teams or Zoom download actually is, and it works directly — no
need to extract the audio first. Verified working: `.mp4`, `.mov`, `.mkv`,
`.webm`, plus every audio container (`.m4a`, `.mp3`, `.wav`, `.flac`, `.opus`).
ffmpeg pulls the audio track out; the video is ignored and never uploaded.

A file with **no** audio track — a muted screen recording — is refused up front,
before any network backend is offered it. That check runs before the egress gate
specifically so a silent video is never uploaded and billed.

## Review is a step, not an optional extra

**Transcription output is a draft. Say so, and make the user check it.**

Machine transcription is confidently wrong in ways no threshold catches. Measured
on real audio in this project: `large-v3` silently dropped a speaker's spoken
correction of a figure, and every backend heard "board pack" as "board packs up".
Both read perfectly fluently. The quality guard catches inventions over silence;
it cannot catch a plausible mis-hearing, and neither can you by reading the
output.

So the workflow is **transcribe → the user checks → then notes**:

1. The transcript `.md` ends with a **"Worth checking"** section — ranked
   timestamps where a listen-back is most likely to pay off (least-confident
   segments, anything containing figures). Point the user at it explicitly.
2. **Names, figures and dates are what to verify.** They are the first thing to
   degrade and the most expensive to get wrong in a record.
3. Fixes go into the `.md`, by hand or with `--replace` on a re-run. Then
   regenerate anything built from it.

Never present a transcript or a set of notes as verified. You did not hear the
audio; the user did. Say what was decoded, say where it is least certain, and let
them confirm.

### Editing and regenerating

The `.md` is the source. Edit it, then rebuild the PDF — no re-transcription:

```bash
"$HOME/.claude/skills/transcribe-summarize/scripts/notes_check.py" notes.md
"$HOME/.claude/skills/transcribe-summarize/scripts/render_pdf.py"  notes.md
```

Run the checker again after any edit. Hand-editing is exactly when a provenance
leak or an inferred "next steps" line gets introduced.

## Writing the notes

When the user wants meeting notes rather than a transcript, **you write them** —
read the transcript and produce the document yourself. There is a
`scripts/summarize.py` fallback that calls a network model, but it is for running
this pipeline outside a Claude Code session; inside one it is a pointless upload.

### Ask first — three things the audio cannot tell you

**Almost no backend here identifies speakers.** Every Whisper engine and Parakeet
return no speaker field at all. The one exception is `--backend elevenlabs`
(Scribe), which diarizes up to 32 speakers — but it returns `speaker_0`,
`speaker_1`, positional labels rather than names, and the register forbids a raw
label in a notes document. So it tells you *how many* people spoke and *which
lines belong together*; it still cannot tell you *who*. Attribution comes from the
user either way — Scribe just makes the mapping possible from the transcript
instead of from memory.

Before writing notes, ask for:

1. **Who was present.** Without names, write the notes with no attribution at all
   — impersonal, "it was stated that…". Never invent a name, a role, or a
   positional label. If the transcript makes it obvious who spoke (someone is
   addressed by name), you may attribute that; ambiguity means no name.
2. **The date of the meeting.** Today's date is a *default*, not a fact. Audio is
   often written up days after the conversation, and a wrong date on a factual
   record is worse than no date. Confirm it.
3. **What the meeting was** — a title in the participants' own terms.

Ask for all three in one message, not one at a time, and offer today's date as
the default so the user only has to correct it. If the user declines to supply
attendees, proceed unattributed and say that is what you did.

**Before writing a single line, read `references/notes-register.md`.** It is short
and it is the whole specification. It is not style advice; the rules in it were
learned by having sections deleted by hand from a real document, and the reasons
are legal rather than aesthetic.

The two rules it turns on:

- **Factual summary only.** What was said. No analysis, no implications, no
  "next steps", no "what wasn't covered", no filling a gap.
- **It reads as a protocol write-up by someone who was there.** It never mentions
  a recording, a transcript, audio, a model, or any decoder artefact — no
  timecodes, no `Speaker 2` labels.

Then check it, every time — the register is not self-enforcing and a model asked
to summarise will add analysis unprompted:

```bash
"$HOME/.claude/skills/transcribe-summarize/scripts/notes_check.py" notes.md
```

Exit 0 is clean; exit 1 lists `file:line` for every violation. Fix them before
handing the document over. Run it on the PDF too — a leak that only exists in the
rendered file still counts.

**Guidance is welcome, but after the notes and outside the document.** Offer your
read in conversation, or write it to a separate file. Never append it.

## The PDF

```bash
"$HOME/.claude/skills/transcribe-summarize/scripts/render_pdf.py" notes.md
```

Headless Chrome, Chromium or Edge, discovered automatically. If none is
installed it writes the HTML, says so, and exits 0 — the HTML is still a
deliverable.

## Provenance

Local decoding is `mlx-whisper` (MLX) or `faster-whisper` (CTranslate2); audio
preparation is `ffmpeg`. The network backends call Groq's and OpenAI's
transcription endpoints directly over stdlib `urllib` — no vendor SDK.

Keeping provenance out of a notes document changes what the document says, not
what happened. Recording a call without consent is a criminal offence in some
jurisdictions — in Germany, StGB §201 — and a well-registered write-up is not a
defence to it. Obtaining consent is the user's responsibility, and if the
question comes up, say so plainly rather than implying the framing provides
cover.

## Tooling

Unit tests in `tests/`, an offline eval in `evals/`. Fixtures are generated by
`fixtures/make_fixtures.py`, never committed — this repository is public and
audio never belongs in it.
