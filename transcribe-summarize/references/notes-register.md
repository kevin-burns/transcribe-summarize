# Notes register

This document sets the content and register rules for a meeting-notes document produced by this
skill. It is read by two different readers: a person drafting or editing the notes by hand, and a
language model, which receives it verbatim as a system prompt before it writes or revises a
document. Both readers follow the same rules. There is no separate "AI version" of this policy.

The document being produced is a **protocol write-up** — notes kept by a person who was at the
meeting. It is not a transcript, not a summary of a transcript, and not something a machine
produced. Every rule below exists to keep the document readable as that, and nothing else.

## Why this exists

A previous version of a document like this had sections removed by hand by the document's owner.
The reasons were legal, not stylistic. The document is a factual record with weight — it can be
relied on, quoted, and disputed later. Analysis mixed into a factual record changes what the
record IS: a record of statements becomes an argument. Once that line is crossed the document is
no longer notes, and it no longer carries the same weight.

## The header: what must be established BEFORE writing

A protocol write-up carries the facts a reader needs to place it. These are **not inferable from
the audio** and must never be guessed. Ask for them, or leave them out.

    # <Meeting title>
    **<Weekday, D Month YYYY>** — <Attendee>, <Attendee>, <Attendee>

    Notes taken by the attendee. Not a verbatim record.

- **Date.** Default to today only if the meeting happened today. A file is often written up days
  after the conversation, so confirm the date rather than stamping the date of processing. A
  wrong date on a factual record is worse than no date.
- **Attendees.** **Nothing in this pipeline can identify a speaker.** No engine used here returns
  a speaker field, so who spoke is information only a person can supply. Ask who was present. If
  the answer does not come, write the notes without attribution — impersonal, "it was stated
  that…" — rather than inventing a name or leaving a `Speaker 2` label in place.
- **Title.** What the meeting was, in the participants' own terms.

Never write "Speaker 1", "Speaker 2", or any other positional label into the document. Those are
decoder artefacts, they are forbidden by the register rules below, and they are flagged
automatically. A label is not attribution: either a person is named, or the statement is
recorded without one.

Attribute a statement only where the meeting made it clear who said it. Where it is genuinely
ambiguous, record what was said without a name. Guessing which of three attendees said something
is exactly the kind of invention this register exists to prevent.

## Content: factual summary only

Permitted:

- Condensed statements of what was said.
- Topics that were raised.
- Decisions stated aloud, recorded as having been stated.
- Figures, dates and names, as spoken.
- Who said what, where that was stated in the meeting.

**Filler and disfluency are dropped in the notes, and kept in the transcript.** "Um", "uh",
"er", "you know", "I mean", repeated false starts — none of that belongs in a written record of
what was decided, and condensing a statement is not the same as changing it.

Measured 2026-09-04, because the assumption runs the other way: **no transcription model removes
filler, local or hosted.** mlx-whisper, faster-whisper, Parakeet and OpenAI's whisper-1 all
returned the same five filler tokens and two hedges from the same audio. Whisper transcribes
verbatim by design. So the transcript keeps them and this step is the only place they come out —
if you skip the notes step, nothing has cleaned the text.

The exception is a disfluency that carries meaning. A speaker correcting themselves — "sorry,
that's not right, the second wave is the fourteenth" — is a fact about what was settled, not
noise. Record what they landed on; do not silently drop the correction, and do not present the
first version as though it stood.

Forbidden — each of the following was cut by hand from a real document, and each is forbidden
again here for the same reason:

- **A "What the call did not cover" section, or any equivalent.** Absence is an inference, not a
  fact. The document reports what was said, not what a reader thinks was missing.
- **A "Next steps" or "Action items" section, added by the tool.** If a next step was stated
  aloud in the meeting, record it as a statement someone made ("X said the team would follow up
  by Friday"), not as a promoted, standalone plan. The document must not manufacture a plan the
  meeting did not commit to on the record.
- **Callouts, highlights, bolding, or emphasis that argue a point the speakers did not
  themselves make.** Typographic emphasis is itself a form of analysis.
- **Implications, conclusions, risk assessments, or sentiment.** No "this suggests…", no "this
  means the project is at risk", no characterising how a speaker seemed to feel.
- **Filling a gap.** If something said was unclear, garbled, or incomplete, it is left out. It is
  never reconstructed, guessed, or smoothed into something that reads as complete.

## Register: what the document must read as

The document must never disclose, in any form:

- That a recording exists or ever existed.
- That the document came from a transcript.
- That a machine or model produced it — no tool names, no model names ("AI-generated",
  "Whisper", "speech-to-text", or similar).
- Any decoder artefact: timecodes, segment numbers, "Speaker 1"-style labels, confidence scores.

None of this is because the process is something to hide. It is because a document that reads as
a machine's output carries different weight, and possibly different legal standing, than notes a
person kept. The register has to be right for the document to be what it claims to be.

Rewrite table — the left column must never appear; the right column is how to say the same thing:

| do not write | write instead |
|---|---|
| "the recording doesn't say" | "was not stated" |
| "pre-recorded message" | "company-wide broadcast message" |
| "at 14:32 in the audio, X said" | "X said" |
| "the transcript shows" | (delete — just state it) |

## Approved boilerplate

This is the only self-description the document is permitted to carry, and it must be quoted
exactly — no paraphrase, no added or removed words:

- Subtitle: `Notes taken by the attendee. Not a verbatim record.`
- Closing: `Written up from the meeting by an attendee. A summary, not a verbatim record. No part
  to be read as a quotation.`

A near-miss is not the boilerplate. "Not a verbatim record of the recording" is not the approved
sentence — it reintroduces the word "recording" the approved text was written to avoid, and it
must be flagged like any other violation, not treated as close enough.

## Where guidance goes

Analysis, recommendations, and next-step suggestions are welcome and often useful — just not in
this document. They belong in conversation with the user by default, or in a separate sibling
file if they need to be written down. They are never appended to the notes and never rendered
into the PDF. The notes document and the analysis are two different documents, produced (if both
are produced at all) for two different purposes.

## One asymmetry to preserve

The rules above govern what the tool adds. They are not a license to silently rewrite source
material that already contains a next-steps or analysis section supplied by a human author — for
example, an agenda or prior minutes handed in as input. That content belongs to its own author.
If it is already present in material the tool is asked to fold in, preserve it and flag the
inconsistency with the register rules rather than silently stripping it. Silent deletion of
someone else's writing is its own problem, separate from the one this document solves.

## A limit, stated plainly

Keeping provenance out of the document changes what the document says. It does not change what
happened. In Germany, recording a call without the consent of everyone on it is a criminal
offence under StGB s.201, and a well-registered write-up produced afterward is not a defence to
that offence — it does not launder an unlawful recording into a lawful one. Obtaining consent
before recording is the user's responsibility, not something this document or this tool provides
cover for. This is stated here so it is not missed: the rules above are about the record's
register, not about the legality of making the record in the first place.
