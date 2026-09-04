# Fixtures

Build these from your own audio or synthesise them. **No client recordings and no client names**
— this directory is in a public repository, so a fixture that leaks one is a disclosure rather
than a test failure.

The case worth constructing first is the silence hallucination: a file with a stretch of
near-silence around -69 dB against speech around -30 dB. That is the condition that produced a
55x word loop with `compression_ratio` 17.38. See `../README.md`, "Why this exists".

`.gitignore` in the parent blocks audio and `.json` here, so fixtures are generated locally
rather than committed. A generator script that produces them deterministically is the thing to
commit instead.
