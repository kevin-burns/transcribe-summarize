"""transcribe-summarize internals.

The core here is stdlib-only and backend-free on purpose: audio preparation,
the hallucination guard, correction rules and document rendering all work the
same whichever engine decoded the audio. Only `tslib.backends` imports anything
third-party, and only for the backend actually selected.
"""

__all__ = ["audio", "backends", "corrections", "quality", "render"]
