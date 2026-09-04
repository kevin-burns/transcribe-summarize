"""Tests for scripts/render_pdf.py. No browser required."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import render_pdf  # noqa: E402

# ---------------------------------------------------------------------------
# Block-level rendering
# ---------------------------------------------------------------------------


def test_headings():
    html = render_pdf.markdown_to_html("# One\n## Two\n### Three\n#### Four", "T")
    assert "<h1>One</h1>" in html
    assert "<h2>Two</h2>" in html
    assert "<h3>Three</h3>" in html
    assert "<h4>Four</h4>" in html


def test_unordered_list():
    html = render_pdf.markdown_to_html("- one\n- two\n* three", "T")
    assert "<ul>" in html
    assert "<li>one</li>" in html
    assert "<li>two</li>" in html
    assert "<li>three</li>" in html


def test_ordered_list():
    html = render_pdf.markdown_to_html("1. first\n2. second\n3. third", "T")
    assert "<ol>" in html
    assert "<li>first</li>" in html
    assert "<li>third</li>" in html


def test_blockquote():
    html = render_pdf.markdown_to_html("> a wise quote\n> continues here", "T")
    assert "<blockquote>" in html
    assert "a wise quote" in html


def test_horizontal_rule():
    html = render_pdf.markdown_to_html("above\n\n---\n\nbelow", "T")
    assert "<hr>" in html


def test_fenced_code_block():
    html = render_pdf.markdown_to_html("```python\nx = 1\n```", "T")
    assert "<pre>" in html
    assert "<code" in html
    assert "x = 1" in html


def test_pipe_table():
    md = "| Name | Role |\n| --- | --- |\n| Alice | Lead |\n| Bob | Eng |"
    html = render_pdf.markdown_to_html(md, "T")
    assert "<table>" in html
    assert "<th>Name</th>" in html
    assert "<th>Role</th>" in html
    assert "<td>Alice</td>" in html
    assert "<td>Bob</td>" in html


# ---------------------------------------------------------------------------
# Inline rendering
# ---------------------------------------------------------------------------


def test_inline_bold_italic_code_link():
    html = render_pdf.markdown_to_html(
        "This is **bold**, *italic*, `code`, and [a link](https://example.com).", "T"
    )
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<code>code</code>" in html
    assert '<a href="https://example.com">a link</a>' in html


# ---------------------------------------------------------------------------
# HTML escaping
# ---------------------------------------------------------------------------


def test_html_escaping_script_tag():
    html = render_pdf.markdown_to_html("<script>alert(1)</script>", "T")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_html_escaping_ampersand():
    html = render_pdf.markdown_to_html("a & b", "T")
    assert "a &amp; b" in html


# ---------------------------------------------------------------------------
# Nothing silently dropped
# ---------------------------------------------------------------------------


def test_unrecognised_line_falls_through():
    weird_line = "   this line has odd    spacing and no markdown syntax at all"
    html = render_pdf.markdown_to_html(weird_line, "T")
    assert "this line has odd" in html
    assert "spacing and no markdown syntax at all" in html


# ---------------------------------------------------------------------------
# Browser discovery
# ---------------------------------------------------------------------------


def test_find_browser_missing_override_returns_none():
    assert render_pdf.find_browser(override="/definitely/not/here") is None


def test_find_browser_respects_chrome_env_var(tmp_path, monkeypatch):
    fake_browser = tmp_path / "fake-chrome"
    fake_browser.write_text("#!/bin/sh\necho fake\n")
    fake_browser.chmod(fake_browser.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv("CHROME", str(fake_browser))
    assert render_pdf.find_browser() == str(fake_browser)


# ---------------------------------------------------------------------------
# Document shell
# ---------------------------------------------------------------------------


def test_document_has_page_rule_and_title():
    html = render_pdf.markdown_to_html("# Hello", "My Report")
    assert "@page" in html
    assert "<title>My Report</title>" in html


# ---------------------------------------------------------------------------
# Full CLI run with no browser available
# ---------------------------------------------------------------------------


def test_cli_run_no_browser_exits_zero_and_writes_html(tmp_path, monkeypatch):
    md_path = tmp_path / "notes.md"
    md_path.write_text("# Meeting Notes\n\nSome content here.\n")

    # Force "no browser found" regardless of what is actually installed on this machine.
    monkeypatch.setattr(render_pdf, "find_browser", lambda override=None: None)

    exit_code = render_pdf.main([str(md_path), "--json"])
    assert exit_code == 0

    html_path = md_path.with_suffix(".html")
    assert html_path.exists()


def test_cli_subprocess_no_browser_json_output(tmp_path):
    md_path = tmp_path / "notes.md"
    md_path.write_text("# Meeting Notes\n\nSome content here.\n")

    script = str(Path(render_pdf.__file__).resolve())

    # --browser is an explicit override and is authoritative (see find_browser): pointing it
    # at a nonexistent path guarantees "no browser found" regardless of what is actually
    # installed on the machine running this test.
    result = subprocess.run(
        [sys.executable, script, str(md_path), "--browser", "/definitely/not/a/real/browser", "--json"],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["pdf"] is None
    assert Path(payload["html"]).exists()


# ------------------------------------------------------------- process-group cleanup
#
# Found by an end-to-end smoke test: after one render, two headless Chrome
# processes were still alive. `process.terminate()` signals the parent only, and
# Chrome forks renderer, GPU and zygote children that outlive it. The fix is to
# put the browser in its own process group/session and signal the group.


def test_the_browser_is_launched_in_its_own_process_group():
    """A grep-shaped guard on the launch flags, since actually leaking processes
    is not something a unit test can assert cheaply."""
    source = Path(render_pdf.__file__).read_text()
    assert "start_new_session" in source, "POSIX: browser must get its own session"
    assert "CREATE_NEW_PROCESS_GROUP" in source, "Windows: browser must get its own group"


def test_termination_signals_the_group_not_just_the_parent():
    source = Path(render_pdf.__file__).read_text()
    assert "killpg" in source, "terminating the parent alone leaves Chrome's children alive"
    assert "taskkill" in source, "Windows has no process groups to signal; taskkill /T walks the tree"
    assert "System32" in source, "taskkill must be resolved absolutely, not through PATH (bandit B607)"


# ------------------------------------------------------------ lazy continuation
#
# Found by looking at a rendered PDF, not by these tests: every list item long
# enough to soft-wrap had its tail rendered as a separate paragraph at the left
# margin, outside the list. Real notes bullets are prose and wrap constantly; the
# short fixtures here never did, so nothing caught it.


def test_a_soft_wrapped_bullet_stays_one_list_item():
    markdown = (
        "- Reyes asked whether the pilot group would keep access to the old system during the\n"
        "  first week. Okonkwo said that had not been decided.\n"
        "- Lindqvist asked who owns the rollback decision.\n"
    )
    html = render_pdf.markdown_to_html(markdown, "t")
    assert html.count("<li>") == 2, "a wrapped line must not become a third item"
    assert "first week. Okonkwo said that had not been decided.</li>" in html
    body = html[html.index("<ul>"):html.index("</ul>")]
    assert "<p>" not in body, "the wrapped tail escaped the list as a paragraph"


def test_a_soft_wrapped_numbered_item_stays_one_item():
    markdown = "1. Staging rebuild, completed before the\n   freeze lifts\n2. Pilot group\n"
    html = render_pdf.markdown_to_html(markdown, "t")
    assert html.count("<li>") == 2
    assert "completed before the freeze lifts</li>" in html


def test_a_heading_after_a_list_still_ends_the_list():
    """Continuation must not swallow the next block."""
    html = render_pdf.markdown_to_html("- one\n- two\n## Next section\nBody.\n", "t")
    assert html.count("<li>") == 2
    assert "<h2>Next section</h2>" in html
    assert "Next section" not in html[html.index("<ul>"):html.index("</ul>")]


def test_a_blank_line_ends_the_list():
    html = render_pdf.markdown_to_html("- one\n\nA new paragraph.\n", "t")
    assert html.count("<li>") == 1
    assert "<p>A new paragraph.</p>" in html
