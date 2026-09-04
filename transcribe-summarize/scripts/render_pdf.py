#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Render a small Markdown subset to a print-ready PDF via headless Chrome/Chromium/Edge.

Stdlib only. The input is Markdown this project's own tooling generates (meeting notes), so
the parser below supports a deliberately small subset rather than being a general-purpose
Markdown implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Inline Markdown -> HTML (bold, italic, code, links), with escaping applied first
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"\*([^*]+)\*")
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")


def render_inline(text: str) -> str:
    """Escape HTML special characters, then apply the supported inline markup.

    Escaping happens BEFORE any markup is recognised, so a stray `<` or `&` in someone's
    notes can never be interpreted as HTML by anything downstream. Matches are pulled out
    into placeholders (not re-scanned) so that, e.g., a `*` inside a `` `code span` `` is
    never mistaken for italic markup.
    """
    escaped = html.escape(text)
    stash: list[str] = []

    def _stash(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    escaped = _CODE_RE.sub(lambda m: _stash(f"<code>{m.group(1)}</code>"), escaped)
    escaped = _LINK_RE.sub(lambda m: _stash(f'<a href="{m.group(2)}">{m.group(1)}</a>'), escaped)
    escaped = _BOLD_RE.sub(lambda m: _stash(f"<strong>{m.group(1)}</strong>"), escaped)
    escaped = _ITALIC_RE.sub(lambda m: _stash(f"<em>{m.group(1)}</em>"), escaped)

    return _PLACEHOLDER_RE.sub(lambda m: stash[int(m.group(1))], escaped)


# ---------------------------------------------------------------------------
# Block-level Markdown -> HTML
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_HR_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_UL_ITEM_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_OL_ITEM_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")


def _split_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _try_parse_table(lines: list[str], i: int) -> tuple[str, int] | None:
    """Try to parse a pipe table starting at lines[i]. Returns (html, next_index) or None."""
    if "|" not in lines[i] or i + 1 >= len(lines):
        return None
    sep_cells = _split_row(lines[i + 1])
    if not sep_cells or not all(_TABLE_SEP_CELL_RE.match(c) for c in sep_cells):
        return None

    header_cells = _split_row(lines[i])
    j = i + 2
    body_rows = []
    while j < len(lines) and "|" in lines[j] and lines[j].strip() != "":
        body_rows.append(_split_row(lines[j]))
        j += 1

    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{render_inline(c)}</th>" for c in header_cells)
    parts.append("</tr></thead><tbody>")
    for row in body_rows:
        parts.append("<tr>")
        parts.extend(f"<td>{render_inline(c)}</td>" for c in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts), j


def _collect_list_items(lines: list[str], start: int, item_re: re.Pattern[str]) -> tuple[list[str], int]:
    """Collect list items from `start`, joining soft-wrapped continuation lines.

    Markdown calls this lazy continuation: a list item wrapped across several
    source lines is still ONE item. Without it, every wrapped bullet renders its
    tail as a separate paragraph at the left margin, outside the list -- which is
    exactly what a real notes document does, because prose bullets are long.
    Caught by looking at a rendered PDF, not by a unit test on short fixtures.

    A line continues the previous item when it is non-blank, is not itself a new
    item, and is not the start of another block (heading, quote, rule, fence,
    table row).
    """
    items: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        match = item_re.match(line)
        if match:
            items.append(match.group(1))
            i += 1
            continue
        stripped = line.strip()
        if (
            items
            and stripped
            and not _UL_ITEM_RE.match(line)
            and not _OL_ITEM_RE.match(line)
            and not stripped.startswith(("#", ">", "|", "```", "---", "***", "___"))
        ):
            items[-1] = f"{items[-1]} {stripped}"
            i += 1
            continue
        break
    return items, i


def render_blocks(lines: list[str]) -> str:
    """Render a list of source lines (block-level) to a string of HTML fragments."""
    output: list[str] = []
    paragraph: list[str] = []
    i = 0
    n = len(lines)

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{render_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # consume the closing fence, if present
            code_html = html.escape("\n".join(code_lines))
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            output.append(f"<pre><code{cls}>{code_html}</code></pre>")
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            output.append(f"<h{level}>{render_inline(heading_match.group(2))}</h{level}>")
            i += 1
            continue

        if _HR_RE.match(stripped):
            flush_paragraph()
            output.append("<hr>")
            i += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            output.append(f"<blockquote>{render_blocks(quote_lines)}</blockquote>")
            continue

        table = _try_parse_table(lines, i)
        if table is not None:
            flush_paragraph()
            table_html, next_i = table
            output.append(table_html)
            i = next_i
            continue

        if _UL_ITEM_RE.match(line):
            flush_paragraph()
            items, i = _collect_list_items(lines, i, _UL_ITEM_RE)
            items_html = "".join(f"<li>{render_inline(t)}</li>" for t in items)
            output.append(f"<ul>{items_html}</ul>")
            continue

        if _OL_ITEM_RE.match(line):
            flush_paragraph()
            items, i = _collect_list_items(lines, i, _OL_ITEM_RE)
            items_html = "".join(f"<li>{render_inline(t)}</li>" for t in items)
            output.append(f"<ol>{items_html}</ol>")
            continue

        # Anything unrecognised falls through as plain paragraph text rather than being
        # silently dropped.
        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    return "\n".join(output)


_DOC_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  @page {
    size: A4;
    margin: 20mm 18mm;
  }
  * { box-sizing: border-box; }
  body {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 100%;
    margin: 0;
    orphans: 3;
    widows: 3;
  }
  h1, h2, h3, h4 {
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 700;
    color: #111111;
    line-height: 1.25;
    page-break-inside: avoid;
    page-break-after: avoid;
  }
  h1 { font-size: 22pt; margin: 0 0 10pt; border-bottom: 1.5pt solid #333; padding-bottom: 6pt; }
  h2 { font-size: 16pt; margin: 22pt 0 8pt; }
  h3 { font-size: 13pt; margin: 16pt 0 6pt; }
  h4 { font-size: 11.5pt; margin: 14pt 0 4pt; font-style: italic; }
  p { margin: 0 0 9pt; orphans: 3; widows: 3; }
  ul, ol { margin: 0 0 9pt; padding-left: 22pt; }
  li { margin: 0 0 3pt; }
  blockquote {
    margin: 0 0 9pt;
    padding: 4pt 14pt;
    border-left: 3pt solid #999;
    color: #444;
    font-style: italic;
    page-break-inside: avoid;
  }
  hr {
    border: none;
    border-top: 0.75pt solid #999;
    margin: 16pt 0;
  }
  code {
    font-family: "SF Mono", Menlo, Consolas, "Courier New", monospace;
    font-size: 9.5pt;
    background: #f2f2f2;
    padding: 1pt 3pt;
    border-radius: 2pt;
  }
  pre {
    background: #f2f2f2;
    padding: 8pt 10pt;
    border-radius: 3pt;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 0 0 9pt;
  }
  pre code { background: none; padding: 0; }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 0 0 12pt;
    font-size: 10pt;
    page-break-inside: avoid;
  }
  th, td {
    text-align: left;
    padding: 5pt 8pt;
    border-bottom: 0.75pt solid #ccc;
    vertical-align: top;
  }
  thead th {
    border-bottom: 1.25pt solid #333;
    font-weight: 700;
  }
  a { color: #1a4d8f; text-decoration: none; }
  a:visited { color: #1a4d8f; }
</style>
</head>
<body>
<article>
__BODY__
</article>
</body>
</html>
"""


def markdown_to_html(text: str, title: str = "Document") -> str:
    """Render the supported Markdown subset in `text` to a complete standalone HTML document."""
    body = render_blocks(text.splitlines())
    return _DOC_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__BODY__", body)


# ---------------------------------------------------------------------------
# Browser discovery
# ---------------------------------------------------------------------------


def _is_executable(path: str | None) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _platform_candidates() -> list[str]:
    candidates: list[str] = []
    system = platform.system()

    if system == "Darwin":
        candidates += [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
    elif system == "Windows":
        program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        program_files_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        candidates += [
            os.path.join(program_files, "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(program_files_x86, "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(local_app_data, "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(program_files, "Chromium", "Application", "chrome.exe"),
            os.path.join(program_files, "Microsoft", "Edge", "Application", "msedge.exe"),
            os.path.join(program_files_x86, "Microsoft", "Edge", "Application", "msedge.exe"),
        ]

    for name in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
        "microsoft-edge-stable",
    ):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    return candidates


def find_browser(override: str | None = None) -> str | None:
    """Locate a Chrome/Chromium/Edge binary usable for headless PDF printing.

    An explicit `override` (the --browser flag or a direct call) is authoritative: if it does
    not point at an executable file, this returns None rather than silently substituting a
    different browser found elsewhere on the machine.
    """
    if override is not None:
        return override if _is_executable(override) else None

    env_override = os.environ.get("CHROME")
    if env_override and _is_executable(env_override):
        return env_override

    for candidate in _platform_candidates():
        if _is_executable(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# HTML -> PDF via headless Chrome
# ---------------------------------------------------------------------------


def html_to_pdf(html_path: str, pdf_path: str, browser: str, timeout: int = 60) -> bool:
    """Render `html_path` to `pdf_path` using headless Chrome/Chromium/Edge at `browser`.

    VERIFIED GOTCHA: Chrome's headless PDF printer writes the file to disk and then does NOT
    exit on its own. A foreground `subprocess.run()` would block for the full `timeout` even
    though the PDF is already complete — so this launches with Popen, polls the output file
    until its size is stable across two consecutive polls (it is written incrementally, so
    merely existing is not enough), and then terminates the process itself rather than
    waiting for it to exit.
    """
    html_abspath = os.path.abspath(html_path)
    pdf_abspath = os.path.abspath(pdf_path)
    file_url = Path(html_abspath).as_uri()

    with tempfile.TemporaryDirectory(prefix="render-pdf-chrome-") as scratch_dir:
        # A scratch --user-data-dir is required: without one, a Chrome instance the user
        # already has running can absorb this invocation as a no-op new-tab request instead
        # of actually launching a fresh headless instance that prints the PDF.
        cmd = [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_abspath}",
            f"--user-data-dir={scratch_dir}",
            file_url,
        ]
        # Own process group / session, so the whole browser can be reaped as a unit.
        # Chrome forks renderer, GPU and zygote children; terminating only the parent
        # leaves those alive as orphans. Measured: two headless processes survived a
        # single render before this was added.
        if os.name == "nt":
            popen_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            popen_kwargs = {"start_new_session": True}

        process = subprocess.Popen(  # noqa: PLW1509 -- no preexec_fn; start_new_session is the safe form
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **popen_kwargs
        )
        try:
            success = _wait_for_stable_file(pdf_abspath, timeout)
        finally:
            _terminate_tree(process)

    return success


def _terminate_tree(process: subprocess.Popen) -> None:
    """Kill the browser and every child it spawned.

    Chrome does not exit once the PDF is written, so it has to be stopped rather
    than waited on. `process.terminate()` alone is not enough: it signals the
    parent only, and Chrome's helper processes outlive it. Signalling the whole
    process group catches them.
    """
    if process.poll() is not None:
        return

    if os.name == "nt":
        # No process groups to signal on Windows; taskkill /T walks the tree.
        # Absolute path, not a bare name: a bare "taskkill" is resolved through
        # PATH, so a taskkill.exe earlier on the PATH would be run instead. The
        # only interpolated value is a PID we created ourselves.
        # SYSTEMROOT, uppercase: os.environ is case-insensitive on Windows and
        # ruff's SIM112 wants the canonical form.
        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        taskkill = os.path.join(system_root, "System32", "taskkill.exe")
        subprocess.run(
            [taskkill, "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    else:
        for signal_number in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(process.pid), signal_number)
            except (ProcessLookupError, PermissionError):
                break
            try:
                process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue

    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


def _wait_for_stable_file(path: str, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    last_size: int | None = None
    while time.monotonic() < deadline:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0 and size == last_size:
                return True
            last_size = size
        time.sleep(0.25)
    return bool(os.path.exists(path) and os.path.getsize(path) > 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _report(as_json: bool, html_path: Path, pdf_path: Path | None, engine: str | None) -> None:
    if as_json:
        print(json.dumps({"html": str(html_path), "pdf": str(pdf_path) if pdf_path else None, "engine": engine}))
        return
    print(f"HTML written to: {html_path}")
    if pdf_path:
        print(f"PDF written to: {pdf_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a Markdown document to a print-ready PDF.")
    parser.add_argument("input", help="Path to the input Markdown file")
    parser.add_argument("--out", help="Output PDF path (default: input path with .pdf suffix)")
    parser.add_argument("--html-only", action="store_true", help="Write the HTML and stop, no PDF")
    parser.add_argument("--browser", help="Path to a Chrome/Chromium/Edge binary (overrides autodetect)")
    parser.add_argument("--timeout", type=int, default=60, help="Hard timeout in seconds for PDF rendering")
    parser.add_argument("--title", help="Document title (default: input filename stem)")
    parser.add_argument("--json", action="store_true", help="Emit a JSON result line instead of plain text")
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    text = input_path.read_text(encoding="utf-8")
    title = args.title or input_path.stem

    out_pdf = Path(args.out).resolve() if args.out else input_path.with_suffix(".pdf")
    out_html = out_pdf.with_suffix(".html")

    out_html.write_text(markdown_to_html(text, title), encoding="utf-8")

    if args.html_only:
        _report(args.json, out_html, None, None)
        return 0

    browser = find_browser(args.browser)
    if browser is None:
        # No PDF engine is not a crash: the HTML is itself a deliverable, and many machines
        # in scope for this skill (fresh Linux/Windows boxes) simply won't have Chrome.
        print(
            f"No PDF engine (Chrome/Chromium/Edge) found. HTML written to: {out_html}",
            file=sys.stderr,
        )
        _report(args.json, out_html, None, None)
        return 0

    ok = html_to_pdf(str(out_html), str(out_pdf), browser, timeout=args.timeout)
    if not ok:
        print(
            f"PDF rendering failed or timed out after {args.timeout}s. HTML written to: {out_html}",
            file=sys.stderr,
        )
        _report(args.json, out_html, None, browser)
        return 1

    _report(args.json, out_html, out_pdf, browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
