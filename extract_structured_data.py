#!/usr/bin/env python3
"""
extract_structured_data.py

Extracts structured data from unstructured documents (HTML, Markdown, plain text)
using the Anthropic Claude API and a user-supplied JSON schema.

Usage:
    python extract_structured_data.py <input_file> <schema_file> [--output <output_file>]

Arguments:
    input_file    Path to the document (.html, .htm, .md, .markdown, .txt, or other)
    schema_file   Path to a JSON file describing the desired output schema
    --output      (Optional) Path to write the extracted JSON. Defaults to stdout.

Environment:
    ANTHROPIC_API_KEY  Your Anthropic API key (required)

Examples:
    python extract_structured_data.py invoice.html schema.json
    python extract_structured_data.py article.md  schema.json --output result.json
    python extract_structured_data.py report.txt  schema.json --output result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Optional dependency: html2text for HTML → Markdown conversion.
# Install with:  pip install html2text
# ---------------------------------------------------------------------------
try:
    import html2text
    _HTML2TEXT_AVAILABLE = True
except ImportError:
    _HTML2TEXT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Anthropic SDK
# Install with:  pip install anthropic
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:
    sys.exit(
        "ERROR: The 'anthropic' package is not installed.\n"
        "       Run:  pip install anthropic"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

HTML_EXTENSIONS  = {".html", ".htm"}
MD_EXTENSIONS    = {".md", ".markdown"}
# Everything else is treated as plain text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_doc_type(path: Path) -> str:
    """Return 'html', 'markdown', or 'text' based on file extension."""
    ext = path.suffix.lower()
    if ext in HTML_EXTENSIONS:
        return "html"
    if ext in MD_EXTENSIONS:
        return "markdown"
    return "text"


def html_to_markdown(html: str) -> str:
    """
    Convert an HTML string to Markdown, retaining as much content as possible.
    Falls back to a simple tag-stripping approach if html2text is unavailable.
    """
    if _HTML2TEXT_AVAILABLE:
        converter = html2text.HTML2Text()
        converter.ignore_links       = False
        converter.ignore_images      = False
        converter.ignore_tables      = False
        converter.body_width         = 0          # no hard line-wraps
        converter.protect_links      = True
        converter.wrap_links         = False
        converter.single_line_break  = False
        converter.mark_code          = True
        return converter.handle(html)

    # --- Fallback: basic tag stripping (lossy but dependency-free) ----------
    import re
    # Replace common structural tags with newlines
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</?(?:p|div|li|tr|h[1-6]|blockquote)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Remove all remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Collapse excessive blank lines
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def load_schema(schema_path: Path) -> dict:
    """Load and validate that the schema file is valid JSON."""
    try:
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: Schema file is not valid JSON: {exc}")
    except OSError as exc:
        sys.exit(f"ERROR: Cannot read schema file: {exc}")
    return schema


def load_document(doc_path: Path) -> tuple[str, str]:
    """
    Read the document file.

    Returns
    -------
    (content, doc_type)
        content   : the (possibly converted) document text
        doc_type  : 'html' | 'markdown' | 'text'
    """
    try:
        raw = doc_path.read_text(encoding="utf-8")
    except OSError as exc:
        sys.exit(f"ERROR: Cannot read document file: {exc}")

    doc_type = detect_doc_type(doc_path)

    if doc_type == "html":
        print(f"[info] Detected HTML document — converting to Markdown …", file=sys.stderr)
        content = html_to_markdown(raw)
        if not _HTML2TEXT_AVAILABLE:
            print(
                "[warn] 'html2text' is not installed; using basic tag-stripping fallback.\n"
                "       For best results run:  pip install html2text",
                file=sys.stderr,
            )
    else:
        content = raw

    return content, doc_type


def build_prompt(content: str, doc_type: str, schema: dict) -> str:
    """Construct the user prompt sent to Claude."""
    schema_str = json.dumps(schema, indent=2)

    doc_type_label = {
        "html":     "Markdown (converted from HTML)",
        "markdown": "Markdown",
        "text":     "plain text",
    }.get(doc_type, "text")

    return f"""You are a precise data-extraction assistant.

Your task is to extract information from the document below and return it as a
single JSON object that strictly conforms to the provided JSON schema.

Rules:
- Return ONLY the JSON object — no prose, no markdown fences, no explanation.
- Every required field in the schema MUST be present in your output.
- Use null for optional fields where the information is not present in the document.
- Do not invent or hallucinate values that are not supported by the document.
- Preserve original formatting for free-text fields (dates, names, amounts, etc.).

---

## JSON Schema

```json
{schema_str}
```

---

## Document ({doc_type_label})

{content}

---

Now output the extracted JSON object:"""


def extract_with_claude(prompt: str, api_key: str) -> str:
    """Call the Claude API and return the raw text response."""
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    # The response content is a list of blocks; grab the first text block.
    for block in message.content:
        if block.type == "text":
            return block.text

    sys.exit("ERROR: Claude returned no text content in its response.")


def parse_json_response(raw: str) -> dict:
    """
    Parse the JSON from Claude's response.
    Handles the case where the model accidentally wraps it in a code fence.
    """
    text = raw.strip()

    # Strip optional markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (```json or ```)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(
            f"[warn] Could not parse Claude's response as JSON: {exc}\n"
            f"[warn] Raw response follows:\n{raw}",
            file=sys.stderr,
        )
        sys.exit("ERROR: Extraction failed — Claude did not return valid JSON.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract structured data from a document using Claude.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_file",  help="Path to the input document (.html, .md, .txt, …)")
    parser.add_argument("schema_file", help="Path to the JSON schema file")
    parser.add_argument(
        "--output", "-o",
        metavar="OUTPUT_FILE",
        default=None,
        help="Write extracted JSON to this file (default: stdout)",
    )
    args = parser.parse_args()

    # --- Resolve paths -------------------------------------------------------
    doc_path    = Path(args.input_file)
    schema_path = Path(args.schema_file)

    if not doc_path.exists():
        sys.exit(f"ERROR: Input file not found: {doc_path}")
    if not schema_path.exists():
        sys.exit(f"ERROR: Schema file not found: {schema_path}")

    # --- API key -------------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "       Export it before running this script."
        )

    # --- Load inputs ---------------------------------------------------------
    schema              = load_schema(schema_path)
    content, doc_type   = load_document(doc_path)

    print(
        f"[info] Document type : {doc_type}\n"
        f"[info] Document chars: {len(content):,}\n"
        f"[info] Calling Claude ({MODEL}) …",
        file=sys.stderr,
    )

    # --- Build prompt & call API ---------------------------------------------
    prompt      = build_prompt(content, doc_type, schema)
    raw_response = extract_with_claude(prompt, api_key)

    # --- Parse & output ------------------------------------------------------
    extracted = parse_json_response(raw_response)
    output_json = json.dumps(extracted, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_json + "\n", encoding="utf-8")
        print(f"[info] Extracted data written to: {out_path}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
