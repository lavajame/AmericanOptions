from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


RE_WHITESPACE = re.compile(r"[ \t\u00A0]+");
RE_MANY_NEWLINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Section:
    title: str
    start_page: int
    end_page: int
    text: str


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_WHITESPACE.sub(" ", text)
    # Keep line breaks, but reduce excessive vertical whitespace
    text = RE_MANY_NEWLINES.sub("\n\n", text)
    # Trim trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        pages.append(_normalize_text(raw))
    return pages


def _find_candidate_pages(pages: list[str], keywords: Iterable[str]) -> list[int]:
    needles = [k.lower() for k in keywords]
    hits: list[int] = []
    for idx, text in enumerate(pages):
        t = text.lower()
        if any(k in t for k in needles):
            hits.append(idx)
    return hits


def _slice_pages(pages: list[str], start: int, end: int) -> str:
    chunk = "\n\n".join(
        f"[page {i+1}]\n{pages[i]}" for i in range(start, end + 1) if pages[i].strip()
    )
    return chunk.strip()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "fang_oosterlee_american.pdf"
    out_path = repo_root / "fang_oosterlee_extracted.txt"

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    pages = extract_pdf_text(pdf_path)

    # Heuristic keywords for “numerical examples” sections in papers.
    keywords = [
        "numerical",
        "example",
        "examples",
        "experiment",
        "experiments",
        "table",
        "figure",
        "bermudan",
        "american",
        "pricing",
        "results",
    ]
    hit_pages = _find_candidate_pages(pages, keywords)

    # Try to narrow around pages that explicitly mention "Numerical" and "Example".
    strong_hits = _find_candidate_pages(pages, ["numerical example", "numerical examples"]) or hit_pages

    # Build a compact-but-complete extract:
    #   - Always include first pages (setup/notation)
    #   - Include the full band where tables/figures are concentrated (common for numerical section)
    #   - Also include a neighborhood around strong hits (backup)
    include: set[int] = set()

    for p in range(min(2, len(pages))):
        include.add(p)

    # Heuristic: if the paper has a dense numerical section, it often spans ~10 pages.
    # Our index for this PDF tends to cluster around the mid/late pages.
    if hit_pages:
        lo = min(hit_pages)
        hi = max(hit_pages)
        # Expand a bit to capture section headers and parameter descriptions.
        lo = max(0, lo - 3)
        hi = min(len(pages) - 1, hi + 3)
        # Cap band size (avoid dumping the whole paper if keywords are everywhere).
        if (hi - lo + 1) <= 20:
            for p in range(lo, hi + 1):
                include.add(p)

    for p in strong_hits:
        for q in range(max(0, p - 2), min(len(pages), p + 3)):
            include.add(q)

    include_list = sorted(include)

    extracted = []
    extracted.append("Fang & Oosterlee American (extracted)")
    extracted.append(f"Source: {pdf_path.name}")
    extracted.append("")

    extracted.append("Included pages (heuristic): " + ", ".join(str(i + 1) for i in include_list))
    extracted.append("")

    for i in include_list:
        t = pages[i]
        if not t.strip():
            continue
        extracted.append(f"--- page {i+1} ---")
        extracted.append(t)
        extracted.append("")

    out_path.write_text("\n".join(extracted), encoding="utf-8")

    # Also write a tiny “index” of likely example/table mentions across all pages.
    index_lines: list[str] = []
    table_re = re.compile(r"\b(Table|TABLE)\s+(\d+)\b")
    fig_re = re.compile(r"\b(Figure|FIGURE|Fig\.)\s*(\d+)\b")
    ex_re = re.compile(r"\b(Example|EXAMPLE)\s*(\d+)\b")

    for i, t in enumerate(pages):
        if not t:
            continue
        tables = sorted({m.group(0) for m in table_re.finditer(t)})
        figs = sorted({m.group(0) for m in fig_re.finditer(t)})
        exs = sorted({m.group(0) for m in ex_re.finditer(t)})
        if tables or figs or exs or ("numerical" in t.lower() and "example" in t.lower()):
            bits = []
            if exs:
                bits.append("Examples: " + ", ".join(exs))
            if tables:
                bits.append("Tables: " + ", ".join(tables))
            if figs:
                bits.append("Figures: " + ", ".join(figs))
            if "numerical" in t.lower() and "example" in t.lower() and not exs:
                bits.append("mentions: numerical example")
            index_lines.append(f"page {i+1}: " + " | ".join(bits))

    (repo_root / "fang_oosterlee_examples_index.txt").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )

    print(f"Wrote: {out_path}")
    print(f"Wrote: {repo_root / 'fang_oosterlee_examples_index.txt'}")


if __name__ == "__main__":
    main()
