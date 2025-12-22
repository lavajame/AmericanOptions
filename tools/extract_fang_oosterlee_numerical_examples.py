from __future__ import annotations

import re
from pathlib import Path


# Require: "Table <num>" not immediately followed by a period, to avoid matches like
# "... presented in Table 8." that can be line-broken to start with "Table 8.".
RE_TABLE = re.compile(r"^Table\s+(\d+)(?!\.)\b(.*)$")
RE_FIG = re.compile(r"^Fig\.?\s*(\d+)\b(.*)$")
RE_PAGE = re.compile(r"^--- page (\d+) ---$")


def _clean_block(block: list[str]) -> list[str]:
    while block and not block[0].strip():
        block.pop(0)
    while block and not block[-1].strip():
        block.pop()
    return block


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "fang_oosterlee_extracted.txt"
    out = repo_root / "fang_oosterlee_numerical_examples.txt"

    if not src.exists():
        raise SystemExit(
            f"Missing {src}. Run tools/extract_fang_oosterlee_examples.py first."
        )

    lines = src.read_text(encoding="utf-8").splitlines()

    table_starts: list[tuple[int, str]] = []
    fig_lines: list[tuple[int, str]] = []
    page_at_line: dict[int, int] = {}

    current_page = None
    for i, line in enumerate(lines):
        mpage = RE_PAGE.match(line.strip())
        if mpage:
            current_page = int(mpage.group(1))
        if current_page is not None:
            page_at_line[i] = current_page

        if RE_TABLE.match(line.strip()):
            table_starts.append((i, line.strip()))
        if RE_FIG.match(line.strip()):
            fig_lines.append((i, line.strip()))

    blocks: list[str] = []
    blocks.append("Fang & Oosterlee (2009) – Numerical Examples (extracted)")
    blocks.append("Source: fang_oosterlee_american.pdf")
    blocks.append("Generated from: fang_oosterlee_extracted.txt")
    blocks.append("")

    if fig_lines:
        blocks.append("Figures (captions found):")
        for idx, cap in fig_lines:
            pg = page_at_line.get(idx)
            where = f"page {pg}" if pg is not None else "(page ?)"
            blocks.append(f"- {where}: {cap}")
        blocks.append("")

    if not table_starts:
        blocks.append("No tables found via simple pattern match.")
        out.write_text("\n".join(blocks) + "\n", encoding="utf-8")
        print(f"Wrote: {out}")
        return

    blocks.append("Tables (blocks extracted):")
    blocks.append("")

    # Extract each table block until the next Table heading or a page delimiter.
    for j, (start, heading) in enumerate(table_starts):
        end = len(lines)
        if j + 1 < len(table_starts):
            end = min(end, table_starts[j + 1][0])

        # Also stop at the next page delimiter if it occurs before the next table.
        for k in range(start + 1, end):
            if RE_PAGE.match(lines[k].strip()):
                end = k
                break

        raw_block = lines[start:end]
        # If the table is embedded in text, keep a reasonable block size.
        raw_block = raw_block[:120]
        block = _clean_block(raw_block)

        pg = page_at_line.get(start)
        where = f"page {pg}" if pg is not None else "(page ?)"

        blocks.append(f"=== {where}: {heading} ===")
        blocks.extend(block)
        blocks.append("")

    # Also capture “reference value” snippets which often encode target numbers.
    blocks.append("Reference-value mentions (snippets):")
    ref_re = re.compile(r"reference value", re.IGNORECASE)
    for i, line in enumerate(lines):
        if ref_re.search(line):
            pg = page_at_line.get(i)
            where = f"page {pg}" if pg is not None else "(page ?)"
            window = lines[max(0, i - 1) : min(len(lines), i + 3)]
            window = [w.strip() for w in window if w.strip()]
            blocks.append(f"- {where}: " + " ".join(window))

    out.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
