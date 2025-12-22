import re
from pathlib import Path
from PyPDF2 import PdfReader

pdf_path = Path('bowen.pdf')
if not pdf_path.exists():
    raise SystemExit('bowen.pdf not found in workspace')

reader = PdfReader(str(pdf_path))
texts = []
for i, p in enumerate(reader.pages):
    try:
        t = p.extract_text() or ''
    except Exception:
        t = ''
    texts.append(t)

full = '\n\n'.join(texts)
# write out a plain text dump
out = Path('bowen_extracted.txt')
out.write_text(full)

# find occurrences of CGMY and show context
lines = full.splitlines()
matches = []
for idx, line in enumerate(lines):
    if 'CGMY' in line or 'CGMY' in line.upper():
        start = max(0, idx-3)
        end = min(len(lines), idx+4)
        matches.append('\n'.join(lines[start:end]))

# print some context
if matches:
    print('=== Found CGMY context snippets ===')
    for m in matches:
        print('---')
        print(m)
else:
    # fallback: print lines mentioning table or examples
    print('No direct CGMY token found; printing lines with "Table" or "Example"')
    for idx, line in enumerate(lines):
        if re.search(r'Table|Example|Parameter', line, re.IGNORECASE):
            start = max(0, idx-2)
            end = min(len(lines), idx+3)
            print('---')
            print('\n'.join(lines[start:end]))

print('\nExtraction saved to bowen_extracted.txt')
