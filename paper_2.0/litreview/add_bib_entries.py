"""Append arXiv-verified entries to ../references.bib without rebuilding the whole file (2026-10-10, revise round 1:
the full rebuild's OpenAlex calls were timing out). Metadata from each arXiv abstract page; adds rows to VERIFICATION.md."""
import html
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = {"kumaran2026confidence": "2603.22161", "ni2025popular": "2505.17537"}
NOTE = {"ni2025popular": "EMNLP 2026 Main (arXiv comment)"}
bib_path = HERE.parent / "references.bib"
bib = bib_path.read_text(encoding="utf-8")
rows = []
for key, aid in NEW.items():
    if "{" + key + "," in bib:
        print("exists", key); continue
    h = urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/abs/{aid}", headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode("utf-8", "replace")
    meta = lambda n: [html.unescape(x) for x in re.findall(r'<meta name="citation_' + n + r'" content="([^"]*)"', h)]
    title, authors, date = meta("title")[0], meta("author"), (meta("date") or meta("online_date"))[0]
    fields = [f"  title = {{{{{title}}}}}", f"  author = {{{' and '.join(authors)}}}", f"  year = {{{date[:4]}}}",
              f"  eprint = {{{aid}}}", "  archivePrefix = {arXiv}", f"  url = {{https://arxiv.org/abs/{aid}}}"]
    if key in NOTE:
        fields.append(f"  note = {{{NOTE[key]}}}")
    bib = bib.rstrip() + "\n\n@misc{" + key + ",\n" + ",\n".join(fields) + "\n}\n"
    rows.append(f"| {key} | https://arxiv.org/abs/{aid} | {title} | {authors[0]} | {date[:4]} |")
    time.sleep(1.5)
bib_path.write_text(bib, encoding="utf-8")
if rows:
    with open(HERE / "VERIFICATION.md", "a", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
print("\n".join(rows))
