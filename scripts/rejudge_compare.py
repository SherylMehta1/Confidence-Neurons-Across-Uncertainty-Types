"""Old against corrected Gemma-27B labels, per stored file and per paper number. Usage: python scripts/rejudge_compare.py SHADOW_DIR (made by
rejudge_shadow.py SHADOW_DIR corrected). Writes results/rejudge_gemma/{labels_by_file,qwen_check,verdicts,macro_changes}.csv and corrected/*.new.
Reads the papers; changes none."""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SH, D = Path(sys.argv[1]).resolve(), REPO / "results/rejudge_gemma"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


NEW, QW = {}, {}
for n, d in (("gemma27b", NEW), ("qwen32b", QW)):
    for r in read(D / f"rejudge_{n}.jsonl"):
        d.setdefault(r["file"], {})[tuple(r["id"])] = r
rows, q = [], []
for f in sorted(x for x in NEW if x.startswith("results/")):
    w = len(next(iter(NEW[f])))
    load = lambda p: {tuple(v for k, v in r.items() if k not in ("raw", "label"))[-w:]: r for r in read(p)}
    old, qs = load(REPO / f), load(REPO / f.replace("gemma27b", "qwen32b"))
    l7 = load(REPO / f.replace("gemma27b", "llama70b")) if (REPO / f.replace("gemma27b", "llama70b")).exists() else None
    ks = [k for k in old if k in qs]; dk = lambda d: [d[k]["label"] == "DONT_KNOW" for k in ks]
    both = lambda g: np.mean([a and b for a, b in zip(dk(g), dk(qs))])
    r = dict(file=f, replies=len(old), unparsed_old=sum(x["label"] == "UNPARSED" for x in old.values()), unparsed_new=sum(x["label"] == "UNPARSED" for x in NEW[f].values()), label_changed=np.mean([old[k]["label"] != NEW[f][k]["label"] for k in old]),
             dont_know_old=np.mean(dk(old)), dont_know_new=np.mean(dk(NEW[f])), kappa_qwen_old=kappa(dk(old), dk(qs)), kappa_qwen_new=kappa(dk(NEW[f]), dk(qs)), both_judges_old=both(old), both_judges_new=both(NEW[f]))
    if l7 and all(k in l7 for k in ks):
        r |= dict(kappa_llama70_old=kappa(dk(old), dk(l7)), kappa_llama70_new=kappa(dk(NEW[f]), dk(l7)))
    rows.append(r)
    s = QW.get(f, {})
    q.append(dict(file=f, checked=len(s), qwen_label_equal_stored=np.mean([s[k]["label"] == qs[k]["label"] for k in s if k in qs]), qwen_dont_know_equal=np.mean([(s[k]["label"] == "DONT_KNOW") == (qs[k]["label"] == "DONT_KNOW") for k in s if k in qs])))
T = pd.DataFrame(rows); T.to_csv(D / "labels_by_file.csv", index=False, float_format="%.5g"); print(T.drop(columns=["unparsed_old", "unparsed_new"]).round(3).to_string(index=False))
Qc = pd.DataFrame(q); Qc.to_csv(D / "qwen_check.csv", index=False, float_format="%.5g"); print(Qc.round(4).to_string(index=False)); print("Qwen overall:", np.average(Qc.qwen_dont_know_equal, weights=Qc.checked))

# corrected outputs, suffix .new so that no other script mistakes them for the committed files
C = D / "corrected"; shutil.rmtree(C, ignore_errors=True); C.mkdir()
for d in ("results/hedge_measure_v2", "results/round3", "paper_2.0/analysis"):
    for p in (SH / d).glob("*"):
        o = REPO / d / p.name
        if p.is_file() and p.suffix in (".txt", ".csv") and o.exists() and o.read_bytes() != p.read_bytes():
            shutil.copy(p, C / f"{Path(d).name}__{p.name}.new")
shutil.copy(SH / "paper_2.0/numbers.tex", C / "paper_2.0__numbers.tex.new")

# verdict lines that change
V = []
pat = re.compile(r"(->\s*[A-Z][A-Z0-9 \-/<>&']{5,}|verdict: .*|licensing: .*|title .*)")
for p in sorted(C.glob("*summary.txt.new")):
    o = REPO / "results" / p.name.split("__")[0] / p.name.split("__")[1][:-4]
    a, b = o.read_text().splitlines(), p.read_text().splitlines()
    if len(a) != len(b):
        print("line counts differ", p.name); continue
    head = ""
    for x, y in zip(a, b):
        if re.match(r"(== )?\[?(llama|qwen|gemma|mistral|olmo2?_?7?b?|gemma9b)\b", x.strip()):
            head = x.strip()[:12].split()[0 if not x.startswith("==") else 1]
        mx, my = pat.search(x), pat.search(y)
        if mx and my and mx[0].strip() != my[0].strip():
            V.append(dict(source=p.name.split("__")[1][:-4], model=head.strip("[]:"), line=re.sub(r"\s+", " ", y.strip())[:60], as_stored=mx[0].strip()[:90], corrected=my[0].strip()[:90]))
V = pd.DataFrame(V); V.to_csv(D / "verdicts.csv", index=False); print(V.to_string(index=False))

# every number macro each paper draft uses
mac = lambda p: {m[1]: (m[2], m[3]) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*% *(.*)", p.read_text(encoding="utf-8"))}
a, b = mac(REPO / "paper_2.0/numbers.tex"), mac(SH / "paper_2.0/numbers.tex")
assert set(a) == set(b)
R = []
for m in a:
    use = {n: len(re.findall(rf"\\{m}(?![A-Za-z])", (REPO / f).read_text(encoding="utf-8"))) for n, f in (("srw", "paper_2.0/srw/main.tex"), ("mere_v2", "paper_2.0/mere/main_v2.tex"), ("unireps", "paper_2.0/main.tex"))}
    if a[m][0] != b[m][0]:
        R.append(dict(macro=m, as_printed=a[m][0].replace("$", "")[:160], corrected=b[m][0].replace("$", "")[:160], source=a[m][1], **use))
R = pd.DataFrame(R); R.to_csv(D / "macro_changes.csv", index=False)
print(f"{len(a)} macros, {len(R)} change; used in a draft and changed: " + ", ".join(f"{n} {(R[n] > 0).sum()}" for n in ("srw", "mere_v2", "unireps")))
print(R[R.srw > 0][["macro", "as_printed", "corrected"]].to_string(index=False))
