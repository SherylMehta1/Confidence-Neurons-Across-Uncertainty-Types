"""Re-run the repo's own read-offs in a shadow copy of the repo, with the stored or the corrected Gemma-27B labels. No existing file is written.
  python scripts/rejudge_shadow.py DIR stored|corrected
DIR gets symlinks to data and untouched inputs and real copies of everything the scripts write. With `stored`, every output must equal the
committed one (that is the check of this script). With `corrected`, the 18 judge_gemma27b* files are replaced by results/rejudge_gemma labels."""
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIR, MODE = Path(sys.argv[1]).resolve(), sys.argv[2]
assert MODE in ("stored", "corrected") and REPO not in [DIR, *DIR.parents]
if DIR.exists():
    shutil.rmtree(DIR)
COPY = ["results/hedge_measure_v2", "results/hedge_judge", "results/round3", "scripts", "shared", "paper_2.0/analysis", "paper_2.0/figures", "results/x0"]
DIR.mkdir(parents=True)
for p in REPO.iterdir():
    if p.name not in ("results", "paper_2.0", "scripts", "shared", ".git", ".venv"):
        (DIR / p.name).symlink_to(p)
for top in ("results", "paper_2.0"):
    (DIR / top).mkdir()
    for p in (REPO / top).iterdir():
        if f"{top}/{p.name}" not in COPY and not (p.is_file() and top == "paper_2.0"):
            (DIR / top / p.name).symlink_to(p)
for c in COPY:
    shutil.copytree(REPO / c, DIR / c, ignore=shutil.ignore_patterns("__pycache__"))
for p in (REPO / "paper_2.0").iterdir():
    if p.is_file():
        shutil.copy(p, DIR / "paper_2.0" / p.name)
(DIR / "Confidence-Neurons-Across-Uncertainty-Types").symlink_to(DIR)  # name two analysis scripts hard-code
if MODE == "corrected":
    new = {}
    for r in map(json.loads, open(REPO / "results/rejudge_gemma/rejudge_gemma27b.jsonl", encoding="utf-8")):
        new.setdefault(r["file"], {})[tuple(r["id"])] = r
    n = 0
    for f, d in new.items():
        if not f.startswith("results/"):
            continue
        old = [json.loads(l) for l in open(REPO / f, encoding="utf-8") if l.strip()]
        keys = [k for k in old[0] if k not in ("raw", "label")][-len(next(iter(d))):]  # E8 and E16 files carry an extra leading "model" field
        assert len(old) == len(d)
        with open(DIR / f, "w", encoding="utf-8") as o:
            for r in old:
                x = d[tuple(r[k] for k in keys)]
                o.write(json.dumps(dict(r, raw=x["raw"], label=x["label"])) + "\n"); n += 1
    print("labels replaced:", n, "in", sum(f.startswith("results/") for f in new), "files")
STEPS = [("scripts/hedge_measure_v2_analysis.py", []), *[("scripts/round3_analysis.py", [e]) for e in ("e1", "e2", "e4", "e5", "e6", "e7", "e8", "e9", "e11", "e12", "e13", "e14", "e15", "e16")],
         *[(f"paper_2.0/analysis/{n}.py", []) for n in ("echo_check", "round2_checks", "round3_label_checks", "lockin_checks", "ambiguity_check", "guessability", "mere_checks", "surprisal_check", "round4_checks", "explore_olmo")],
         ("paper_2.0/make_numbers.py", [])]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONWARNINGS="ignore")
for s, a in STEPS:
    t = time.time()
    r = subprocess.run([sys.executable, str(DIR / s), *a], cwd=DIR, env=env, capture_output=True, text=True)
    print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {s} {' '.join(a)} {time.time() - t:.0f}s" + ("" if r.returncode == 0 else "\n" + r.stderr[-700:]), flush=True)
nm = lambda s: re.sub(r"np\.int64\((\d+)\)", r"\1", s)
OUTS = sorted({p.relative_to(DIR) for d in ("results/hedge_measure_v2", "results/round3", "paper_2.0/analysis") for p in (DIR / d).glob("*") if p.suffix in (".txt", ".csv") and not p.is_symlink()} | {Path("paper_2.0/numbers.tex"), Path("paper_2.0/tab_onsets.tex")})
same = 0
for o in OUTS:
    if not (REPO / o).exists():
        print("new file", o); continue
    a, b = nm((REPO / o).read_text(encoding="utf-8-sig")).splitlines(), nm((DIR / o).read_text(encoding="utf-8-sig")).splitlines()
    if a == b:
        same += 1
    else:
        d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        print(f"differs: {o} ({len(d)} lines)")
        if MODE == "stored":
            print("\n".join("    " + l[:200] for l in d[:6]))
print(f"{same} of {len(OUTS)} outputs identical to the committed files")
