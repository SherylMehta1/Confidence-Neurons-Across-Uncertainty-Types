"""Synthetic-data tests for score_human.py (same folder). Uses made-up data only; never opens a real key or label file.
Needs numpy, pandas and scikit-learn. From the repo root:
  python results/human_validation_v1/test_score_human.py [<temp dir outside the repo>]
Without an argument it writes its fake files to a fresh system temp folder."""
import csv, importlib.util, os, random, re, subprocess, sys, tempfile
from sklearn.metrics import cohen_kappa_score

sys.dont_write_bytecode = True  # importing the scorer must not leave __pycache__ in the folder
HERE = os.path.dirname(os.path.abspath(__file__))
SCORER = os.path.join(HERE, "score_human.py")
REPO = os.path.dirname(os.path.dirname(HERE))
T = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="score_human_test_")
os.makedirs(T, exist_ok=True)
assert os.path.commonpath([T, REPO]) != REPO, "temp dir must be outside the repo"
COL = "label_DONT_KNOW_or_CLARIFY_or_ANSWER"
CELLS = ["control/HighlyKnown", "control/Unknown", "uncertain/HighlyKnown", "uncertain/Unknown"]
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def run(*files):
    p = subprocess.run([sys.executable, SCORER, "--key", f"{T}/key.csv", *files], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


# ---------- fake key: random verdicts ----------
r = random.Random(2026)
key = [dict(id=f"FAKE{i:03d}", cell=CELLS[i // 30], qwen32b=r.choice(["DONT_KNOW", "ANSWER", "CLARIFY"]),
            gemma27b=r.choice(["DONT_KNOW", "ANSWER", "CLARIFY"])) for i in range(120)]
for k in key:  # make the judges correlated so kappas are not all near zero
    if r.random() < 0.6: k["gemma27b"] = k["qwen32b"]
write(f"{T}/key.csv", key, ["id", "cell", "qwen32b", "gemma27b"])
both = [k["qwen32b"] == "DONT_KNOW" and k["gemma27b"] == "DONT_KNOW" for k in key]

# ---------- two label files: both-judges rule with an exact, known number of flipped replies ----------
FIELDS = ["id", "question", "reply", COL, "note"]
def labels(n_flip, seed):
    rr = random.Random(seed); flip = set(rr.sample(range(120), n_flip)); rows = []
    for i, k in enumerate(key):
        dk = both[i] != (i in flip)
        v = "DONT_KNOW" if dk else rr.choice(["ANSWER", "CLARIFY"])
        rows.append(dict(id=k["id"], question="q", reply="r", **{COL: rr.choice([v, v.lower(), f" {v} "])}, note="x" if i % 17 == 0 else ""))
    rr.shuffle(rows)  # row order differs from the key: forces join on id
    return rows
L1, L2 = labels(12, 1), labels(20, 2)
write(f"{T}/lab1.csv", L1, FIELDS); write(f"{T}/lab2.csv", L2, FIELDS)
h1 = {x["id"]: x[COL].strip().upper() == "DONT_KNOW" for x in L1}
h2 = {x["id"]: x[COL].strip().upper() == "DONT_KNOW" for x in L2}
H1 = [h1[k["id"]] for k in key]; H2 = [h2[k["id"]] for k in key]
Q = [k["qwen32b"] == "DONT_KNOW" for k in key]; G = [k["gemma27b"] == "DONT_KNOW" for k in key]

check("known noise: labeller 1 differs from both-judges rule on exactly 12", sum(a != b for a, b in zip(H1, both)) == 12)
check("known noise: labeller 2 differs from both-judges rule on exactly 20", sum(a != b for a, b in zip(H2, both)) == 20)


def hand_kappa(a, b):
    """Cohen's kappa from the 2x2 table, written out independently of the scorer."""
    n = len(a); n11 = sum(x and y for x, y in zip(a, b)); n00 = sum((not x) and (not y) for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b)); n01 = n - n11 - n00 - n10
    po = (n11 + n00) / n
    pe = ((n11 + n10) / n) * ((n11 + n01) / n) + ((n00 + n01) / n) * ((n00 + n10) / n)
    return (po - pe) / (1 - pe)


def skl(a, b):
    return cohen_kappa_score(["DK" if x else "NO" for x in a], ["DK" if x else "NO" for x in b])


# ---------- run the scorer and parse its kappas ----------
rc, out, err = run(f"{T}/lab1.csv", f"{T}/lab2.csv")
check("scorer runs on valid synthetic files", rc == 0, err.strip()[-200:])
NUM = r"(-?\d+\.\d+) \[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"
def judge_kappas(text):
    d = {}
    for name in ["both-judges rule", "qwen32b", "gemma27b"]:
        m = re.search(rf"^{re.escape(name)}\n  kappa \[95% CI\]\s+{NUM}\s+{NUM}", text, re.M)
        d[name] = (m.group(1, 2, 3), m.group(4, 5, 6))
    d["inter"] = re.search(rf"inter-human kappa \[95% CI\]:\s+{NUM}", text).group(1, 2, 3)
    d["agreed"] = re.search(rf"both-judges rule kappa \[95% CI\]:\s+{NUM}", text).group(1, 2, 3)
    return d
K = judge_kappas(out)

# full-precision kappa from the scorer's own function
spec = importlib.util.spec_from_file_location("score_human", SCORER); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
agree = [a == b for a, b in zip(H1, H2)]
A1 = [x for x, g in zip(H1, agree) if g]; AB = [x for x, g in zip(both, agree) if g]
pairs = {("both-judges rule", 0): (H1, both), ("both-judges rule", 1): (H2, both), ("qwen32b", 0): (H1, Q), ("qwen32b", 1): (H2, Q),
         ("gemma27b", 0): (H1, G), ("gemma27b", 1): (H2, G), ("inter", None): (H1, H2), ("agreed", None): (A1, AB)}
for (name, col), (a, b) in pairs.items():
    printed = float((K[name][col] if col is not None else K[name])[0])
    sk, hd, sc = skl(a, b), hand_kappa(a, b), mod.kappa(a, b)
    label = f"{name}" + (f", labeller {col + 1}" if col is not None else "")
    check(f"kappa {label}: scorer = sklearn = hand", abs(sc - sk) < 1e-12 and abs(sc - hd) < 1e-12 and printed == round(sk, 2),
          f"printed {printed:.2f}, sklearn {sk:.6f}, hand {hd:.6f}, scorer {sc:.6f}")

# ---------- same seed, same intervals ----------
rc2, out2, _ = run(f"{T}/lab1.csv", f"{T}/lab2.csv")
ci = lambda t: re.findall(r"\[\s*-?\d+\.\d+,\s*-?\d+\.\d+\]", t)
check("same seed gives identical intervals on a second run", rc2 == 0 and ci(out) == ci(out2) and len(ci(out)) == 8, f"{len(ci(out))} intervals")
import numpy as np
i0a, i0b, i1 = mod.boot_idx(120), mod.boot_idx(120), np.random.default_rng(1).integers(0, 120, size=(mod.B, 120))
check("bootstrap: 10,000 draws, seed 0 reproducible, a different seed differs", i0a.shape == (10_000, 120) and (i0a == i0b).all() and not (i0a == i1).all())

# ---------- invalid inputs stop the scorer with the right error ----------
def bad(name, rows):
    write(f"{T}/{name}.csv", rows, FIELDS); return f"{T}/{name}.csv"
cases = {
    "blank label": bad("bad_blank", [dict(x, **{COL: ""}) if i == 3 else x for i, x in enumerate(L2)]),
    "typo DONT KNOW": bad("bad_typo", [dict(x, **{COL: "DONT KNOW"}) if i == 3 else x for i, x in enumerate(L2)]),
    "missing id": bad("bad_missing", L2[:-1]),
    "duplicate id": bad("bad_dup", L2 + [dict(L2[0])]),
}
for name, path in cases.items():
    for pos, files in [(2, (f"{T}/lab1.csv", path)), (1, (path, f"{T}/lab2.csv"))]:
        rc, o, e = run(*files)
        want = f"ERROR: labeller {pos}: 1 bad row(s)"
        check(f"{name} (as labeller {pos}) stops with '{want} ...'", rc == 1 and e.strip().startswith(want) and "kappa" not in o and "FAKE" not in e + o, e.strip())
rc, o, e = run(f"{T}/bad_blank.csv", f"{T}/bad_dup.csv")
check("bad rows in both files: first file reported, nothing scored", rc == 1 and "labeller 1: 1 bad row(s)" in e and "kappa" not in o)

# ---------- swapping the label files swaps the columns, inter-human unchanged ----------
rc, outs, _ = run(f"{T}/lab2.csv", f"{T}/lab1.csv")
Ks = judge_kappas(outs)
check("swap: every judge's kappa and CI swap columns exactly", all(Ks[n] == (K[n][1], K[n][0]) for n in ["both-judges rule", "qwen32b", "gemma27b"]))
def two_cols(text, label):
    return [tuple(re.split(r"\s{2,}", l.strip())[1:3]) for l in text.splitlines() if l.startswith(label)]
check("swap: agreement, precision, recall, human DONT_KNOW rate swap columns",
      all([(b, a) for a, b in two_cols(out, f"  {m}")] == two_cols(outs, f"  {m}") for m in ["agreement", "precision", "recall"])
      and re.search(r"labeller 1\s+(\S+) \| labeller 2\s+(\S+)", out).groups()[::-1] == re.search(r"labeller 1\s+(\S+) \| labeller 2\s+(\S+)", outs).groups())
check("swap: inter-human kappa and interval unchanged", Ks["inter"] == K["inter"], f"{K['inter']}")

# ---------- no ids or per-reply verdicts in output ----------
check("no id appears in any output", "FAKE" not in out + out2 + outs)

print(f"\n{sum(ok for _, ok in results)} of {len(results)} checks passed")
sys.exit(0 if all(ok for _, ok in results) else 1)
