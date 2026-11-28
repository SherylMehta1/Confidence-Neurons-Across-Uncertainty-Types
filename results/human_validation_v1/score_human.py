"""Score the two hedge judges against two human labellers (results/human_validation_v1/README.md).

Copy of scripts/hedge_measure_v2_human_score.py, finalised for the v1 human check. The metric definitions are
unchanged: DONT_KNOW is one outcome and ANSWER/CLARIFY the other, the kappa formula is the same, and the
both-judges rule is qwen32b == DONT_KNOW and gemma27b == DONT_KNOW.

Usage (from the repo root):
  .venv/bin/python results/human_validation_v1/score_human.py --key <key.csv> <labeller1.csv> <labeller2.csv>

Prints aggregates only. It never prints an id or an individual reply's verdict, including in error messages.
"""
import argparse
import hashlib
import sys

import numpy as np
import pandas as pd

COL = "label_DONT_KNOW_or_CLARIFY_or_ANSWER"
NOTE = "note"
VALID = {"DONT_KNOW", "CLARIFY", "ANSWER"}
N_EXPECTED = 120
B, SEED = 10_000, 0


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pa, pb = a.mean(), b.mean(); pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kappa_rows(a, b):
    """The same formula as kappa(), applied to each row of 2-D boolean arrays (one row per bootstrap draw)."""
    po = (a == b).mean(1); pa, pb = a.mean(1), b.mean(1); pe = pa * pb + (1 - pa) * (1 - pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pe < 1, (po - pe) / (1 - pe), np.nan)


def boot_idx(n):
    """B resamples of n replies with replacement (seed SEED). Every kappa on the same set of replies uses the
    same resamples, so intervals do not depend on the order in which kappas are computed or files are given."""
    return np.random.default_rng(SEED).integers(0, n, size=(B, n))


def boot_ci(a, b, idx):
    """95% percentile interval for kappa(a, b) over the resamples idx."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    k = kappa_rows(a[idx], b[idx])
    ok = ~np.isnan(k)
    lo, hi = (np.percentile(k[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan))
    return lo, hi, int((~ok).sum())


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def load_key(path):
    k = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = {"id", "cell", "qwen32b", "gemma27b"} - set(k.columns)
    if missing:
        fail(f"key is missing {len(missing)} required column(s): {sorted(missing)}")
    k["id"] = k["id"].str.strip()
    if len(k) != N_EXPECTED or k["id"].duplicated().any() or (k["id"] == "").any():
        bad = int(k["id"].duplicated(keep="first").sum() + (k["id"] == "").sum())
        fail(f"key must have {N_EXPECTED} distinct non-blank ids; it has {len(k)} rows, {bad} bad row(s)")
    return k


def load_labels(path, key_ids, name):
    s = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "id" not in s.columns or COL not in s.columns:
        fail(f"{name}: needs columns 'id' and '{COL}'")
    s["id"] = s["id"].str.strip()
    s[COL] = s[COL].str.strip().str.upper()
    dup = s["id"].duplicated(keep="first")  # each extra copy of an id is one bad row
    extra = ~s["id"].isin(key_ids)
    badlab = ~s[COL].isin(VALID)
    missing = len(key_ids - set(s["id"]))
    n_bad = int((dup | extra | badlab).sum()) + missing
    if n_bad:
        fail(f"{name}: {n_bad} bad row(s) (blank or invalid label, duplicate, unknown or missing id); nothing scored")
    if NOTE not in s.columns:
        s[NOTE] = ""
    return s[["id", COL, NOTE]]


def pct(x):
    return f"{100 * x:5.1f}%" if not np.isnan(x) else "   n/a"


def fk(k, ci):
    lo, hi, nan = ci
    extra = f" ({nan} draws undefined)" if nan else ""
    return f"{k:5.2f} [{lo:5.2f}, {hi:5.2f}]{extra}" if not np.isnan(k) else "  n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="key CSV with columns id, cell, qwen32b, gemma27b")
    ap.add_argument("labeller1")
    ap.add_argument("labeller2")
    a = ap.parse_args()

    print(f"script sha256: {sha256(__file__)}")
    print(f"key    sha256: {sha256(a.key)}  ({a.key})")

    key = load_key(a.key)
    ids = set(key["id"])
    l1 = load_labels(a.labeller1, ids, "labeller 1")
    l2 = load_labels(a.labeller2, ids, "labeller 2")

    # Join on id only; row order in any file is irrelevant.
    m = (key[["id", "cell", "qwen32b", "gemma27b"]]
         .merge(l1.rename(columns={COL: "h1", NOTE: "n1"}), on="id", how="inner", validate="1:1")
         .merge(l2.rename(columns={COL: "h2", NOTE: "n2"}), on="id", how="inner", validate="1:1"))
    assert len(m) == N_EXPECTED

    idx = boot_idx(len(m))
    q = (m.qwen32b == "DONT_KNOW").to_numpy()
    g = (m.gemma27b == "DONT_KNOW").to_numpy()
    both = q & g
    judges = [("both-judges rule", both), ("qwen32b", q), ("gemma27b", g)]
    hs = [(m.h1 == "DONT_KNOW").to_numpy(), (m.h2 == "DONT_KNOW").to_numpy()]

    print(f"\nreplies scored: {len(m)} (joined on id); outcome = DONT_KNOW vs (ANSWER or CLARIFY)")
    print(f"bootstrap: 95% percentile intervals, {B:,} draws resampling replies, seed {SEED}")

    print("\n== Judges vs each labeller (all 120 replies) ==")
    print(f"human DONT_KNOW rate: labeller 1 {pct(hs[0].mean())} | labeller 2 {pct(hs[1].mean())}")
    w = 32
    print(f"{'judge / metric':<20}{'labeller 1':<{w}}{'labeller 2':<{w}}")
    for name, j in judges:
        print(name)
        row = lambda f: "".join(f"{f(h):<{w}}" for h in hs)
        print(f"{'  kappa [95% CI]':<20}" + row(lambda h: fk(kappa(h, j), boot_ci(h, j, idx))))
        print(f"{'  agreement':<20}" + row(lambda h: pct((h == j).mean())))
        print(f"{'  precision':<20}" + row(lambda h: pct(h[j].mean() if j.any() else np.nan)))
        print(f"{'  recall':<20}" + row(lambda h: pct(j[h].mean() if h.any() else np.nan)))
    print(f"judge DONT_KNOW rate: both-judges {pct(both.mean())} | qwen32b {pct(q.mean())} | gemma27b {pct(g.mean())}")

    print("\n== Labeller 1 vs labeller 2 (all 120 replies) ==")
    print(f"inter-human kappa [95% CI]: {fk(kappa(hs[0], hs[1]), boot_ci(hs[0], hs[1], idx))}")
    print(f"inter-human agreement (DONT_KNOW vs not): {pct((hs[0] == hs[1]).mean())}")
    print(f"exact agreement on all three labels: {pct((m.h1 == m.h2).mean())}")

    agree = hs[0] == hs[1]
    print("\n== Replies where the labellers agree (DESCRIPTIVE ONLY: selected on human agreement) ==")
    print(f"count: {int(agree.sum())} of {len(m)}")
    if agree.sum():
        ha, ba = hs[0][agree], both[agree]
        print(f"both-judges rule kappa [95% CI]: {fk(kappa(ha, ba), boot_ci(ha, ba, boot_idx(len(ha))))}")

    print("\n== Disagreements between labellers (DONT_KNOW vs not) ==")
    dis = m.loc[~agree, "cell"].value_counts()
    print(f"count: {int((~agree).sum())}")
    for c in sorted(m.cell.unique()):
        print(f"  {c:<28}{int(dis.get(c, 0)):>4}")

    print("\n== CLARIFY and note counts ==")
    print(f"{'':<28}{'labeller 1':>12}{'labeller 2':>12}")
    print(f"{'CLARIFY labels':<28}{int((m.h1 == 'CLARIFY').sum()):>12}{int((m.h2 == 'CLARIFY').sum()):>12}")
    print(f"{'replies with a note':<28}{int((m.n1.str.strip() != '').sum()):>12}{int((m.n2.str.strip() != '').sum()):>12}")

    print("\n== Per cell DONT_KNOW rate, human vs both-judges rule (DESCRIPTIVE ONLY) ==")
    print(f"{'cell':<28}{'n':>4}{'labeller 1':>12}{'labeller 2':>12}{'both-judges':>13}")
    for c in sorted(m.cell.unique()):
        s = (m.cell == c).to_numpy()
        print(f"{c:<28}{int(s.sum()):>4}{pct(hs[0][s].mean()):>12}{pct(hs[1][s].mean()):>12}{pct(both[s].mean()):>13}")


if __name__ == "__main__":
    main()
