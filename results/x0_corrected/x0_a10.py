"""X0 A10 with corrected Gemma labels: same choices as results/x0/A10_choices.csv, every repo number parsed from the shadow's regenerated
output (REPO), so nothing is typed. Columns *_as_printed give the old-label number from results/x0/A10_choices.csv for comparison."""
from x0_common import *

log = Log("x0_a10_log.txt")
N = r"([+-]?\d+\.\d+)"


def num(path, pat, d=2, ci_=True, sign=True):
    m = re.search(pat, (REPO / path).read_text(encoding="utf-8"), re.S)
    assert m, (path, pat)
    f = lambda x: (f"{float(x):+.{d}f}" if sign else f"{float(x):.{d}f}")
    return f"{f(m[1])} [{f(m[2])}, {f(m[3])}]" if ci_ else f(m[1])


I = rf"\[{N},\s*{N}\]"
S, R4, R3_ = "results/hedge_measure_v2/summary.txt", "paper_2.0/analysis/round4_checks.txt", "results/round3/"
x = lambda f: pd.read_csv(OUT / f)
A4, A6, A5, A1 = x("A4_margin_by_judge.csv").set_index("measure"), x("A6_summary.csv"), x("A5_unknown_gap.csv"), x("A1_intervals.csv")
P = "Two judges (primary; adopted post hoc)"
m = lambda k: (lambda r: f"{r['margin']:+.2f} [{r['margin lo']:+.2f}, {r['margin hi']:+.2f}]")(A4.loc[k])
w = A6[A6.measure.str.startswith("regex, 20")].set_index("statistic")
wr = lambda k: (lambda r: f"{r.estimate:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] ({r.relations})")(w.loc[f"mean margin over relations with every cell >= {k}"])
a5 = A5[A5.measure == P].set_index("correction")
prim = num(S, rf"primary \(both DONT_KNOW\).*?pop-know {N} \[{N},{N}\]")
kap = A1[A1.quantity.str.startswith("kappa")].iloc[0]
GC = pd.read_csv(REPO / "paper_2.0/analysis/guessability.csv").set_index("tag").loc["judge_noguess"]
jl = re.search(rf"preference {N}, within-cell bootstrap \[\s*{N}\s+{N}\s*\]; n", (REPO / R4).read_text())
NEW = {
    "Primary measure: both judges": (prim, num("results/hedge_judge/summary.txt", rf"pop - know {N} {I}")),
    "Reply abstains only if both": (prim, "; ".join(m(k) for k in ("Qwen2.5-32B alone", "Gemma-2-27B alone", "either judge"))),
    "DONT_KNOW only": (prim, num(S, rf"secondary \(\+CLARIFY\).*?pop-know {N} \[{N},{N}\]")),
    "Agreement bar": (f"{kap.estimate:.2f} ({'passes' if kap.estimate >= 0.6 else 'fails'})", f"{kap.lo:.2f} ({'passes' if kap.lo >= 0.6 else 'fails'})"),
    "Third judge": (num(R3_ + "e5_summary.txt", rf"diff {N} {I}"), prim[:5]),
    "Strict declines": (num(R3_ + "e7_summary.txt", rf"T7c DECLINE.*?diff {N} {I}"), prim[:5]),
    "Regex on the judged": (num(S, rf"regex on new replies.*?pop-know {N} \[{N},{N}\]"), prim[:5]),
    "Unknown facts regraded": (num(R3_ + "e7_summary.txt", rf"T7a two-judge.*?diff {N} {I}"), prim[:5]),
    "Popular-unknown questions with": (num("paper_2.0/analysis/round3_label_checks.txt", rf"worst case\): {N} {I}"), prim[:5]),
    "Guessable facts removed": (f"{GC['diff']:+.2f} [{GC.diff_lo:+.2f}, {GC.diff_hi:+.2f}]", num(R3_ + "e11_summary.txt", rf"primary \(drop.*?diff {N} {I}")),
    "All three label corrections": (f"{float(jl[1]):+.2f} [{float(jl[2]):+.2f}, {float(jl[3]):+.2f}]", prim),
    "Knowledge relabelled": (num(R3_ + "e1_summary.txt", rf"T1 two-judge.*?diff {N} {I}") + " (27 obscure known)", num("paper_2.0/analysis/round3_label_checks.txt", rf"both labellings agree: {N} {I}")),
    "Intervals resample questions": (prim.split(" ", 1)[1], (lambda a, b: f"[{float(a[1]):+.2f}, {float(a[2]):+.2f}]; [{b.lo:+.2f}, {b.hi:+.2f}]")(re.search(rf"relation-clustered bootstrap \(5,000\) {I}", (REPO / R4).read_text()), A1[(A1.quantity == "margin") & (A1.method == "relations, then questions")].iloc[0])),
    "Namesake covariate": (num("paper_2.0/analysis/ambiguity_check.txt", rf"LPM obscure {N} \[{N},{N}\]  unknown \S+ \S+  log_hom_z"), num("paper_2.0/analysis/ambiguity_check.txt", rf"LPM obscure {N} \[{N},{N}\]  unknown")),
    "Within-relation mean": (num("paper_2.0/analysis/within_relation_check.txt", rf"pref: {N} {I}") + " (composer, country, sport)", wr(1) + "; " + wr(5)),
    "Margin outside": (num("paper_2.0/analysis/mere_checks.txt", rf"outside: preference {N} {I}") + " (two judges, 7 facts)", (lambda r: f"{r.estimate:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]")(w.loc["margin, outside country, capital, sport"])),
    "Gold aliases inside": (num("paper_2.0/analysis/echo_check.txt", rf"obscure HighlyKnown.*?alias excl. echoes {N}", ci_=False, sign=False) + " (obscure known)", num("paper_2.0/analysis/echo_check.txt", rf"obscure HighlyKnown.*?alias incl. name echoes {N}", ci_=False, sign=False)),
    "Unknown-fact popularity gap": tuple(f"{a5.loc[k, 'gap']:+.2f} [{a5.loc[k, 'lo']:+.2f}, {a5.loc[k, 'hi']:+.2f}]" for k in ("no correction", "Three label corrections (post hoc)")),
    "Fresh replies": (num(R3_ + "e13_summary.txt", rf"diff {N} {I}"), prim[:5]),
    "New questions": (num(R3_ + "e14_summary.txt", rf"diff {N} {I}"), (lambda a: f"[{float(a[1]):+.2f}, {float(a[2]):+.2f}]")(re.search(rf"preference {I}", (REPO / (R3_ + "e14_summary.txt")).read_text()))),
    "All X0 analyses": ("results/x0_corrected", "-"),
}
O = pd.read_csv(HERE.parent / "x0/A10_choices.csv")
assert len(O) == len(NEW)
O = O.rename(columns={"number_used": "number_used_as_printed", "number_under_alternative": "number_under_alternative_as_printed"})
for i, c in enumerate(O.choice):
    k = [n for n in NEW if c.startswith(n)]
    assert len(k) == 1, c
    O.loc[i, "number_used"], O.loc[i, "number_under_alternative"] = NEW[k[0]]
O.loc[O.choice.str.startswith("Primary measure"), "status"] += "; Gemma labels corrected 24 Nov 2026 (post hoc; same rule)"
new = pd.DataFrame([dict(choice="Gemma-2-27B judge run with soft-capping (eager attention; vLLM equal)", status="Post hoc correction of a software fault found by the X2 pilot; the two-judge rule is unchanged", plan="x2_amendment_1.md", number_used=prim, alternative="Labels as printed (sdpa attention, no soft-capping)",
                          number_under_alternative=O.number_used_as_printed[0], number_used_as_printed=O.number_used_as_printed[0], number_under_alternative_as_printed="-", script_line="scripts/rejudge_run.py", source="results/rejudge_gemma/")])
O = pd.concat([new, O])
save(O, "A10_choices.csv")
for r in O.itertuples():
    log(f"{r.choice[:70]:<70} | printed {r.number_used_as_printed} | corrected {r.number_used} | alt {r.number_under_alternative}")
log.save()
