"""X0 A10: table of analysis choices. Every number quoted from the repo is asserted to be in its source file; every script line is
found by searching for a marker, so it stays traceable. Also checks the pre-registration hashes and marks each plan."""
import hashlib
from x0_common import *

log = Log("x0_a10_log.txt")


def line(path, marker):
    for i, l in enumerate((REPO / path).read_text(encoding="utf-8").splitlines(), 1):
        if marker in l:
            return f"{path}:{i}"
    raise AssertionError(f"marker {marker!r} not in {path}")


def src(path, *needles):
    t = (REPO / path).read_text(encoding="utf-8")
    for n in needles:
        assert n in t, f"{n!r} not in {path}"
    return path


x = lambda f: pd.read_csv(OUT / f)
A4, A6, A5, A1 = x("A4_margin_by_judge.csv").set_index("measure"), x("A6_summary.csv"), x("A5_unknown_gap.csv"), x("A1_intervals.csv")
g2 = lambda v: f"{v:+.2f}"
m = lambda r: f"{r['margin']:+.2f} [{r['margin lo']:+.2f}, {r['margin hi']:+.2f}]"
w = A6[A6.measure.str.startswith("regex, 20")].set_index("statistic")
wr = lambda k: (lambda r: f"{r.estimate:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] ({r.relations})")(w.loc[f"mean margin over relations with every cell >= {k}"])
a5 = A5[A5.measure.str.startswith("both judges (primary)")].set_index("correction")
gp = lambda k: f"{a5.loc[k, 'gap']:+.2f} [{a5.loc[k, 'lo']:+.2f}, {a5.loc[k, 'hi']:+.2f}]"
S, R4 = "results/hedge_measure_v2/summary.txt", "paper_2.0/analysis/round4_checks.txt"
HM_, RA = "scripts/hedge_measure_v2_analysis.py", "scripts/round3_analysis.py"
rows = [
    # choice, status, plan, number used, alternative, number under alternative, script line, source
    ("Primary measure: both judges say DONT_KNOW, three 96-token replies", "Designed and fixed in writing (7 Oct) before its run, but after the one-judge corrected +0.13 was seen", "hedge_measure_v2_criterion.md (81acfc9f)",
     "+0.30 [+0.20, +0.40]", "First pre-specified judge measure: regex rates on 24-token replies, corrected by Qwen2.5-32B", "+0.13 [+0.03, +0.24]", line("scripts/hedge_judge_analysis.py", "rate[a, k] = "),
     src(S, "+0.300 [+0.204,+0.398]") + "; " + src("results/hedge_judge/summary.txt", "+0.132 [+0.027, +0.239]")),
    ("Reply abstains only if both judges agree", "Pre-specified", "hedge_measure_v2_criterion.md (81acfc9f)", "+0.30 [+0.20, +0.40]", "Qwen alone; Gemma alone; either (X0, post hoc)",
     "; ".join(m(A4.loc[k]) for k in ("Qwen2.5-32B alone", "Gemma-2-27B alone", "either judge")), line(HM_, "prim=("), "results/x0/A4_margin_by_judge.csv"),
    ("DONT_KNOW only; clarification requests not counted", "Pre-specified (secondary measure)", "hedge_measure_v2_criterion.md (81acfc9f)", "+0.30 [+0.20, +0.40]", "DONT_KNOW or CLARIFY", "+0.31 [+0.21, +0.41]", line(HM_, "sec=("), src(S, "+0.311 [+0.209,+0.414]")),
    ("Agreement bar kappa >= 0.60 read on the point estimate", "Pre-specified", "hedge_measure_v2_criterion.md (81acfc9f)", "0.62 (passes)", "Bar on the lower end of the interval", "0.57 (fails)", line(HM_, "ok_k = "), src("paper_2.0/analysis/lockin_checks.txt", "0.619 [0.568, 0.668]")),
    ("Third judge, majority of three", "Pre-specified (robustness)", "e5_third_judge.md (f35aa4d9)", "+0.44 [+0.32, +0.55]", "Two judges", "+0.30", line(RA, "majority = >= 2 of 3"), src("results/round3/e5_summary.txt", "+0.437 [+0.322, +0.550]")),
    ("Strict declines (70B judge, no candidate named)", "Pre-specified", "e7_label_and_decline_audit.md (1498be91)", "+0.43 [+0.31, +0.54]", "Two judges", "+0.30", line(RA, "def e7"), src("results/round3/e7_summary.txt", "+0.429 [+0.312, +0.542]")),
    ("Regex on the judged 96-token replies (margin)", "Post hoc (its recall against the judges was pre-specified)", "-", "+0.43 [+0.31, +0.55]", "Two judges", "+0.30", line(HM_, "regex on new replies"), src(S, "+0.429 [+0.306,+0.547]")),
    ("Unknown facts regraded by meaning", "Pre-specified", "e7_label_and_decline_audit.md (1498be91)", "+0.23 [+0.12, +0.35]", "Alias grading", "+0.30", line(RA, "def e7"), src("results/round3/e7_summary.txt", "+0.234 [+0.119, +0.348]")),
    ("Popular-unknown questions with the gold answer in a free reply dropped", "Post hoc", "-", "+0.19 [+0.07, +0.31]", "Kept", "+0.30", line("paper_2.0/analysis/round3_label_checks.py", "worst case"), src("paper_2.0/analysis/round3_label_checks.txt", "+0.189 [+0.068, +0.308]")),
    ("Guessable facts removed", "Post hoc", "-", "+0.30 [+0.19, +0.41]", "Pre-specified pseudo-name control (E11, f716d396)", "+0.36 [+0.25, +0.48]", line("paper_2.0/analysis/guessability.py", "def guess"),
     src("paper_2.0/analysis/guessability.csv", "0.2997076023391813") + "; " + src("results/round3/e11_summary.txt", "+0.364 [+0.250, +0.479]")),
    ("All three label corrections jointly", "Joint rule post hoc; the regrade inside it is pre-specified (E7)", "-", "+0.14 [-0.01, +0.28]", "No corrections", "+0.30 [+0.20, +0.40]", line("paper_2.0/analysis/round4_checks.py", "# (3) all label corrections"), src(R4, "+0.140", "[-0.005  0.283]")),
    ("Knowledge relabelled with a paraphrased prompt", "Pre-specified", "e1_independent_labels.md (af50f53f)", "+0.32 [+0.16, +0.49] (27 obscure known)", "Only where both labelings agree (post hoc)", "+0.31 [+0.15, +0.48]", line(RA, "def e1"),
     src("results/round3/e1_summary.txt", "+0.322 [+0.157, +0.490]") + "; " + src("paper_2.0/analysis/round3_label_checks.txt", "+0.314 [+0.147, +0.480]")),
    ("Intervals resample questions within cells", "Question bootstrap pre-specified; relation clustering post hoc", "hedge_measure_v2_criterion.md (81acfc9f)", "[+0.20, +0.40]", "Relations resampled (paper, 5,000 draws); relations then questions (X0, 10,000)",
     "[-0.10, +0.43]; " + (lambda r: f"[{r.lo:+.2f}, {r.hi:+.2f}]")(A1[(A1.quantity == "margin") & (A1.method == "relations, then questions")].iloc[0]), line("paper_2.0/analysis/round4_checks.py", "# (2) relation-clustered"), src(R4, "[-0.099, +0.432]")),
    ("Namesake covariate (coefficient of the obscure arm, not a margin)", "Post hoc", "-", "+0.34 [+0.27, +0.41]", "Without it, same 384 questions", "+0.36 [+0.29, +0.43]", line("paper_2.0/analysis/ambiguity_check.py", '"log_hom_z"], ["obscure"'), src("paper_2.0/analysis/ambiguity_check.txt", "+0.340 [+0.267,+0.412]", "+0.355 [+0.286,+0.426]")),
    ("Within-relation mean keeps relations with at least 3 questions in every cell", "Post hoc", "-", "-0.04 [-0.20, +0.11] (composer, country, sport)", "Minimum 1 question; minimum 5 questions (X0)", wr(1) + "; " + wr(5), line("paper_2.0/analysis/within_relation_check.py", ">= 3).all()]"), src("paper_2.0/analysis/within_relation_check.txt", "pref: -0.040 [-0.197, +0.113]")),
    ("Margin outside country, capital and sport", "Post hoc", "-", "+0.00 [-0.18, +0.31] (two judges, 7 facts)", "Regex on the full pool, same 7 facts", (lambda r: f"{r.estimate:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]")(w.loc["margin, outside country, capital, sport"]), line("paper_2.0/analysis/mere_checks.py", "TOP = "), src("paper_2.0/analysis/mere_checks.txt", "outside: preference +0.004 [-0.181, +0.307]")),
    ("Gold aliases inside the subject's name not counted in free replies", "Post hoc; replaced an earlier count that included them", "-", "0.25 (obscure known)", "Name echoes counted", "0.53", line("paper_2.0/analysis/echo_check.py", "alias_no_echo=any"), src("paper_2.0/analysis/echo_check.txt", "alias excl. echoes 0.25", "alias incl. name echoes 0.53")),
    ("Unknown-fact popularity gap used as the headline", "Post hoc", "-", gp("no correction"), "After the three label corrections", gp("all three"), line("paper_2.0/analysis/round4_checks.py", "# (5) the label-robust"), src(R4, "+0.343 [+0.253, +0.430]", "+0.179 [+0.043, +0.309]")),
    ("Fresh replies, same questions", "Pre-specified", "e13_e14_replications.md (fbbc8c25)", "+0.29 [+0.20, +0.38]", "Original replies", "+0.30", line(RA, "def e13"), src("results/round3/e13_summary.txt", "+0.290 [+0.197, +0.381]")),
    ("New questions, 10th-30th against 70th-90th percentile", "Pre-specified", "e13_e14_replications.md (fbbc8c25)", "+0.10 [+0.01, +0.20]", "Relation-clustered interval", "[-0.12, +0.27]", line(RA, "def e14"), src("results/round3/e14_summary.txt", "+0.103 [+0.010, +0.197]", "[-0.118, +0.274]")),
    ("All X0 analyses (A1 to A9)", "Post hoc: run on existing data after the results were known", "-", "results/x0", "-", "-", "results/x0/*.py", "results/x0/"),
]
T = pd.DataFrame(rows, columns=["choice", "status", "plan", "number_used", "alternative", "number_under_alternative", "script_line", "source"])
save(T, "A10_choices.csv")
log(f"A10: {len(T)} choices; every quoted repo number found in its source file")

rec = dict(l.split()[:2][::-1] for l in (REPO / "paper_2.0/PREREG/HASHES_round3.txt").read_text().splitlines() if l.strip())
what = {"cross_model_agreement.md": "Post hoc: plan written after the point estimates had been seen (says so itself)", "e17_nla_qwen.md": "Exploratory; result lost, nothing reported",
        "e4_amendment_third_judge.md": "Pre-specified amendment (before judging); secondary scorer only"}
rows = []
for p in sorted((REPO / "paper_2.0/PREREG").glob("*.md")) + [REPO / "hedge_judge_criterion.md", REPO / "hedge_measure_v2_criterion.md"]:
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    stored = rec.get(p.name) or {"hedge_judge_criterion.md": "ae0bb807", "hedge_measure_v2_criterion.md": "81acfc9f"}.get(p.name, "")
    rows.append(dict(plan=p.relative_to(REPO).as_posix(), sha256_prefix=h, recorded=stored or "none recorded", matches=(h == stored) if stored else "", status=what.get(p.name, "Pre-specified"),
                     title=p.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")))
P = pd.DataFrame(rows); save(P, "A10_prereg_status.csv")
log(P[["plan", "sha256_prefix", "recorded", "matches", "status"]].to_string(index=False))
log("The hashes are self-recorded. Git shows criterion notes and results committed on the same day, so git does not prove the order.")
log.save()
