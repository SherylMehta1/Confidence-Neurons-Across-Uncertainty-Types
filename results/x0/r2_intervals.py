"""R2: index of every bootstrap interval in results/x0 and in the scripts its numbers come from, with draws and seed read from the code.
Nothing is resampled here: estimates and bounds are copied from saved outputs, then compared with them again at the end."""
import ast
from x0_common import HERE, OUT, REPO, pd, re, save

rows = []


def const(path, pattern, cast=int, after=None):
    """(value, 'file:line') of the first source line matching pattern; None if the code does not state it."""
    on = after is None
    for i, l in enumerate((REPO / path).read_text(encoding="utf-8").splitlines(), 1):
        on = on or after in l
        m = re.search(pattern, l)
        if m and on:
            return cast(m[1].replace("_", "")), f"{path}:{i}"
    return None, f"{path}: not stated"


X0 = "results/x0/"
for n in ast.parse((HERE / "x0_common.py").read_text()).body:
    if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "(SEED, B)":
        SEED, B = ast.literal_eval(n.value); LINE = f"{X0}x0_common.py:{n.lineno}"


def add(block, name, measure, method, draws, seed, est, lo, hi, output, code, used="", note=""):
    if pd.isna(lo) and pd.isna(hi):
        return
    rows.append(dict(block=block, name=name, measure=measure, method=method, draws=draws, seed=seed, draws_used=used, estimate=est, low=lo, high=hi, output_file=output, code=code, note=note))


# ---------- block 1: intervals computed by X0 itself (constants B and SEED in x0_common.py) ----------
Q, RO, RQ = "questions within cell", "whole relations", "whole relations, then questions"
NREC = "draws dropped for an empty cell are not recorded"
PRIMARY = "Two judges (primary; adopted post hoc)"


def x0(f, name, measure, est, lo, hi, method, used="", note="", script="x0_analyses.py"):
    D = pd.read_csv(OUT / f, float_precision="round_trip")
    for r in D.to_dict("records"):
        nm, ms, mt, us = (v(r) if callable(v) else v for v in (name, measure, method, used))
        add("X0", nm, ms, mt, B, SEED, r[est], r[lo], r[hi], f"{X0}{f}", f"{LINE}; {X0}{script}", us, note(r) if callable(note) else note)


x0("A1_intervals.csv", lambda r: f"A1 {r['quantity']}", lambda r: "entropy, full pool" if "entropy" in r["quantity"] else ("two judges, kappa" if "kappa" in r["quantity"] else PRIMARY), "estimate", "lo", "hi",
   lambda r: {"relations only": RO, "relations, then questions": RQ, "questions": "questions (their 3 replies kept together)"}.get(r["method"], Q), used=lambda r: r["resamples_used"])
x0("A3_reply_types.csv", lambda r: f"A3 share of replies with a gold alias, {r['cell']} ({r['subset']}, {r['alias_rule']})", "alias match", "any_gold", "any_gold_lo", "any_gold_hi", Q)
for e in ("popularity effect", "knowledge effect", "margin"):
    x0("A4_margin_by_judge.csv", f"A4 {e}", lambda r: r["measure"], e, f"{e} lo", f"{e} hi", Q)
x0("A4_margin_by_judge.csv", "A4 margin", lambda r: r["measure"], "margin", "margin_relation_lo", "margin_relation_hi", RO, note=NREC)
for a, b in (("qwen", "gemma"), ("qwen", "llama70"), ("gemma", "llama70")):
    k = f"kappa_{a}_{b}"
    x0("A4_judges_by_cell.csv", lambda r: f"A4 kappa {a}-{b}, {r['cell']}", "judge agreement", k, k + "_lo", k + "_hi", "questions (their 3 replies kept together)", note="resamples with undefined kappa are skipped, count not recorded")
x0("A4_margin_steps.csv", lambda r: f"A4 margin, step {r['step']}", lambda r: r["step"][3:], "margin", "lo", "hi", lambda r: "judged replies within regex stratum" if r["step"][0] == "2" else Q,
   note=lambda r: "X0's own 10,000-draw recomputation; the paper's One judge, corrected row uses 5,000 (block 2)" if r["step"][0] == "2" else "")
for lo, hi, mt, nt in (("lo", "hi", Q, ""), ("rel_lo", "rel_hi", RO, NREC), ("rel_q_lo", "rel_q_hi", RQ, NREC)):
    x0("A5_unknown_gap.csv", lambda r: f"A5 unknown-fact gap, {r['correction']}", lambda r: r["measure"], "gap", lo, hi, mt, note=nt)
x0("A5_unknown_gap.csv", lambda r: f"A5 margin, {r['correction']}", lambda r: r["measure"], "margin", "margin_lo", "margin_hi", Q)
x0("A5_leave_one_relation_out.csv", lambda r: f"A5 unknown-fact gap without {r['left_out']}, {r['correction']}", lambda r: r["measure"], "gap", "lo", "hi", Q)
x0("A5_within_relation_gap.csv", lambda r: f"A5 within-relation unknown-fact gap, equal weight, min {r['min_n']}", lambda r: r["measure"], "equal_weight", "eq_lo", "eq_hi", "questions within relation and cell")
x0("A5_within_relation_gap.csv", lambda r: f"A5 within-relation unknown-fact gap, precision weight, min {r['min_n']}", lambda r: r["measure"], "precision_weight", "pw_lo", "pw_hi", "questions within relation and cell")
x0("A6_within_relation.csv", lambda r: f"A6 margin in {r['relation']}", lambda r: r["measure"], "margin", "lo", "hi", "questions within relation and cell")
x0("A6_summary.csv", lambda r: f"A6 {r['statistic']}", lambda r: r["measure"], "estimate", "lo", "hi", lambda r: "questions within relation and cell" if r["statistic"].startswith("mean") else Q)
for nm, es, lo, hi in (("log views", "b_views", "views_lo", "views_hi"), ("entropy", "b_entropy", "entropy_lo", "entropy_hi"), ("|views| - |entropy|", "abs_views_minus_abs_entropy", "diff_lo", "diff_hi")):
    x0("A7_popularity_vs_entropy.csv", lambda r: f"A7 {nm}, {r['sample']}, {r['model']}", lambda r: r["outcome"], es, lo, hi, "questions, whole sample (regression)", script="x0_pool.py")
x0("A7_popularity_vs_entropy.csv", lambda r: f"A7 |views| - |entropy|, {r['sample']}, {r['model']}", lambda r: r["outcome"], "abs_views_minus_abs_entropy", "diff_rel_lo", "diff_rel_hi", "whole relations (regression)", script="x0_pool.py")
x0("A8_thresholds.csv", lambda r: f"A8 margin at {r['cutoff']}, greedy {r['greedy_condition']}", "regex, 20 x 24-token replies, full pool", "regex24_margin", "regex_lo", "regex_hi", Q, script="x0_a8.py")
x0("A8_thresholds.csv", lambda r: f"A8 margin at {r['cutoff']}, greedy {r['greedy_condition']}", PRIMARY, "two_judge_margin", "two_lo", "two_hi", Q, script="x0_a8.py")
x0("within_relation_by_min_cell.csv", lambda r: f"R3 within-relation margin, min {r['min_questions_per_cell']} per cell", "regex, 20 x 24-token replies, full pool", "mean_margin", "lo", "hi", "questions within relation and cell", script="r3_within_relation.py")
x0("within_relation_by_min_cell.csv", lambda r: f"R3 within-relation margin, min {r['min_questions_per_cell']} per cell (the paper's interval, copied)", "regex, 20 x 24-token replies, full pool", "mean_margin", "paper_lo", "paper_hi", "questions within relation and cell", script="r3_within_relation.py")
for r in rows:  # that last family is the repo's interval, not an X0 bootstrap
    if "copied" in r["name"]:
        r["draws"], r["code"] = const("paper_2.0/analysis/within_relation_check.py", r"for _ in range\((\d+)\)")
        r["block"] = "repo script, quoted"

# ---------- block 2: intervals printed by the repo scripts X0 re-runs or quotes ----------
IV = re.compile(r"\[\s*([+-]?\d+\.\d+)\s*,?\s*([+-]?\d+\.\d+)\s*\]")
NUM = re.compile(r"(?<![\w.(])[+-]?\d+\.\d+(?![\w.])")
AN, RA = "paper_2.0/analysis/", "scripts/round3_analysis.py"
# output file -> (block label, seed pattern source, list of (regex on the line/label, method, script, draws pattern))
D10 = (RA, r"def effects\(.*B=([\d_]+)")
SRC = {
    "results/hedge_measure_v2/summary.txt": [(".", Q, "scripts/hedge_measure_v2_analysis.py", r"^B = (\d+)")],
    "results/hedge_judge/summary.txt": [(".", "judged replies within regex stratum", "scripts/hedge_judge_analysis.py", r"^B = (\d+)")],
    AN + "round4_checks.txt": [("relation-clustered", RO, AN + "round4_checks.py", r"for _ in range\((\d+)\)"), ("popularity gap|after all", Q, AN + "round4_checks.py", r"def gap_unknown\(df, B=([\d_]+)"), (".", Q, AN + "round4_checks.py", r"def within_boot\(df, B=([\d_]+)")],
    AN + "within_relation_check.txt": [(".", "questions within relation and cell", AN + "within_relation_check.py", r"for _ in range\((\d+)\)")],
    AN + "mere_checks.txt": [("inside known gap", Q, AN + "mere_checks.py", r"for _ in range\(([\d_]+)\)\]"), (".", Q, AN + "mere_checks.py", r"def pref\(df, B=([\d_]+)")],
    AN + "lockin_checks.txt": [("kappa", "questions (their replies kept together)", AN + "lockin_checks.py", r"for _ in range\((\d+)\)"), (".", Q, AN + "lockin_checks.py", r"def boot\(v, B=([\d_]+)")],
    AN + "round2_checks.txt": [(".", "questions, whole sample (linear probability model)", AN + "round2_checks.py", r"def lpm\(.*B=(\d+)")],
    AN + "round3_label_checks.txt": [(".", Q, AN + "round3_label_checks.py", r"def effects\(.*B=([\d_]+)")],
    AN + "ambiguity_check.txt": [(".", "questions, whole sample (linear probability model)", AN + "ambiguity_check.py", r"def lpm\(.*B=(\d+)")],
    "results/round3/e1_summary.txt": [(".", Q, *D10)], "results/round3/e5_summary.txt": [(".", Q, *D10)], "results/round3/e7_summary.txt": [(".", Q, *D10)], "results/round3/e9_summary.txt": [(".", Q, *D10)],
    "results/round3/e3_summary.txt": [(".", Q, RA, r"e, n = effects\(M, y, B=(\d+)\)")],
    "results/round3/e6_summary.txt": [(".", "questions, whole sample (linear probability model)", RA, r"def lpm\(.*B=(\d+)")],
    "results/round3/e11_summary.txt": [("relation-clustered", RO, RA, r"for _ in range\((\d+)\)"), (".", Q, *D10)],
    "results/round3/e13_summary.txt": [(".", Q, *D10)],
    "results/round3/e14_summary.txt": [("relation-clustered", RO, RA, r"def _clustered\(df, B=(\d+)"), (".", Q, *D10)],
    "results/round3/e16_summary.txt": [(".", "steering questions", RA, r"rng.integers\(0, len\(piv\), \(([\d_]+),")],
}
for out, rules in SRC.items():
    script = rules[0][2]
    seed, sloc = const(script, r"default_rng\((\d+)\)")
    if seed is None:
        seed, sloc = const(script, r"seed=(\d+)")
    for i, l in enumerate((REPO / out).read_text(encoding="utf-8").splitlines(), 1):
        prev = 0
        for m in IV.finditer(l):
            label = l[prev:m.start()]
            if re.search(r"rel depth|\brange\s*$", label):  # the lens window and a min-max range are not bootstrap intervals
                prev = m.end(); continue
            nums = NUM.findall(IV.sub(" ", l[:m.start()]))
            own = NUM.findall(label)
            clustered = "clustered" in label or "bootstrap" in label
            est = float(own[-1]) if own else (float(nums[-1]) if nums and clustered else None)
            key = label if re.search("clustered|bootstrap", label) else label + (" " + l.split(";")[0] if out.endswith("lockin_checks.txt") and "kappa" in l else "")
            method, sc, pat = next((a, b, c) for rx, a, b, c in rules if re.search(rx, key))
            draws, loc = const(sc, pat, after="def e16" if "e16" in out else None)
            note = "estimate is not printed on this line" if est is None else ""
            if "within_relation" in out:
                note = "seed 0 generates one child seed per cell and draw"
            if clustered and "clustered" in label:
                note = (note + "; " if note else "") + NREC
            add("repo script", f"{out}:{i} " + re.sub(r"\s+", " ", label).strip(" ;|")[-90:], "as labelled", method, draws, seed, est, float(m[1]), float(m[2]), f"{out}:{i}", f"{loc}; seed {sloc}", note=note)
            prev = m.end()
g = pd.read_csv(REPO / (AN + "guessability.csv"), float_precision="round_trip")
dr, loc = const(AN + "guessability.py", r"B=(\d+)"); sd, sl = const(AN + "guessability.py", r"default_rng\((\d+)\)")
for r in g.to_dict("records"):
    for c, n in (("pop", "popularity effect"), ("know", "knowledge effect"), ("diff", "margin"), ("popU", "unknown-fact gap")):
        add("repo script", f"guessability {r['tag']} {n}", r["tag"], Q, dr, sd, r[c], r[c + "_lo"], r[c + "_hi"], AN + "guessability.csv", f"{loc}; seed {sl}", note="one random stream shared by all rows")
g = pd.read_csv(REPO / "results/factorial_4models/effects.csv", float_precision="round_trip")
dr, loc = const("scripts/qwen_factorial_analysis.py", r"^B = ([\d_]+)"); sd, sl = const("scripts/qwen_factorial_analysis.py", r"default_rng\((\d+)\)")
for r in g.to_dict("records"):
    for c, n in (("pop", "popularity effect"), ("know", "knowledge effect"), ("diff", "margin")):
        add("repo script", f"four families: {r['model']} {r['readout']} {n}", r["readout"], Q, dr, sd, r[c], r[c + "_lo"], r[c + "_hi"], "results/factorial_4models/effects.csv", f"{loc}; seed {sl}", note="entropy rows are Figure 1c" if r["readout"] == "entropy" else "")
dr, loc = const("paper_2.0/figures/fig_dissociation.py", r"def ci\(v, B=(\d+)"); sd, sl = const("paper_2.0/figures/fig_dissociation.py", r"default_rng\((\d+)\)")
rows.append(dict(block="repo script", name="Figure 1a and 1b error bars", measure="two-judge abstention; entropy", method=Q, draws=dr, seed=sd, draws_used="", estimate=None, low=None, high=None, output_file="paper_2.0/figures/fig_dissociation.pdf",
                 code=f"{loc}; seed {sl}", note="drawn in the figure; Table 1's own intervals are the 10,000-draw ones"))
T = pd.DataFrame(rows)
T.to_csv(OUT / "intervals_index.csv", index=False)

# ---------- checks: every row equals its saved output ----------
bad = 0
for r in T[T.low.notna()].to_dict("records"):
    f, _, ln = r["output_file"].partition(".txt:")
    if ln:
        l = (REPO / (f + ".txt")).read_text(encoding="utf-8").splitlines()[int(ln) - 1]
        bad += not any(float(a) == r["low"] and float(b) == r["high"] for a, b in IV.findall(l)) or (pd.notna(r["estimate"]) and not any(float(x) == r["estimate"] for x in NUM.findall(l)))
    else:
        V = set(pd.read_csv(REPO / r["output_file"], float_precision="round_trip").select_dtypes("number").to_numpy().ravel())
        bad += r["low"] not in V or r["high"] not in V or r["estimate"] not in V
print(f"{len(T)} rows; rows differing from the saved output: {bad}; without draws: {T.draws.isna().sum()}; without seed: {T.seed.isna().sum()}")
assert bad == 0
print(T.groupby(["block", "method", "draws", "seed"], dropna=False).size().to_string())

# ---------- every interval macro used in the SRW paper -> its row, draws and seed ----------
tex = (REPO / "paper_2.0/srw/main.tex").read_text(encoding="utf-8")
mac = {m[1]: (m[2], m[3]) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*% *(.*)", (REPO / "paper_2.0/numbers.tex").read_text(encoding="utf-8"))}
WHERE = {"tc": "lockin_checks", "lk": "lockin_checks", "rc": "round4_checks", "ms": "mere_checks", "wr": "within_relation_check", "hm": "hedge_measure_v2", "hj": "hedge_judge", "tj": "e5_summary", "fr": "e13_summary", "nq": "e14_summary",
         "nc": "e11_summary", "se": "e7_summary", "ga": "e16_summary", "pm": "e9_summary"}
used = list(dict.fromkeys(m for m in re.findall(r"\\([A-Za-z]+)", tex) if m in mac and m.endswith("CI")))
S = []
for m in used:
    v = re.findall(r"-?\d+\.\d+", mac[m][0].replace("$-$", "-").replace("$+$", ""))
    c = T[T.output_file.str.contains(WHERE.get(m[:2], "?")) & ~T.output_file.str.contains("results/x0") & T.low.notna()]
    c = c[[f"{abs(a):.2f}" == f"{abs(float(v[0])):.2f}" and f"{abs(b):.2f}" == f"{abs(float(v[1])):.2f}" for a, b in zip(c.low, c.high)]]
    if len(c) == 0:  # rounding edge: macro rounded from the file's three decimals
        c = T[T.output_file.str.contains(WHERE.get(m[:2], "?")) & ~T.output_file.str.contains("results/x0") & T.low.notna()]
        c = c[[abs(a - float(v[0])) <= 0.0051 and abs(b - float(v[1])) <= 0.0051 for a, b in zip(c.low, c.high)]]
    S.append(dict(macro=m, value=mac[m][0].replace("$", ""), source="; ".join(sorted(set(c.output_file.str.replace(r":\d+$", "", regex=True)))), method="; ".join(sorted(set(c.method))), draws="; ".join(sorted({str(int(x)) for x in c.draws.dropna()})), seed="; ".join(sorted({str(int(x)) for x in c.seed.dropna()})), matching_rows=len(c)))
lw = re.findall(r"\[(.*?), (.*?)\]", mac["lwRows"][0]) if "lwRows" in mac else []
S.append(dict(macro="lwRows (lens table)", value=f"{len(lw)} intervals", source="results/round3/e3_summary.txt", method=Q, draws=str(const(*SRC["results/round3/e3_summary.txt"][0][2:])[0]), seed="0", matching_rows=len(lw)))
S = pd.DataFrame(S); S.to_csv(OUT / "intervals_srw_paper.csv", index=False)
print(S.to_string(index=False))
