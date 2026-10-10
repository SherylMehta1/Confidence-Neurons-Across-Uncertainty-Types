"""
Generate paper/numbers.tex for the UniReps extended abstract from the repo's result files (invariant I1).
Every macro carries a comment naming its source file and field. Read-only on the repo (invariant I3).

  python make_numbers.py            # writes numbers.tex next to this script
  python make_numbers.py --check    # print macros, do not write
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE.parent, HERE.parent / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
R = REPO / "results"
OUT = []


def mac(name, value, src):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    OUT.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {src}")


def f2(x, signed=False):
    return (f"{x:+.2f}" if signed else f"{x:.2f}").replace("+", "$+$") if signed else f"{x:.2f}"


def sgn(x, d=2):
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grab(text, pattern, src):
    m = re.search(pattern, text, re.S)
    if not m:
        raise SystemExit(f"pattern not found in {src}: {pattern}")
    return m


# ---------------------------------------------------------------- C1: two-judge hedging measurement (Llama)
src = "results/hedge_measure_v2/summary.txt"
t = (REPO / src).read_text(encoding="utf-8")
blk = t[t.index("== llama"): t.index("== qwen")]
m = grab(blk, r"(\d+) items, (\d+) samples; kappa\(DONT_KNOW\) = ([\d.]+)", src)
mac("hmLlamaItems", m.group(1), src); mac("hmLlamaSamples", f"{int(m.group(2)):,}".replace(",", "{,}"), src); mac("hmLlamaKappa", m.group(3), src)
num = r"([+-][\d.]+)"
pm = grab(blk, r"primary \(both DONT_KNOW\)\s+cells unc/Unkn ([\d.]+) unc/High ([\d.]+) con/Unkn ([\d.]+) con/High ([\d.]+) \| pop " + num + r" \[" + num + "," + num + r"\] know " + num + r" \[" + num + "," + num + r"\] pop-know " + num + r" \[" + num + "," + num + r"\]", src)
g = [float(x) for x in pm.groups()]
for nm, v in zip(["hmLlamaObsUnk", "hmLlamaObsKnown", "hmLlamaPopUnk", "hmLlamaPopKnown"], g[:4]):
    mac(nm, f"{v:.2f}", src + " primary cells")
mac("hmLlamaPopEff", sgn(g[4]), src + " primary pop"); mac("hmLlamaPopEffCI", f"[{sgn(g[5])}, {sgn(g[6])}]", src)
mac("hmLlamaKnowEff", sgn(g[7]), src + " primary know"); mac("hmLlamaKnowEffCI", f"[{sgn(g[8])}, {sgn(g[9])}]", src)
mac("hmLlamaDiff", sgn(g[10]), src + " primary pop-know"); mac("hmLlamaDiffCI", f"[{sgn(g[11])}, {sgn(g[12])}]", src)
rr = grab(blk, r"regex recall on 96-token replies = ([\d.]+)", src); mac("hmLlamaRegexRecall", rr.group(1), src)
rg = grab(blk, r"regex on new replies.*?pop-know " + num, src); mac("hmLlamaRegexDiff", sgn(float(rg.group(1))), src + " regex on 96-token replies pop-know")
sec = grab(blk, r"secondary \(\+CLARIFY\).*?pop-know " + num, src); mac("hmLlamaSecDiff", sgn(float(sec.group(1))), src + " secondary pop-know")
for mdl, nm in (("qwen", "Qwen"), ("gemma", "Gemma"), ("mistral", "Mistral")):
    b = t[t.index(f"== {mdl}"):]
    k = grab(b, r"kappa\(DONT_KNOW\) = ([\d.]+)", src); mac(f"hm{nm}Kappa", k.group(1), src + f" {mdl}")
    c = grab(b, r"primary \(both DONT_KNOW\)\s+cells unc/Unkn ([\d.]+)", src); mac(f"hm{nm}ObsUnk", f"{float(c.group(1)):.2f}", src + f" {mdl} primary obscure-unknown")
    r2 = grab(b, r"regex on new replies\s+cells unc/Unkn ([\d.]+)", src); mac(f"hm{nm}RegexObsUnk", f"{float(r2.group(1)):.2f}", src + f" {mdl} regex obscure-unknown")
    s2 = grab(b, r"secondary \(\+CLARIFY\)\s+cells unc/Unkn ([\d.]+)", src); mac(f"hm{nm}SecObsUnk", f"{float(s2.group(1)):.2f}", src + f" {mdl} secondary obscure-unknown")

# ---------------------------------------------------------------- C1 instrument robustness: regex 24-token and first judge (Llama)
src = "results/factorial_4models/effects.csv"
eff = {(r["model"], r["readout"]): r for r in read_csv(REPO / src)}
e = eff["llama", "hedge_rate"]
mac("faLlamaRegexPop", f"{float(e['pop']):.2f}", src + " llama hedge_rate pop"); mac("faLlamaRegexKnow", f"{float(e['know']):.2f}", src + " llama hedge_rate know")
mac("faLlamaRegexDiff", sgn(float(e["diff"])), src + " llama hedge_rate diff")
srcj = "results/hedge_judge/summary.txt"
tj = (REPO / srcj).read_text(encoding="utf-8")
jm = grab(tj[tj.index("== llama"):], r"pop - know " + num, srcj); mac("hjLlamaDiff", sgn(float(jm.group(1))), srcj + " llama corrected pop - know")
jr = grab(tj, r"== qwen: regex recall ([\d.]+)", srcj); mac("hjQwenRecall", jr.group(1), srcj)
jg = grab(tj, r"== gemma: regex recall ([\d.]+)", srcj); mac("hjGemmaRecall", jg.group(1), srcj)
jmi = grab(tj, r"== mistral: regex recall ([\d.]+)", srcj); mac("hjMistralRecall", jmi.group(1), srcj)

# ---------------------------------------------------------------- C2: entropy follows knowledge, 4 families
for mdl, nm in (("llama", "Llama"), ("qwen", "Qwen"), ("gemma", "Gemma"), ("mistral", "Mistral")):
    e = eff[mdl, "entropy"]
    mac(f"ent{nm}Pop", sgn(float(e["pop"])), src + f" {mdl} entropy pop")
    mac(f"ent{nm}Know", sgn(float(e["know"])), src + f" {mdl} entropy know")
    mac(f"ent{nm}KnowCI", f"[{sgn(float(e['know_lo']))}, {sgn(float(e['know_hi']))}]", src + f" {mdl} entropy know CI")
    mac(f"ent{nm}PopCI", f"[{sgn(float(e['pop_lo']))}, {sgn(float(e['pop_hi']))}]", src + f" {mdl} entropy pop CI")
    h = eff[mdl, "hedge_rate"]
    mac(f"hr{nm}Pool", f"{0:.2f}", "placeholder")  # replaced below
OUT[:] = [l for l in OUT if "placeholder" not in l]
cells = read_csv(REPO / "results/factorial_4models/cells.csv")
for mdl, nm in (("llama", "Llama"), ("qwen", "Qwen"), ("gemma", "Gemma"), ("mistral", "Mistral")):
    c = {r["cell"]: r for r in cells if r["model"] == mdl and r["readout"] == "hedge_rate"}
    mac(f"hr{nm}ObsUnk", f"{float(c['uncertain/Unknown']['mean']):.2f}", f"results/factorial_4models/cells.csv {mdl} hedge_rate uncertain/Unknown")
    n = sum(int(r["n"]) for r in c.values())
    mac(f"fa{nm}N", f"{n:,}".replace(",", "{,}"), f"results/factorial_4models/cells.csv {mdl} sum of cell n (HighlyKnown+Unknown)")
for mdl, nm in (("gemma", "Gemma"),):
    e = eff[mdl, "hedge_logodds"]
    mac(f"lo{nm}Pop", sgn(float(e["pop"])), src + f" {mdl} hedge_logodds pop"); mac(f"lo{nm}Know", sgn(float(e["know"])), src + f" {mdl} hedge_logodds know")
e = eff["llama", "hedge_logodds"]
mac("loLlamaPop", sgn(float(e["pop"])), src + " llama hedge_logodds pop"); mac("loLlamaKnow", sgn(float(e["know"])), src + " llama hedge_logodds know")

# ---------------------------------------------------------------- C3: name-spelling controls (Llama)
srcu = "results/unireps_checks/check1_pairs.csv"
pairs = read_csv(REPO / srcu)
pk = [p for p in pairs if p["contrast"] == "pop_given_known"]
mac("scPopKnownPairs", str(len(pk)), srcu)
mac("scPopKnownGap", sgn(sum(float(p["d_hedge"]) for p in pk) / len(pk)), srcu + " mean d_hedge pop_given_known")
srcs = "results/unireps_checks/summary.txt"
ts = (REPO / srcs).read_text(encoding="utf-8")
m = grab(ts, r"pop_given_known: .*?hedge gap at EQUAL surface form.*?: " + num + r" \[" + num + ", " + num + r"\]", srcs)
mac("scPopKnownGapSurface", sgn(float(m.group(1))), srcs); mac("scPopKnownGapSurfaceCI", f"[{sgn(float(m.group(2)))}, {sgn(float(m.group(3)))}]", srcs)
m = grab(ts, r"pop_given_known: .*?excluding name-inferable obscure twins: n=(\d+), hedge gap " + num, srcs)
mac("scNonInferN", m.group(1), srcs); mac("scNonInferGap", sgn(float(m.group(2))), srcs)
m = grab(ts, r"M1 popularity \+ knowledge.*?logpop_z\s+" + num, srcs); mac("scPoolPop", sgn(float(m.group(1))), srcs + " M1 logpop_z (hedge GLM)")
m = grab(ts, r"M2 \+ surface form of the name.*?logpop_z\s+" + num, srcs); mac("scPoolPopSurface", sgn(float(m.group(1))), srcs + " M2 logpop_z (hedge GLM)")

# ---------------------------------------------------------------- C4/C5: steering (direction reuse v2) and the no-hedge control
for mdl, nm in (("llama", "Llama"), ("gemma9b", "Gemma")):
    srcd = f"results/direction_reuse_v2/{mdl}/summary.txt"
    td = (REPO / srcd).read_text(encoding="utf-8")
    k = grab(td, r"k\* = ([\d.]+) x natural norm", srcd); mac(f"st{nm}Dose", k.group(1).rstrip("0").rstrip(".") if "." in k.group(1) else k.group(1), srcd)
    ly = grab(td, r"l\* = (\d+) of (\d+);[^\n]*n = (\d+)", srcd)
    mac(f"st{nm}Layer", ly.group(1), srcd + " l*"); mac(f"st{nm}Layers", ly.group(2), srcd + " layers"); mac(f"st{nm}N", ly.group(3), srcd + " n")
    hs = dict(re.findall(r"(\w+) ([\d.]+)", grab(td, r"S  hedge at k\*: ([^\n]*)", srcd).group(1)))
    mac(f"st{nm}None", f"{float(hs['none']):.2f}", srcd + " none"); mac(f"st{nm}Fam", f"{float(hs['own_fam']):.2f}", srcd + " own_fam")
    mac(f"st{nm}Know", f"{float(hs['inst_know']):.2f}", srcd + " inst_know"); mac(f"st{nm}Base", f"{float(hs['base_fam']):.2f}", srcd + " base_fam")
    rnd = [float(v) for kk, v in hs.items() if kk.startswith("random")]
    mac(f"st{nm}Rand", f"{sum(rnd) / len(rnd):.2f}", srcd + " mean of random0-4")
    dd = grab(td, r"DD hedge\(fam\)-hedge\(know\) " + num + r" \[" + num + "," + num + r"\]; entropy\(know\)-entropy\(fam\) " + num + r" \[" + num + "," + num + r"\]", srcd)
    v = [float(x) for x in dd.groups()]
    mac(f"st{nm}DH", sgn(v[0]), srcd + " DD hedge"); mac(f"st{nm}DHCI", f"[{sgn(v[1])}, {sgn(v[2])}]", srcd)
    mac(f"st{nm}DE", sgn(v[3]), srcd + " DD entropy"); mac(f"st{nm}DECI", f"[{sgn(v[4])}, {sgn(v[5])}]", srcd)
    es = dict(re.findall(r"(\w+) ([\d.]+)", grab(td, r"E  entropy at k\*: ([^\n]*)", srcd).group(1)))
    mac(f"st{nm}EntNone", f"{float(es['none']):.2f}", srcd + " entropy none"); mac(f"st{nm}EntKnow", f"{float(es['inst_know']):.2f}", srcd + " entropy inst_know")
    mac(f"st{nm}EntFam", f"{float(es['own_fam']):.2f}", srcd + " entropy own_fam")
srcn = "results/direction_nohedge/llama/summary.txt"
tn = (REPO / srcn).read_text(encoding="utf-8")
hs = dict(re.findall(r"(\w+) ([\d.]+)", grab(tn, r"hedge rate: ([^\n]*)", srcn).group(1)))
mac("nhNoHedge", f"{float(hs['d_nohedge_U']):.2f}", srcn + " d_nohedge_U")
m = grab(tn, r"d_nohedge_U - random " + num + r" \[" + num + ", " + num + r"\]", srcn)
mac("nhOverRand", sgn(float(m.group(1))), srcn); mac("nhOverRandCI", f"[{sgn(float(m.group(2)))}, {sgn(float(m.group(3)))}]", srcn)
logn = (REPO / "results/direction_nohedge/llama/log.txt").read_text(encoding="utf-8")
m = grab(logn, r"cos\(d_nohedge_U, d_fam\) = " + num, "results/direction_nohedge/llama/log.txt"); mac("nhCos", f"{float(m.group(1)):.2f}", "results/direction_nohedge/llama/log.txt")

# ---------------------------------------------------------------- C6: base-model familiarity signal (lens study)
srco = "results/jlens_cascade/outcomes.csv"
oc = {r["model"]: r for r in read_csv(REPO / srco)}
for mdl, nm in (("llama3.1-8b", "LlamaBase"), ("gemma-2-9b", "GemmaNineBase"), ("gemma-2-2b", "GemmaTwoBase"), ("olmo-3-1025-7b", "OlmoBase"),
                ("llama3.1-8b-it", "LlamaInst"), ("gemma-2-9b-it", "GemmaNineInst"), ("gemma-2-2b-it", "GemmaTwoInst"), ("qwen2.5-7b-it", "QwenInst")):
    r = oc[mdl]
    mac(f"jl{nm}MidF", sgn(float(r["midF"])), srco + f" {mdl} midF")
    mac(f"jl{nm}MidFCI", f"[{sgn(float(r['midF_lo']))}, {sgn(float(r['midF_hi']))}]", srco + f" {mdl} midF CI")
    mac(f"jl{nm}Outcome", r["outcome"], srco + f" {mdl} outcome")
counts = {k: sum(1 for r in oc.values() if r["outcome"] == k) for k in ("SUPPORTS", "CONTRADICTS", "UNDECIDED")}
mac("jlSupports", str(counts["SUPPORTS"]), srco); mac("jlContradicts", str(counts["CONTRADICTS"]), srco); mac("jlUndecided", str(counts["UNDECIDED"]), srco)
rel = sorted(float(r["rel_F"]) for r in oc.values() if r["rel_F"]) + sorted(float(r["rel_K"]) for r in oc.values() if r["rel_K"])
mac("jlOnsetLo", f"{min(rel):.2f}", srco + " min of rel_F, rel_K"); mac("jlOnsetHi", f"{max(rel):.2f}", srco + " max of rel_F, rel_K")

# ---------------------------------------------------------------- C7: lens timing with a single BOS (Llama familiarity)
srct = "results/singlebos/circuit_familiarity/jlens_summary.txt"
tt = (REPO / srct).read_text(encoding="utf-8")
m = grab(tt, r"log-odds L(\d+), entropy L(\d+)", srct); mac("ltHedgeHalf", m.group(1), srct); mac("ltEntHalf", m.group(2), srct)
mac("ltLead", str(int(m.group(2)) - int(m.group(1))), srct + " difference")

# ---------------------------------------------------------------- C9: circuit (old pipeline numbers, from the repo's own numbers.tex)
oldn = (REPO / "paper/numbers.tex").read_text(encoding="utf-8")
for old, new in (("circuitHedgeCUVThreeFull", "ciRecovery"), ("circuitHedgeCUVThreeFullCI", "ciRecoveryCI")):
    m = re.search(r"\\newcommand\{\\" + old + r"\}\{([^}]*)\}", oldn)
    if m:
        mac(new, m.group(1), f"paper/numbers.tex \\{old}")
fc = [r for r in read_csv(REPO / "paper/figures/fig6_familiarity_v3_full_cells.csv")
      if r["readout"] == "logodds_rec" and r["direction"] == "uncertain_to_control" and r["series"].startswith("full set")][0]
mac("ciRecoveryUC", f"{float(fc['mean']):.2f}", "paper/figures/fig6_familiarity_v3_full_cells.csv uncertain_to_control full set")
mac("ciRecoveryUCCI", f"[{float(fc['ci_lo']):.2f}, {float(fc['ci_hi']):.2f}]", "same, CI")
srcb = "results/circuit_familiarity_v3/behavioral_summary.txt"
tb = (REPO / srcb).read_text(encoding="utf-8")
fr = re.findall(r"flip-to-source-behaviour rate\s+circuit ([\d.]+)", tb)
if len(fr) >= 2:
    mac("ciRemoveFlip", f"{float(fr[0]):.2f}", srcb + " control_to_uncertain circuit flip"); mac("ciInjectFlip", f"{float(fr[1]):.2f}", srcb + " uncertain_to_control circuit flip")

# ---------------------------------------------------------------- pool facts
gr = json.load(open(REPO / "data/familiarity_v3_qwen/gate_report.json"))
mac("poolCandidates", f"{gr['n_candidates']:,}".replace(",", "{,}"), "data/familiarity_v3_qwen/gate_report.json n_candidates")
srcf = "results/factorial_4models/summary.txt"
tf = (REPO / srcf).read_text(encoding="utf-8")
m = grab(tf, r"LLAMA  \(familiarity_v3\): (\d+) scored items", srcf)
mac("poolLlama", f"{int(m.group(1)):,}".replace(",", "{,}"), srcf + " llama scored items")
srck = "hedge_measure_v2_criterion.md"
m = grab((REPO / srck).read_text(encoding="utf-8"), r"Cohen's κ ≥ ([\d.]+)", srck)
mac("kappaBar", m.group(1), srck + " (criterion bar, fixed before the run)")

# ---------------------------------------------------------------- revise round 1 (2026-10-10): new macros
def ci(lo, hi):
    return f"[{sgn(float(lo))}, {sgn(float(hi))}]"


# Table 1 intervals and Table 2 Llama row (hedge_measure_v2 summary, effects.csv, hedge_judge summary)
src = "results/hedge_measure_v2/summary.txt"
t = (REPO / src).read_text(encoding="utf-8")
blk = t[t.index("== llama"): t.index("== qwen")]
for tag, nm in (("secondary \\(\\+CLARIFY\\)", "hmLlamaSec"), ("regex on new replies", "hmLlamaRegex")):
    m = grab(blk, tag + r"\s+cells unc/Unkn ([\d.]+)[^\n]*pop-know " + num + r" \[" + num + "," + num + r"\]", src)
    mac(nm + "ObsUnk", f"{float(m.group(1)):.2f}", src + " llama obscure-unknown")
    mac(nm + "DiffCI", ci(m.group(3), m.group(4)), src + " llama pop-know CI")
for fam, nm in (("qwen", "Qwen"), ("gemma", "Gemma"), ("mistral", "Mistral")):
    m = grab(t, r"== " + fam + r": [^\n]*regex recall on 96-token replies = ([\d.]+)", src)
    mac(f"hm{nm}RegexRecall", m.group(1), src + f" {fam} regex recall")
e = eff["llama", "hedge_rate"]
mac("faLlamaRegexDiffCI", ci(e["diff_lo"], e["diff_hi"]), "results/factorial_4models/effects.csv llama hedge_rate diff CI")
srcj = "results/hedge_judge/summary.txt"
tj = (REPO / srcj).read_text(encoding="utf-8")
m = grab(tj[tj.index("== llama"):], r"pop - know " + num + r" \[" + num + ", " + num + r"\]", srcj)
mac("hjLlamaDiffCI", ci(m.group(2), m.group(3)), srcj + " llama corrected pop - know CI")

# forced-answer hedge log-odds, all four families (effects.csv)
for mdl, nm in (("llama", "Llama"), ("qwen", "Qwen"), ("gemma", "Gemma"), ("mistral", "Mistral")):
    e = eff[mdl, "hedge_logodds"]
    if nm in ("Qwen", "Mistral"):
        mac(f"lo{nm}Pop", sgn(float(e["pop"])), src_e := f"results/factorial_4models/effects.csv {mdl} hedge_logodds pop")
        mac(f"lo{nm}Know", sgn(float(e["know"])), f"results/factorial_4models/effects.csv {mdl} hedge_logodds know")
    mac(f"lo{nm}PopCI", ci(e["pop_lo"], e["pop_hi"]), f"results/factorial_4models/effects.csv {mdl} hedge_logodds pop CI")
    mac(f"lo{nm}KnowCI", ci(e["know_lo"], e["know_hi"]), f"results/factorial_4models/effects.csv {mdl} hedge_logodds know CI")
    mac(f"lo{nm}Diff", sgn(float(e["diff"])), f"results/factorial_4models/effects.csv {mdl} hedge_logodds diff")
    mac(f"lo{nm}DiffCI", ci(e["diff_lo"], e["diff_hi"]), f"results/factorial_4models/effects.csv {mdl} hedge_logodds diff CI")

# guessability control (paper/analysis/guessability.csv, exploratory)
g = {r["tag"]: r for r in read_csv(HERE / "analysis/guessability.csv")}
srcg = "paper/analysis/guessability.csv"
mac("gsObsKnownGuess", str(int(float(g["judge_all"]["obs_known_guessable"]))), srcg + " obscure-known guessable (pool)")
mac("gsObsKnownTotal", str(int(float(g["judge_all"]["obs_known_total"]))), srcg + " obscure-known total (pool)")
r = g["judge_noguess"]
mac("gsN", str(int(float(r["n_total"]))), srcg + " judge_noguess n")
mac("gsPop", sgn(float(r["pop"])), srcg + " judge_noguess pop"); mac("gsKnow", sgn(float(r["know"])), srcg + " judge_noguess know")
mac("gsDiff", sgn(float(r["diff"])), srcg + " judge_noguess diff"); mac("gsDiffCI", ci(r["diff_lo"], r["diff_hi"]), srcg + " judge_noguess diff CI")
mac("gsObsKnownRate", f"{float(r['mean_obs_known']):.2f}", srcg + " judge_noguess obscure-known mean")
mac("gsObsKnownN", str(int(float(r["n_obs_known"]))), srcg + " judge_noguess obscure-known n")
r = g["regex_noguess"]
mac("gsRegexN", f"{int(float(r['n_total'])):,}".replace(",", "{,}"), srcg + " regex_noguess n")
mac("gsRegexDiff", sgn(float(r["diff"])), srcg + " regex_noguess diff"); mac("gsRegexDiffCI", ci(r["diff_lo"], r["diff_hi"]), srcg + " regex_noguess diff CI")
r = g["judge_all"]
mac("hmLlamaPopUnkOnly", sgn(float(r["popU"])), srcg + " judge_all popularity gap among unknown facts")
mac("hmLlamaPopUnkOnlyCI", ci(r["popU_lo"], r["popU_hi"]), srcg + " judge_all popU CI")

# cross-model agreement (paper/analysis/cross_model_agreement.csv, exploratory)
cm = read_csv(HERE / "analysis/cross_model_agreement.csv")
srcc = "paper/analysis/cross_model_agreement.csv"
for other, nm in (("qwen", "Qwen"), ("gemma", "Gemma"), ("mistral", "Mistral")):
    for subset, sn in (("knowledge differs", "Diff"), ("same knowledge", "Same")):
        for readout, rn in (("hedge_logodds", "Hedge"), ("entropy", "Ent"), ("hedge_rate", "Regex")):
            r = [x for x in cm if x["pair"] == f"llama-{other}" and x["subset"] == subset and x["readout"] == readout][0]
            mac(f"cm{nm}{sn}{rn}", sgn(float(r["rho"])), srcc + f" {other} {subset} {readout} rho")
            mac(f"cm{nm}{sn}{rn}CI", ci(r["lo"], r["hi"]), srcc + f" {other} {subset} {readout} CI")
        r = [x for x in cm if x["pair"] == f"llama-{other}" and x["subset"] == subset][0]
        mac(f"cm{nm}{sn}N", f"{int(r['n']):,}".replace(",", "{,}"), srcc + f" {other} {subset} n")
txt = (HERE / "analysis/cross_model_agreement.txt").read_text(encoding="utf-8")
for other, nm in (("qwen", "Qwen"), ("gemma", "Gemma"), ("mistral", "Mistral")):
    m = grab(txt, r"llama-" + other + r":[^\n]*Delta " + num + r" \[" + num + "," + num + r"\]\s+-> ([A-Z ]+)", "paper/analysis/cross_model_agreement.txt")
    mac(f"cm{nm}Delta", sgn(float(m.group(1))), "paper/analysis/cross_model_agreement.txt Delta")
    mac(f"cm{nm}DeltaCI", ci(m.group(2), m.group(3)), "paper/analysis/cross_model_agreement.txt Delta CI")
    mac(f"cm{nm}Rule", "met" if "HOLDS" in m.group(4) else "not met", "paper/analysis/cross_model_agreement.txt verdict")

# base-model lens: mid-window knowledge effect (descriptive; same window as midF) and onsets per model
cv = read_csv(REPO / "results/jlens_cascade/curves.csv")
for mdl, nm in (("llama3.1-8b", "LlamaBase"), ("gemma-2-9b", "GemmaNineBase"), ("gemma-2-2b", "GemmaTwoBase"), ("olmo-3-1025-7b", "OlmoBase")):
    k = [float(x["K"]) for x in cv if x["model"] == mdl and 1 / 3 <= float(x["rel_depth"]) <= 2 / 3]
    f = [float(x["F"]) for x in cv if x["model"] == mdl and 1 / 3 <= float(x["rel_depth"]) <= 2 / 3]
    mac(f"jl{nm}MidK", sgn(sum(k) / len(k)), f"results/jlens_cascade/curves.csv {mdl} mean K over rel_depth in [1/3, 2/3]")
    assert abs(sum(f) / len(f) - float(oc[mdl]["midF"])) < 0.01, mdl  # window matches the tested statistic
ONS = [("llama3.1-8b-it", "Llama-3.1-8B-Instruct"), ("llama3.1-8b", "Llama-3.1-8B"), ("qwen2.5-7b-it", "Qwen2.5-7B-Instruct"),
       ("gemma-2-9b-it", "Gemma-2-9B-it"), ("gemma-2-9b", "Gemma-2-9B"), ("gemma-2-2b-it", "Gemma-2-2B-it"),
       ("gemma-2-2b", "Gemma-2-2B"), ("olmo-3-1025-7b", "OLMo-3-7B")]
tab = ["% Generated by make_numbers.py from results/jlens_cascade/outcomes.csv. Do not edit by hand."]
def di(x):
    v = int(round(float(x)))
    return "0" if v == 0 else (f"$+${v}" if v > 0 else f"$-${-v}")


for mdl, name in ONS:
    r = oc[mdl]
    tab.append(f"{name} & {r['n_layers']} & {r['onset_F']} & {r['onset_K']} & {di(r["D"])} [{di(r["D_lo"])}, {di(r["D_hi"])}] & {r['outcome'].lower()} & {sgn(float(r['midF']))} \\\\")
(HERE / "tab_onsets.tex").write_text("\n".join(tab) + "\n", encoding="utf-8")
mac("jlOnsetRows", " ".join(tab[1:]), "results/jlens_cascade/outcomes.csv onset table rows (also tab_onsets.tex)")

# criterion hashes (sha256 prefix of each criterion file at the repo root)
import hashlib
for fn, nm in (("hedge_measure_v2_criterion.md", "HedgeMeasure"), ("qwen_factorial_criterion.md", "QwenFactorial"),
               ("gemma_mistral_factorial_criterion.md", "GemmaMistralFactorial"), ("direction_reuse_v2_criterion.md", "Steering"),
               ("nohedge_direction_criterion.md", "NoHedge"), ("jlens_cascade_criterion.md", "Lens"), ("l5_behavioral_criterion.md", "Circuit")):
    mac(f"hash{nm}", hashlib.sha256((REPO / fn).read_bytes()).hexdigest()[:8], fn + " sha256 prefix")

# ---------------------------------------------------------------- round 3 experiments (2026-10-10; PREREG e1-e5)
r3 = "results/round3/"
t1 = (REPO / (r3 + "e1_summary.txt")).read_text(encoding="utf-8")
blk = t1[t1.index("T1 two-judge"): t1.index("T2 original")]
m = grab(blk, r"pop " + num + r" \[" + num + ", " + num + r"\]\s+know " + num + r" \[" + num + ", " + num + r"\]\s+diff " + num + r" \[" + num + ", " + num + r"\]", r3 + "e1_summary.txt T1")
v = [float(x) for x in m.groups()]
mac("eoPop", sgn(v[0]), r3 + "e1_summary.txt T1 pop"); mac("eoPopCI", ci(v[1], v[2]), r3 + "e1_summary.txt T1 pop CI")
mac("eoKnow", sgn(v[3]), r3 + "e1_summary.txt T1 know"); mac("eoKnowCI", ci(v[4], v[5]), r3 + "e1_summary.txt T1 know CI")
mac("eoDiff", sgn(v[6]), r3 + "e1_summary.txt T1 diff"); mac("eoDiffCI", ci(v[7], v[8]), r3 + "e1_summary.txt T1 diff CI")
m = grab(blk, r'"unc/High": ([\d.]+)', r3 + "e1_summary.txt T1"); mac("eoObsKnown", f"{float(m.group(1)):.2f}", r3 + "e1_summary.txt T1 obscure-known mean")
m = grab(blk, r"'unc/High': (\d+)", r3 + "e1_summary.txt T1"); mac("eoObsKnownN", m.group(1), r3 + "e1_summary.txt T1 obscure-known n")
m = grab(t1, r"know - pop " + num + r" \[" + num + ", " + num + r"\]", r3 + "e1_summary.txt T2")
mac("eoEntKmP", sgn(float(m.group(1))), r3 + "e1_summary.txt T2 know - pop"); mac("eoEntKmPCI", ci(m.group(2), m.group(3)), r3 + "e1_summary.txt T2 CI")
for tag, nm in (("T4 entropy_free", "eoFreeEnt"), ("T4 logodds_I_free", "eoFreeI")):
    b = t1[t1.index(tag):]
    m = grab(b, r"pop " + num + r" \[" + num + ", " + num + r"\]\s+know " + num + r" \[" + num + ", " + num + r"\]\s+diff " + num + r" \[" + num + ", " + num + r"\]", r3 + "e1_summary.txt " + tag)
    v = [float(x) for x in m.groups()]
    d = 1 if nm == "eoFreeI" else 2
    mac(nm + "Pop", sgn(v[0], d), r3 + f"e1_summary.txt {tag} pop"); mac(nm + "Know", sgn(v[3], d), r3 + f"e1_summary.txt {tag} know")
    mac(nm + "Diff", sgn(v[6], d), r3 + f"e1_summary.txt {tag} diff"); mac(nm + "DiffCI", f"[{sgn(v[7], d)}, {sgn(v[8], d)}]", r3 + f"e1_summary.txt {tag} diff CI")
t2 = (REPO / (r3 + "e2_summary.txt")).read_text(encoding="utf-8")
m = grab(t2, r"Spearman\(surprisal, log page views\) = " + num, r3 + "e2_summary.txt S1"); mac("esRho", sgn(float(m.group(1)), 3), r3 + "e2_summary.txt S1")
m = grab(t2, r"S2 \(a\) two judges[^\n]*surprisal_z " + num + r" \[" + num + "," + num + r"\]", r3 + "e2_summary.txt S2a")
mac("esJudge", sgn(float(m.group(1)), 3), r3 + "e2_summary.txt S2 (a) surprisal coef"); mac("esJudgeCI", f"[{sgn(float(m.group(2)), 3)}, {sgn(float(m.group(3)), 3)}]", r3 + "e2_summary.txt S2 (a) CI")
for fam in read_csv(REPO / (r3 + "e5_summary.csv")):
    nm = fam["family"].capitalize()
    mac(f"tj{nm}Kappa", f"{float(fam['fleiss']):.2f}", r3 + f"e5_summary.csv {fam['family']} Fleiss kappa")
    mac(f"tj{nm}Diff", sgn(float(fam["diff"])), r3 + f"e5_summary.csv {fam['family']} majority pop-know")
    mac(f"tj{nm}DiffCI", ci(fam["diff_lo"], fam["diff_hi"]), r3 + f"e5_summary.csv {fam['family']} CI")
    mac(f"tj{nm}Verdict", fam["verdict"].lower(), r3 + f"e5_summary.csv {fam['family']} verdict")
    mac(f"tj{nm}ObsUnk", f"{float(fam['obs_unk']):.2f}", r3 + f"e5_summary.csv {fam['family']} obscure-unknown majority rate")
for fam in read_csv(REPO / (r3 + "e5_summary.csv")):
    if fam["family"] == "llama":
        mac("tjLlamaQwenLlamaK", f"{float(fam['k_qwen32b-llama70b']):.2f}", r3 + "e5_summary.csv llama pairwise kappa qwen32b-llama70b")
# round-3 criterion hashes (paper/PREREG; recorded in PREREG/HASHES_round3.txt before the runs)
for fn, nm in (("e1_independent_labels.md", "EOne"), ("e2_name_surprisal.md", "ETwo"), ("e3_lens_split.md", "EThree"),
               ("e4_steering_v3.md", "EFour"), ("e4_amendment_third_judge.md", "EFourA"), ("e5_third_judge.md", "EFive")):
    mac(f"hash{nm}", hashlib.sha256((HERE / "PREREG" / fn).read_bytes()).hexdigest()[:8], "paper/PREREG/" + fn + " sha256 prefix")
# E3 window effects table (results/round3/e3_window_effects.csv), base models
p3 = REPO / (r3 + "e3_window_effects.csv")
if p3.exists():
    w3 = read_csv(p3)
    rows = []
    for mdl, name in (("llama3.1-8b", "Llama-3.1-8B"), ("gemma-2-9b", "Gemma-2-9B"), ("gemma-2-2b", "Gemma-2-2B"), ("olmo-3-1025-7b", "OLMo-3-7B")):
        g = {x["term"]: x for x in w3 if x["model"] == mdl}
        cells = [f"{sgn(float(g[t][e]))} {ci(g[t][e + '_lo'], g[t][e + '_hi'])}" for t in ("logodds", "hedge_term") for e in ("pop", "know")]
        rows.append(name + " & " + " & ".join(cells) + " \\\\")
    mac("lwRows", " ".join(rows), r3 + "e3_window_effects.csv base-model rows (logodds and hedge_term, pop and know)")
# E4 steering v3 (results/round3/e4_summary.txt)
t4 = (REPO / (r3 + "e4_summary.txt")).read_text(encoding="utf-8")
for mdl, nm in (("llama", "Llama"), ("gemma9b", "Gemma")):
    b = t4[t4.index(f"[{mdl}]"):]
    b = b[: b.index("[gemma9b]")] if mdl == "llama" else b
    m = grab(b, r"cos\(fam,abst\) " + num + r"; k\* = ([\d.]+)", r3 + f"e4_summary.txt {mdl}")
    mac(f"sv{nm}Cos", f"{float(m.group(1)):.2f}", r3 + f"e4_summary.txt {mdl} cos(fam,abst)")
    k = float(m.group(2)); mac(f"sv{nm}Dose", f"{k:g}", r3 + f"e4_summary.txt {mdl} k*")
    for c, cn in (("none", "None"), ("fam", "Fam"), ("know", "Know"), ("abst", "Abst"), ("fam_perp", "Perp")):
        m = grab(b, r"EVAL " + c + r" +judges ([\d.]+)  regex ([\d.]+)", r3 + f"e4_summary.txt {mdl} {c}")
        mac(f"sv{nm}{cn}", f"{float(m.group(1)):.2f}", r3 + f"e4_summary.txt {mdl} EVAL {c} judges")
        mac(f"sv{nm}{cn}Regex", f"{float(m.group(2)):.2f}", r3 + f"e4_summary.txt {mdl} EVAL {c} regex")
    for tag, cn in (("D1 maj fam - random", "MajDOne"), ("D2 maj fam_perp - random", "MajDTwo")):
        m = re.search(re.escape(tag) + r" " + num + r" \[" + num + "," + num + r"\]", b)
        if m:  # secondary scorer, PREREG e4_amendment_third_judge.md
            mac(f"sv{nm}{cn}", sgn(float(m.group(1))), r3 + f"e4_summary.txt {mdl} {tag} (secondary)")
            mac(f"sv{nm}{cn}CI", ci(m.group(2), m.group(3)), r3 + f"e4_summary.txt {mdl} {tag} CI (secondary)")
    rnd = [float(x) for x in re.findall(r"EVAL random\d +judges ([\d.]+)", b)]
    mac(f"sv{nm}Rand", f"{sum(rnd) / len(rnd):.3f}", r3 + f"e4_summary.txt {mdl} mean of random0-4 judges")
    for tag, cn in (("D1 fam - random", "DOne"), ("D2 fam_perp - random", "DTwo"), ("D3 fam - know", "DThree"), ("D4 entropy know - fam", "DFour")):
        m = grab(b, re.escape(tag) + r" +" + num + r" \[" + num + "," + num + r"\]", r3 + f"e4_summary.txt {mdl} {tag}")
        mac(f"sv{nm}{cn}", sgn(float(m.group(1))), r3 + f"e4_summary.txt {mdl} {tag}")
        mac(f"sv{nm}{cn}CI", ci(m.group(2), m.group(3)), r3 + f"e4_summary.txt {mdl} {tag} CI")
# E4 natural norms (geometry json) and Gemma's SELECT regex rates at 1x and 2x (chair K29)
for mdl, nm in (("llama", "Llama"), ("gemma9b", "Gemma")):
    geo = json.load(open(R / f"round3/e4/geometry_{mdl}.json"))
    for k, cn in (("nat_norm_fam", "Fam"), ("nat_norm_know", "Know"), ("nat_norm_abst", "Abst")):
        mac(f"sv{nm}Norm{cn}", f"{geo[k]:.1f}", f"results/round3/e4/geometry_{mdl}.json {k}")
bg = t4[t4.index("[gemma9b]"):]
m = grab(bg, r"\n +fam +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+)", r3 + "e4_summary.txt gemma SELECT fam row")
for i, cn in ((0, "Half"), (1, "One"), (2, "Two"), (3, "Four")):
    mac(f"svGemmaSel{cn}", f"{float(m.group(i + 1)):.2f}", r3 + f"e4_summary.txt gemma SELECT fam regex at dose index {i}")
# E6 entity-level knowledge (results/round3/e6_summary.txt)
t6 = (REPO / (r3 + "e6_summary.txt")).read_text(encoding="utf-8")
m = grab(t6, r"Spearman\(entity_known, log page views\) = " + num, r3 + "e6_summary.txt K3"); mac("ekRho", sgn(float(m.group(1))), r3 + "e6_summary.txt K3 rho")
m = grab(t6, r"unc/High ([\d.]+)  con/Unkn ([\d.]+)  con/High ([\d.]+)", r3 + "e6_summary.txt K3 cells")
mac("ekObsKnown", f"{float(m.group(1)):.2f}", r3 + "e6_summary.txt obscure-known mean entity_known"); mac("ekPopKnown", f"{float(m.group(3)):.2f}", r3 + "e6_summary.txt popular-known mean entity_known")
m = grab(t6, r"\(a\) two judges n=\d+ with: +obscure " + num + r" \[" + num + "," + num + r"\][^\n]*entity_known_z " + num + r" \[" + num + "," + num + r"\]  \| obscure coefficient shrinks by (\d+)%", r3 + "e6_summary.txt (a) with")
mac("ekObs", sgn(float(m.group(1))), r3 + "e6_summary.txt (a) obscure coef with entity_known"); mac("ekObsCI", ci(m.group(2), m.group(3)), r3 + "e6_summary.txt (a) CI")
mac("ekCoef", sgn(float(m.group(4))), r3 + "e6_summary.txt (a) entity_known_z coef"); mac("ekCoefCI", ci(m.group(5), m.group(6)), r3 + "e6_summary.txt (a) CI")
mac("ekShrink", m.group(7) + "\\%", r3 + "e6_summary.txt (a) shrink")
m = grab(t6, r"\(b\) regex n=\d+ with:[^\n]*shrinks by (\d+)%", r3 + "e6_summary.txt (b)"); mac("ekShrinkRegex", m.group(1) + "\\%", r3 + "e6_summary.txt (b) shrink")
mac("hashESix", hashlib.sha256((HERE / "PREREG" / "e6_entity_knowledge.md").read_bytes()).hexdigest()[:8], "paper/PREREG/e6_entity_knowledge.md sha256 prefix")
# round-2 exploratory checks (paper/analysis/round2_checks.txt)
t3 = (HERE / "analysis/round2_checks.txt").read_text(encoding="utf-8")
m = grab(t3, r"obscure  HighlyKnown replies +\d+ +correct ([\d.]+) +\| non-abstaining replies +\d+ correct ([\d.]+) +\| questions with both a correct reply and an abstention: (\d+) of (\d+)", "paper/analysis/round2_checks.txt (1)")
mac("rcObsKnownCorrect", m.group(1), "paper/analysis/round2_checks.txt obscure-known free replies correct"); mac("rcObsKnownNonAbsCorrect", m.group(2), "paper/analysis/round2_checks.txt non-abstaining correct")
mac("rcBothN", m.group(3), "paper/analysis/round2_checks.txt questions with correct reply and abstention"); mac("rcBothOf", m.group(4), "paper/analysis/round2_checks.txt of")
m = grab(t3, r"two judges \(n=387\)\s+obscure " + num + r" \[" + num + "," + num + r"\]\s+unknown " + num + r" \[" + num + "," + num + r"\]\s+entropy_z " + num, "paper/analysis/round2_checks.txt (2)")
v = [float(x) for x in m.groups()]
mac("rcPopEnt", sgn(v[0]), "paper/analysis/round2_checks.txt judges obscure coef with entropy"); mac("rcPopEntCI", ci(v[1], v[2]), "paper/analysis/round2_checks.txt CI")
mac("rcKnowEnt", sgn(v[3]), "paper/analysis/round2_checks.txt judges unknown coef with entropy"); mac("rcKnowEntCI", ci(v[4], v[5]), "paper/analysis/round2_checks.txt CI")
mac("rcEntCoef", sgn(v[6]), "paper/analysis/round2_checks.txt judges entropy_z coef")

# ---------------------------------------------------------------- lock-in batch (2026-10-10; PREREG e7-e10) and camera-ready tabulations
def cis(m, i):  # signed value and its interval from a "x [lo, hi]" regex match starting at group i
    return sgn(float(m.group(i))), ci(m.group(i + 1), m.group(i + 2))


def ciu(lo, hi):  # unsigned interval
    return f"[{float(lo):.2f}, {float(hi):.2f}]"


rng3 = num + r" \[" + num + r", " + num + r"\]"
for nm, fn in (("ESeven", "e7_label_and_decline_audit.md"), ("EEight", "e8_steering_within_level.md"), ("ENine", "e9_permissive_prompt.md"), ("ETen", "e10_instruct_lens.md")):
    mac(f"hash{nm}", hashlib.sha256((HERE / "PREREG" / fn).read_bytes()).hexdigest()[:8], f"paper/PREREG/{fn} sha256 prefix")
mac("hashHedgeJudge", hashlib.sha256((REPO / "hedge_judge_criterion.md").read_bytes()).hexdigest()[:8], "hedge_judge_criterion.md sha256 prefix")
# four-family confirmatory verdicts (results/factorial_4models/summary.txt)
tv = (R / "factorial_4models/summary.txt").read_text(encoding="utf-8")
for fam in ("Qwen", "Gemma", "Mistral"):
    mac(f"ff{fam}Verdict", grab(tv, r"VERDICT \(" + fam + r", per [^)]*\): (\w+)", "results/factorial_4models/summary.txt").group(1), f"results/factorial_4models/summary.txt VERDICT {fam}")
# E7 (results/round3/e7_summary.txt)
t7 = (R / "round3/e7_summary.txt").read_text(encoding="utf-8")
for arm, a in (("control", "PU"), ("uncertain", "OU")):
    blk = t7[t7.index("    " + arm if arm == "control" else "    uncertain"):]
    for lab, l in (("HighlyKnown", "Known"), ("Unknown", "Stay"), ("intermediate", "Inter")):
        mac(f"se{a}{l}", grab(blk, lab + r" +(\d+)", "e7_summary.txt movement").group(1), f"results/round3/e7_summary.txt T7a movement {arm} {lab}")
m = grab(t7, r"T7a two-judge[^\n]*\n  pop[^\n]*diff " + rng3, "e7_summary.txt T7a"); mac("seDiff", cis(m, 1)[0], "e7_summary.txt T7a diff"); mac("seDiffCI", cis(m, 1)[1], "e7_summary.txt T7a CI")
m = grab(t7, r"T7c DECLINE rate[^\n]*\n  pop[^\n]*diff " + rng3, "e7_summary.txt T7c"); mac("sdDiff", cis(m, 1)[0], "e7_summary.txt T7c diff"); mac("sdDiffCI", cis(m, 1)[1], "e7_summary.txt T7c CI")
m = grab(t7, r"T7c DECLINE rate[^\n]*\n[^\n]*\n  cell means \{\"unc/Unkn\": ([\d.]+), \"unc/High\": ([\d.]+), \"con/Unkn\": ([\d.]+), \"con/High\": ([\d.]+)\}", "e7_summary.txt T7c cells")
for nm, v in zip(("sdObsUnk", "sdObsKnown", "sdPopUnk", "sdPopKnown"), m.groups()):
    mac(nm, f"{float(v):.2f}", f"results/round3/e7_summary.txt T7c cell {nm}")
hs = [float(x) for x in re.findall(r"(?:unc|con)/\w{4} ([\d.]+) \(of \d+\)", t7)]
mac("sdHedgedMax", f"{max(hs):.2f}", "results/round3/e7_summary.txt max per-cell HEDGED_ANSWER share of both-judge DONT_KNOW")
# E9 (results/round3/e9_summary.txt)
t9 = (R / "round3/e9_summary.txt").read_text(encoding="utf-8")
for mdl, nm in (("llama", "Llama"), ("gemma", "Gemma")):
    m = grab(t9, mdl + r" +replies \d+  Fleiss kappa ([\d.]+)[^\n]*\n  majority-rule[^\n]*diff " + rng3 + r"\n  cells \{\"unc/Unkn\": ([\d.]+), \"unc/High\": ([\d.]+), \"con/Unkn\": ([\d.]+), \"con/High\": ([\d.]+)\}  -> ([A-Z ]+)", f"e9_summary.txt {mdl}")
    mac(f"pm{nm}Kappa", f"{float(m.group(1)):.2f}", f"results/round3/e9_summary.txt {mdl} Fleiss kappa")
    mac(f"pm{nm}Diff", cis(m, 2)[0], f"results/round3/e9_summary.txt {mdl} diff"); mac(f"pm{nm}DiffCI", cis(m, 2)[1], f"results/round3/e9_summary.txt {mdl} CI")
    for c, v in zip(("ObsUnk", "ObsKnown", "PopUnk", "PopKnown"), m.groups()[4:8]):
        mac(f"pm{nm}{c}", f"{float(v):.2f}", f"results/round3/e9_summary.txt {mdl} cell {c}")
# E10 (results/round3/e10_summary.csv): hedge term and full readout, popularity - knowledge, base and instruct
t10 = read_csv(R / "round3/e10_summary.csv")
rows = []
for base, nm in (("llama3.1-8b", "Llama-3.1-8B"), ("gemma-2-9b", "Gemma-2-9B"), ("gemma-2-2b", "Gemma-2-2B")):
    cells = []
    for term in ("logodds", "hedge_term"):
        for mdl in (base, [x["model"] for x in t10 if x["pair"] == base and x["model"] != base][0]):
            x = [r for r in t10 if r["model"] == mdl and r["term"] == term][0]
            cells.append(f"{sgn(float(x['diff']))} {ci(x['diff_lo'], x['diff_hi'])}")
    rows.append(nm + " & " + " & ".join(cells) + " \\\\")
mac("itRows", " ".join(rows), "results/round3/e10_summary.csv popularity - knowledge, base vs instruct, full readout and hedge term")
# ambiguity (paper/analysis/ambiguity_check.txt, post hoc)
ta = (HERE / "analysis/ambiguity_check.txt").read_text(encoding="utf-8")
for arm, k, nm in (("popular", "HighlyKnown", "PopKnown"), ("popular", "Unknown", "PopUnk"), ("obscure", "HighlyKnown", "ObsKnown"), ("obscure", "Unknown", "ObsUnk")):
    mac(f"am{nm}Hom", grab(ta, arm + r" +" + k + r" +n +\d+ +median homonyms (\d+)", "ambiguity_check.txt").group(1), f"paper/analysis/ambiguity_check.txt median homonyms {arm} {k}")
m = grab(ta, r"LPM obscure " + num + r" \[" + num + "," + num + r"\]  unknown [^\n]*\n  LPM obscure " + num + r" \[" + num + "," + num + r"\]  unknown [^\n]*log_hom_z " + num + r" \[" + num + "," + num + r"\]\n", "ambiguity_check.txt LPM")
mac("amObsBase", sgn(float(m.group(1))), "paper/analysis/ambiguity_check.txt obscure coef without homonyms")
mac("amObs", sgn(float(m.group(4))), "paper/analysis/ambiguity_check.txt obscure coef with log homonyms"); mac("amObsCI", ci(m.group(5), m.group(6)), "paper/analysis/ambiguity_check.txt CI")
m = grab(ta, r"single-homonym items ([\d.]+) \(n=(\d+)\)", "ambiguity_check.txt single"); mac("amSingle", f"{float(m.group(1)):.2f}", "paper/analysis/ambiguity_check.txt obscure-known single-homonym abstention"); mac("amSingleN", m.group(2), "paper/analysis/ambiguity_check.txt n")
# label checks (paper/analysis/round3_label_checks.txt, post hoc)
tl = (HERE / "analysis/round3_label_checks.txt").read_text(encoding="utf-8")
m = grab(tl, r"drop such questions from popular-unknown only \(worst case\): " + rng3, "round3_label_checks.txt (1)"); mac("lcAliasDrop", cis(m, 1)[0], "round3_label_checks.txt worst-case drop"); mac("lcAliasDropCI", cis(m, 1)[1], "round3_label_checks.txt CI")
mac("lcPUAlias", f"{float(grab(tl, r'any free reply: unc [\d.]+  con ([\d.]+)', 'round3_label_checks.txt').group(1)):.2f}", "round3_label_checks.txt popular-unknown share with alias in a free reply")
m = grab(tl, r"labels under both \(n=(\d+)\): agreement ([\d.]+), Cohen's kappa ([\d.]+)", "round3_label_checks.txt (2)")
mac("lcBothN", m.group(1), "round3_label_checks.txt n both labels"); mac("lcAgree", f"{float(m.group(2)):.2f}", "round3_label_checks.txt agreement"); mac("lcKappa", f"{float(m.group(3)):.2f}", "round3_label_checks.txt kappa")
m = grab(tl, r"HighlyKnown +(\d+) +(\d+)\n +Unknown +(\d+) +(\d+)", "round3_label_checks.txt crosstab")
for nm, v in zip(("lcKK", "lcKU", "lcUK", "lcUU"), m.groups()):
    mac(nm, v, "round3_label_checks.txt cross-tab " + nm)
m = grab(tl, r"both labellings agree: " + rng3 + r"  n \{'unc/Unkn': (\d+), 'unc/High': (\d+), 'con/Unkn': (\d+), 'con/High': (\d+)\}", "round3_label_checks.txt consensus")
mac("lcCons", cis(m, 1)[0], "round3_label_checks.txt consensus contrast"); mac("lcConsCI", cis(m, 1)[1], "round3_label_checks.txt CI")
mac("lcConsN", " / ".join(m.groups()[3:7]), "round3_label_checks.txt consensus cell n (OU / OK / PU / PK)")
dk = [float(x) for x in re.findall(r"(?:unc|con)/\w{4} ([\d.]+) \(n=\d+\)", tl.split("(3)")[1])]
mac("lcDKAliasMax", f"{max(dk):.2f}", "round3_label_checks.txt max per-cell share of DONT_KNOW replies with gold alias")
# camera-ready tabulations (paper/analysis/lockin_checks.txt, post hoc)
tk = (HERE / "analysis/lockin_checks.txt").read_text(encoding="utf-8")
for arm, nm in (("obscure", "Obs"), ("popular", "Pop")):
    m = grab(tk, arm + r" known forced-answer entropy ([\d.]+) \[([\d.]+), ([\d.]+)\]", "lockin_checks.txt (1)")
    mac(f"lk{nm}KnownEnt", f"{float(m.group(1)):.2f}", f"lockin_checks.txt {arm} known entropy"); mac(f"lk{nm}KnownEntCI", ciu(m.group(2), m.group(3)), "lockin_checks.txt CI")
    m = grab(tk, arm + r" known two-judge abstention ([\d.]+) \[([\d.]+), ([\d.]+)\]", "lockin_checks.txt (1)")
    mac(f"lk{nm}KnownAbsCI", ciu(m.group(2), m.group(3)), f"lockin_checks.txt {arm} known abstention CI")
m = grab(tk, r"two-judge kappa\(DONT_KNOW\) [\d.]+ \[([\d.]+), ([\d.]+)\]", "lockin_checks.txt (2)"); mac("lkKappaCI", ciu(m.group(1), m.group(2)), "lockin_checks.txt kappa CI")
m = grab(tk, r"qwen32b ([\d.]+)  gemma27b ([\d.]+)  llama70b ([\d.]+)", "lockin_checks.txt (2) majority")
for nm, v in zip(("lkMajQwen", "lkMajGemma", "lkMajLlama"), m.groups()):
    mac(nm, f"{float(v):.2f}", "lockin_checks.txt agreement with the three-judge majority " + nm)
mac("lkSteerOverlap", grab(tk, r"also in the 387-question sample (\d+)", "lockin_checks.txt (3)").group(1), "lockin_checks.txt steering EVAL questions also in the 387 sample")
for arm, nm in (("uncertain", "Obs"), ("control", "Pop")):
    m = grab(tk, r"pool " + arm + r": HighlyKnown (\d+)  Maybe (\d+)  Unknown (\d+)", "lockin_checks.txt (3)")
    mac(f"lk{nm}PoolKnown", m.group(1), "lockin_checks.txt pool"); mac(f"lk{nm}PoolMaybe", m.group(2), "lockin_checks.txt pool"); mac(f"lk{nm}PoolUnk", m.group(3), "lockin_checks.txt pool")
# E2 regex outcomes and the chair-N6 check (results/round3/e2_summary.txt; paper/analysis/surprisal_check.txt, post hoc)
for tag, nm in (("S2", "esRegex"), ("S3", "esRegexPop")):
    m = grab(t2, tag + r" \(b\) regex n=\d+: surprisal_z " + num + r" \[" + num + "," + num + r"\]", r3 + "e2_summary.txt " + tag)
    mac(nm, sgn(float(m.group(1))), r3 + f"e2_summary.txt {tag} (b) regex surprisal_z"); mac(nm + "CI", ci(m.group(2), m.group(3)), "CI")
tsc = (HERE / "analysis/surprisal_check.txt").read_text(encoding="utf-8")
mac("esArmR", sgn(float(grab(tsc, r"point-biserial\(surprisal, obscure arm\) " + num, "surprisal_check.txt").group(1))), "paper/analysis/surprisal_check.txt point-biserial with obscure arm")
mac("esArmPartial", sgn(float(grab(tsc, r"\| relation, unknown\) n=\d+: " + num, "surprisal_check.txt").group(1))), "paper/analysis/surprisal_check.txt partial correlation")
# E8 (results/round3/e8_summary.txt), added when the corrected-dose run is in
p8 = R / "round3/e8_summary.txt"
if p8.exists():
    t8 = p8.read_text(encoding="utf-8")
    for mdl, nm in (("llama", "Llama"), ("gemma9b", "Gemma")):
        b = t8[t8.index(f"[{mdl}]"):]
        b = b[: b.index("[gemma9b]")] if mdl == "llama" and "[gemma9b]" in b else b
        mac(f"wl{nm}Cos", f"{float(grab(b, r'cos\(fam, abst_within\) ' + num, 'e8_summary.txt').group(1)):.2f}", f"results/round3/e8_summary.txt {mdl} cos(fam, abst_within)")
        m = grab(b, r"abst_within fit on (\d+) vs (\d+)", "e8_summary.txt"); mac(f"wl{nm}Hi", m.group(1), "e8_summary.txt n hi"); mac(f"wl{nm}Lo", m.group(2), "e8_summary.txt n lo")
        pr = b[b.index("[primary"):b.index("[secondary")]
        for tag, cn in (("D1w fam - random", "DOne"), ("D2w fam_perp_w - random", "DTwo"), ("abst_within - random", "DAbst")):
            m = grab(pr, re.escape(tag) + r" +" + num + r" \[" + num + "," + num + r"\]", f"e8_summary.txt {mdl} {tag}")
            mac(f"wl{nm}{cn}", sgn(float(m.group(1))), f"results/round3/e8_summary.txt {mdl} primary {tag}"); mac(f"wl{nm}{cn}CI", ci(m.group(2), m.group(3)), "CI")
        rates = dict(re.findall(r"(\w+) ([\d.]+)", pr.split("rates:")[1].split("(n questions")[0]))
        for c, cn in (("none", "None"), ("fam", "Fam"), ("abst_within", "Abst"), ("fam_perp_w", "Perp")):
            mac(f"wl{nm}{cn}", f"{float(rates[c]):.2f}", f"results/round3/e8_summary.txt {mdl} primary rate {c}")
        mac(f"wl{nm}Rand", f"{sum(float(rates[f'random{j}']) for j in range(5)) / 5:.2f}", f"results/round3/e8_summary.txt {mdl} mean random rate")

text = "% Generated by Sher/paper/make_numbers.py from the repo's result files. Do not edit by hand.\n" + "\n".join(OUT) + "\n"
if "--check" in sys.argv:
    print(text)
else:
    (HERE / "numbers.tex").write_text(text, encoding="utf-8")
    print(f"wrote {len(OUT)} macros to {HERE / 'numbers.tex'}")
