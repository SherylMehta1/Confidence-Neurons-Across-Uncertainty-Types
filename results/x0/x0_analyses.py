"""X0 A1 to A6 and A9 on the tidy tables written by build.py."""
from x0_common import *

log = Log("x0_analyses_log.txt")
pool = pd.read_csv(OUT / "tidy_pool.csv", keep_default_na=False)
lab_pool = pool[pool.cell.isin(CELLS)]
Q = pd.read_csv(OUT / "tidy_questions.csv", keep_default_na=False)
R = pd.read_csv(OUT / "tidy_replies.csv", keep_default_na=False)
NAME = {"both": "both judges (primary)", "qwen": "Qwen2.5-32B alone", "gemma": "Gemma-2-27B alone", "llama70": "Llama-3.3-70B alone",
        "either": "either judge", "majority3": "majority of three judges", "both_clar": "both judges, DONT_KNOW or CLARIFY",
        "decline70": "70B strict decline", "regex96": "regex, same 96-token replies", "regex24": "regex, 20 x 24-token replies, full pool"}
DATA = {m: (Q, m) for m in NAME if m != "regex24"} | {"regex24": (lab_pool, "regex24")}

# ---------------- A1 ----------------
log("== A1 exact intervals (primary measure; seed 0, 10,000 resamples)")
A1 = []


def add(quantity, est, lo, hi, method, n_valid=B):
    A1.append(dict(quantity=quantity, estimate=est, lo=lo, hi=hi, method=method, resamples_used=n_valid))


pt, bs = question_boot(Q, "both")
for c in CELLS:
    add(f"abstention, {CELL_NAME[c]}", pt[c], *ci(bs[c]), "questions within cell")
pe, be = effects(pt), effects(bs)
for k in pe:
    add(k, *((pe[k], *ci(be[k])) if k != "gap among unknown" else gap_ci(Q, "both")), "questions within cell")  # same draws as A5 and round4_checks
for two, nm in ((False, "relations only"), (True, "relations, then questions")):
    e = effects(relation_boot(Q, "both", two))
    for k in pe:
        add(k, pe[k], *ci(e[k]), nm, int(np.isfinite(e[k]).sum()))
pt2, bs2 = question_boot(lab_pool, "entropy")
for c in CELLS:
    add(f"forced-answer entropy (pool), {CELL_NAME[c]}", pt2[c], *ci(bs2[c]), "questions within cell")
# judge kappa, resampling questions (3 replies stay together)
wide = R.pivot(index="key", columns="s", values=["qwen", "gemma"])
a, b = wide["qwen"].to_numpy(bool), wide["gemma"].to_numpy(bool)
rng = np.random.default_rng(SEED)
kb = [kappa(a[i].ravel(), b[i].ravel()) for i in rng.integers(0, len(a), (B, len(a)))]
add("kappa(DONT_KNOW), two judges", kappa(a.ravel(), b.ravel()), *ci(kb), "questions")
A1 = pd.DataFrame(A1)
save(A1, "A1_intervals.csv")
log(A1.round(3).to_string(index=False))

# ---------------- A2 ----------------
log("\n== A2 relation by cell")
rows = []
for p in sorted(pool.prop.unique()):
    d = dict(relation=p)
    for c in CELLS:
        q, l = Q[(Q.prop == p) & (Q.cell == c)], lab_pool[(lab_pool.prop == p) & (lab_pool.cell == c)]
        d |= {f"{c}_n": len(q), f"{c}_abst": q.both.mean() if len(q) else np.nan, f"{c}_flag": "empty" if len(q) == 0 else ("<5" if len(q) < 5 else ""),
              f"{c}_pool_n": len(l), f"{c}_pool_regex24": l.regex24.mean() if len(l) else np.nan}
    rows.append(d)
A2 = pd.DataFrame(rows)
save(A2, "A2_relation_by_cell.csv")
log(A2[["relation"] + [f"{c}_{s}" for c in CELLS for s in ("n", "abst")]].round(2).to_string(index=False))
ok = Q[Q.cell == "OK"].prop.value_counts()
log(f"obscure known: {len(Q[Q.cell == 'OK'])} facts in {len(ok)} relations; {ok.to_dict()}; top three {ok[list(BIG3)].sum()}")
log(f"popular unknown in those three relations: {Q[(Q.cell == 'PU') & Q.prop.isin(BIG3)].shape[0]} of 100")

# ---------------- A3 ----------------
log("\n== A3 free-prompt reply types (exclusive: abstain > gold > clarify > wrong > judges split)")


def rtype(r, gold):
    if r.both:
        return "abstain"
    if r[gold]:
        return "gold answer"
    if r.lab_qwen == r.lab_gemma == "CLARIFY":
        return "clarify"
    if r.lab_qwen == r.lab_gemma == "ANSWER":
        return "answer, no gold"
    return "judges split, no gold"


rows = []
info = Q.set_index("key")
R["alias_in_name"], R["alias_outside_name"] = R.key.map(info.alias_in_name).astype(str) == "True", R.key.map(info.alias_outside_name).astype(str) == "True"
subsets = {"all questions": R.index == R.index, "no gold alias inside the name": ~R.alias_in_name, "a gold alias inside the name": R.alias_in_name}
for sname, mask in subsets.items():
    for gold in ("gold_excl", "gold_incl"):
        S = R[mask].copy()
        S["type"] = S.apply(rtype, axis=1, gold=gold)
        for c in CELLS:
            s = S[S.cell == c]
            if len(s) == 0:
                continue
            q = s.groupby("key")[[gold, "both"]].max()
            rows.append(dict(subset=sname, alias_rule="name echoes excluded" if gold == "gold_excl" else "name echoes counted", cell=CELL_NAME[c],
                             questions=s.key.nunique(), replies=len(s), any_gold=s[gold].mean(), any_gold_lo=ci(boot_mean(s.groupby("key")[gold].mean(), np.random.default_rng(SEED)))[0],
                             any_gold_hi=ci(boot_mean(s.groupby("key")[gold].mean(), np.random.default_rng(SEED)))[1],
                             **{t: (s.type == t).mean() for t in ("abstain", "gold answer", "answer, no gold", "clarify", "judges split, no gold")},
                             gold_given_not_abstaining=s.loc[~s.both, gold].mean(), q_with_gold_reply=q[gold].mean(),
                             q_with_gold_and_abstain=int((q[gold] & q.both).sum())))
A3 = pd.DataFrame(rows)
save(A3, "A3_reply_types.csv")
log(A3.drop(columns=["any_gold_lo", "any_gold_hi"]).round(2).to_string(index=False))

# ---------------- A4 ----------------
log("\n== A4 judges")
rows = []
for m in NAME:
    df, y = DATA[m]
    pt, bs = question_boot(df, y)
    e, eb = effects(pt), effects(bs)
    rc = effects(relation_boot(df, y, False))
    rows.append(dict(measure=NAME[m], n=len(df), **{c: pt[c] for c in CELLS},
                     **{f"{k}{s}": v for k in ("popularity effect", "knowledge effect", "margin") for s, v in zip(("", " lo", " hi"), (e[k], *ci(eb[k])))},
                     margin_relation_lo=ci(rc["margin"])[0], margin_relation_hi=ci(rc["margin"])[1]))
A4 = pd.DataFrame(rows)
save(A4, "A4_margin_by_judge.csv")
for r in rows:
    log(f"  {r['measure']:<42} OK {r['OK']:.3f} OU {r['OU']:.3f} PK {r['PK']:.3f} PU {r['PU']:.3f} | margin {fmt(r['margin'], r['margin lo'], r['margin hi'])}"
        f" relation-clustered [{r['margin_relation_lo']:+.3f}, {r['margin_relation_hi']:+.3f}]")
rows = []
for c in CELLS + ["all"]:
    s = R if c == "all" else R[R.cell == c]
    keys = s.key.unique()
    d = dict(cell="all" if c == "all" else CELL_NAME[c], replies=len(s), **{f"{m}_rate": s[m].mean() for m in ("qwen", "gemma", "llama70", "both", "either", "regex96")})
    for x, y in (("qwen", "gemma"), ("qwen", "llama70"), ("gemma", "llama70")):
        w = s.pivot(index="key", columns="s", values=[x, y])
        a, b = w[x].to_numpy(bool), w[y].to_numpy(bool)
        rng = np.random.default_rng(SEED)
        kb = [kappa(a[i].ravel(), b[i].ravel()) for i in rng.integers(0, len(a), (B, len(a)))]
        k0 = kappa(a.ravel(), b.ravel())
        d |= {f"kappa_{x}_{y}": k0, f"kappa_{x}_{y}_lo": ci(kb)[0] if np.isfinite(k0) else np.nan, f"kappa_{x}_{y}_hi": ci(kb)[1] if np.isfinite(k0) else np.nan}
    d |= dict(agree_qwen_gemma=(s.qwen == s.gemma).mean(), qwen_yes_gemma_no=(s.qwen & ~s.gemma).mean(), gemma_yes_qwen_no=(~s.qwen & s.gemma).mean())
    rows.append(d)
save(pd.DataFrame(rows), "A4_judges_by_cell.csv")
log(pd.DataFrame(rows).round(2).T.to_string(header=False))
t = R[R.qwen & ~R.gemma]
log(f"Qwen DONT_KNOW but Gemma not: {len(t)} replies; Gemma's label {t.lab_gemma.value_counts().to_dict()}; 70B agrees with Qwen on {t.llama70.mean():.2f}")

log("\n-- A4: why +0.13 against +0.30 (judge-corrected 24-token rates re-derived from results/hedge_judge)")
S = pd.DataFrame(read(REPO / "results/hedge_judge/sample.jsonl")).merge(pd.DataFrame(read(REPO / "results/hedge_judge/judgments.jsonl")), on="sid")
S = S[S.model == "llama"].assign(jh=lambda d: (d.judge == "HEDGE").astype(float), cell=lambda d: [CELL_OF[x] for x in zip(d.arm, d.slick)])
tot = {(CELL_OF[x["arm"], x["slick"]], bool(x["regex"])): x["n_total"] for x in json.load(open(REPO / "results/hedge_judge/stratum_counts.json")) if x["model"] == "llama"}
rng = np.random.default_rng(SEED)
prec, miss, pb, mb, share = {}, {}, {}, {}, {}
for c in CELLS:
    for rx, pt_, bs_ in ((True, prec, pb), (False, miss, mb)):
        v = S[(S.cell == c) & (S.regex == rx)].jh.to_numpy()
        pt_[c], bs_[c] = v.mean(), boot_mean(v, rng)
    share[c] = tot[c, True] / (tot[c, True] + tot[c, False])
corr = {c: share[c] * prec[c] + (1 - share[c]) * miss[c] for c in CELLS}
corb = {c: share[c] * pb[c] + (1 - share[c]) * mb[c] for c in CELLS}
steps = []


def step(name, pt, bs, n):
    e, eb = effects(pt), effects(bs)
    steps.append(dict(step=name, n_questions=n, **{c: pt[c] for c in CELLS}, margin=e["margin"], lo=ci(eb["margin"])[0], hi=ci(eb["margin"])[1]))


step("1. regex, 24-token replies, full pool", *question_boot(lab_pool, "regex24"), len(lab_pool))
step("2. step 1 corrected by one judge (Qwen2.5-32B, HEDGE/ANSWER prompt)", corr, corb, len(lab_pool))
step("3. regex, 24-token replies, the 387 questions", *question_boot(Q, "regex24"), len(Q))
step("4. regex, 96-token replies, the 387 questions", *question_boot(Q, "regex96"), len(Q))
for m, t in (("qwen", "5. Qwen2.5-32B alone, 96 tokens (DONT_KNOW/CLARIFY/ANSWER prompt)"), ("gemma", "6. Gemma-2-27B alone, 96 tokens"), ("both", "7. both judges, 96 tokens (primary)")):
    step(t, *question_boot(Q, m), len(Q))
ST = pd.DataFrame(steps)
save(ST, "A4_margin_steps.csv")
log(ST.round(3).to_string(index=False))
ST2 = pd.DataFrame([dict(cell=CELL_NAME[c], regex_share=share[c], precision=prec[c], miss_rate=miss[c], corrected=corr[c], judged_regex_hedge=len(S[(S.cell == c) & S.regex]), judged_regex_answer=len(S[(S.cell == c) & ~S.regex])) for c in CELLS])
save(ST2, "A4_correction_parts.csv")
log(ST2.round(3).to_string(index=False))

# ---------------- A5 ----------------
log("\n== A5 popularity gap among unknown facts")
J = Q.copy()  # all label corrections jointly, rule as in paper_2.0/analysis/round4_checks.py
G = {(x["key"], x["i"]): x["label"] for x in read(R3 / "e7_grades.jsonl")}
sem = {}
for x in read(R3 / "e7_answers.jsonl"):
    g = [G.get((x["key"], t)) for t in x["answers"]]
    if any(v is None for v in g):
        continue
    k = [v == "CORRECT" for v in g]
    sem[x["key"]] = "K" if (k[0] and sum(k[1:]) >= 9) else ("U" if sum(k[1:]) == 0 else "band")
rawp = {f"{r['prop']}::{r['subj']}": r for r in read(POOL) if r.get("subj") and r["slick"] in ("HighlyKnown", "Unknown")}
mode = {}
for p in {r["prop"] for r in rawp.values()}:
    vc = pd.Series([r["obj"] for r in rawp.values() if r["prop"] == p]).value_counts()
    mode[p] = (vc.index[0], vc.iloc[0] / vc.sum())
ng = lambda s: re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def guess(r):
    s = " " + ng(r["subj"]) + " "
    top, sh = mode[r["prop"]]
    return any(len(ng(a)) >= 3 and (" " + ng(a) + " ") in s for a in pool_aliases(r)) or (r["obj"] == top and sh >= 0.2)


Q["guessable"] = [guess(rawp[k]) for k in Q.key]
Q["semlab"] = [sem.get(k, "ungraded") if c[1] == "U" else "K" for k, c in zip(Q.key, Q.cell)]
Q["alias_hit"] = Q.gold_excl > 0
J = {"no correction": Q}
d = Q.copy(); d["cell"] = [c[0] + s if s in ("K", "U") else "band" for c, s in zip(d.cell, d.semlab)]
J["1 meaning-based regrade"] = d[d.cell != "band"]
J["2 popular unknown with gold in a reply dropped"] = Q[~((Q.cell == "PU") & Q.alias_hit)]
J["3 guessable dropped"] = Q[~Q.guessable]
d = J["1 meaning-based regrade"]
J["all three"] = d[~((d.cell == "PU") & d.alias_hit) & ~d.guessable]
rows = []
for m in NAME:
    for cname, d in (J.items() if m != "regex24" else [("no correction", lab_pool)]):
        y = m
        est, lo, hi = gap_ci(d, y)
        for_rel = relation_boot(d, y, False, cells=["OU", "PU"])
        rl, rh = ci(for_rel["OU"] - for_rel["PU"])
        tl, th = ci((lambda z: z["OU"] - z["PU"])(relation_boot(d, y, True, cells=["OU", "PU"])))
        mg = effects({c: d.loc[d.cell == c, y].mean() for c in CELLS})["margin"]
        _, mb_ = question_boot(d, y)
        rows.append(dict(measure=NAME[m], correction=cname, n_OU=(d.cell == "OU").sum(), n_PU=(d.cell == "PU").sum(), n_OK=(d.cell == "OK").sum(), gap=est, lo=lo, hi=hi,
                         rel_lo=rl, rel_hi=rh, rel_q_lo=tl, rel_q_hi=th, margin=mg, margin_lo=ci(effects(mb_)["margin"])[0], margin_hi=ci(effects(mb_)["margin"])[1]))
A5 = pd.DataFrame(rows)
save(A5, "A5_unknown_gap.csv")
for r in rows:
    log(f"  {r['measure']:<42} {r['correction']:<46} gap {fmt(r['gap'], r['lo'], r['hi'])} rel [{r['rel_lo']:+.3f}, {r['rel_hi']:+.3f}] n {r['n_OU']}/{r['n_PU']} | margin {fmt(r['margin'], r['margin_lo'], r['margin_hi'])}")
rows = []
for m in NAME:
    for cname in (("no correction", "all three") if m != "regex24" else ("no correction",)):
        d = J[cname] if m != "regex24" else lab_pool
        for p in sorted(d.prop.unique()):
            s = d[d.prop != p]
            rows.append(dict(measure=NAME[m], correction=cname, left_out=p, n_OU=(s.cell == "OU").sum(), n_PU=(s.cell == "PU").sum(), **dict(zip(("gap", "lo", "hi"), gap_ci(s, m)))))
L = pd.DataFrame(rows)
save(L, "A5_leave_one_relation_out.csv")
for (m, c), g in L.groupby(["measure", "correction"], sort=False):
    i, j = g.gap.idxmin(), g.gap.idxmax()
    log(f"  LORO {m:<42} {c:<14} gap {g.gap.min():+.3f} (without {g.left_out[i]}) to {g.gap.max():+.3f} (without {g.left_out[j]}); lowest lower bound {g.lo.min():+.3f}; intervals including 0: {(g.lo <= 0).sum()}")
# relation-standardised unknown gap: within-relation gaps
rows = []
for m, d, need in (("both", Q, 1), ("both", Q, 3), ("regex24", lab_pool, 5)):
    g = d[d.cell.isin(["OU", "PU"])].groupby(["prop", "cell"])[m].agg(["mean", "size"]).unstack()
    g = g.dropna(); g = g[(g["size"] >= need).all(axis=1)]
    w = 1 / (1 / g["size"]["OU"] + 1 / g["size"]["PU"])
    diff = g["mean"]["OU"] - g["mean"]["PU"]
    rng = np.random.default_rng(SEED); bs = []
    for p in g.index:
        bs.append(boot_mean(d[(d.prop == p) & (d.cell == "OU")][m], rng) - boot_mean(d[(d.prop == p) & (d.cell == "PU")][m], rng))
    bs = np.array(bs)
    rows.append(dict(measure=NAME[m], min_n=need, relations=len(g), equal_weight=diff.mean(), eq_lo=ci(bs.mean(0))[0], eq_hi=ci(bs.mean(0))[1],
                     precision_weight=np.average(diff, weights=w), pw_lo=ci(np.average(bs, axis=0, weights=w))[0], pw_hi=ci(np.average(bs, axis=0, weights=w))[1],
                     relations_positive=int((diff > 0).sum())))
WR = pd.DataFrame(rows); save(WR, "A5_within_relation_gap.csv"); log(WR.round(3).to_string(index=False))

# ---------------- A6 ----------------
log("\n== A6 margin inside relations")
rows, summ = [], []
for m in ("regex24", "both", "regex96"):
    d = DATA[m][0]
    for p in sorted(d.prop.unique()):
        n = {c: int(((d.prop == p) & (d.cell == c)).sum()) for c in CELLS}
        if min(n.values()) < 1:
            continue
        pt, bs = question_boot(d[d.prop == p], m)
        e, eb = effects(pt), effects(bs)
        rows.append(dict(measure=NAME[m], key=m, relation=p, **{f"n_{c}": n[c] for c in CELLS}, min_cell=min(n.values()), margin=e["margin"], lo=ci(eb["margin"])[0], hi=ci(eb["margin"])[1],
                         gap_known=e["gap among known"], gap_unknown=e["gap among unknown"], _b=eb["margin"]))
    for k in (1, 3, 5):
        rr = [r for r in rows if r["key"] == m and r["min_cell"] >= k]
        if rr:
            mb_ = np.mean([r["_b"] for r in rr], axis=0)
            summ.append(dict(measure=NAME[m], statistic=f"mean margin over relations with every cell >= {k}", relations=", ".join(r["relation"] for r in rr), n_OK=sum(r["n_OK"] for r in rr),
                             estimate=np.mean([r["margin"] for r in rr]), lo=ci(mb_)[0], hi=ci(mb_)[1]))
    for nm, s in (("outside country, capital, sport", d[~d.prop.isin(BIG3)]), ("inside country, capital, sport", d[d.prop.isin(BIG3)]), ("all relations", d)):
        pt, bs = question_boot(s, m)
        summ.append(dict(measure=NAME[m], statistic=f"margin, {nm}", relations="", n_OK=int((s.cell == "OK").sum()), estimate=effects(pt)["margin"], lo=ci(effects(bs)["margin"])[0], hi=ci(effects(bs)["margin"])[1]))
save(pd.DataFrame(rows).drop(columns=["_b", "key"]), "A6_within_relation.csv")
save(pd.DataFrame(summ), "A6_summary.csv")
log(pd.DataFrame(rows).drop(columns=["_b", "key"]).round(3).to_string(index=False))
log(pd.DataFrame(summ).round(3).to_string(index=False))

# ---------------- A9 ----------------
log("\n== A9 regex against the two-judge measure, same replies")
rows = []
for c in CELLS + ["all"]:
    s = R if c == "all" else R[R.cell == c]
    rows.append(dict(cell="all" if c == "all" else CELL_NAME[c], replies=len(s), regex=s.regex96.mean(), both=s.both.mean(), agree=(s.regex96 == s.both).mean(), kappa=kappa(s.regex96, s.both),
                     regex_only=(s.regex96 & ~s.both).mean(), both_only=(~s.regex96 & s.both).mean(), recall_vs_both=s.loc[s.both, "regex96"].mean() if s.both.any() else np.nan,
                     precision_vs_both=s.loc[s.regex96, "both"].mean() if s.regex96.any() else np.nan, kappa_vs_qwen=kappa(s.regex96, s.qwen), kappa_vs_gemma=kappa(s.regex96, s.gemma),
                     kappa_vs_llama70=kappa(s.regex96, s.llama70), precision_vs_llama70=s.loc[s.regex96, "llama70"].mean() if s.regex96.any() else np.nan,
                     n_regex_only=int((s.regex96 & ~s.both).sum()), regex_only_70B_dont_know=s.loc[s.regex96 & ~s.both, "llama70"].mean() if (s.regex96 & ~s.both).any() else np.nan,
                     regex_only_clarify_either=((s.lab_qwen == "CLARIFY") | (s.lab_gemma == "CLARIFY"))[s.regex96 & ~s.both].mean() if (s.regex96 & ~s.both).any() else np.nan))
A9 = pd.DataFrame(rows)
save(A9, "A9_regex_by_cell.csv")
log(A9.round(2).T.to_string(header=False))
t = R[R.regex96 & ~R.both]
log("regex-only replies by (Qwen, Gemma) labels: " + str(t.groupby(["lab_qwen", "lab_gemma"]).size().sort_values(ascending=False).to_dict()))
log.save()
