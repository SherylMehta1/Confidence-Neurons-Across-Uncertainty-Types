"""X0 step 1: tidy tables from the raw result files, integrity checks, and the Table 1 reproduction."""
from x0_common import *

log = Log("build_log.txt")
raw = read(POOL)
log(f"pool rows {len(raw)}; labelled known/unknown {sum(r['slick'] in ('HighlyKnown', 'Unknown') for r in raw)}")
empty = [r for r in raw if not r.get("subj")]
for r in empty:
    log(f"  dropped, empty subject: {r['q']!r} ({r['arm']}, {r['slick']})")
rows, bad = [], dict(frac=0, label=0, regex=0)
for r in raw:
    if not r.get("subj"):
        continue
    al = pool_aliases(r)
    ok = [is_correct(s, al) for s in r["samples"]]
    g = is_correct(r["greedy"], al)
    lab = slick_class(g, sum(ok) / len(ok))
    rx = np.mean([bool(HEDGE_RE.search(t)) for t in r["free"]])
    bad["frac"] += abs(sum(ok) / len(ok) - r["frac_correct"]) > 1e-9
    bad["label"] += lab != r["slick"]
    bad["regex"] += abs(rx - r["hedge_rate"]) > 1e-9
    subj = nrm(r["subj"])
    na = [nrm(a).strip() for a in al if len(str(a).strip()) >= 3]
    rows.append(dict(key=f"{r['prop']}::{r['subj']}", prop=r["prop"], arm=r["arm"], know=r["slick"],
                     cell=CELL_OF.get((r["arm"], r["slick"]), "band"), s_pop=r["s_pop"], log_views=np.log(max(r["s_pop"], 1)),
                     n_correct=sum(ok), greedy_ok=g, ok_bits="".join("1" if x else "0" for x in ok), entropy=r["entropy"],
                     regex24=r["hedge_rate"], alias_in_name=any((" " + a + " ") in subj for a in na),
                     alias_outside_name=any((" " + a + " ") not in subj for a in na)))
pool = pd.DataFrame(rows)
assert pool.key.is_unique
log(f"after dropping empty subjects: {len(pool)} rows, {(pool.cell != 'band').sum()} labelled: {pool.cell.value_counts().to_dict()}")
log(f"re-grading with the repo's grader disagrees with the stored file on: {bad}")
log(f"s_pop < 1: {(pool.s_pop < 1).sum()}; missing entropy: {pool.entropy.isna().sum()}; relations: {pool.prop.nunique()}")
log("relations: " + ", ".join(sorted(pool.prop.unique())))
save(pool, "tidy_pool.csv")

P = {r["key"]: r for r in ({**r, "key": f"{r['prop']}::{r['subj']}"} for r in raw if r.get("subj"))}
lab = {j: {(x["key"], x["s"]): x["label"] for x in read(HM / f"judge_{j}_llama.jsonl")} for j in ("qwen32b", "gemma27b", "llama70b")}
dec = {(x["key"], x["i"]): x["label"] for x in read(R3 / "e7_declines.jsonl")}
rows = []
for g in read(HM / "gens_llama.jsonl"):
    r = P[g["key"]]
    assert r["arm"] == g["arm"] and r["slick"] == g["slick"]
    subj = nrm(r["subj"])
    na = [nrm(a).strip() for a in pool_aliases(r) if len(str(a).strip()) >= 3]
    for s, t in enumerate(g["samples"]):
        k = (g["key"], s)
        q, m, l = lab["qwen32b"][k], lab["gemma27b"][k], lab["llama70b"][k]
        n = nrm(t)
        dk = [x == "DONT_KNOW" for x in (q, m, l)]
        rows.append(dict(key=g["key"], s=s, prop=r["prop"], cell=CELL_OF[g["arm"], g["slick"]], lab_qwen=q, lab_gemma=m, lab_llama70=l,
                         lab_decline=dec.get(k), qwen=dk[0], gemma=dk[1], llama70=dk[2], both=dk[0] and dk[1], either=dk[0] or dk[1],
                         majority3=sum(dk) >= 2, both_clar=all(x in ("DONT_KNOW", "CLARIFY") for x in (q, m)),
                         decline70=dec.get(k) == "DECLINE", regex96=bool(HEDGE_RE.search(t)),
                         gold_incl=any(" " + a + " " in n for a in na),
                         gold_excl=any(" " + a + " " in n for a in na if (" " + a + " ") not in subj)))
R = pd.DataFrame(rows)
log(f"replies {len(R)}; decline labels missing {R.lab_decline.isna().sum()}; unparsed qwen {(R.lab_qwen == 'UNPARSED').sum()} "
    f"gemma {(R.lab_gemma == 'UNPARSED').sum()} llama70 {(R.lab_llama70 == 'UNPARSED').sum()}")
save(R, "tidy_replies.csv")
MEAS = ["qwen", "gemma", "llama70", "both", "either", "majority3", "both_clar", "decline70", "regex96", "gold_incl", "gold_excl"]
Q = R.groupby("key")[MEAS].mean().reset_index().merge(pool, on="key", how="left")
assert len(Q) == 387 and Q.cell.isin(CELLS).all()
ref = pd.read_csv(HM / "item_rates.csv").query("model == 'llama'").set_index("key")
assert np.allclose(Q.both, Q.key.map(ref.prim)) and np.allclose(Q.regex96, Q.key.map(ref.regex)) and np.allclose(Q.both_clar, Q.key.map(ref.sec))
log("item rates equal results/hedge_measure_v2/item_rates.csv (both, both_clar, regex96)")
save(Q, "tidy_questions.csv")

paper = {"OK": (0.43, 0.95, 87, 87), "OU": (0.48, 3.51, 100, 1179), "PK": (0.00, 0.86, 100, 676), "PU": (0.13, 2.58, 100, 361)}
T = []
for c in CELLS:
    q, p = Q[Q.cell == c], pool[pool.cell == c]
    T.append(dict(cell=CELL_NAME[c], abstention=q.both.mean(), abstention_paper=paper[c][0], n_abstention=len(q),
                  entropy_pool=p.entropy.mean(), entropy_paper=paper[c][1], n_pool=len(p), entropy_387_only=q.entropy.mean(),
                  match=round(q.both.mean(), 2) == paper[c][0] and round(p.entropy.mean(), 2) == paper[c][1]
                  and len(q) == paper[c][2] and len(p) == paper[c][3]))
T = pd.DataFrame(T)
save(T, "table1_reproduction.csv")
log(T.round(3).to_string(index=False))
assert T.match.all(), "Table 1 does not reproduce: stop"
log.save()
