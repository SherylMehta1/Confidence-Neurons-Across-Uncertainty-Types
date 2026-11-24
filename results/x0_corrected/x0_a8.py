"""X0 A8: fragility of the known / unknown labels."""
from x0_common import *

log = Log("x0_a8_log.txt")
pool = pd.read_csv(OUT / "tidy_pool.csv", keep_default_na=False, dtype={"ok_bits": str})
Q = pd.read_csv(OUT / "tidy_questions.csv", keep_default_na=False)
pool["greedy_ok"] = pool.greedy_ok.astype(str) == "True"
Q["greedy_ok"] = Q.greedy_ok.astype(str) == "True"
bits = np.array([[int(c) for c in s] for s in pool.ok_bits])
unk = (pool.n_correct == 0) & ~pool.greedy_ok

log("(a) known cut-off at 8, 9 and 10 of 10 (with and without the greedy condition; unknown: 0 of 10, greedy wrong)")
rows = []
for t in (8, 9, 10):
    for greedy in (True, False):
        kn = (pool.n_correct >= t) & (pool.greedy_ok | (not greedy))
        d = pool[kn | unk].copy()
        d["cell"] = np.where(d.arm == "uncertain", "O", "P") + np.where(kn[d.index], "K", "U").astype(object)
        pt, bs = question_boot(d, "regex24")
        n = {c: int((d.cell == c).sum()) for c in CELLS}
        r = dict(cutoff=f"{t} of 10", greedy_condition=greedy, n_known=int(kn.sum()), n_unknown=int(unk.sum()), **{f"n_{c}": n[c] for c in CELLS},
                 regex24_margin=effects(pt)["margin"], regex_lo=ci(effects(bs)["margin"])[0], regex_hi=ci(effects(bs)["margin"])[1])
        if t >= 9:  # two-judge replies exist only for the 387 sampled questions, all labelled at 9 of 10
            q = Q[(Q.cell.str[1] == "U") | (Q.n_correct >= t)]
            pt, bs = question_boot(q, "both")
            r |= dict(two_judge_margin=effects(pt)["margin"], two_lo=ci(effects(bs)["margin"])[0], two_hi=ci(effects(bs)["margin"])[1],
                      two_judge_n_OK=int((q.cell == "OK").sum()), two_judge_n_PK=int((q.cell == "PK").sum()))
        rows.append(r)
A = pd.DataFrame(rows); save(A, "A8_thresholds.csv"); log(A.round(3).to_string(index=False))
log("two-judge margins at 8 of 10 cannot be computed: questions at exactly 8 of 10 were never given judged free replies")

log("\n(b) random halves of the 10 stored samples (10,000 splits, same 9-of-10 rule = 5 of 5; greedy condition ignored in halves)")
lab = lambda k: np.where(k == 5, 2, np.where(k == 0, 0, 1))  # 2 known, 0 unknown, 1 band
rng = np.random.default_rng(SEED)
rows = []
for c in CELLS:
    X = bits[(pool.cell == c).to_numpy()]
    want = 2 if c[1] == "K" else 0
    k = X.sum(1)
    # exact under random splits: first-half count is hypergeometric
    h = rng.hypergeometric(k[None, :], (10 - k)[None, :], 5, size=(B, len(k)))
    a, b = lab(h), lab(k[None, :] - h)
    rows.append(dict(cell=CELL_NAME[c], n=len(X), n_10_of_10=int((k == 10).sum()), n_9_of_10=int((k == 9).sum()), n_0_of_10=int((k == 0).sum()), half_gives_stored_label=(a == want).mean(),
                     halves_agree=(a == b).mean(), half_gives_band=(a == 1).mean(), half_gives_opposite_label=(a == (2 - want)).mean(),
                     any_half_flips=((a != want) | (b != want)).mean()))
A = pd.DataFrame(rows); save(A, "A8_label_stability.csv"); log(A.round(3).to_string(index=False))
log("These shares are fixed by the stored counts: a 10/10 or 0/10 question can never change, and a 9/10 question always changes in exactly one")
log("half, always to the dropped band and never to the opposite label. So '100% stable' for unknown is true by construction, not evidence.")
for c in ("OK", "PK"):
    p = pool[pool.cell == c]
    log(f"  {CELL_NAME[c]}: {(p.n_correct == 9).sum()} of {len(p)} facts at 9 of 10 ({(p.n_correct == 9).mean():.3f}); half of that is the share of halves that miss")

log("\n(c) fresh evidence: test-retest of unknown labels on new forced answers (E7), re-graded with the same string grader")
P = pool.set_index("key")
rows = []
for x in read(R3 / "e7_answers.jsonl"):
    ok = [is_correct(t, x["aliases"]) for t in x["answers"]]
    rows.append(dict(key=x["key"], cell=P.cell[x["key"]] if x["key"] in P.index else "?", new=slick_class(ok[0], sum(ok[1:]) / 10)))
T = pd.DataFrame(rows)
A = T.groupby(["cell", "new"]).size().unstack(fill_value=0)
A["share_same"] = A["Unknown"] / A.sum(axis=1)
save(A.reset_index(), "A8_unknown_test_retest.csv"); log(A.round(3).to_string())

log("\n(d) fresh evidence: labels from a paraphrased prompt (E1)")
E1 = pd.DataFrame(read(R3 / "e1_llama.jsonl"))
E1["cell"] = E1.key.map(P.cell)
A = E1[E1.cell.isin(CELLS)].groupby(["cell", "slick_para"]).size().unstack(fill_value=0)
A["share_same"] = [A.loc[c, "HighlyKnown" if c[1] == "K" else "Unknown"] / A.loc[c].sum() for c in A.index]
A["share_opposite"] = [A.loc[c, "Unknown" if c[1] == "K" else "HighlyKnown"] / A.loc[c, ["HighlyKnown", "Maybe", "Unknown"]].sum() for c in A.index]
save(A.reset_index(), "A8_paraphrase_retest.csv"); log(A.round(3).to_string())
q = Q.copy(); q["para"] = q.key.map(E1.set_index("key").slick_para)
for c in ("OK",):
    s = q[q.cell == c]
    log("  obscure known, two-judge abstention by paraphrase label: " + "; ".join(f"{k} {g.both.mean():.3f} (n={len(g)})" for k, g in s.groupby("para")))
log.save()
