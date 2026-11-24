"""R3: within-relation margin (obscure known minus popular unknown), regex, full 2,303-question pool, by minimum cell size.
Main row: k = 3, the paper's own row, with the paper's own interval (5,000 draws). Other k: extra appendix rows.
Intervals resample questions within relation and cell; they do not cover variation between relations."""
from x0_common import *

d = pd.read_csv(OUT / "tidy_pool.csv", keep_default_na=False)
d = d[d.cell.isin(CELLS)]
assert len(d) == 2303
paper = re.search(r"pref: ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", (REPO / "paper_2.0/analysis/within_relation_check.txt").read_text())
n = d.groupby(["prop", "cell"]).size().unstack(fill_value=0)
rows = []
for k in (3, 1, 5, 10):
    rel = sorted(n.index[(n.OK >= k) & (n.PU >= k)])
    assert rel == sorted(n.index[(n[CELLS] >= k).all(axis=1)]), "the two-cell rule and the paper's every-cell rule differ"
    est, bs = [], []
    rng = np.random.default_rng(SEED)
    for p in rel:
        s = d[d.prop == p]
        ok, pu = s[s.cell == "OK"].regex24.to_numpy(float), s[s.cell == "PU"].regex24.to_numpy(float)
        est.append(ok.mean() - pu.mean())
        bs.append(boot_mean(ok, rng) - boot_mean(pu, rng))
    lo, hi = ci(np.mean(bs, axis=0))
    one = len(rel) == 1
    rows.append(dict(min_questions_per_cell=k, role="main (paper's row)" if k == 3 else "extra appendix row", n_relations=len(rel), relations=", ".join(rel), margin_per_relation="; ".join(f"{p} {e:+.3f}" for p, e in zip(rel, est)),
                     n_obscure_known=int(n.OK[rel].sum()), n_popular_unknown=int(n.PU[rel].sum()), smallest_obscure_known_cell=int(n.OK[rel].min()), smallest_popular_unknown_cell=int(n.PU[rel].min()),
                     mean_margin=np.mean(est), lo=np.nan if one else lo, hi=np.nan if one else hi, draws=B, seed=SEED, paper_lo=float(paper[2]) if k == 3 else np.nan, paper_hi=float(paper[3]) if k == 3 else np.nan,
                     paper_draws=5000 if k == 3 else np.nan,
                     note=(f"one relation: no interval for a mean over relations; resampling only its questions gives [{lo:+.3f}, {hi:+.3f}]" if one else "questions resampled within relation and cell")))
T = pd.DataFrame(rows); save(T, "within_relation_by_min_cell.csv"); print(T.T.to_string(header=False))
assert [round(x, 3) for x in T.mean_margin] == [-0.040, 0.212, 0.240, 0.687] or [f"{x:.3f}" for x in T.mean_margin] == ["-0.040", "0.212", "0.240", "0.687"] , list(T.mean_margin)
assert list(T.n_relations) == [3, 6, 2, 1] and abs(T.mean_margin[0] - float(paper[1])) < 5e-4
print("R3 checks passed")
