"""R1: known / unknown counts at 8, 9 and 10 of 10. Counted straight from the raw file with the repo's grader, not from tidy tables.
Known: greedy answer correct and at least t of 10 samples correct. Unknown: no sample correct and greedy wrong. Main row: 9 of 10."""
from x0_common import *

rows, dropped = [], 0
for r in read(POOL):
    if not r.get("subj"):
        dropped += 1
        continue
    al = pool_aliases(r)
    assert len(r["samples"]) == 10
    rows.append(("obscure" if r["arm"] == "uncertain" else "popular", sum(is_correct(s, al) for s in r["samples"]), is_correct(r["greedy"], al)))
D = pd.DataFrame(rows, columns=["arm", "n", "greedy"])
out = []
for t in (9, 8, 10):
    k, u = (D.n >= t) & D.greedy, (D.n == 0) & ~D.greedy
    c = lambda m, a: int((m & (D.arm == a)).sum())
    out.append(dict(cutoff=f"{t} of 10", role="main (paper's rule)" if t == 9 else "extra context", obscure_known=c(k, "obscure"), obscure_unknown=c(u, "obscure"), popular_known=c(k, "popular"),
                    popular_unknown=c(u, "popular"), known_total=int(k.sum()), unknown_total=int(u.sum()), labelled_total=int((k | u).sum()), dropped_band=int((~(k | u)).sum()), rows_with_empty_subject_excluded=dropped))
T = pd.DataFrame(out); save(T, "counts_by_threshold.csv"); print(T.T.to_string(header=False))
m = T.iloc[0]
assert list(T.known_total) == [763, 801, 707] and (T.unknown_total == 1540).all()
assert (m.obscure_known, m.obscure_unknown, m.popular_known, m.popular_unknown, m.labelled_total) == (87, 1179, 676, 361, 2303)
print("R1 checks passed")
