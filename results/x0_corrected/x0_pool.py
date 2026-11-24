"""X0 A7: does regex abstention follow popularity more than entropy? Full labelled pool (2,303), then the 387 questions."""
from x0_common import *

log = Log("x0_pool_log.txt")
pool = pd.read_csv(OUT / "tidy_pool.csv", keep_default_na=False)
pool = pool[pool.cell.isin(CELLS)].reset_index(drop=True)
Q = pd.read_csv(OUT / "tidy_questions.csv", keep_default_na=False)
rows = []


def fit(df, y, controls, label, sample):
    """OLS of y (0-1 scale) on z-scored log views and entropy (+ controls). Bootstrap over questions: multinomial weights,
    regressors z-scored once on the sample."""
    z = lambda x: (x - x.mean()) / x.std(ddof=0)
    Y = df[y].to_numpy(float)
    X = [np.ones(len(df)), z(df.log_views.to_numpy(float)), z(df.entropy.to_numpy(float))]
    if "knowledge" in controls:
        X.append(df.cell.str[1].eq("U").to_numpy(float))
    if "relation" in controls:
        d = pd.get_dummies(df.prop, drop_first=True).to_numpy(float)
        X += list(d.T)
    X = np.column_stack(X)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    rng = np.random.default_rng(SEED)
    bs = np.empty((B, 2))
    for i in range(B):
        w = np.bincount(rng.integers(0, len(Y), len(Y)), minlength=len(Y))
        Xw = X * w[:, None]
        bs[i] = np.linalg.lstsq(Xw.T @ X, Xw.T @ Y, rcond=None)[0][1:3]
    # relation resampling: sums of X'X and X'y per relation, weighted by how often each relation is drawn
    pr = sorted(df.prop.unique())
    XX = np.array([X[(df.prop == p).to_numpy()].T @ X[(df.prop == p).to_numpy()] for p in pr])
    XY = np.array([X[(df.prop == p).to_numpy()].T @ Y[(df.prop == p).to_numpy()] for p in pr])
    W = np.random.default_rng(SEED).multinomial(len(pr), np.full(len(pr), 1 / len(pr)), size=B)
    br = np.array([np.linalg.lstsq(np.tensordot(w, XX, 1), w @ XY, rcond=None)[0][1:3] for w in W])
    dr = np.abs(br[:, 0]) - np.abs(br[:, 1])
    # partial correlations given the other regressors
    pc = []
    for j in (1, 2):
        o = np.delete(X, j, 1)
        rx = X[:, j] - o @ np.linalg.lstsq(o, X[:, j], rcond=None)[0]
        ry = Y - o @ np.linalg.lstsq(o, Y, rcond=None)[0]
        pc.append(np.corrcoef(rx, ry)[0, 1])
    d = np.abs(bs[:, 0]) - np.abs(bs[:, 1])
    r = dict(sample=sample, outcome=y, model=label, n=len(df), b_views=beta[1], views_lo=ci(bs[:, 0])[0], views_hi=ci(bs[:, 0])[1], b_entropy=beta[2],
             entropy_lo=ci(bs[:, 1])[0], entropy_hi=ci(bs[:, 1])[1], abs_views_minus_abs_entropy=abs(beta[1]) - abs(beta[2]), diff_lo=ci(d)[0], diff_hi=ci(d)[1], diff_rel_lo=ci(dr)[0], diff_rel_hi=ci(dr)[1],
             beta_views_y_standardised=beta[1] / Y.std(), beta_entropy_y_standardised=beta[2] / Y.std(), partial_r_views=pc[0], partial_r_entropy=pc[1],
             corr_views_entropy=np.corrcoef(X[:, 1], X[:, 2])[0, 1])
    rows.append(r)
    log(f"{sample:<34} {y:<8} {label:<34} n {len(df):4d} | views {fmt(beta[1], *ci(bs[:, 0]))} entropy {fmt(beta[2], *ci(bs[:, 1]))} | |views|-|entropy| "
        f"{fmt(r['abs_views_minus_abs_entropy'], *ci(d))} rel [{ci(dr)[0]:+.3f}, {ci(dr)[1]:+.3f}] | partial r {pc[0]:+.3f} {pc[1]:+.3f}")


for nm, d in (("all labelled questions", pool), ("known only", pool[pool.cell.str[1] == "K"]), ("unknown only", pool[pool.cell.str[1] == "U"]),
              ("obscure only", pool[pool.cell.str[0] == "O"]), ("popular only", pool[pool.cell.str[0] == "P"])):
    for c in ((), ("relation",)) + ((("knowledge",),) if nm.startswith("all") else ()) + ((("knowledge", "relation"),) if nm.startswith("all") else ()):
        fit(d, "regex24", c, "views + entropy" + "".join(f" + {x}" for x in c), nm)
for y in ("both", "regex96"):
    fit(Q, y, (), "views + entropy", "387 questions")
    fit(Q, y, ("relation",), "views + entropy + relation", "387 questions")
save(pd.DataFrame(rows), "A7_popularity_vs_entropy.csv")
log("Coefficients: change in the abstention rate per standard deviation of the regressor. The arms are the extreme deciles of page views, so the")
log("spread of log views (and hence its per-SD coefficient) is set by the design; entropy is not sampled that way.")
log.save()
