"""R0 follow-up: is [0.203, 0.398] against [0.204, 0.397] only Monte Carlo noise? Same data, same method (questions within cell),
50 seeds x 10,000 draws and one run of 1,000,000 draws. Diagnostic only; no reported interval is changed."""
from x0_common import *

Q = pd.read_csv(OUT / "tidy_questions.csv", keep_default_na=False)


def run(seed, size):
    rng = np.random.default_rng(seed)
    d = boot_mean(Q[Q.cell == "OK"].both, rng, size) - boot_mean(Q[Q.cell == "PU"].both, rng, size)
    return np.percentile(d, [2.5, 97.5])


x = np.array([run(s, B) for s in range(50)])
big = run(0, 1_000_000)
T = pd.DataFrame([dict(setting="50 seeds, 10,000 draws each", lo_min=x[:, 0].min(), lo_max=x[:, 0].max(), hi_min=x[:, 1].min(), hi_max=x[:, 1].max()),
                  dict(setting="seed 0, 1,000,000 draws", lo_min=big[0], lo_max=big[0], hi_min=big[1], hi_max=big[1])])
save(T, "r0_margin_montecarlo.csv")
print(T.round(4).to_string(index=False))
