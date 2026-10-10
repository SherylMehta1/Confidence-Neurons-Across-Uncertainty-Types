"""Cross-model agreement of the abstention readout vs answer entropy (plan: ../PREREG/cross_model_agreement.md; exploratory).
Reads the repo's per-item cross-model files (read-only) and writes analysis/cross_model_agreement.{csv,txt}.
Spearman rho = Pearson correlation of ranks; ranks are computed once on the full subset (bootstrap resamples reuse them,
a standard approximation), 10,000 item bootstraps and 2,000 within-relation permutations, seed 0."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
R = next(p for p in (HERE.parents[1], HERE.parents[1] / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir()) / "results" / "factorial_4models"
PAIRS = {"qwen": "cross_model_items.csv", "gemma": "cross_model_items_gemma.csv", "mistral": "cross_model_items_mistral.csv"}
READOUTS = ["hedge_logodds", "hedge_rate", "entropy"]
B, P = 10_000, 2_000


def corr_rows(X, Y):
    """Row-wise Pearson correlation of matrices X, Y (each row one resample)."""
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    den = np.sqrt((Xc ** 2).sum(1) * (Yc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc * Yc).sum(1) / den


def rho(a, b):
    return float(corr_rows(rankdata(a)[None], rankdata(b)[None])[0])


def main():
    rows, lines = [], []
    for other, f in PAIRS.items():
        d = pd.read_csv(R / f).dropna(subset=[f"{r}_{s}" for r in READOUTS for s in "lq"]).reset_index(drop=True)
        for subset in ("all", "same knowledge", "knowledge differs"):
            sel = d if subset == "all" else d[d.agree == subset].reset_index(drop=True)
            n = len(sel)
            rng = np.random.default_rng(0)
            idx = rng.integers(0, n, (B, n))
            groups = [np.flatnonzero(sel.prop.to_numpy() == g) for g in sel.prop.unique()]
            prng = np.random.default_rng(0)
            perms = np.tile(np.arange(n), (P, 1))
            for p in range(P):
                for g in groups:
                    perms[p, g] = prng.permutation(g)
            ranks = {}
            for r in READOUTS:
                a, b = rankdata(sel[f"{r}_l"]), rankdata(sel[f"{r}_q"])
                ranks[r] = (a, b)
                pt = rho(sel[f"{r}_l"], sel[f"{r}_q"])
                bs = corr_rows(a[idx], b[idx])
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                null = corr_rows(np.tile(a, (P, 1)), b[perms])
                pval = (np.sum(np.abs(null) >= abs(pt)) + 1) / (P + 1)
                within = {arm: rho(g[f"{r}_l"], g[f"{r}_q"]) if len(g) > 2 else np.nan for arm, g in sel.groupby("arm_l")}
                rows.append(dict(pair=f"llama-{other}", subset=subset, readout=r, n=n, rho=pt, lo=lo, hi=hi, perm_p=pval,
                                 rho_obscure=within.get("uncertain"), rho_popular=within.get("control")))
            if subset == "knowledge differs":
                (ha, hb), (ea, eb) = ranks["hedge_logodds"], ranks["entropy"]
                delta = corr_rows(ha[idx], hb[idx]) - corr_rows(ea[idx], eb[idx])
                pt = rho(sel.hedge_logodds_l, sel.hedge_logodds_q) - rho(sel.entropy_l, sel.entropy_q)
                lo, hi = np.nanpercentile(delta, [2.5, 97.5])
                h = rows[-3]
                holds = h["lo"] > 0 and lo > 0
                lines.append(f"llama-{other}: knowledge-differs n={n}  rho(hedge_logodds) {h['rho']:+.2f} [{h['lo']:+.2f},{h['hi']:+.2f}]  "
                             f"rho(entropy) {rows[-1]['rho']:+.2f} [{rows[-1]['lo']:+.2f},{rows[-1]['hi']:+.2f}]  "
                             f"Delta {pt:+.2f} [{lo:+.2f},{hi:+.2f}]  -> {'PREDICTION HOLDS' if holds else 'NOT SUPPORTED'}")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "cross_model_agreement.csv", index=False)
    txt = out.to_string(index=False, float_format=lambda v: f"{v:+.3f}") + "\n\n" + "\n".join(lines) + "\n"
    (HERE / "cross_model_agreement.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
