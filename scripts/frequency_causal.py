"""
Causal test for token-frequency neurons (Stolfo et al. 2024, second class).

A token-frequency neuron shifts the output distribution toward (or away from) the
unigram distribution u. We clamp each neuron at mean + k*sigma for k in --sigma-levels
(k=0 is in-distribution mean-ablation) at the last position, and measure on every
prompt how the next-token distribution p moves relative to u:

  elogu      = E_p[log u]          mass on frequent tokens (up = toward unigram)
  kl_unigram = KL(p || u)          distance from the unigram distribution
  entropy    = H(p)

Per neuron we report the dose-response slope of each metric per sigma (mean over
prompts of the per-prompt least-squares slope across levels), the mean-ablation
delta, sign-flip permutation p-values, and -- the actual test -- whether the
WEIGHT-SPACE frequency score (results/token_frequency_neurons.csv, freq_corr) predicts
the CAUSAL slope across neurons (Pearson r over neurons; enrichment of
FREQ-NEURON-LIKE candidates vs random control neurons).

Usage (from any CWD):
  python scripts/frequency_causal.py --candidates results/candidate_neurons_bf16.json,candidate_neurons.json,results/candidates_old15.json \
      --control-neurons 10 --categories lack_of_knowledge --out results/frequency_causal.csv
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import glob
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats

from _common import (REPO_ROOT, add_model_args, guard_output, load_category, load_model_from_args,
                     parse_categories, parse_neurons)
from behavioral_test import activation_stats
from shared.ablation import activation_sweep_and_get_probs, get_probs_and_activation
from shared.detection import load_candidate_neurons
from shared.provenance import build_provenance, write_provenance



def align_logfreq(logfreq, vocab_rows):
    """Align a unigram log-frequency vector to the unembedding's row count: truncate if the
    tokenizer has more entries than the matrix, pad with the minimum (unseen-token) value if the
    matrix is larger (padded vocab, e.g. Qwen: 152064 rows vs 151665 tokens)."""
    import numpy as _np
    lf = _np.asarray(logfreq, dtype=_np.float64)
    if len(lf) >= vocab_rows:
        return lf[:vocab_rows]
    return _np.concatenate([lf, _np.full(vocab_rows - len(lf), lf.min())])

def dist_metrics(p, logu):
    """p: fp32 probs [V]; logu: log unigram [V] (same device). Returns (entropy, elogu, kl)."""
    p = p.float()
    logp = torch.log(p.clamp_min(1e-30))
    ent = -(torch.xlogy(p, p)).sum().item()
    elogu = (p * logu).sum().item()
    kl = (p * (logp - logu)).sum().item()
    return ent, elogu, kl


def temp_matched_elogu(p_clean, target_entropy, logu, iters=40):
    """E_{p_T}[log u] for the temperature-scaled CLEAN distribution p_T = softmax(log p_clean / T)
    whose entropy equals target_entropy (bisection on log T). This is what a pure
    temperature change of the same size would do to the frequency readout; the
    frequency-specific effect is elogu_clamped - this."""
    logp = torch.log(p_clean.float().clamp_min(1e-30))
    lo, hi = -4.0, 4.0  # log T in [e^-4, e^4]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pt = torch.softmax(logp / float(np.exp(mid)), dim=-1)
        ent = -(torch.xlogy(pt, pt)).sum().item()
        if ent < target_entropy:
            lo = mid  # need higher T (more entropy)
        else:
            hi = mid
    pt = torch.softmax(logp / float(np.exp(0.5 * (lo + hi))), dim=-1)
    return (pt * logu).sum().item(), float(np.exp(0.5 * (lo + hi)))


def sweep_metrics(model, tokenizer, prompt, layer, idx, values, logu):
    probs = activation_sweep_and_get_probs(model, tokenizer, prompt, layer, idx, values)
    return [dist_metrics(p, logu) for p in probs]


def per_prompt_slopes(g, levels_col, metric_col):
    """least-squares slope of metric on sigma level, one per prompt"""
    out = []
    for _, pp in g.groupby("prompt_id"):
        if pp[levels_col].nunique() > 1:
            out.append(np.polyfit(pp[levels_col].to_numpy(float), pp[metric_col].to_numpy(float), 1)[0])
    return np.array(out)


def sign_flip_p(x, rng, n=5000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan
    obs = abs(x.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(x)))
    return float(((np.abs((signs * x).mean(1)) >= obs).sum() + 1) / (n + 1))


def summarize(df, freq_scores=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for nid, g in df.groupby("neuron_id"):
        u = g[~g.is_control]
        r = dict(neuron_id=nid, is_candidate=bool(g.is_candidate.iloc[0]), n_prompts=u.prompt_id.nunique())
        for m in ("d_elogu", "d_elogu_beyond_temp", "d_kl", "d_entropy"):
            if m not in u:
                continue
            s = per_prompt_slopes(u, "sigma_level", m)
            r[f"{m}_slope_per_sigma"] = s.mean() if len(s) else np.nan
            r[f"{m}_slope_p"] = sign_flip_p(s, rng) if len(s) else np.nan
            z = u[u.sigma_level == 0][m]
            r[f"{m}_mean_ablation"] = z.mean() if len(z) else np.nan
            r[f"{m}_mean_ablation_p"] = sign_flip_p(z, rng) if len(z) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    if freq_scores is not None:
        t = t.merge(freq_scores, on="neuron_id", how="left")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--candidates", default="results/candidate_neurons_bf16.json,candidate_neurons.json,results/candidates_old15.json")
    ap.add_argument("--neurons", default=None, help="explicit comma list instead of --candidates")
    ap.add_argument("--control-neurons", type=int, default=10)
    ap.add_argument("--categories", default="lack_of_knowledge")
    ap.add_argument("--sigma-levels", default="-2,0,2")
    ap.add_argument("--unigram", default=None, help="path to unigram_logfreq_*.npy (default: newest in results/)")
    ap.add_argument("--freq-scores", default="results/token_frequency_neurons.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/frequency_causal.csv")
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    guard_output(out, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    uni = args.unigram or sorted(glob.glob(str(REPO_ROOT / "results/unigram_logfreq_*.npy")))[-1]
    logu_np = np.load(uni)
    vocab_rows = model.get_output_embeddings().weight.shape[0]
    logu = torch.tensor(align_logfreq(logu_np, vocab_rows), dtype=torch.float32, device=model.device)
    levels = [float(x) for x in args.sigma_levels.split(",")]

    if args.neurons:
        neurons = parse_neurons(args.neurons)
    else:
        seen = {}
        for f in args.candidates.split(","):
            for c in load_candidate_neurons(REPO_ROOT / f.strip()):
                seen[(c["layer"], c["neuron_idx"])] = True
        neurons = sorted(seen)
    rng = random.Random(args.seed)
    layers = sorted({l for l, _ in neurons})
    controls = []
    taken = set(neurons)
    while len(controls) < args.control_neurons:
        p = (rng.choice(layers), rng.randrange(model.config.intermediate_size))
        if p not in taken:
            taken.add(p)
            controls.append(p)
    all_neurons = neurons + controls
    cand_ids = {f"L{l}_N{n}" for l, n in neurons}

    rows = []
    for cat in parse_categories(args.categories):
        prompts, ctrls = load_category(cat)
        recs = [dict(r, is_control=False) for r in prompts] + [dict(r, is_control=True) for r in ctrls]
        if args.limit:
            recs = [r for r in recs if not r["is_control"]][: args.limit] + [r for r in recs if r["is_control"]][: args.limit]
        pooled = [r["chat_formatted_prompt"] for r in recs if r["split"] == "working"]
        print(f"[{cat}] {len(recs)} prompts, {len(all_neurons)} neurons ({len(controls)} random controls), levels {levels}")
        st = activation_stats(model, tokenizer, pooled, all_neurons)
        clean, clean_probs = {}, {}
        for r in recs:
            p, _ = get_probs_and_activation(model, tokenizer, r["chat_formatted_prompt"], all_neurons[0][0], all_neurons[0][1])
            clean[r["prompt_id"]] = dist_metrics(p, logu)
            clean_probs[r["prompt_id"]] = p.float()
        for j, (l, n) in enumerate(all_neurons):
            nid = f"L{l}_N{n}"
            mu, sd = st[(l, n)]
            values = [mu + k * sd for k in levels]
            for r in recs:
                ms = sweep_metrics(model, tokenizer, r["chat_formatted_prompt"], l, n, values, logu)
                e0, u0, k0 = clean[r["prompt_id"]]
                for k, v, (e, eu, kl) in zip(levels, values, ms):
                    eu_t, T = temp_matched_elogu(clean_probs[r["prompt_id"]], e, logu)
                    rows.append(dict(neuron_id=nid, is_candidate=nid in cand_ids, category=cat, prompt_id=r["prompt_id"],
                                     split=r["split"], is_control=r["is_control"], sigma_level=k, clamp_value=v,
                                     baseline_mean=mu, baseline_sigma=sd, entropy=e, elogu=eu, kl_unigram=kl,
                                     d_entropy=e - e0, d_elogu=eu - u0, d_kl=kl - k0,
                                     elogu_tempmatched=eu_t, matched_T=T, d_elogu_beyond_temp=eu - eu_t))
            if (j + 1) % 5 == 0 or j == len(all_neurons) - 1:
                print(f"  {j + 1}/{len(all_neurons)} neurons done")
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    fs = None
    fsp = REPO_ROOT / args.freq_scores
    if fsp.exists():
        f = pd.read_csv(fsp)
        fs = f[f.kind == "candidate"][["neuron_id", "freq_corr", "abs_freq_corr_pctile"]]
    summ = summarize(df, fs, seed=args.seed)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)

    lines = ["Frequency-neuron causal test", f"unigram: {uni}", ""]
    c = summ[summ.is_candidate]
    k = summ[~summ.is_candidate]
    if "freq_corr" in c and c.freq_corr.notna().sum() > 3:
        cc = c.dropna(subset=["freq_corr"])
        for m, label in (("d_elogu_slope_per_sigma", "raw d_elogu"), ("d_elogu_beyond_temp_slope_per_sigma", "d_elogu BEYOND temperature-matched baseline")):
            if m not in cc:
                continue
            r_, p_ = stats.pearsonr(cc.freq_corr, cc[m])
            rs, ps = stats.spearmanr(cc.freq_corr, cc[m])
            sa = (np.sign(cc[cc.abs_freq_corr_pctile >= 99].freq_corr) == np.sign(cc[cc.abs_freq_corr_pctile >= 99][m])).mean()
            lines.append(f"Weight-space freq_corr vs causal slope [{label}] across {len(cc)} candidates: Pearson r={r_:.3f} (p={p_:.2g}), Spearman rho={rs:.3f} (p={ps:.2g}); sign agreement among FREQ-NEURON-LIKE {sa:.2f}")
        fl = cc[cc.abs_freq_corr_pctile >= 99]
        nf = cc[cc.abs_freq_corr_pctile < 95]
        lines.append(f"|causal elogu slope| per sigma: FREQ-NEURON-LIKE (n={len(fl)}) {fl.d_elogu_slope_per_sigma.abs().mean():.4f} | "
                     f"other candidates (n={len(nf)}) {nf.d_elogu_slope_per_sigma.abs().mean():.4f} | random controls (n={len(k)}) {k.d_elogu_slope_per_sigma.abs().mean():.4f}")
        sign_ok = (np.sign(fl.freq_corr) == np.sign(fl.d_elogu_slope_per_sigma)).mean() if len(fl) else np.nan
        lines.append(f"sign agreement (weight score vs causal slope) among FREQ-NEURON-LIKE: {sign_ok:.2f}")
    lines.append("")
    cols = ["neuron_id", "is_candidate", "freq_corr", "abs_freq_corr_pctile", "d_elogu_beyond_temp_slope_per_sigma", "d_elogu_beyond_temp_slope_p", "d_elogu_slope_per_sigma", "d_elogu_slope_p",
            "d_kl_slope_per_sigma", "d_kl_slope_p", "d_entropy_slope_per_sigma", "d_elogu_mean_ablation", "d_elogu_mean_ablation_p"]
    cols = [x for x in cols if x in summ]
    pd.set_option("display.width", 250)
    sort_key = "d_elogu_beyond_temp_slope_per_sigma" if "d_elogu_beyond_temp_slope_per_sigma" in summ else "d_elogu_slope_per_sigma"
    lines.append(summ.sort_values(sort_key, key=lambda s: -s.abs())[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    text = "\n".join(lines)
    print(text)
    out.with_name(out.stem + "_summary.txt").write_text(text + "\n")
    write_provenance(out, build_provenance(model, script="scripts/frequency_causal.py", unigram=str(uni), sigma_levels=levels,
                                           candidates=args.candidates, neurons=sorted(cand_ids),
                                           control_neurons=[f"L{l}_N{n}" for l, n in controls], seed=args.seed,
                                           categories=args.categories, limit=args.limit))
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
