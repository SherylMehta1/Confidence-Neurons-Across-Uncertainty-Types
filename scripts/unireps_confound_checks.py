"""
Two CPU-only checks requested before the UniReps 2026 extended abstract (no model, no GPU).

CHECK 1 -- is "familiarity" the ENTITY or the SURFACE FORM of its name?
  The popularity x knowledge factorial says hedging follows entity popularity (PopQA s_pop) while
  entropy follows knowledge. Two confounds a reviewer can raise:
    (a) token rarity: obscure names are spelled with rarer words / more sub-word pieces, and the
        model's correlation-selected "confidence neurons" are token-frequency neurons;
    (b) name-inferable answers: in pop_given_known the obscure-but-known items are mostly
        "capital of X District -> X", answerable from the name itself, so "known" may mean "guessable".
  Data: data/familiarity_v3/all_measured.jsonl (every scored item, HighlyKnown and Unknown only,
  Maybe excluded as in the factorial) plus the four factorial_v3 contrasts.
  Surface measures: wordfreq Zipf frequency of the name's words (mean, min); number of Llama-3.1
  tokens for " "+subject; mean log BPE id of those tokens (a merge-order proxy for token frequency).
  Models: binomial GLM of hedge counts (no-prefill free generations) and OLS of first-token entropy,
  relation fixed effects, standard errors clustered by relation.

CHECK 2 -- T1 from litreview/I_novelty_sweep_and_asymmetry.md section 2.4.
  Injection (uncertain_to_control) fails on known targets. Is that because known targets sit further
  from the hedge/answer tipping point (threshold account), or because a second input -- the retrieved
  answer -- vetoes the hedge (gate account)? Uses the committed behavioral.csv and faithfulness csvs.
  Target "margin" proxies come from the dataset records (top1 probability and entropy of the target
  prompt WITH the prefill, i.e. the prompt that was patched).

Run:  PYTHONPATH=<dir with wordfreq> python scripts/unireps_confound_checks.py
      (needs: numpy, pandas, statsmodels, tokenizers, wordfreq; a Llama-3.1 tokenizer.json via --tokenizer)
Writes results/unireps_checks/{check1_*,check2_*}.csv and summary.txt. Overwrites only that folder.
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import ast
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tokenizers import Tokenizer
from wordfreq import zipf_frequency

REPO = _Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "unireps_checks"
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def lit(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s)


def name_inferable(subj, aliases):
    """True when a gold alias can be read off the subject's own name (e.g. Belorechensky District -> Belorechensk)."""
    sn = norm(subj)
    swords = sn.split()
    for a in aliases:
        an = norm(a).strip()
        if len(an) < 4:
            continue
        if f" {an} " in f" {sn} ":
            return True
        first = an.split()[0] if an.split() else ""
        if len(first) >= 5 and any(w.startswith(first[:5]) for w in swords):
            return True
    return False


def surface(subj, tok):
    words = re.findall(r"[^\W\d_]+", subj)
    z = [zipf_frequency(w.lower(), "en") for w in words] or [0.0]
    ids = tok.encode(" " + subj, add_special_tokens=False).ids
    return dict(zipf_mean=float(np.mean(z)), zipf_min=float(np.min(z)), n_tok=len(ids),
                tok_logid=float(np.mean(np.log1p(ids))) if ids else 0.0,
                tok_per_word=len(ids) / max(1, len(words)))


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


# ----------------------------------------------------------------------------- check 1
def check1(tok, pool="familiarity_v3", fprefix="factorial_v3", tag=""):
    say("=" * 88)
    say("CHECK 1 -- entity popularity vs surface form of the name (familiarity_v3 pool, Llama-3.1-8B-Instruct)")
    say("=" * 88)
    rows = [json.loads(l) for l in open(REPO / "data" / pool / "all_measured.jsonl", encoding="utf8") if l.strip()]
    recs = []
    for r in rows:
        if r.get("slick") not in ("HighlyKnown", "Unknown") or r.get("s_pop") is None or not r.get("subj"):
            continue
        free = lit(r.get("free"))
        n_free = len(free) if free else 20
        aliases = lit(r.get("aliases")) or lit(r.get("possible_answers"))
        recs.append(dict(subj=r["subj"], rel=r["prop"], arm=r["arm"], known=int(r["slick"] == "HighlyKnown"),
                         logpop=math.log10(r["s_pop"] + 1), hedge_rate=r["hedge_rate"], n_free=n_free,
                         hedges=round(r["hedge_rate"] * n_free), entropy=r["entropy"], top1=r["top1"],
                         inferable=int(name_inferable(r["subj"], aliases)), **surface(r["subj"], tok)))
    d = pd.DataFrame(recs)
    d.to_csv(OUT / f"check1{tag}_items.csv", index=False)
    say(f"items: {len(d)} (HighlyKnown {d.known.sum()}, Unknown {(1 - d.known).sum()}); relations {d.rel.nunique()}; "
        f"free generations per item: {sorted(d.n_free.unique())}")
    say(f"name-inferable answers: {d.inferable.mean():.1%} of items "
        f"(known {d[d.known == 1].inferable.mean():.1%}, unknown {d[d.known == 0].inferable.mean():.1%})")

    say("\nSpearman correlations among predictors:")
    cols = ["logpop", "known", "zipf_mean", "zipf_min", "n_tok", "tok_logid", "inferable"]
    say(d[cols].corr(method="spearman").round(2).to_string())

    for c in ["logpop", "zipf_mean", "zipf_min", "n_tok", "tok_logid"]:
        d[c + "_z"] = z(d[c])
    surf = "zipf_mean_z + zipf_min_z + n_tok_z + tok_logid_z"
    specs = {
        "M1 popularity + knowledge": "logpop_z + known",
        "M2 + surface form of the name": f"logpop_z + known + {surf}",
        "M3 + name-inferable answer": f"logpop_z + known + {surf} + inferable",
        "M0 surface only (no popularity)": f"known + {surf} + inferable",
    }
    out = []
    for outcome in ["hedge", "entropy"]:
        say(f"\n--- outcome: {'hedge rate (binomial GLM on free-generation hedge counts)' if outcome == 'hedge' else 'first-token entropy (OLS)'};"
            " relation fixed effects; SE clustered by relation")
        for name, rhs in specs.items():
            f = f"{'hedge_rate' if outcome == 'hedge' else 'entropy'} ~ {rhs} + C(rel)"
            if outcome == "hedge":
                m = smf.glm(f, d, family=sm.families.Binomial(), var_weights=d.n_free).fit(
                    cov_type="cluster", cov_kwds={"groups": pd.factorize(d.rel)[0]})
                fit = f"deviance {m.deviance:.1f}"
            else:
                m = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.rel)[0]})
                fit = f"R2 {m.rsquared:.3f}"
            terms = [t for t in m.params.index if not t.startswith("C(rel)") and t != "Intercept"]
            say(f"  {name}  ({fit})")
            for t in terms:
                ci = m.conf_int().loc[t]
                say(f"     {t:<12} {m.params[t]:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={m.pvalues[t]:.2g}")
                out.append(dict(outcome=outcome, model=name, term=t, coef=m.params[t], lo=ci[0], hi=ci[1], p=m.pvalues[t]))
    pd.DataFrame(out).to_csv(OUT / f"check1{tag}_models.csv", index=False)

    say("\nNatural decoupling among KNOWN items: hedge rate by entity popularity x name word frequency (median splits)")
    k = d[d.known == 1].copy()
    k["pop"] = np.where(k.logpop >= k.logpop.median(), "popular", "obscure")
    k["name"] = np.where(k.zipf_mean >= k.zipf_mean.median(), "common-word name", "rare-word name")
    tab = k.groupby(["pop", "name"]).agg(n=("hedge_rate", "size"), hedge=("hedge_rate", "mean"), entropy=("entropy", "mean"))
    say(tab.round(3).to_string())
    tab.to_csv(OUT / f"check1{tag}_decoupling_known.csv")

    say("\nPair-level, factorial_v3 popularity contrasts (obscure twin minus popular twin):")
    pair_rows = []
    for contrast in ["pop_given_known", "pop_given_unknown"]:
        P = {r["twin_id"]: r for r in map(json.loads, open(REPO / f"data/{fprefix}_{contrast}/prompts.jsonl", encoding="utf8"))}
        C = {r["twin_id"]: r for r in map(json.loads, open(REPO / f"data/{fprefix}_{contrast}/controls.jsonl", encoding="utf8"))}
        recs2 = []
        for tid, u in P.items():
            c = C.get(tid)
            if c is None:
                continue
            su, sc = surface(u["subject"], tok), surface(c["subject"], tok)
            recs2.append(dict(contrast=contrast, twin_id=tid, rel=u["relation"],
                              d_hedge=u["hedge_rate"] - c["hedge_rate"], d_entropy=u["entropy"] - c["entropy"],
                              d_zipf=su["zipf_mean"] - sc["zipf_mean"], d_ntok=su["n_tok"] - sc["n_tok"],
                              d_logid=su["tok_logid"] - sc["tok_logid"],
                              inferable_u=int(name_inferable(u["subject"], u["gold"])),
                              inferable_c=int(name_inferable(c["subject"], c["gold"]))))
        p = pd.DataFrame(recs2)
        pair_rows.append(p)
        say(f"\n  {contrast}: {len(p)} pairs; relations: {p.rel.value_counts().head(4).to_dict()}")
        say(f"    raw hedge gap {p.d_hedge.mean():+.3f}; entropy gap {p.d_entropy.mean():+.3f}; "
            f"obscure twin name-inferable {p.inferable_u.mean():.0%}, popular twin {p.inferable_c.mean():.0%}")
        m = smf.ols("d_hedge ~ d_zipf + d_ntok + d_logid", p).fit(cov_type="HC3")
        say(f"    hedge gap at EQUAL surface form (intercept of d_hedge ~ d_zipf + d_ntok + d_logid): "
            f"{m.params['Intercept']:+.3f} [{m.conf_int().loc['Intercept'][0]:+.3f}, {m.conf_int().loc['Intercept'][1]:+.3f}]")
        ni = p[p.inferable_u == 0]
        if len(ni) >= 5:
            b = np.random.default_rng(0)
            boots = [ni.d_hedge.sample(len(ni), replace=True, random_state=b.integers(1 << 31)).mean() for _ in range(5000)]
            say(f"    excluding name-inferable obscure twins: n={len(ni)}, hedge gap {ni.d_hedge.mean():+.3f} "
                f"[{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}], entropy gap {ni.d_entropy.mean():+.3f}")
        else:
            say(f"    excluding name-inferable obscure twins: only {len(ni)} pairs left")
    pd.concat(pair_rows).to_csv(OUT / f"check1{tag}_pairs.csv", index=False)


# ----------------------------------------------------------------------------- check 2
ARMS = {  # results dir: (data category, faithfulness file)
    "circuit_familiarity_v3": ("familiarity_v3", "faithfulness_limit141.csv"),
    "circuit_factorial_v3_know_given_hi": ("factorial_v3_know_given_hi", "faithfulness.csv"),
    "circuit_factorial_v3_pop_given_unknown": ("factorial_v3_pop_given_unknown", "faithfulness.csv"),
    "circuit_uu_hedge_confab_v3": ("uu_hedge_confab_v3", "faithfulness.csv"),
}
HEDGE_START = re.compile(r"^\W*(i\b|i'm|i am|unfortunately|sorry|not\b|unknown|there is no|i don|i can|i could|cannot|unable)", re.I)


def load_pairs(category, limit=141, split="held_out"):
    """Mirror of circuit_common.load_pairs without importing torch: pair by twin_id, filter split, take limit."""
    P = [json.loads(l) for l in open(REPO / f"data/{category}/prompts.jsonl", encoding="utf8")]
    C = [json.loads(l) for l in open(REPO / f"data/{category}/controls.jsonl", encoding="utf8")]
    by = {c["twin_id"]: c for c in C}
    pairs = [(p, by[p["twin_id"]]) for p in P if p["twin_id"] in by]
    pairs = [(u, c) for u, c in pairs if u.get("split") == split]
    return pairs[:limit]


def check2():
    say("\n" + "=" * 88)
    say("CHECK 2 -- T1: why does injecting the circuit state fail on known targets? (uncertain_to_control)")
    say("=" * 88)
    allrows = []
    for rdir, (cat, ffile) in ARMS.items():
        pairs = load_pairs(cat)
        b = pd.read_csv(REPO / "results" / rdir / "behavioral.csv")
        f = pd.read_csv(REPO / "results" / rdir / ffile)
        bc = b[b.set == "circuit"]
        fc = f[(f.set == "circuit") & (f.direction == "uncertain_to_control")].set_index("pair")
        inj = bc[bc.direction == "uncertain_to_control"].set_index("pair")
        rem = bc[bc.direction == "control_to_uncertain"].set_index("pair")
        assert inj.index.max() < len(pairs), (rdir, inj.index.max(), len(pairs))
        for i, (u, c) in enumerate(pairs):
            if i not in inj.index:
                continue
            tgt_h, src_h = bool(inj.loc[i, "hedged_clean"]), bool(rem.loc[i, "hedged_clean"])
            allrows.append(dict(arm=rdir.replace("circuit_", ""), pair=i, discordant=tgt_h != src_h,
                                flip=bool(inj.loc[i, "hedged_patched"]) == src_h,
                                tgt_known=int(c.get("frac_correct", 0) >= 0.9), tgt_frac_correct=c.get("frac_correct"),
                                tgt_top1=c.get("top1"), tgt_entropy=c.get("entropy"),
                                src_top1=u.get("top1"), src_entropy=u.get("entropy"),
                                logodds_rec=fc.loc[i, "logodds_rec"] if i in fc.index else np.nan,
                                gen_patched=str(inj.loc[i, "gen_patched"]), gen_clean=str(inj.loc[i, "gen_clean"])))
    d = pd.DataFrame(allrows)
    d.to_csv(OUT / "check2_injection_pairs.csv", index=False)
    dd = d[d.discordant].copy()
    say("Discordant injection pairs (target twin does not hedge, source twin does):")
    g = dd.groupby("arm").agg(n=("flip", "size"), flip=("flip", "mean"), target_known=("tgt_known", "mean"),
                              tgt_top1=("tgt_top1", "mean"), tgt_entropy=("tgt_entropy", "mean"),
                              first_token_recovery=("logodds_rec", "median"))
    say(g.round(3).to_string())

    say("\n(a) Does the target's distance from the tipping point predict failure WITHIN the all-known familiarity arm?")
    fam = dd[dd.arm == "familiarity_v3"].copy()
    for v in ["tgt_top1", "tgt_entropy", "logodds_rec"]:
        x = fam[[v, "flip"]].dropna()
        m = smf.logit(f"flip ~ {v}", x.assign(flip=x.flip.astype(int))).fit(disp=0)
        say(f"    flip ~ {v:<12} n={len(x)}  coef {m.params[v]:+.3f}  p={m.pvalues[v]:.2g}   "
            f"mean {v}: flipped {x[x.flip][v].mean():.3f} vs not {x[~x.flip][v].mean():.3f}")

    say("\n(b) Across arms: does target knowledge survive controlling for the target's top-1 probability and entropy?")
    x = dd.dropna(subset=["tgt_top1", "tgt_entropy"]).assign(flip=lambda t: t.flip.astype(int))
    for f in ["flip ~ tgt_known", "flip ~ tgt_top1 + tgt_entropy", "flip ~ tgt_known + tgt_top1 + tgt_entropy"]:
        m = smf.logit(f, x).fit(disp=0, cov_type="HC1")
        say(f"    {f:<45} " + "  ".join(f"{k} {m.params[k]:+.2f} (p={m.pvalues[k]:.2g})" for k in m.params.index if k != "Intercept")
            + f"   pseudo-R2 {m.prsquared:.3f}")
    say("    caveat: knowledge is constant within each arm, so (b) leans on 4 arms; read it with (a).")
    say("    target top-1 probability by knowledge: known "
        f"{x[x.tgt_known == 1].tgt_top1.mean():.3f} vs unknown {x[x.tgt_known == 0].tgt_top1.mean():.3f}")

    say("\n(c) Was the decision point moved even when generation did not flip? first-token log-odds recovery (median, IQR):")
    for (arm, fl), s in dd.groupby(["arm", "flip"]).logodds_rec:
        say(f"    {arm:<32} flipped={str(fl):<5} n={len(s):>3}  median {s.median():+.2f}  IQR [{s.quantile(.25):+.2f}, {s.quantile(.75):+.2f}]")

    say("\n(d) How do failed injections on known targets start? (first 60 chars of patched generation)")
    fail = dd[(~dd.flip) & (dd.tgt_known == 1)]
    starts_hedge = fail.gen_patched.str.match(HEDGE_START).mean() if len(fail) else float("nan")
    say(f"    {len(fail)} failures; share whose patched text OPENS with hedge-like words: {starts_hedge:.1%}")
    for s in fail.gen_patched.head(12):
        say(f"      | {s[:60]!r}")


def check3():
    """Sensitivity of the L5 behavioral flip rates to a gap in behavioral_test.HEDGE_RE.

    HEDGE_RE matches "not available" / "not clear" / "not known" only with nothing in between, so
    "not readily available in my database" is scored as an ANSWER. The locked analysis (59f5ec5) is
    unchanged; this reports what the amended detector gives, as a sensitivity row beside it."""
    say("\n" + "=" * 88)
    say("CHECK 3 -- behavioral flip rates under the original vs amended hedge detector (sensitivity only)")
    say("=" * 88)
    src = (REPO / "scripts/behavioral_test.py").read_text(encoding="utf8")
    block = src[src.index("HEDGE_RE = re.compile("):]
    ns = {}
    exec("import re\n" + block[:block.index("\n)\n") + 3], ns)
    h0 = ns["HEDGE_RE"]
    gap = re.compile(r"\bnot (?:readily|immediately|currently|publicly|entirely|widely) (?:available|clear|known)\b", re.I)
    detectors = {"original": lambda s: bool(h0.search(str(s))),
                 "amended": lambda s: bool(h0.search(str(s)) or gap.search(str(s)))}
    rng = np.random.default_rng(0)
    rows = []
    for rdir in ARMS:
        b = pd.read_csv(REPO / "results" / rdir / "behavioral.csv")
        repro = (b.gen_patched.map(detectors["original"]) == b.hedged_patched).mean()
        say(f"{rdir.replace('circuit_', '')}  (original detector reproduces stored labels: {repro:.3f})")
        for name, h in detectors.items():
            b["hc"], b["hp"] = b.gen_clean.map(h), b.gen_patched.map(h)
            c = b[b.set == "circuit"]
            clean = {dn: c[c.direction == dn].set_index("pair").hc for dn in ("control_to_uncertain", "uncertain_to_control")}
            parts = []
            for dn, other, label in [("control_to_uncertain", "uncertain_to_control", "removal"),
                                     ("uncertain_to_control", "control_to_uncertain", "injection")]:
                g = c[c.direction == dn].copy()
                g["tgt"], g["src"] = g.pair.map(clean[dn]), g.pair.map(clean[other])
                g = g[g.tgt != g.src]
                arr = (g.hp == g.src).values
                bs = [rng.choice(arr, len(arr)).mean() for _ in range(5000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                parts.append(f"{label} {arr.mean():.3f} [{lo:.3f}, {hi:.3f}] n={len(arr)}")
                rows.append(dict(arm=rdir, detector=name, direction=label, flip=arr.mean(), lo=lo, hi=hi, n=len(arr)))
            say(f"    {name:<9} " + " ; ".join(parts))
    pd.DataFrame(rows).to_csv(OUT / "check3_detector_sensitivity.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to the tokenizer.json of the model that produced --pool")
    ap.add_argument("--pool", default="familiarity_v3", help="data/<pool>/all_measured.jsonl for check 1")
    ap.add_argument("--factorial-prefix", default="factorial_v3", help="data/<prefix>_<contrast>/ for check 1 pairs")
    ap.add_argument("--check1-only", action="store_true", help="run check 1 only (e.g. on a second model's pool)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    tag = "" if args.pool == "familiarity_v3" else f"_{args.pool}"
    say(f"pool: {args.pool}; tokenizer: {args.tokenizer}")
    check1(tok, args.pool, args.factorial_prefix, tag)
    if not args.check1_only:
        check2()
        check3()
    (OUT / f"summary{tag}.txt").write_text("\n".join(LINES), encoding="utf8")


if __name__ == "__main__":
    main()
