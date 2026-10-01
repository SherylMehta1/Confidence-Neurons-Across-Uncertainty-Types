#!/usr/bin/env python3
"""
power_analysis_uu.py

Power / minimum-detectable-effect (MDE) calculator for the U/U (hedger vs.
confabulator) falsification test named in the pre-registration commit
(preregistration_commit.md, point C).

WHY THIS EXISTS
----------------
The pre-registered falsification rule is: the confabulator circuit must clear
`recovery > 0.7` in BOTH directions (removal and injection) on the held-out
`uu_hedge_confab_v3` pairs (N = 23) to count as confirmed.

Before running that test against the held-out pairs, this script answers two
questions:

  1. POWER:  if the confabulator circuit's true recovery sits some distance
     above or below the 0.7 bar, what's the probability that a one-shot test
     on N=23 pairs actually detects that -- as opposed to missing it by chance
     because the sample is small?
  2. MDE:    for a target power (e.g. 80%), how far from 0.7 does the true
     recovery need to be for this test to be capable of noticing at all?

This matters because N=23 is small and fixed (it's whatever's left in the
held-out slice -- not something to casually grow after the fact). Running
this BEFORE the real test is what lets a doomed-from-the-start test be caught
and fixed (widen the pool, or downgrade the claim to exploratory) instead of
being discovered only after an ambiguous null result.

TWO MODES
---------
--mode bound   (default, matches the committed criterion directly)
    One-sample test: is the mean recovery over N held-out pairs above the
    0.7 bound? This is what the pre-registration commit actually specifies.

--mode diff
    Two-sample test: does the confabulator-circuit recovery differ from the
    hedger-circuit recovery (or from some other reference circuit) by at
    least a given margin? Use this if the falsification question ever gets
    reframed as a comparison between two measured circuits rather than a
    single circuit against a fixed bound.

TWO RULES (--mode bound only) -- READ THIS BEFORE USING THE OUTPUT
--------------------------------------------------------------------
--rule point  (default -- this is what the pre-registration commit locks)
    The commit's criterion is a bare point-estimate crossing: does the
    sample mean recovery exceed 0.7? No significance test, no alpha, no
    confidence interval -- the same convention circuit_faithfulness.py
    already uses elsewhere (a bare mean() > 0.7). "Power" here means
    P(sample mean > bound | true mean), computed from the central
    t-distribution of the sample mean. There is no alpha in this mode.

--rule ci
    An EARLIER VERSION OF THIS SCRIPT computed power as a one-sided t-TEST
    at level alpha (critical value t_(1-alpha, df)). That is algebraically
    identical to requiring the one-sided (1-alpha) CI lower bound to exceed
    the bound -- i.e. it silently implemented the stricter rule the
    pre-registration commit explicitly rejects (see commit point C: "not
    the CI-lower-bound form"). Kept here only for comparison / in case the
    commit is ever revised to lock that stricter rule instead; --rule point
    is the one that matches the commit as written.

    Caught in review 2026-10-01: the doc's original table (MDE 0.14 @ 80%
    power) was computed with this rule even though the commit's prose
    argued for --rule point. The corrected --rule point table gives a
    substantially smaller MDE (0.047 @ 80% power) -- i.e. the test is
    better powered, and the "ambiguous" band under the locked rule is
    roughly a third the width the original table implied.

CAVEAT: SINGLE-DIRECTION POWER, NOT THE JOINT TEST
---------------------------------------------------
The commit requires the bound in BOTH directions (removal and injection) on
the same held-out pairs. This script's power/MDE numbers are for ONE
direction. Because both directions are measured on the same pairs, their
per-pair recovery values are correlated (estimated ~0.51 from the one real
circuit-evaluation file available, familiarity_v3's faithfulness_limit141
-- again a borrowed, provisional number). The joint (both-directions) power
is lower than the single-direction number reported here, bounded below by
max(0, 2*p-1) and above by min(p1,p2) via Frechet bounds, or computable
directly under a bivariate-normal approximation with --joint-rho (see
`joint_power_point`). Treat the single-direction table as an upper bound on
the true, harder joint requirement, not as the final answer.

STATISTICS
----------
--rule point (the locked, default rule) uses the exact CENTRAL t-distribution
of the sample mean -- appropriate given N=23, no large-sample normal
approximation needed or used (except as a fallback when scipy is missing, or
inside --joint-rho, see below).

--rule ci (kept for comparison only) uses the exact NONCENTRAL t-distribution,
since it's a significance test against a null rather than a direct read of
the sample mean's distribution.

--joint-rho (--rule point only) uses a bivariate-NORMAL approximation, not an
exact bivariate-t -- at N=23 (df=22) the normal is slightly optimistic. When
--joint-rho is set, the script switches the single-direction column to the
same normal approximation so the two columns are directly comparable; it will
therefore read marginally higher than the exact --rule point number reported
elsewhere. Falls back to a normal approximation with a printed warning
everywhere else if scipy is not available.

DEFAULT SD
----------
The per-pair recovery SD from the one existing real circuit-evaluation file
in this pipeline, `results/circuit_familiarity_v3/faithfulness_limit141.csv`
(set == "circuit", control_to_uncertain direction, N=141 pairs): SD = 0.262.
This is the familiarity circuit, not the (not-yet-run) U/U circuit, so treat
it as a plausible working estimate, not a measured value for the actual test
-- override with --sd once the U/U circuit has been run even once.

USAGE
-----
    # Power to detect a few plausible effect sizes at N=23, bound=0.7
    python power_analysis_uu.py power --n 23 --bound 0.7 \\
        --effects 0.1 0.2 0.3 0.4 0.5

    # MDE at 70/80/90% power, N=23, bound=0.7
    python power_analysis_uu.py mde --n 23 --bound 0.7 \\
        --power-targets 0.7 0.8 0.9

    # Two-sample framing: hedger vs. confabulator circuit recovery
    python power_analysis_uu.py power --mode diff --n1 23 --n2 23 \\
        --effects 0.1 0.2 0.3 0.4 0.5
"""

import argparse
import sys

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

DEFAULT_SD = 0.262  # see DEFAULT SD note above
DEFAULT_N = 23       # uu_hedge_confab_v3 held-out size
DEFAULT_ALPHA = 0.05
DEFAULT_BOUND = 0.7


def _warn_normal_fallback():
    print(
        "WARNING: scipy not found -- falling back to a normal approximation "
        "instead of the exact noncentral-t calculation. Results will be "
        "slightly optimistic (too high power / too low MDE) at N=23. "
        "Install scipy for the exact version: pip install scipy",
        file=sys.stderr,
    )


def power_one_sample_bound(n, true_mean, bound, sd, alpha=DEFAULT_ALPHA, rule="point"):
    """
    Power to detect that a one-sample mean (N pairs, population SD = sd)
    exceeds `bound`, true mean = true_mean.

    rule="point" (default, matches the pre-registration commit):
        the decision rule is the bare point estimate, sample_mean > bound,
        with no significance test and no alpha. Power = P(sample mean >
        bound | true_mean), from the CENTRAL t-distribution of the sample
        mean.

    rule="ci":
        the decision rule is a one-sided t-test at level alpha, equivalently
        "one-sided (1-alpha) CI lower bound > bound". This is the form the
        commit's prose explicitly argues AGAINST using (see module
        docstring) -- kept only for comparison.
    """
    se = sd / (n ** 0.5)
    df = n - 1
    effect = true_mean - bound

    if rule == "point":
        if HAVE_SCIPY:
            power = stats.t.cdf(effect / se, df)
        else:
            _warn_normal_fallback()
            power = _norm_cdf(effect / se)
        return power

    elif rule == "ci":
        ncp = effect / se  # noncentrality parameter
        if HAVE_SCIPY:
            t_crit = stats.t.ppf(1 - alpha, df)
            # power = P(T' > t_crit) where T' ~ noncentral t(df, ncp)
            power = 1 - stats.nct.cdf(t_crit, df, ncp)
        else:
            _warn_normal_fallback()
            z_crit = _norm_ppf(1 - alpha)
            power = 1 - _norm_cdf(z_crit - ncp)
        return power

    else:
        raise ValueError(f"unknown rule: {rule!r} (expected 'point' or 'ci')")


def mde_one_sample_bound(n, bound, sd, alpha=DEFAULT_ALPHA, power_target=0.8, rule="point"):
    """
    Minimum true-mean distance above `bound` needed for `power_target`
    power at sample size n.

    rule="point" has a closed form: effect = se * t.ppf(power_target, df).
    rule="ci" has no closed form here and is solved by bisection.
    """
    se = sd / (n ** 0.5)
    df = n - 1

    if rule == "point":
        if HAVE_SCIPY:
            return se * stats.t.ppf(power_target, df)
        else:
            _warn_normal_fallback()
            return se * _norm_ppf(power_target)

    elif rule == "ci":
        lo, hi = 0.0, 3.0  # search window; generous on purpose
        for _ in range(100):
            mid = (lo + hi) / 2
            p = power_one_sample_bound(n, bound + mid, bound, sd, alpha, rule="ci")
            if p < power_target:
                lo = mid
            else:
                hi = mid
        return hi

    else:
        raise ValueError(f"unknown rule: {rule!r} (expected 'point' or 'ci')")


def joint_power_point(n, effect, sd, rho, bound=DEFAULT_BOUND):
    """
    Approximate power for requiring BOTH directions (removal and injection)
    to each exceed `bound`, under rule="point", when the two directions'
    per-pair recovery values are correlated (same held-out pairs measured
    both ways). Uses a bivariate-normal approximation (not exact small-N
    bivariate-t) via scipy.stats.multivariate_normal -- adequate for a
    sanity check, not a substitute for the real joint calculation once the
    U/U circuit's actual two-direction correlation is measured.

    Assumes both directions have the same effect size and SD (symmetric
    case) -- pass the more conservative (smaller) of the two if they differ.
    """
    if not HAVE_SCIPY:
        raise RuntimeError("joint_power_point requires scipy (multivariate_normal)")
    from scipy.stats import multivariate_normal
    se = sd / (n ** 0.5)
    k = effect / se
    cov = [[1.0, rho], [rho, 1.0]]
    # P(Z1 > -k, Z2 > -k) = 1 - Phi(-k) - Phi(-k) + Phi2(-k, -k; rho)
    phi_k = stats.norm.cdf(-k)
    phi2 = multivariate_normal(mean=[0, 0], cov=cov).cdf([-k, -k])
    return 1 - 2 * phi_k + phi2


def power_two_sample_diff(n1, n2, mean_diff, sd, alpha=DEFAULT_ALPHA):
    """
    Power for a two-sample (unpaired, equal-variance) one-sided test that
    the difference between two circuits' recovery means is at least
    `mean_diff`, at sample sizes n1, n2 and common SD `sd`.
    """
    se = sd * ((1.0 / n1 + 1.0 / n2) ** 0.5)
    df = n1 + n2 - 2
    ncp = mean_diff / se

    if HAVE_SCIPY:
        t_crit = stats.t.ppf(1 - alpha, df)
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    else:
        _warn_normal_fallback()
        z_crit = _norm_ppf(1 - alpha)
        power = 1 - _norm_cdf(z_crit - ncp)

    return power


def mde_two_sample_diff(n1, n2, sd, alpha=DEFAULT_ALPHA, power_target=0.8):
    lo, hi = 0.0, 3.0
    for _ in range(100):
        mid = (lo + hi) / 2
        p = power_two_sample_diff(n1, n2, mid, sd, alpha)
        if p < power_target:
            lo = mid
        else:
            hi = mid
    return hi


# --- normal-approximation fallbacks, only used if scipy is unavailable ---
def _norm_cdf(x):
    import math
    return 0.5 * (1 + math.erf(x / (2 ** 0.5)))


def _norm_ppf(p):
    # Acklam's approximation, good to ~1e-9, avoids a scipy dependency
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = (-2 * __import__("math").log(p)) ** 0.5
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = (-2 * __import__("math").log(1 - p)) ** 0.5
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = dict(
        n=DEFAULT_N, sd=DEFAULT_SD, alpha=DEFAULT_ALPHA, bound=DEFAULT_BOUND,
    )

    p_power = sub.add_parser("power", help="Compute power for assumed effect size(s).")
    p_power.add_argument("--mode", choices=["bound", "diff"], default="bound")
    p_power.add_argument("--rule", choices=["point", "ci"], default="point",
                          help="'point' (default) = locked commit rule, bare "
                               "mean > bound, no alpha. 'ci' = one-sided test "
                               "at --alpha, equivalent to CI-lower-bound > "
                               "bound -- the form the commit rejects. "
                               "--mode bound only.")
    p_power.add_argument("--n", type=int, default=common["n"], help="N for --mode bound")
    p_power.add_argument("--n1", type=int, default=common["n"], help="group 1 N for --mode diff")
    p_power.add_argument("--n2", type=int, default=common["n"], help="group 2 N for --mode diff")
    p_power.add_argument("--sd", type=float, default=common["sd"])
    p_power.add_argument("--alpha", type=float, default=common["alpha"],
                          help="used only by --rule ci")
    p_power.add_argument("--bound", type=float, default=common["bound"],
                          help="falsification bound, --mode bound only")
    p_power.add_argument("--effects", type=float, nargs="+",
                          default=[0.1, 0.2, 0.3, 0.4, 0.5],
                          help="assumed effect sizes to scan: distance above "
                               "bound (--mode bound) or mean difference "
                               "(--mode diff)")
    p_power.add_argument("--joint-rho", type=float, default=None,
                          help="if set (--mode bound --rule point only), also "
                               "report the joint both-directions power at this "
                               "cross-direction correlation (try 0.51, the "
                               "provisional estimate from faithfulness_limit141.csv)")

    p_mde = sub.add_parser("mde", help="Compute MDE for target power level(s).")
    p_mde.add_argument("--mode", choices=["bound", "diff"], default="bound")
    p_mde.add_argument("--rule", choices=["point", "ci"], default="point")
    p_mde.add_argument("--n", type=int, default=common["n"])
    p_mde.add_argument("--n1", type=int, default=common["n"])
    p_mde.add_argument("--n2", type=int, default=common["n"])
    p_mde.add_argument("--sd", type=float, default=common["sd"])
    p_mde.add_argument("--alpha", type=float, default=common["alpha"],
                        help="used only by --rule ci")
    p_mde.add_argument("--bound", type=float, default=common["bound"])
    p_mde.add_argument("--power-targets", type=float, nargs="+",
                        default=[0.7, 0.8, 0.9])

    args = parser.parse_args()

    if not HAVE_SCIPY:
        _warn_normal_fallback()

    if args.command == "power":
        rule_note = "" if args.mode != "bound" else f", rule={args.rule}"
        print(f"# Power analysis -- mode={args.mode}{rule_note}, sd={args.sd}")
        if args.mode == "bound":
            if args.rule == "point":
                print(f"# N={args.n}, bound={args.bound} "
                      f"(locked commit rule: P(sample mean > bound | true mean), no alpha)")
            else:
                print(f"# N={args.n}, bound={args.bound}, alpha={args.alpha} "
                      f"(one-sided test / CI-lower-bound form -- NOT the locked rule)")
            if args.joint_rho is not None and args.rule == "point":
                # Both columns must come from the same approximation (normal)
                # so single vs. joint is an apples-to-apples comparison --
                # otherwise single-direction (exact central-t) can show as
                # LOWER than joint (bivariate-normal), which is impossible
                # for an actual conjunction and was flagged in review.
                print("# NOTE: with --joint-rho set, the single-direction column "
                      "below is also computed via the normal approximation "
                      "(not the exact central-t used elsewhere), so it's "
                      "directly comparable to the joint column. It will read "
                      "slightly higher than the exact --rule point number "
                      "elsewhere in this script -- that gap is the "
                      "t-vs-normal approximation, not part of the joint penalty.")
            header = f"{'effect (above bound)':>22} | {'true mean':>10} | {'power':>8}"
            if args.joint_rho is not None:
                header += f" | {'joint power (rho=%.2f)' % args.joint_rho:>24}"
            print(header)
            for eff in args.effects:
                true_mean = args.bound + eff
                if args.joint_rho is not None and args.rule == "point":
                    se = args.sd / (args.n ** 0.5)
                    p = stats.norm.cdf(eff / se) if HAVE_SCIPY else _norm_cdf(eff / se)
                else:
                    p = power_one_sample_bound(args.n, true_mean, args.bound, args.sd,
                                                args.alpha, rule=args.rule)
                row = f"{eff:>22.3f} | {true_mean:>10.3f} | {p:>8.3f}"
                if args.joint_rho is not None:
                    if args.rule != "point":
                        row += f" | {'(joint only for --rule point)':>24}"
                    else:
                        jp = joint_power_point(args.n, eff, args.sd, args.joint_rho, args.bound)
                        row += f" | {jp:>24.3f}"
                print(row)
        else:
            print(f"# N1={args.n1}, N2={args.n2} (testing: mean difference >= effect)")
            print(f"{'effect (mean diff)':>20} | {'power':>8}")
            for eff in args.effects:
                p = power_two_sample_diff(args.n1, args.n2, eff, args.sd, args.alpha)
                print(f"{eff:>20.3f} | {p:>8.3f}")

    elif args.command == "mde":
        rule_note = "" if args.mode != "bound" else f", rule={args.rule}"
        print(f"# MDE analysis -- mode={args.mode}{rule_note}, sd={args.sd}")
        if args.mode == "bound":
            tag = "locked commit rule" if args.rule == "point" else "CI-lower-bound form -- NOT the locked rule"
            print(f"# N={args.n}, bound={args.bound} ({tag})")
            print(f"{'power target':>12} | {'MDE (above bound)':>18} | {'true mean needed':>17}")
            for pt in args.power_targets:
                mde = mde_one_sample_bound(args.n, args.bound, args.sd, args.alpha, pt, rule=args.rule)
                print(f"{pt:>12.2f} | {mde:>18.3f} | {args.bound + mde:>17.3f}")
        else:
            print(f"# N1={args.n1}, N2={args.n2}")
            print(f"{'power target':>12} | {'MDE (mean diff)':>16}")
            for pt in args.power_targets:
                mde = mde_two_sample_diff(args.n1, args.n2, args.sd, args.alpha, pt)
                print(f"{pt:>12.2f} | {mde:>16.3f}")


if __name__ == "__main__":
    main()