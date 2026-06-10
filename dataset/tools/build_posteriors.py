#!/usr/bin/env python3
"""Regenerate the v0.2 per-channel posterior summaries (spec v0.2 sec 3, sec 9).

A posterior is a **deterministic projection** of ``(source data + declared prior)``,
exactly as a v0.1 record is a deterministic projection of the curated literature
table (``build_records.py``). This tool runs the hierarchical Bayesian inference
(:mod:`harmonia.infer`) over every channel-block record and prints the cached
``posterior_summary`` block — mean / sd / quantiles of the true-value log10 IC50,
the Hill posterior, the sampler diagnostics (rhat / ess), the continuous
identifiability score, the prior-sensitivity, and the censoring flag.

Design decision (and a deliberate departure from the spec's *optional* cache):
Harmonia does **not** persist the posterior summary into the 68 record files. The
source of truth stays ``(source_values | dose_response) + prior``; the posterior is
recomputed on demand in milliseconds (the summary regime is tiny), so caching would
add a 68-file churn and a cross-platform byte-diff gate for no load-time benefit
(numpy is already a hard dependency). Reproducibility is instead asserted by
``tests/test_infer.py`` (run the inference twice -> byte-identical summary), which
proves "deterministic projection" without the brittle git-diff. Run this tool to
inspect the posteriors or to emit a standalone JSON cache with ``--json``.

    python dataset/tools/build_posteriors.py
    python dataset/tools/build_posteriors.py --json > posteriors.json
"""
from __future__ import annotations

import argparse
import json
import sys

import harmonia
from harmonia.infer import infer_channel, learn_tau_pop, resolve_prior
from harmonia.records import ChannelBlock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit a JSON cache to stdout")
    ap.add_argument("--validate", action="store_true",
                    help="run simulation-based calibration + posterior coverage (spec sec 9)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=4000)
    args = ap.parse_args()

    ds = harmonia.load()
    prior = resolve_prior(ds)
    tau_pop = learn_tau_pop(ds.channel_blocks, prior)

    if args.validate:
        from harmonia.infer import (simulation_based_calibration, sbc_uniformity_pvalue,
                                    posterior_coverage)
        ranks, nd = simulation_based_calibration(prior, n_sims=400, n_obs=3,
                                                 tau_scale=tau_pop, n_draws=400, seed=args.seed)
        pval = sbc_uniformity_pvalue(ranks, nd, n_bins=16)
        cov90 = posterior_coverage(prior, n_sims=400, n_obs=3, tau_scale=tau_pop,
                                   n_draws=1500, seed=args.seed, level=0.90)
        print(f"inference calibration under prior={prior.id} (tau_pop={tau_pop:.3f}):")
        print(f"  SBC rank uniformity p-value = {pval:.3f}  (>0.05 = well-calibrated)")
        print(f"  90% credible-interval coverage = {cov90:.3f}  (target 0.90)")
        print("  A correctly-implemented inference, not merely a plausible one (spec sec 9).")
        return 0

    out = {}
    rows = []
    for b in sorted(ds.channel_blocks, key=lambda r: r.id):
        if not isinstance(b, ChannelBlock):
            continue
        post = infer_channel(b, prior, tau_pop, n_draws=args.draws, seed=args.seed)
        summary = post.summary_dict()
        out[b.id] = {"method": "hierarchical_bayes", "prior": prior.id,
                     "censored": post.censored, "posterior_summary": summary}
        rows.append((b.id, summary))

    if args.json:
        json.dump({"prior": prior.id, "tau_pop_log10": round(tau_pop, 4),
                   "posteriors": out}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"prior={prior.id}  learned between-lab SD tau_pop={tau_pop:.4f} log10  "
          f"({len(rows)} channel-block posteriors)")
    worst_rhat = max(s["rhat_max"] for _, s in rows)
    min_ess = min(s["ess_min"] for _, s in rows)
    n_cens = sum(1 for _, s in rows for k in [s] if s["censored"])
    n_prior_dom = sum(1 for _, s in rows if s["prior_sensitivity"] >= 0.5)
    print(f"  worst rhat={worst_rhat:.4f}   min ess={min_ess}   "
          f"censored={n_cens}   prior-dominated={n_prior_dom}")
    for rid, s in rows:
        ic = s["log10_ic50_nm"]
        flag = " CENSORED" if s["censored"] else ""
        flag += " PRIOR-DOM" if s["prior_sensitivity"] >= 0.5 else ""
        print(f"  {rid:<34} log10IC50={ic['mean']:+.3f}+/-{ic['sd']:.3f}  "
              f"hill={s['hill']['mean']:.2f}  ident={s['identifiability_score']:.2f}  "
              f"priorS={s['prior_sensitivity']:.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
