"""The ``harmonia`` command-line interface.

    harmonia version
    harmonia validate
    harmonia info
    harmonia simulate dofetilide --ap-model cipaordv1.0 --mc 200
    harmonia flip verapamil
    harmonia export --format cellml --output exports/cellml/
    harmonia export --all --output exports/
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import List, Optional

from . import __version__


def _load(args):
    from .load import load
    return load(path=getattr(args, "dataset", None))


def cmd_version(args) -> int:
    print(f"harmonia {__version__}")
    return 0


def cmd_validate(args) -> int:
    from .validate import validate_dataset
    report = validate_dataset(path=getattr(args, "dataset", None))
    print(report)
    return 0 if report.ok else 1


def cmd_info(args) -> int:
    ds = _load(args)
    print(f"Harmonia dataset {__version__}  ({ds.root})")
    print(f"  records: {len(ds)}   citations: {len(ds.citations)}")

    by_kind = Counter(r.kind for r in ds)
    by_sub = Counter(r.subsystem for r in ds)
    by_tier = Counter(r.tier for r in ds)
    by_status = Counter(r.review_status for r in ds)

    def line(title, counter):
        items = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
        print(f"  {title:14s} {items}")

    line("by kind", by_kind)
    line("by subsystem", by_sub)
    line("by tier", by_tier)
    line("by review", by_status)

    verified = by_status.get("verified", 0)
    print(f"  VERIFIED: {verified}/{len(ds)} records "
          f"(human-confirmed against the source PDF; LLM-assisted extraction never "
          f"promotes to verified — spec §9)")
    pending = by_status.get("pending_human_review", 0)
    if pending:
        print(f"  PENDING HUMAN REVIEW: {pending}/{len(ds)} records "
              f"(value machine-corroborated against an independent published source, "
              f"awaiting human confirmation — sourced, NOT verified; spec v0.8.2)")

    try:
        from .crosscheck import cross_check
        cc = cross_check(ds)
        print(f"  MACHINE-CROSS-CHECKED: {cc.n_cross_checked}/{len(cc.checks)} channel-block "
              f"records agree with the published CiPA reference (≠ verified; spec v0.8)"
              + (f"; {len(cc.divergent)} DIVERGENT — see `harmonia crosscheck`"
                 if cc.divergent else ""))
    except FileNotFoundError:
        pass

    drugs = ds.drugs()
    print(f"  drugs ({len(drugs)}): {', '.join(drugs)}")

    unident = [b.id for b in ds.channel_blocks if not b.identifiable]
    if unident:
        print(f"  UNIDENTIFIABLE IC50 (max block < 60% -> Tier D): {', '.join(unident)}")
    pops = ds.populations
    if pops:
        print(f"  populations (HYPOTHESIS-TIER, not for prediction): "
              f"{', '.join(p.id.split('.', 1)[1] for p in pops)}")
    print("\n  NOT a clinical tool / NOT a regulatory determination. "
          "Outputs are risk distributions, never verdicts.")
    return 0


def cmd_simulate(args) -> int:
    from .simulate import assess
    ds = _load(args)
    res = assess(ds, args.drug, ap_model=args.ap_model, n_mc=args.mc, metric=args.metric,
                 exposure_multiple=args.exposure_multiple, seed=args.seed,
                 herg_dynamic=args.dynamic, uq=args.uq)
    print(res.summary())
    return 0


def cmd_infer(args) -> int:
    """Show the v0.2 per-channel Bayesian posteriors + sampler diagnostics."""
    from .infer import infer_channel, resolve_prior, learn_tau_pop
    from .records import ChannelBlock
    ds = _load(args)
    pr = resolve_prior(ds, args.prior)
    tau_pop = learn_tau_pop(ds.channel_blocks, pr)
    blocks = [b for b in ds.blocks_for(args.drug) if isinstance(b, ChannelBlock)]
    if not blocks:
        print(f"no channel-block records for drug '{args.drug}'", file=sys.stderr)
        return 1
    print(f"posteriors  drug={args.drug.lower()}  prior={pr.id}  "
          f"learned between-lab SD tau_pop={tau_pop:.3f} log10")
    print(f"  {'channel':<7} {'n':>2} {'IC50 (nM) q05/med/q95':>26}  {'hill':>10}  "
          f"{'ident':>5} {'priorS':>6} {'rhat':>5} {'ess':>5}  flags")
    import numpy as np
    for b in blocks:
        p = infer_channel(b, pr, tau_pop, n_draws=4000, seed=args.seed)
        q = 10 ** np.quantile(p.log10_ic50, [0.05, 0.5, 0.95])
        flags = []
        if p.censored:
            flags.append("CENSORED")
        if p.prior_dominated:
            flags.append("PRIOR-DOMINATED")
        print(f"  {b.channel:<7} {p.n_sources:>2} "
              f"{q[0]:>7.1f}/{q[1]:>7.1f}/{q[2]:>8.1f}  "
              f"{p.hill_mean:>5.2f}+/-{p.hill_sd:<4.2f}  {p.identifiability_score:>5.2f} "
              f"{p.prior_sensitivity:>6.2f} {p.rhat_max:>5.3f} {p.ess_min:>5.0f}  "
              f"{' '.join(flags)}")
    print("\n  A posterior is not a point estimate. Outputs are distributions + diagnostics, "
          "never a verdict.")
    return 0


def cmd_priors(args) -> int:
    """List the prior registry and which records use each prior."""
    ds = _load(args)
    if not ds.priors:
        print("no priors registered (dataset/priors/ is empty)")
        return 0
    for pid, pr in sorted(ds.priors.items()):
        d = pr.raw or {}
        print(f"{pid}  (v{d.get('version', '?')})")
        print(f"  channel-level true log10 IC50 ~ Normal(m0={pr.m0}, s0={pr.s0})")
        print(f"  between-lab SD ~ HalfNormal(scale={pr.tau_scale} log10)")
        print(f"  hill ~ LogNormal(mu={pr.hill_mu}, sigma={pr.hill_sigma})   "
              f"predictive={d.get('predictive')}")
        cites = ", ".join(d.get("citations", [])) or "(none)"
        print(f"  citations: {cites}")
        if d.get("rationale"):
            print(f"  rationale: {d['rationale'][:200]}{'...' if len(d['rationale']) > 200 else ''}")
    print("\n  Priors are declared inputs, not hidden choices (spec v0.2 sec 7). "
          "prior_sensitivity is reported per channel; no prior may carry a risk conclusion.")
    return 0


def cmd_flip(args) -> int:
    from .simulate import flip_view
    ds = _load(args)
    models = args.ap_models.split(",") if args.ap_models else None
    kw = {"ap_models": models} if models else {}
    fv = flip_view(ds, args.drug, n_mc=args.mc, seed=args.seed, **kw)
    print(fv.summary())
    return 0


def cmd_sensitivity(args) -> int:
    from .simulate import flip_sensitivity
    ds = _load(args)
    res = flip_sensitivity(ds, args.drug, ap_model=args.ap_model, metric=args.metric,
                           n_mc=args.mc, seed=args.seed,
                           exposure_multiple=args.exposure_multiple,
                           method="sobol" if args.sobol else "oat", uq=args.uq)
    print(res.summary())
    return 0


def cmd_combo(args) -> int:
    from .simulate import assess_combination
    ds = _load(args)
    res = assess_combination(ds, args.drugs, ap_model=args.ap_model, n_mc=args.mc,
                             metric=args.metric, exposure_multiple=args.exposure_multiple,
                             seed=args.seed)
    print(res.summary())
    return 0


def cmd_population(args) -> int:
    from .populations import assess_population
    ds = _load(args)
    res = assess_population(ds, args.drug, population=args.population,
                           ap_model=args.ap_model, n_models=args.n, metric=args.metric,
                           exposure_multiple=args.exposure_multiple, seed=args.seed)
    print(res.summary())
    return 0


def cmd_performance(args) -> int:
    from .performance import score
    ds = _load(args)
    for cipa_set in (["training", "validation", "all"] if args.set == "report" else [args.set]):
        rep = score(ds, ap_model=args.ap_model, cipa_set=cipa_set, metric=args.metric,
                    herg_dynamic=args.dynamic, exposure_multiple=args.exposure_multiple)
        print(rep.summary())
        print()
    return 0


def cmd_crosscheck(args) -> int:
    from .crosscheck import cross_check, cross_check_binding
    ds = _load(args)
    rep = cross_check(ds, drug=args.drug)
    print(rep.summary())
    binding_bad = False
    if not args.drug:  # the binding check spans the 12 dynamic-fit drugs, not one
        try:
            brep = cross_check_binding(ds)
            print()
            print(brep.summary())
            binding_bad = bool(brep.mismatched)
        except FileNotFoundError:
            pass
    return 1 if (args.strict and (rep.divergent or binding_bad)) else 0


def cmd_export(args) -> int:
    from .export import registry, combine, cellml, sedml, sbml
    ds = _load(args)
    if args.all:
        out = args.output or "exports/"
        written = registry.build_all(ds, out, dataset_version=__version__)
        print(f"wrote {len(written)} artifacts under {out}")
        # Exports are generated artifacts, never hand-edited: validate them so any
        # drift between the dataset/kernel and the exports fails loudly (spec.md
        # §6, §7). Covers the CiPA numeric round trip, the parameter round trip,
        # the ODE round trip (the AST re-integrates to the kernel), CellML unit
        # conformance, SED-ML cross-reference resolution, and OMEX manifest
        # consistency, and SBML validity (libSBML, where installed).
        errors = list(registry.roundtrip_cipa(ds))
        for ap in registry.list_ap_models(ds):
            errors += registry.roundtrip_parameters(ds, ap)
            errors += registry.roundtrip_ode(ds, ap)
            errors += cellml.conformance_violations(cellml.build(ds, ap))
            errors += cellml.validity_violations(cellml.build(ds, ap))
            errors += sbml.consistency_violations(sbml.build(ds, ap))
            errors += sedml.reference_violations(
                registry.build_text(ds, "sedml", ap_model=ap))
            errors += combine.manifest_violations(combine.build_bytes(ds, ap))
        if errors:
            print("export validation FAILED:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print(f"validated: CiPA + parameters + ODE round trips, CellML units + "
              f"libCellML validity, SBML validity, SED-ML refs, OMEX manifests "
              f"across {len(registry.list_ap_models(ds))} AP models")
        return 0

    fmt = args.format
    if not fmt:
        print("error: pass --format or --all", file=sys.stderr)
        return 2

    if fmt == "omex":
        out = args.output or "exports/harmonia.omex"
        combine.build(ds, out, ap_model=args.ap_model, dataset_version=__version__)
        print(f"wrote {out}")
        return 0

    text = registry.build_text(ds, fmt, ap_model=args.ap_model, dataset_version=__version__)
    if args.output:
        import pathlib
        p = pathlib.Path(args.output)
        if p.is_dir() or args.output.endswith("/"):
            p.mkdir(parents=True, exist_ok=True)
            p = p / f"{args.ap_model}{registry.TEXT_FORMATS.get(fmt, '.txt')}"
        p.write_text(text, encoding="utf-8")
        print(f"wrote {p}")
    else:
        sys.stdout.write(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harmonia",
        description="Curated cardiac ion-channel drug-block data + in-silico "
                    "proarrhythmia risk DISTRIBUTIONS (never safety verdicts).")
    p.add_argument("--dataset", help="path to the dataset/ directory")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("version", help="print version").set_defaults(func=cmd_version)
    sub.add_parser("validate", help="JSON-Schema- and semantically validate the dataset"
                   ).set_defaults(func=cmd_validate)
    sub.add_parser("info", help="counts by subsystem / tier / review status"
                   ).set_defaults(func=cmd_info)

    s = sub.add_parser("simulate", help="risk-metric distribution for one drug")
    s.add_argument("drug")
    s.add_argument("--ap-model", default="cipaordv1.0", dest="ap_model")
    s.add_argument("--metric", default="qnet", choices=["qnet", "apd90"])
    s.add_argument("--mc", type=int, default=200, help="Monte-Carlo draws")
    s.add_argument("--exposure-multiple", type=float, default=4.0, dest="exposure_multiple")
    s.add_argument("--dynamic", action="store_true", help="use dynamic hERG binding where available")
    s.add_argument("--uq", default="moments", choices=["moments", "bayes"],
                   help="uncertainty engine: v0.1 method-of-moments or v0.2 hierarchical Bayes")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_simulate)

    inf = sub.add_parser("infer", help="v0.2 per-channel Bayesian posteriors + diagnostics")
    inf.add_argument("drug")
    inf.add_argument("--prior", default=None, help="prior id (default: harmonia-ic50-prior-v1)")
    inf.add_argument("--seed", type=int, default=0)
    inf.set_defaults(func=cmd_infer)

    sub.add_parser("priors", help="list the v0.2 prior registry"
                   ).set_defaults(func=cmd_priors)

    f = sub.add_parser("flip", help="classification-flip view across AP-model variants")
    f.add_argument("drug")
    f.add_argument("--ap-models", default="", help="comma-separated model ids")
    f.add_argument("--mc", type=int, default=200)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_flip)

    se = sub.add_parser("sensitivity",
                        help="attribute the classification-flip to each channel's IC50 spread")
    se.add_argument("drug")
    se.add_argument("--ap-model", default="cipaordv1.0", dest="ap_model")
    se.add_argument("--metric", default="qnet", choices=["qnet", "apd90"])
    se.add_argument("--mc", type=int, default=100,
                    help="Monte-Carlo draws per scenario (runs ~2*n_channels+1 scenarios)")
    se.add_argument("--exposure-multiple", type=float, default=4.0, dest="exposure_multiple")
    se.add_argument("--sobol", action="store_true",
                    help="variance-based (Sobol) indices with interactions, not one-at-a-time")
    se.add_argument("--uq", default="moments", choices=["moments", "bayes"],
                    help="uncertainty engine for the Sobol sampler")
    se.add_argument("--seed", type=int, default=0)
    se.set_defaults(func=cmd_sensitivity)

    cb = sub.add_parser("combo", help="assess a DRUG COMBINATION (polypharmacy proarrhythmia)")
    cb.add_argument("drugs", nargs="+", help="two or more drug names")
    cb.add_argument("--ap-model", default="cipaordv1.0", dest="ap_model")
    cb.add_argument("--metric", default="qnet", choices=["qnet", "apd90"])
    cb.add_argument("--mc", type=int, default=200, help="Monte-Carlo draws")
    cb.add_argument("--exposure-multiple", type=float, default=4.0, dest="exposure_multiple")
    cb.add_argument("--seed", type=int, default=0)
    cb.set_defaults(func=cmd_combo)

    pop = sub.add_parser("population",
                         help="population-of-models spread (HYPOTHESIS-TIER, not for prediction)")
    pop.add_argument("drug")
    pop.add_argument("--population", default="illustrative_v0",
                     help="illustrative_v0 (variability) or a disease background: "
                          "lqt1 / lqt2 / lqt3 (v0.3, hypothesis-tier)")
    pop.add_argument("--ap-model", default="cipaordv1.0", dest="ap_model")
    pop.add_argument("--metric", default="qnet", choices=["qnet", "apd90"])
    pop.add_argument("--n", type=int, default=None, help="number of virtual myocytes")
    pop.add_argument("--exposure-multiple", type=float, default=4.0, dest="exposure_multiple")
    pop.add_argument("--seed", type=int, default=0)
    pop.set_defaults(func=cmd_population)

    pf = sub.add_parser("performance", help="score the kernel's classification vs CiPA expert labels")
    pf.add_argument("--ap-model", default="cipaordv1.0", dest="ap_model")
    pf.add_argument("--set", default="report",
                    choices=["training", "validation", "all", "report"],
                    help="'report' prints training, validation, and all")
    pf.add_argument("--metric", default="qnet", choices=["qnet", "apd90"])
    pf.add_argument("--dynamic", action="store_true", help="use dynamic hERG binding where available")
    pf.add_argument("--exposure-multiple", type=float, default=4.0, dest="exposure_multiple")
    pf.set_defaults(func=cmd_performance)

    cc = sub.add_parser("crosscheck",
                        help="diff transcribed IC50/Hill vs the published CiPA reference "
                             "(machine cross-check, NOT human verification)")
    cc.add_argument("drug", nargs="?", default=None,
                    help="one drug, or omit for every channel-block record")
    cc.add_argument("--strict", action="store_true",
                    help="exit non-zero if any record diverges >5x from the published value")
    cc.set_defaults(func=cmd_crosscheck)

    e = sub.add_parser("export", help="generate export artifacts")
    e.add_argument("--format", choices=["cellml", "myokit", "sbml", "sedml", "cipa",
                                        "csv", "bibtex", "omex"])
    e.add_argument("--all", action="store_true", help="write every format")
    e.add_argument("--ap-model", default="cipaordv1.0", dest="ap_model")
    e.add_argument("--output", help="output file or directory")
    e.set_defaults(func=cmd_export)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
