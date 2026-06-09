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
          f"(LLM-assisted extraction never promotes to verified — spec §9)")

    drugs = ds.drugs()
    print(f"  drugs ({len(drugs)}): {', '.join(drugs)}")

    unident = [b.id for b in ds.channel_blocks if not b.identifiable]  # type: ignore[attr-defined]
    if unident:
        print(f"  UNIDENTIFIABLE IC50 (max block < 60% -> Tier D): {', '.join(unident)}")
    print("\n  NOT a clinical tool / NOT a regulatory determination. "
          "Outputs are risk distributions, never verdicts.")
    return 0


def cmd_simulate(args) -> int:
    from .simulate import assess
    ds = _load(args)
    res = assess(ds, args.drug, ap_model=args.ap_model, n_mc=args.mc, metric=args.metric,
                 exposure_multiple=args.exposure_multiple, seed=args.seed,
                 herg_dynamic=args.dynamic)
    print(res.summary())
    return 0


def cmd_flip(args) -> int:
    from .simulate import flip_view
    ds = _load(args)
    models = args.ap_models.split(",") if args.ap_models else None
    kw = {"ap_models": models} if models else {}
    fv = flip_view(ds, args.drug, n_mc=args.mc, seed=args.seed, **kw)
    print(fv.summary())
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


def cmd_export(args) -> int:
    from .export import registry, combine
    ds = _load(args)
    if args.all:
        out = args.output or "exports/"
        written = registry.build_all(ds, out, dataset_version=__version__)
        print(f"wrote {len(written)} artifacts under {out}")
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
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_simulate)

    f = sub.add_parser("flip", help="classification-flip view across AP-model variants")
    f.add_argument("drug")
    f.add_argument("--ap-models", default="", help="comma-separated model ids")
    f.add_argument("--mc", type=int, default=200)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_flip)

    pf = sub.add_parser("performance", help="score the kernel's classification vs CiPA expert labels")
    pf.add_argument("--ap-model", default="cipaordv1.0", dest="ap_model")
    pf.add_argument("--set", default="report",
                    choices=["training", "validation", "all", "report"],
                    help="'report' prints training, validation, and all")
    pf.add_argument("--metric", default="qnet", choices=["qnet", "apd90"])
    pf.add_argument("--dynamic", action="store_true", help="use dynamic hERG binding where available")
    pf.add_argument("--exposure-multiple", type=float, default=4.0, dest="exposure_multiple")
    pf.set_defaults(func=cmd_performance)

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
