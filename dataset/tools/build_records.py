#!/usr/bin/env python3
"""Generate Harmonia dataset records + citations from the curated CiPA table.

This tool is the *provenance log* for the v0.1 channel-block dataset: the literal
literature table below, plus the deterministic rules that turn it into records
(tier assignment, variability computation, failure-mode flagging). The emitted
JSON files under ../records/ and ../citations/ are the committed source of truth;
re-running this tool must reproduce them byte-for-byte (CI checks this).

IMPORTANT — honesty posture (spec.md §9): no record is ever auto-promoted to
`review_status: "verified"`. The values are literature-derived (Crumb 2016,
Li 2017, Kramer 2013, and the CiPA working-group EFTPCs) but NOBODY has opened
the source PDFs inside Harmonia yet. LLMs may assist extraction; they never
promote a record to `verified`. The verified count is reported honestly as 0.

A channel-block value that AGREES with the independent published CiPA reference
(see `corroborate` below) ships as `review_status: "pending_human_review"` — an
honest middle state: machine-corroborated against a real published source, but
still awaiting human confirmation against the PDF. Everything uncorroborated (and
all non-channel-block records) ships as `"unverified"`.

All IC50 values are in nanomolar (nM). All EFTPCs are free (unbound) Cmax in nM.
"""
from __future__ import annotations

import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RECORDS = HERE.parent / "records"
CITATIONS = HERE.parent / "citations"

# --------------------------------------------------------------------------- #
# Citations (Crossref-resolvable DOIs). review_status here means "metadata
# transcribed"; the *numeric values* drawn from them are still unverified.
# --------------------------------------------------------------------------- #
CITATIONS_TABLE = [
    dict(key="crumb-2016", type="article",
         title="An evaluation of 30 clinical drugs against the comprehensive set of "
               "cardiac ion channels using a chip-based, electrophysiologic platform",
         authors="Crumb WJ, Vicente J, Johannesen L, Strauss DG",
         journal="Journal of Pharmacological and Toxicological Methods", year=2016,
         doi="10.1016/j.vascn.2016.03.009", pmid="27060526"),
    dict(key="li-2017", type="article",
         title="Improving the In Silico Assessment of Proarrhythmia Risk by Combining "
               "hERG Channel-Drug Binding Kinetics and Multichannel Pharmacology",
         authors="Li Z, Dutta S, Sheng J, Tran PN, Wu W, Chang K, Mdluli T, Strauss DG, Colatsky T",
         journal="Circulation: Arrhythmia and Electrophysiology", year=2017,
         doi="10.1161/CIRCEP.116.004628", pmid="28202629"),
    dict(key="kramer-2013", type="article",
         title="MICE models: superior to the HERG model in predicting Torsade de Pointes",
         authors="Kramer J, Obejero-Paz CA, Myatt G, Kuryshev YA, Bruening-Wright A, Verducci JS, Brown AM",
         journal="Scientific Reports", year=2013,
         doi="10.1038/srep02100", pmid="23811994"),
    dict(key="ohara-2011", type="article",
         title="Simulation of the undiseased human cardiac ventricular action potential: "
               "model formulation and experimental validation",
         authors="O'Hara T, Virag L, Varro A, Rudy Y",
         journal="PLoS Computational Biology", year=2011,
         doi="10.1371/journal.pcbi.1002061", pmid="21637795"),
    dict(key="dutta-2017", type="article",
         title="Optimization of an In Silico Cardiac Cell Model for Proarrhythmia Risk Assessment",
         authors="Dutta S, Chang KC, Beattie KA, Sheng J, Tran PN, Wu WW, Wu M, Strauss DG, Colatsky T, Li Z",
         journal="Frontiers in Physiology", year=2017,
         doi="10.3389/fphys.2017.00616", pmid="28868040"),
    dict(key="tomek-2019", type="article",
         title="Development, calibration, and validation of a novel human ventricular "
               "myocyte model in health, disease, and drug block",
         authors="Tomek J, Bueno-Orovio A, Passini E, Zhou X, Minchole A, Britton O, Bartolucci C, Severi S, Shrier A, Virag L, Varro A, Rodriguez B",
         journal="eLife", year=2019,
         doi="10.7554/eLife.48890", pmid="31453809"),
    dict(key="li-2019", type="article",
         title="Assessment of an In Silico Mechanistic Model for Proarrhythmia Risk "
               "Prediction Under the CiPA Initiative",
         authors="Li Z, Ridder BJ, Han X, Wu WW, Sheng J, Tran PN, Wu M, Randolph A, Johnstone RH, Mirams GR, Kuryshev Y, Kramer J, Wu C, Crumb WJ, Strauss DG",
         journal="Clinical Pharmacology & Therapeutics", year=2019,
         doi="10.1002/cpt.1184", pmid="30033627"),
    dict(key="ridder-2020", type="article",
         title="A systematic strategy for estimating hERG block potency and its implications "
               "in a new cardiac safety paradigm",
         authors="Ridder BJ, Leishman DJ, Bridgland-Taylor M, Samieegohar M, Han X, Wu WW, "
                 "Randolph A, Tran P, Mirams GR, Strauss DG, Li Z, et al.",
         journal="Toxicology and Applied Pharmacology", year=2020,
         doi="10.1016/j.taap.2020.114961", pmid="32209365"),
    dict(key="chang-2017", type="article",
         title="Uncertainty Quantification Reveals the Importance of Data Variability and "
               "Experimental Design Considerations for in Silico Proarrhythmia Risk Assessment",
         authors="Chang KC, Dutta S, Mirams GR, Beattie KA, Sheng J, Tran PN, Wu M, Wu WW, Colatsky T, Strauss DG, Li Z",
         journal="Frontiers in Physiology", year=2017,
         doi="10.3389/fphys.2017.00917", pmid="29209226"),
    dict(key="colatsky-2016", type="article",
         title="The Comprehensive in Vitro Proarrhythmia Assay (CiPA) initiative — Update on progress",
         authors="Colatsky T, Fermini B, Gintant G, Pierson JB, Sager P, Sekino Y, Strauss DG, Stockbridge N",
         journal="Journal of Pharmacological and Toxicological Methods", year=2016,
         doi="10.1016/j.vascn.2016.06.002", pmid="27263822"),
    dict(key="fermini-2016", type="article",
         title="A New Perspective in the Field of Cardiac Safety Testing through the "
               "Comprehensive In Vitro Proarrhythmia Assay Paradigm",
         authors="Fermini B, Hancox JC, Abi-Gerges N, Bridgland-Taylor M, Chaudhary KW, Colatsky T, et al.",
         journal="Journal of Biomolecular Screening", year=2016,
         doi="10.1177/1087057115594589", pmid="26170255"),
    dict(key="britton-2013", type="article",
         title="Experimentally calibrated population of models predicts and explains "
               "intersubject variability in cardiac cellular electrophysiology",
         authors="Britton OJ, Bueno-Orovio A, Van Ammel K, Lu HR, Towart R, Gallacher DJ, Rodriguez B",
         journal="Proceedings of the National Academy of Sciences", year=2013,
         doi="10.1073/pnas.1304382110", pmid="23690584"),
    dict(key="passini-2017", type="article",
         title="Human In Silico Drug Trials Demonstrate Higher Accuracy than Animal "
               "Models in Predicting Clinical Pro-Arrhythmic Cardiotoxicity",
         authors="Passini E, Britton OJ, Lu HR, Rohrbacher J, Hermans AN, Gallacher DJ, Greig RJH, Bueno-Orovio A, Rodriguez B",
         journal="Frontiers in Physiology", year=2017,
         doi="10.3389/fphys.2017.00668", pmid="28955244"),
    # v0.2 (Bayesian dose-response UQ) methodological precedents.
    dict(key="johnstone-2016", type="article",
         title="Hierarchical Bayesian inference for ion channel screening dose-response data",
         authors="Johnstone RH, Bardenet R, Gavaghan DJ, Mirams GR",
         journal="Wellcome Open Research", year=2016,
         doi="10.12688/wellcomeopenres.9945.1", pmid="27918599"),
    dict(key="elkins-2013", type="article",
         title="Variability in high-throughput ion-channel screening data and "
               "consequences for cardiac safety assessment",
         authors="Elkins RC, Davies MR, Brough SJ, Gavaghan DJ, Cui Y, Abi-Gerges N, Mirams GR",
         journal="Journal of Pharmacological and Toxicological Methods", year=2013,
         doi="10.1016/j.vascn.2013.04.007", pmid="23651875"),
    # v0.3 (disease/genetic population backgrounds) mechanism reference.
    dict(key="moss-2005", type="article",
         title="Long QT syndrome: from channels to cardiac arrhythmias",
         authors="Moss AJ, Kass RS",
         journal="Journal of Clinical Investigation", year=2005,
         doi="10.1172/JCI25537", pmid="16075042"),
]

# --------------------------------------------------------------------------- #
# The curated channel-block table.
#
# Each drug: cipa risk category, free Cmax (EFTPC, nM), and per-channel block.
# A channel maps to either:
#   ic50, hill, max_block  (single source -> source_values has 1 entry), or
#   a `sources` list of (ic50_nm, platform, citation) tuples that drives the
#   multi-source variability fields.
# `primary` names the citation used as the headline source for that channel.
# --------------------------------------------------------------------------- #
NaN = None
DRUGS = {
    # ----- HIGH risk -----
    "dofetilide": dict(category="high", unii="R4Z9X1N42Q", eftpc=2.0, fu=0.36, channels={
        "IKr": dict(hill=0.9, max_block=95, primary="crumb-2016",
                    sources=[(4.9, "manual", "crumb-2016"),
                             (6.6, "automated", "kramer-2013"),
                             (4.0, "manual", "li-2017")],
                    # Dofetilide is the prototypical TRAPPED hERG blocker (slow
                    # off-rate; bound drug is trapped in the closed channel).
                    # kon/koff set so IC50 = koff/kon ~= 5 nM (matches above).
                    dynamic=dict(kon=1.0e-5, koff=5.0e-5, trapping=True,
                                 citation="li-2017")),
    }),
    "bepridil": dict(category="high", unii="VWA7N2DT4P", eftpc=33.0, fu=0.01, channels={
        "IKr": dict(hill=0.9, max_block=92, primary="crumb-2016",
                    sources=[(55.0, "manual", "crumb-2016"),
                             (92.0, "automated", "kramer-2013")]),
        "ICaL": dict(ic50=2808.0, hill=0.6, max_block=82, primary="crumb-2016"),
        "INaL": dict(ic50=2929.0, hill=1.4, max_block=78, primary="crumb-2016"),
        "INa":  dict(ic50=2929.0, hill=1.2, max_block=75, primary="crumb-2016"),
        "IKs":  dict(ic50=28628.0, hill=1.0, max_block=65, primary="crumb-2016"),
        "Ito":  dict(ic50=8594.0, hill=3.5, max_block=70, primary="crumb-2016"),
    }),
    "quinidine": dict(category="high", unii="ITX08688JL", eftpc=3237.0, fu=0.13, channels={
        "IKr": dict(hill=0.8, max_block=90, primary="crumb-2016",
                    sources=[(411.0, "manual", "crumb-2016"),
                             (700.0, "automated", "kramer-2013"),
                             (320.0, "manual", "li-2017")]),
        "INa":  dict(ic50=12329.0, hill=1.5, max_block=80, primary="crumb-2016"),
        "INaL": dict(ic50=9417.0, hill=1.3, max_block=75, primary="crumb-2016"),
        "ICaL": dict(ic50=15000.0, hill=1.0, max_block=60, primary="crumb-2016"),
        "Ito":  dict(ic50=3487.0, hill=1.3, max_block=85, primary="crumb-2016"),
        "IKs":  dict(ic50=4937.0, hill=1.2, max_block=70, primary="crumb-2016"),
    }),
    "sotalol": dict(category="high", unii="A6D97U294I", eftpc=14690.0, fu=1.0, channels={
        "IKr": dict(ic50=110600.0, hill=0.8, max_block=80, primary="crumb-2016",
                    sources=[(110600.0, "manual", "crumb-2016"),
                             (319000.0, "automated", "kramer-2013")]),
    }),

    # ----- INTERMEDIATE risk -----
    "cisapride": dict(category="intermediate", unii="UVL329170W", eftpc=2.6, fu=0.025, channels={
        "IKr": dict(hill=0.9, max_block=93, primary="crumb-2016",
                    sources=[(6.5, "manual", "crumb-2016"),
                             (20.0, "automated", "kramer-2013"),
                             (4.0, "manual", "li-2017")]),
        # The v0.8 machine cross-check flagged a >5x divergence from the published
        # Li-2017 value; the raw Crumb-2016 dose-response (the cited source) settles
        # it: cisapride blocks ICaL <=2.5% at the 125 nM top dose tested, so the IC50
        # is UNIDENTIFIABLE (max block << 60%). The point estimate is the Li-2017
        # extrapolated fit, kept for provenance only -> Tier D (cf. ranolazine.ical).
        "ICaL": dict(ic50=9258075.75, hill=0.43, max_block=3, primary="crumb-2016"),
        "INa":  dict(ic50=15000.0, hill=1.0, max_block=60, primary="crumb-2016"),
    }),
    "terfenadine": dict(category="intermediate", unii="73334008C7", eftpc=9.0, fu=0.03, channels={
        "IKr": dict(hill=0.8, max_block=91, primary="crumb-2016",
                    sources=[(8.5, "manual", "crumb-2016"),
                             (56.0, "automated", "kramer-2013"),
                             (20.0, "manual", "li-2017")]),
        "ICaL": dict(ic50=700.0, hill=0.7, max_block=85, primary="crumb-2016"),
        # v0.8 cross-check flag, confirmed against the raw Crumb-2016 data: terfenadine
        # blocks INaL only ~15% at the 800 nM top dose, so the IC50 is UNIDENTIFIABLE
        # (max block << 60%). Point estimate is the Li-2017 extrapolated fit -> Tier D.
        "INaL": dict(ic50=20056.02, hill=0.6, max_block=15, primary="crumb-2016"),
        "INa":  dict(ic50=1303.0, hill=0.9, max_block=80, primary="crumb-2016"),
    }),
    "ondansetron": dict(category="intermediate", unii="4AF302ESOS", eftpc=139.0, fu=0.27, channels={
        "IKr": dict(hill=0.9, max_block=86, primary="crumb-2016",
                    sources=[(1320.0, "manual", "crumb-2016"),
                             (900.0, "manual", "li-2017")]),
        "ICaL": dict(ic50=22551.0, hill=0.8, max_block=65, primary="crumb-2016"),
        "INa":  dict(ic50=19444.0, hill=1.0, max_block=62, primary="crumb-2016"),
        "INaL": dict(ic50=19000.0, hill=1.0, max_block=60, primary="crumb-2016"),
    }),
    "chlorpromazine": dict(category="intermediate", unii="U42B7VYA4P", eftpc=38.0, fu=0.04, channels={
        "IKr": dict(hill=0.8, max_block=82, primary="crumb-2016",
                    sources=[(1470.0, "manual", "crumb-2016"),
                             (1000.0, "manual", "li-2017")]),
        "ICaL": dict(ic50=8191.0, hill=0.8, max_block=72, primary="crumb-2016"),
        "INaL": dict(ic50=4559.0, hill=0.9, max_block=70, primary="crumb-2016"),
        "INa":  dict(ic50=4710.0, hill=2.0, max_block=68, primary="crumb-2016"),
    }),

    # ----- LOW risk -----
    "diltiazem": dict(category="low", unii="EE92BBP03H", eftpc=122.0, fu=0.2, channels={
        "IKr": dict(ic50=13150.0, hill=0.9, max_block=70, primary="crumb-2016"),
        "ICaL": dict(ic50=112.0, hill=0.7, max_block=90, primary="crumb-2016"),  # strong CaL → protective
        "INaL": dict(ic50=21000.0, hill=0.7, max_block=62, primary="crumb-2016"),
    }),
    "mexiletine": dict(category="low", unii="1U511HHV4Z", eftpc=4129.0, fu=0.55, channels={
        # Borderline IKr block: max ~60% -> identifiable but only just; Tier C.
        "IKr": dict(ic50=28900.0, hill=0.9, max_block=60, primary="crumb-2016"),
        "INaL": dict(ic50=8957.0, hill=1.4, max_block=80, primary="crumb-2016"),  # late-Na block → protective
        "INa":  dict(ic50=30329.0, hill=1.2, max_block=62, primary="crumb-2016"),
    }),
    "ranolazine": dict(category="low", unii="A6IEZ5M406", eftpc=1948.0, fu=0.38, channels={
        "IKr": dict(ic50=12000.0, hill=0.9, max_block=70, primary="crumb-2016"),
        "INaL": dict(ic50=7884.0, hill=0.9, max_block=78, primary="crumb-2016"),  # late-Na block → protective
        # Deliberate Tier-D example: ranolazine is a weak L-type Ca blocker; the
        # maximum block observed is below ~60%, so the IC50 is UNIDENTIFIABLE.
        "ICaL": dict(ic50=296000.0, hill=1.0, max_block=35, primary="crumb-2016"),
    }),
    "verapamil": dict(category="low", unii="CJ0O37KU29", eftpc=81.0, fu=0.1, channels={
        "IKr": dict(hill=1.0, max_block=85, primary="crumb-2016",
                    sources=[(250.0, "manual", "crumb-2016"),
                             (143.0, "automated", "kramer-2013"),
                             (450.0, "manual", "li-2017")],
                    # Verapamil is a fast-off, NON-trapped hERG blocker; IC50 =
                    # koff/kon ~= 262 nM (geomean above). Fast off-rate -> dynamic
                    # block ~= static Hill block (no use-dependent accumulation).
                    dynamic=dict(kon=1.91e-5, koff=5.0e-3, trapping=False,
                                 citation="li-2017")),
        "ICaL": dict(ic50=202.0, hill=1.1, max_block=90, primary="crumb-2016"),  # balances hERG → low risk
        "INaL": dict(ic50=7000.0, hill=1.0, max_block=62, primary="crumb-2016"),
    }),

    # ======================================================================= #
    # CiPA VALIDATION SET (16 compounds). Values are literature-derived from the
    # CiPA-era datasets (Crumb 2016 / Li 2017, 2019) and ship unverified. EFTPC =
    # free therapeutic Cmax. UNIIs are left null where not confirmed (honest).
    # ======================================================================= #
    # ----- HIGH risk (validation) -----
    "azimilide": dict(category="high", cipa_set="validation", eftpc=70.0, channels={
        "IKr": dict(ic50=1130.0, hill=0.9, max_block=85, primary="crumb-2016"),
        "IKs": dict(ic50=2300.0, hill=1.0, max_block=70, primary="crumb-2016"),
    }),
    "disopyramide": dict(category="high", cipa_set="validation", eftpc=742.0, channels={
        "IKr": dict(ic50=7240.0, hill=0.9, max_block=80, primary="crumb-2016"),
        "INa": dict(ic50=16000.0, hill=1.0, max_block=70, primary="crumb-2016"),
        "ICaL": dict(ic50=24000.0, hill=1.0, max_block=62, primary="crumb-2016"),
    }),
    "ibutilide": dict(category="high", cipa_set="validation", eftpc=0.5, channels={
        "IKr": dict(ic50=14.0, hill=0.9, max_block=90, primary="crumb-2016"),
    }),
    "vandetanib": dict(category="high", cipa_set="validation", eftpc=256.0, channels={
        "IKr": dict(ic50=400.0, hill=0.9, max_block=85, primary="kramer-2013"),
    }),

    # ----- INTERMEDIATE risk (validation) -----
    "astemizole": dict(category="intermediate", cipa_set="validation", eftpc=0.3, channels={
        "IKr": dict(ic50=0.9, hill=0.9, max_block=92, primary="kramer-2013"),
        "ICaL": dict(ic50=2400.0, hill=1.0, max_block=65, primary="crumb-2016"),
    }),
    "clarithromycin": dict(category="intermediate", cipa_set="validation", eftpc=1206.0, channels={
        "IKr": dict(ic50=32900.0, hill=0.9, max_block=70, primary="crumb-2016"),
    }),
    "clozapine": dict(category="intermediate", cipa_set="validation", eftpc=71.0, channels={
        "IKr": dict(ic50=3200.0, hill=0.8, max_block=75, primary="crumb-2016"),
        "INa": dict(ic50=9900.0, hill=1.0, max_block=62, primary="crumb-2016"),
        "ICaL": dict(ic50=6900.0, hill=0.9, max_block=65, primary="crumb-2016"),
    }),
    "domperidone": dict(category="intermediate", cipa_set="validation", eftpc=19.0, channels={
        "IKr": dict(ic50=57.0, hill=0.9, max_block=88, primary="kramer-2013"),
    }),
    "droperidol": dict(category="intermediate", cipa_set="validation", eftpc=6.3, channels={
        "IKr": dict(ic50=60.0, hill=0.9, max_block=88, primary="kramer-2013"),
    }),
    "pimozide": dict(category="intermediate", cipa_set="validation", eftpc=0.43, channels={
        "IKr": dict(ic50=18.0, hill=0.9, max_block=90, primary="kramer-2013"),
        "ICaL": dict(ic50=45.0, hill=1.0, max_block=80, primary="crumb-2016"),
    }),
    "risperidone": dict(category="intermediate", cipa_set="validation", eftpc=1.8, channels={
        "IKr": dict(ic50=298.0, hill=0.9, max_block=85, primary="kramer-2013"),
    }),

    # ----- LOW risk (validation) -----
    "loratadine": dict(category="low", cipa_set="validation", eftpc=0.45, channels={
        "IKr": dict(ic50=9200.0, hill=0.9, max_block=72, primary="crumb-2016"),
        "ICaL": dict(ic50=28000.0, hill=1.0, max_block=60, primary="crumb-2016"),
    }),
    "metoprolol": dict(category="low", cipa_set="validation", eftpc=1800.0, channels={
        "IKr": dict(ic50=28000.0, hill=0.9, max_block=62, primary="crumb-2016"),
    }),
    "nifedipine": dict(category="low", cipa_set="validation", eftpc=7.7, channels={
        "IKr": dict(ic50=44000.0, hill=0.9, max_block=62, primary="crumb-2016"),
        "ICaL": dict(ic50=60.0, hill=1.0, max_block=92, primary="crumb-2016"),  # strong CaL → protective
    }),
    "nitrendipine": dict(category="low", cipa_set="validation", eftpc=3.0, channels={
        "IKr": dict(ic50=10000.0, hill=0.9, max_block=65, primary="crumb-2016"),
        "ICaL": dict(ic50=100.0, hill=1.0, max_block=90, primary="crumb-2016"),  # strong CaL → protective
    }),
    "tamoxifen": dict(category="low", cipa_set="validation", eftpc=25.0, channels={
        "IKr": dict(ic50=800.0, hill=0.9, max_block=80, primary="crumb-2016"),
        "ICaL": dict(ic50=6800.0, hill=1.0, max_block=62, primary="crumb-2016"),
    }),
}


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def quartiles(xs):
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return [s[0], s[0]]

    def q(p):
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return [round(q(0.25), 4), round(q(0.75), 4)]


def assign_tier(n_sources, fold_range, max_block):
    """Data-quality tier (orthogonal to review_status)."""
    if max_block is not None and max_block < 60:
        return "D"  # IC50 unidentifiable
    if n_sources >= 3 and fold_range is not None and fold_range <= 2.5:
        return "A"
    if n_sources >= 2:
        return "B"
    if max_block is not None and max_block >= 70:
        return "B"
    return "C"


# --------------------------------------------------------------------------- #
# Published-reference corroboration -> review_status (spec v0.8.2).
#
# A channel-block value that AGREES (within 5x — the dataset's own divergence
# threshold) with the INDEPENDENT published CiPA reference table
# (dataset/references/cipa_block_reference.json) is promoted from "unverified"
# (literature-transcribed, uncorroborated) to "pending_human_review" (machine-
# corroborated against a real published source, awaiting human confirmation
# against the source PDF). This is NEVER "verified" — an LLM/automated process
# may not promote to verified (spec.md §9); pending_human_review is the honest
# middle state that distinguishes sourced data from illustrative placeholders.
#
# Only IDENTIFIABLE channels (max block >= 60%) are eligible: a Tier-D channel's
# IC50 is an extrapolation, so "sourced and awaiting confirmation" would mislead.
# --------------------------------------------------------------------------- #
CORROBORATION_FOLD = 5.0
_REFERENCE_INDEX = None


def _load_references():
    """Merge the published reference tables into one index:
    (drug, channel) -> (ic50_nm, citation_key, source_text). Training (Li 2017,
    12 drugs) takes precedence over the validation table (Ridder 2020 hERG +
    Li 2019 ICaL, 16 drugs) where both exist (they do not overlap in practice)."""
    global _REFERENCE_INDEX
    if _REFERENCE_INDEX is not None:
        return _REFERENCE_INDEX
    idx = {}
    refdir = HERE.parent / "references"
    train = refdir / "cipa_block_reference.json"
    if train.is_file():
        data = json.loads(train.read_text(encoding="utf-8"))
        src = data.get("source", {})
        cit = src.get("citation", "li-2017")
        stext = src.get("published_table", "published CiPA reference table")
        for e in data["entries"]:
            if e.get("ic50_nm"):
                idx[(e["drug"].lower(), e["channel"].lower())] = (float(e["ic50_nm"]), cit, stext)
    valn = refdir / "cipa_validation_reference.json"
    if valn.is_file():
        data = json.loads(valn.read_text(encoding="utf-8"))
        sources = data.get("sources", {})
        for e in data["entries"]:
            if e.get("ic50_nm"):
                stext = sources.get(e["channel"], {}).get("source", "published reference")
                idx.setdefault((e["drug"].lower(), e["channel"].lower()),
                               (float(e["ic50_nm"]), e.get("citation", ""), stext))
    _REFERENCE_INDEX = idx
    return idx


def corroborate(drug, channel, central, max_block):
    """If the value is identifiable and agrees (<=5x) with an independent
    published reference, return (review_status, corroboration_dict); else
    (None, None)."""
    if max_block is not None and max_block < 60:
        return None, None  # unidentifiable -> not eligible (its IC50 is extrapolated)
    if not central:
        return None, None
    entry = _load_references().get((drug.lower(), channel.lower()))
    if not entry:
        return None, None
    ref_ic50, citation, source_text = entry
    fold = max(central / ref_ic50, ref_ic50 / central)
    if fold > CORROBORATION_FOLD:
        return None, None
    return "pending_human_review", {
        "source": source_text,
        "citation": citation,
        "ic50_fold_diff": round(fold, 3),
        "note": "IC50 machine-corroborated against an independent published value; "
                "a human has NOT confirmed it against the source PDF, so it is "
                "pending_human_review, not verified (spec.md §9).",
    }


def _channel_extraction(drug, channel, central, max_block):
    """Build the channel-block extraction block, setting review_status to
    pending_human_review (with provenance) when the value is corroborated against
    the published reference, else unverified. Never verified (spec.md §9)."""
    status, corroboration = corroborate(drug, channel, central, max_block)
    if status == "pending_human_review":
        return {
            "review_status": "pending_human_review",
            "method": "literature value transcribed by maintainer, then machine-corroborated "
                      "against the independent published CiPA reference table",
            "verified_by": [],
            "corroboration": corroboration,
            "notes": "IC50 agrees with an independent published value (see corroboration). "
                     "Still awaiting human confirmation against the source PDF — "
                     "pending_human_review, NOT verified (spec.md §9).",
        }
    return {
        "review_status": "unverified",
        "method": "literature table transcribed by maintainer; values from CiPA-era patch-clamp datasets",
        "verified_by": [],
        "notes": "Values are literature-derived (see primary_citation / source_values). "
                 "No source PDF has been confirmed inside Harmonia, and no independent "
                 "published reference corroborates this value, so review_status stays "
                 "unverified per spec §9.",
    }


# --------------------------------------------------------------------------- #
# CiPA dynamic-hERG binding kinetics (spec v0.6).
# The optimized per-drug parameters of the Li et al. 2017 IKr-Markov drug-binding
# model (saturating, voltage-dependent trapping), validated in Li et al. 2019.
# Values are the FDA/CiPA repository optimal fits
# (https://github.com/FDA/CiPA, hERG_fitting/results/<drug>/pars.txt):
#   Kmax     max binding-rate scale (dimensionless)
#   Ku       unbinding rate (ms^-1)
#   n        Hill coefficient of the concentration-dependent binding rate
#   halfmax  n-th power of the half-maximal concentration (nM^n)
#   Vhalf    trapping half-voltage (mV); near 0 => trapped, very negative => washes out
# Kt (the closed-channel trapping rate) is fixed at 3.5e-5 ms^-1 for all drugs.
# ONLY the 12 CiPA compounds with published Milnes-protocol dynamic fits appear here;
# the other 16 have no dynamic data and keep static Hill block (never fabricated).
CIPA_KT = 3.5e-05
CIPA_HERG = {
    "dofetilide":     dict(Kmax=35.1,      Ku=1.816e-05, n=1.08,   halfmax=216.6,       Vhalf=-1.0),
    "bepridil":       dict(Kmax=5594000.0, Ku=0.0001719, n=0.9374, halfmax=147200000.0, Vhalf=-61.34),
    "cisapride":      dict(Kmax=10.22,     Ku=0.0004161, n=0.9615, halfmax=42.32,       Vhalf=-167.4),
    "chlorpromazine": dict(Kmax=157900.0,  Ku=0.04671,   n=0.8871, halfmax=43510000.0,  Vhalf=-14.45),
    "diltiazem":      dict(Kmax=182500.0,  Ku=0.282,     n=0.9382, halfmax=667700000.0, Vhalf=-90.65),
    "mexiletine":     dict(Kmax=15.0,      Ku=0.07114,   n=1.139,  halfmax=723000.0,    Vhalf=-87.51),
    "ondansetron":    dict(Kmax=172000.0,  Ku=0.02324,   n=0.891,  halfmax=52240000.0,  Vhalf=-82.2),
    "quinidine":      dict(Kmax=275.7,     Ku=0.004103,  n=0.8488, halfmax=53830.0,     Vhalf=-61.35),
    "ranolazine":     dict(Kmax=52.84,     Ku=0.02035,   n=0.9532, halfmax=143000.0,    Vhalf=-94.99),
    "sotalol":        dict(Kmax=96190.0,   Ku=0.02225,   n=0.7513, halfmax=385600000.0, Vhalf=-51.5),
    "terfenadine":    dict(Kmax=102200.0,  Ku=7.788e-05, n=0.6502, halfmax=409500.0,    Vhalf=-81.63),
    "verapamil":      dict(Kmax=1694000.0, Ku=0.0008165, n=1.043,  halfmax=335600000.0, Vhalf=-97.08),
}


def build_channel_block(drug, dmeta, channel, cmeta):
    primary = cmeta["primary"]
    max_block = cmeta.get("max_block")

    if "sources" in cmeta:
        src = cmeta["sources"]
        ic50s = [s[0] for s in src]
        source_values = [
            dict(ic50_nm=s[0], platform=s[1], citation=s[2],
                 hill=cmeta.get("hill"), temperature_c=37)
            for s in src
        ]
        central = round(geomean(ic50s), 4)
        fold_range = round(max(ic50s) / min(ic50s), 3)
        n_sources = len(ic50s)
        variability = dict(
            fold_range=fold_range, n_sources=n_sources,
            iqr_nm=quartiles(ic50s), geomean_nm=central,
        )
        low, high = round(min(ic50s), 4), round(max(ic50s), 4)
    else:
        central = cmeta["ic50"]
        source_values = [dict(ic50_nm=central, platform="manual",
                              citation=primary, hill=cmeta.get("hill"),
                              temperature_c=37)]
        fold_range = 1.0
        n_sources = 1
        variability = dict(fold_range=1.0, n_sources=1, iqr_nm=[central, central],
                           geomean_nm=central)
        low = high = central

    tier = assign_tier(n_sources, fold_range, max_block)

    failure_modes = []
    if max_block is not None and max_block < 60:
        failure_modes.append(dict(
            condition="max_block_observed_percent < 60",
            behavior="IC50 not identifiable; any point estimate is an extrapolation and is misleading",
            action="tier_down_to_D and warn; exclude from point classification",
            citation="chang-2017"))
    if fold_range is not None and fold_range > 5:
        failure_modes.append(dict(
            condition="inter-source IC50 fold-range > 5",
            behavior="platform/lab-dependent block potency; risk classification can flip with the source chosen",
            action="propagate the full source distribution (Monte-Carlo), never a single value",
            citation="chang-2017"))

    dyn = cmeta.get("dynamic")
    rec = {
        "id": f"channel_block.{drug}.{channel.lower()}",
        "kind": "channel_block",
        "subsystem": "channel_block",
        "tier": tier,
        "primary_citation": primary,
        "drug": {"name": drug, "unii": dmeta.get("unii"), "chembl": None},
        "channel": channel,
        "block_model": "dynamic_binding" if dyn else "hill",
        "parameters": [
            {"symbol": "IC50", "label": "half-maximal inhibitory concentration",
             "value": {"central": central, "low": low, "high": high, "units": "nM"},
             "tier": tier, "primary_citation": primary},
            {"symbol": "h", "label": "Hill coefficient",
             "value": {"central": cmeta.get("hill", 1.0), "low": None, "high": None,
                       "units": "dimensionless"},
             "tier": tier, "primary_citation": primary},
        ],
        "assay_context": {
            "platform": "mixed" if n_sources > 1 else "manual_patch_clamp",
            "temperature_c": 37,
            "expression_system": "HEK293 / CHO (heterologous)",
            "max_block_observed_percent": max_block,
            "holding_protocol": "depolarizing step protocol per source assay",
        },
        "source_values": source_values,
        "variability": variability,
        "known_failure_modes": failure_modes,
        "extraction": _channel_extraction(drug, channel, central, max_block),
    }
    if dyn:
        rec["dynamic_binding"] = {
            "kon": dyn["kon"], "koff": dyn["koff"], "trapping": dyn["trapping"],
            "citation": dyn.get("citation", "li-2017"),
        }
    if channel == "IKr" and drug in CIPA_HERG:
        rec["cipa_binding"] = {
            **CIPA_HERG[drug], "Kt": CIPA_KT,
            "model": "CiPA IKr-Markov drug binding (Li 2017), saturating + "
                     "voltage-dependent trapping",
            "source": "FDA/CiPA hERG_fitting optimal fit "
                      "(github.com/FDA/CiPA, hERG_fitting/results/<drug>/pars.txt)",
            "citation": "li-2017", "validation_citation": "li-2019",
        }
    if tier == "D":
        rec["notes"] = ("Max block observed < 60% — the IC50 is UNIDENTIFIABLE. The value stored "
                        "is an extrapolation kept only for provenance; it is excluded from point "
                        "classification and flagged in every assessment that touches this channel.")
    return rec


def build_drug_reference(drug, dmeta):
    cipa_set = dmeta.get("cipa_set", "training")
    cit = "li-2017" if cipa_set == "training" else "li-2019"
    rec = {
        "id": f"drug_reference.{drug}",
        "kind": "drug_reference",
        "subsystem": "drug_reference_sets",
        "tier": "B",
        "primary_citation": cit,
        "drug": {"name": drug, "unii": dmeta.get("unii"), "chembl": None},
        "cipa_set": cipa_set,
        "expert_risk_label": dmeta["category"],
        "eftpc_nm": {"central": dmeta["eftpc"], "units": "nM", "kind": "free",
                     "citation": cit},
        "extraction": {
            "review_status": "unverified",
            "method": f"CiPA working-group {cipa_set} set + free Cmax (EFTPC) from CiPA references",
            "verified_by": [],
            "notes": f"Expert risk label and EFTPC from the CiPA {cipa_set}-set literature; unverified.",
        },
        "notes": "Expert consensus risk label is GROUND TRUTH for classifier calibration/scoring only — "
                 "it is NOT a Harmonia output and NOT a clinical determination.",
    }
    fu = dmeta.get("fu")
    if fu is not None:
        # free = fraction_unbound * total; store total derived from the free EFTPC
        rec["protein_binding"] = {
            "fraction_unbound": fu,
            "total_cmax_nm": round(dmeta["eftpc"] / fu, 3),
            "citation": cit,
        }
    return rec


def build_ap_models():
    common_currents = ["INa", "INaL", "Ito", "ICaL", "IKr", "IKs", "IK1"]
    models = []
    models.append({
        "id": "ap_model.ord",
        "kind": "ap_model",
        "subsystem": "ap_models",
        "tier": "C",
        "primary_citation": "ohara-2011",
        "model": {
            "name": "O'Hara-Rudy (ORd) — reduced reference kernel",
            "lineage": "O'Hara-Rudy",
            "formulation": "human ventricular endocardial AP; Hodgkin-Huxley + algebraic IK1",
            "currents": common_currents,
        },
        "model_parameters": [
            {"symbol": "g_scale_IKr", "label": "IKr conductance scale", "value": 1.0, "units": "dimensionless"},
            {"symbol": "g_scale_ICaL", "label": "ICaL conductance scale", "value": 1.0, "units": "dimensionless"},
            {"symbol": "g_scale_INaL", "label": "INaL conductance scale", "value": 1.0, "units": "dimensionless"},
        ],
        "validation": {"training_set_reproduced": None, "validation_set_accuracy": None,
                       "notes": "Bundled kernel is a REDUCED reference implementation of the ORd "
                                "lineage — structurally faithful (7 named currents, Hill block) and "
                                "numerically stable, but not bit-exact to the published CellML. "
                                "Tier C until cross-checked vs canonical CellML (Phase F)."},
        "extraction": {"review_status": "unverified", "method": "reduced kernel implemented from the ORd lineage",
                       "verified_by": [], "notes": "Equations not yet cross-checked vs canonical CellML."},
    })
    models.append({
        "id": "ap_model.cipaordv1.0",
        "kind": "ap_model",
        "subsystem": "ap_models",
        "tier": "C",
        "primary_citation": "dutta-2017",
        "model": {
            "name": "IKr-dynamic CiPAORd v1.0 — reduced reference variant",
            "lineage": "O'Hara-Rudy",
            "formulation": "ORd re-optimized for CiPA; IKr re-scaled. Dynamic hERG binding "
                           "(Langmuir kon/koff with trapping) is available at simulation time "
                           "for drugs carrying dynamic_binding kinetics (assess(..., herg_dynamic=True)).",
            "currents": common_currents,
        },
        "model_parameters": [
            {"symbol": "g_scale_IKr", "label": "IKr conductance scale", "value": 1.20, "units": "dimensionless"},
            {"symbol": "g_scale_ICaL", "label": "ICaL conductance scale", "value": 1.0, "units": "dimensionless"},
            {"symbol": "g_scale_INaL", "label": "INaL conductance scale", "value": 1.30, "units": "dimensionless"},
        ],
        "validation": {"training_set_reproduced": None, "validation_set_accuracy": None,
                       "notes": "Default classification model. The reduced-kernel default metric is qNet "
                                "(discriminating since the Na-Ca exchanger was added and excluded from the "
                                "qNet sum, Phase C): 10/12 CiPA training labels and zero two-category errors "
                                "across all 28 compounds; APD90 selectable. Run `harmonia performance` for the "
                                "live, honest confusion matrix. Not a qualified regulatory classifier (Tier C)."},
        "extraction": {"review_status": "unverified", "method": "reduced CiPAORd variant with dynamic-hERG option",
                       "verified_by": [], "notes": "Dynamic Langmuir hERG binding implemented (Phase B); "
                       "full CiPA Markov hERG model + published optimized kinetics pending Phase C."},
    })
    models.append({
        "id": "ap_model.tor_ord",
        "kind": "ap_model",
        "subsystem": "ap_models",
        "tier": "C",
        "primary_citation": "tomek-2019",
        "model": {
            "name": "Tomek-O'Hara-Rudy (ToR-ORd) — reduced reference variant",
            "lineage": "O'Hara-Rudy",
            "formulation": "updated ORd (reformulated INaL, ICaL, IKr, Ca handling). "
                           "v0.1 kernel approximates it with a rebalanced-conductance variant.",
            "currents": common_currents,
        },
        "model_parameters": [
            {"symbol": "g_scale_IKr", "label": "IKr conductance scale", "value": 0.90, "units": "dimensionless"},
            {"symbol": "g_scale_ICaL", "label": "ICaL conductance scale", "value": 1.10, "units": "dimensionless"},
            {"symbol": "g_scale_INaL", "label": "INaL conductance scale", "value": 0.85, "units": "dimensionless"},
        ],
        "validation": {"training_set_reproduced": None, "validation_set_accuracy": None,
                       "notes": "ToR-ORd reformulation is the Phase-C deliverable."},
        "extraction": {"review_status": "unverified", "method": "reduced ToR-ORd variant",
                       "verified_by": [], "notes": "Full reformulation pending Phase C."},
    })
    return models


# shared illustrative inter-individual CVs (loosely inspired by Britton 2013 / Passini
# 2017; deliberately round, NOT calibrated). The disease populations recenter this same
# spread on a diseased mean (spec v0.3 §2).
_POP_CV = {"IKr": 0.22, "IKs": 0.28, "ICaL": 0.18, "INaL": 0.25,
           "Ito": 0.30, "IK1": 0.15, "INaCa": 0.20}

# v0.3 congenital long-QT channelopathies: (gene/mechanism, per-current MEAN shift).
# Illustrative heterozygous-scale magnitudes, NOT genotype-calibrated (spec v0.3 §2).
_LQTS = [
    ("lqt1", "Congenital LQT1 (KCNQ1 loss of function) — illustrative",
     "KCNQ1 loss of function reduces the slow delayed-rectifier IKs",
     {"IKs": 0.5}),
    ("lqt2", "Congenital LQT2 (KCNH2/hERG loss of function) — illustrative",
     "KCNH2 (hERG) loss of function reduces the rapid delayed-rectifier IKr",
     {"IKr": 0.5}),
    ("lqt3", "Congenital LQT3 (SCN5A gain of function) — illustrative",
     "SCN5A gain of function increases the late sodium current INaL",
     {"INaL": 2.0}),
]


def build_populations():
    """Population-of-models specs. HYPOTHESIS-TIER (Tier D) — illustrative
    inter-individual conductance variability and (v0.3) illustrative disease/genetic
    backgrounds, NOT calibrated to human data and NOT for prediction (spec.md §3, §10;
    spec v0.3). The CVs and disease mean shifts are deliberately round, illustrative
    numbers inspired by the literature (Britton 2013; Passini 2017; Moss & Kass 2005)."""
    pops = [{
        "id": "population.illustrative_v0",
        "kind": "population",
        "subsystem": "populations",
        "tier": "D",
        "primary_citation": "britton-2013",
        "population": {
            "name": "Illustrative inter-individual conductance variability (v0)",
            "n_default": 100,
            "conductance_cv": dict(_POP_CV),
            "predictive": False,
        },
        "extraction": {
            "review_status": "unverified",
            "method": "illustrative conductance CVs inspired by population-of-models studies",
            "verified_by": [],
            "notes": "CVs are illustrative, NOT calibrated to experimental population data.",
        },
        "notes": "HYPOTHESIS-TIER — DO NOT USE FOR PREDICTION. A population of virtual "
                 "myocytes built by sampling conductances at these CVs illustrates how "
                 "inter-individual repolarization-reserve variability could spread a "
                 "drug's risk classification across a population. The CVs are not "
                 "calibrated to human data; every population assessment is capped at "
                 "Tier D and labelled non-predictive.",
    }]
    # v0.5 experimentally-calibrated population (Britton et al. 2013): the same
    # illustrative variability cloud, but a candidate myocyte is admitted only if
    # its DRUG-FREE AP biomarkers fall in plausible ranges (and it repolarizes),
    # which removes the abnormal repolarization tail of the raw prior. The ranges
    # are kernel-plausibility bounds (this kernel's own units; baseline APD90 ~272
    # ms, triangulation ~42 ms), bracketing the physiological bulk and rejecting
    # the long/triangular-repolarization outliers — NOT a fit to patient data, so
    # the assessment stays Tier D / non-predictive.
    pops.append({
        "id": "population.calibrated_v0",
        "kind": "population",
        "subsystem": "populations",
        "tier": "D",
        "primary_citation": "britton-2013",
        "population": {
            "name": "Experimentally-calibrated population (drug-free AP biomarker acceptance, v0)",
            "n_default": 100,
            "conductance_cv": dict(_POP_CV),
            "calibration": {
                "method": "drug-free AP biomarker acceptance / calibration-by-rejection "
                          "(Britton et al. 2013): keep only candidate myocytes whose "
                          "drug-free APD90, resting and peak potential, and triangulation "
                          "fall within plausible ranges and that repolarize",
                "citation": "britton-2013",
                "max_oversample": 40,
                "biomarkers": {
                    "apd90_ms":          {"min": 180.0, "max": 360.0},
                    "vrest_mv":          {"min": -90.0, "max": -80.0},
                    "vpeak_mv":          {"min": 40.0,  "max": 70.0},
                    "triangulation_ms":  {"min": 10.0,  "max": 120.0},
                },
            },
            "predictive": False,
        },
        "extraction": {
            "review_status": "unverified",
            "method": "calibration ranges are kernel-plausibility bounds bracketing the "
                      "reduced kernel's drug-free biomarker bulk; methodology after Britton 2013",
            "verified_by": [],
            "notes": "Ranges are kernel-specific plausibility bounds, NOT a fit to human "
                     "population electrophysiology data; the methodology (not the numbers) "
                     "is Britton et al. 2013.",
        },
        "notes": "HYPOTHESIS-TIER — DO NOT USE FOR PREDICTION. The experimentally-calibrated "
                 "populations-of-models method (Britton et al. 2013) admits a virtual myocyte "
                 "only when its DRUG-FREE action potential is physiologically plausible, so the "
                 "population excludes the abnormal repolarization tail that the raw prior cloud "
                 "(illustrative_v0) contains. Here the acceptance ranges are bounds on THIS "
                 "kernel's own biomarkers (not borrowed regulatory or patient-fit values), so "
                 "the calibration buys physiological plausibility, not predictiveness: every "
                 "assessment is still capped at Tier D and labelled non-predictive. The "
                 "drug-response thresholds remain the healthy reference.",
    })
    for pid, name, mechanism, scale in _LQTS:
        shift = ", ".join(f"{c}x{v:g}" for c, v in scale.items())
        pops.append({
            "id": f"population.{pid}",
            "kind": "population",
            "subsystem": "populations",
            "tier": "D",
            "primary_citation": "moss-2005",
            "population": {
                "name": name,
                "n_default": 100,
                "conductance_cv": dict(_POP_CV),
                "conductance_scale": scale,
                "predictive": False,
            },
            "extraction": {
                "review_status": "unverified",
                "method": "illustrative congenital long-QT background: the dominant "
                          "channel defect applied as a mean conductance shift under the "
                          "shared illustrative CVs",
                "verified_by": [],
                "notes": "Heterozygous-scale, illustrative magnitude — NOT genotype-calibrated.",
            },
            "notes": f"HYPOTHESIS-TIER — DO NOT USE FOR PREDICTION. {mechanism} "
                     f"(mean shift {shift}), here recentering the illustrative population "
                     f"spread on a reduced-repolarization-reserve background. The magnitude "
                     f"is an illustrative heterozygous-scale shift, NOT a genotype-calibrated "
                     f"parameter; qNet/APD thresholds remain the HEALTHY reference. A "
                     f"mechanism demonstration of the gene-drug repolarization-reserve "
                     f"interaction, never a per-patient or per-genotype safety claim. Every "
                     f"assessment is capped at Tier D and labelled non-predictive.",
        })
    return pops


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    RECORDS.mkdir(parents=True, exist_ok=True)
    CITATIONS.mkdir(parents=True, exist_ok=True)

    # wipe previously generated files so removals propagate
    for p in RECORDS.glob("*.json"):
        p.unlink()
    for p in CITATIONS.glob("*.json"):
        p.unlink()

    n_rec = 0
    for drug, dmeta in DRUGS.items():
        for channel, cmeta in dmeta["channels"].items():
            rec = build_channel_block(drug, dmeta, channel, cmeta)
            write_json(RECORDS / f"{rec['id']}.json", rec)
            n_rec += 1
        ref = build_drug_reference(drug, dmeta)
        write_json(RECORDS / f"{ref['id']}.json", ref)
        n_rec += 1

    for m in build_ap_models():
        write_json(RECORDS / f"{m['id']}.json", m)
        n_rec += 1

    for pop in build_populations():
        write_json(RECORDS / f"{pop['id']}.json", pop)
        n_rec += 1

    for c in CITATIONS_TABLE:
        cid = c["key"]
        obj = {
            "key": cid,
            "type": c["type"],
            "title": c["title"],
            "authors": c["authors"],
            "journal": c["journal"],
            "year": c["year"],
            "doi": c["doi"],
            "pmid": c.get("pmid"),
            "url": f"https://doi.org/{c['doi']}",
        }
        write_json(CITATIONS / f"{cid}.json", obj)

    print(f"wrote {n_rec} records and {len(CITATIONS_TABLE)} citations")


if __name__ == "__main__":
    main()
