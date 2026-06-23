#!/usr/bin/env python3
"""Generate the published-CiPA block-potency reference table used by the
machine cross-check layer (spec v0.8).

WHAT THIS IS. A second, *independent* copy of the canonical CiPA multichannel
block table — IC50 (nM) and Hill coefficient per drug x current for the 12 CiPA
training drugs — transcribed from **Li et al. 2017** (Circ Arrhythm
Electrophysiol; DOI 10.1161/CIRCEP.116.004628, the ``li-2017`` citation already
in Harmonia). It exists so ``harmonia crosscheck`` can diff every channel-block
record's transcribed IC50/Hill against a published value that did NOT come from
the same transcription pass — turning "trust the curator's number" into a
measured agreement (or a flagged divergence).

WHY THIS IS NOT CIRCULAR. Harmonia's records were transcribed by the maintainer
from the primary literature. The values below were obtained, in machine-readable
form, from the FDA/CiPA reference implementation
(github.com/FDA/CiPA, ``Hill_fitting/data/Li2017_IC50.csv`` @ commit e881df6).
Agreement between the two independent renderings of the same published table is
real evidence the transcription is faithful; divergence flags a record for human
review. A machine cross-check is explicitly NOT a human ``verified`` stamp
(spec.md §9): nobody has opened the source PDF inside Harmonia. It is a weaker,
honestly-labeled provenance signal.

LICENSING. The FDA/CiPA *software* repository is GPL-3.0. The numeric IC50/Hill
*values* reproduced here are uncopyrightable factual measurements originating in
the Li 2017 publication; they are redistributed under Harmonia's CC-BY-4.0
dataset license with attribution to Li 2017. No GPL-licensed file is vendored —
only the facts, with the FDA/CiPA repository credited as the machine-readable
conduit.

REPRODUCIBILITY. The committed ``dataset/references/cipa_block_reference.json``
must be byte-identical to a fresh run of this tool (CI checks this), exactly like
``build_records.py``.
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REFERENCES = HERE.parent / "references"

# Provenance of the machine-readable conduit (NOT the copyright source — that is
# the Li 2017 paper). Pinned to the exact commit the values were read from.
SOURCE = {
    "citation": "li-2017",
    "published_table": "Li et al. 2017, Circ Arrhythm Electrophysiol, "
                       "DOI 10.1161/CIRCEP.116.004628 — multichannel IC50/Hill for the "
                       "12 CiPA training drugs.",
    "machine_readable_conduit": "github.com/FDA/CiPA, Hill_fitting/data/Li2017_IC50.csv",
    "conduit_commit": "e881df6766d1067a08b65184d7a320804aef73d6",
    "conduit_repo_license": "GPL-3.0 (software); the numeric values reproduced here are "
                            "uncopyrightable facts from the Li 2017 publication, "
                            "redistributed under CC-BY-4.0 with attribution.",
}

# The published Li 2017 table, verbatim facts (FDA/CiPA Li2017_IC50.csv). IC50 in
# nM, Hill dimensionless; "NA" = the current was not fit for that drug. Header
# order is the source file's; hERG maps to Harmonia's IKr channel name.
LI2017_IC50_CSV = """\
drug,ICaL_h,ICaL_IC50,Ito_h,Ito_IC50,IK1_h,IK1_IC50,IKs_h,IKs_IC50,INaL_h,INaL_IC50,INa_h,INa_IC50,hERG_IC50,hERG_h
dofetilide,1.16,260.32,0.77,18.82,0.77,394.26,NA,NA,0.26,753160.41,0.89,380.5,4.87,0.93
bepridil,0.648641112,2808.055446,3.54,8594.04,NA,NA,0.706125727,28628.30722,1.42,1813.91,1.163653755,2929.272579,50,0.92
sotalol,0.86510548,7061526.92,0.66,43143455.32,1.204371947,3050260.348,1.167218474,4221855.541,NA,NA,0.51,1.14e+09,110600,0.7678
quinidine,0.589208762,51592.34546,1.28,3487.43,0.35,39589919,1.36305201,4898.902693,1.34,9416.98,1.493724623,12329.0369,992,0.8199
cisapride,0.43,9258075.75,0.24,219112.35,0.51,29498.04,0.29,81192862,NA,NA,NA,NA,10.1,0.73
terfenadine,0.660062997,700.4003343,0.26,239960.82,NA,NA,0.54,399754,0.6,20056.02,1.015027823,4803.18322,23,0.6468
ondansetron,0.752643717,22551.37898,0.99,1023377.67,NA,NA,0.65344884,569807.4225,1.03,19180.8,1.019672018,57666.4315,1320,0.9134
chlorpromazine,0.844107446,8191.915057,0.37,17616711.07,0.687753884,9269.939647,NA,NA,0.94,4559.55,1.995393819,4535.599203,929.2,0.85
verapamil,1.096809669,201.7832944,0.82,13429.25,0.27,348913215.6,NA,NA,1.03,7028.01,NA,NA,288,0.96
ranolazine,NA,NA,NA,NA,NA,NA,0.52,36155020,0.94,7884.47,1.424957384,68774.53398,8270,0.8866
mexiletine,1.030674129,38243.58316,NA,NA,NA,NA,NA,NA,1.41,8956.75,NA,NA,28880,0.9246
diltiazem,0.714205489,112.1303862,0.17,2821526904,NA,NA,NA,NA,0.68,21868.47,0.70224184,110859.7524,13150,0.9254
"""

# Source column-name (current) -> Harmonia channel id. "hERG" is the CiPA name
# for the rapid delayed-rectifier K current Harmonia records as IKr.
CHANNEL_MAP = {
    "ICaL": "ICaL",
    "Ito": "Ito",
    "IK1": "IK1",
    "IKs": "IKs",
    "INaL": "INaL",
    "INa": "INa",
    "hERG": "IKr",
}


def _f(token: str):
    token = token.strip()
    if token == "" or token.upper() == "NA":
        return None
    return float(token)


def build_entries():
    lines = [ln for ln in LI2017_IC50_CSV.strip().splitlines() if ln.strip()]
    header = lines[0].split(",")
    # Map "<current>_IC50" / "<current>_h" column indices.
    cols = {name: i for i, name in enumerate(header)}
    currents = sorted({h.rsplit("_", 1)[0] for h in header if h != "drug"})
    entries = []
    for line in lines[1:]:
        cells = line.split(",")
        drug = cells[0].strip()
        for cur in currents:
            ic50 = _f(cells[cols[f"{cur}_IC50"]]) if f"{cur}_IC50" in cols else None
            hill = _f(cells[cols[f"{cur}_h"]]) if f"{cur}_h" in cols else None
            if ic50 is None and hill is None:
                continue  # current not fit for this drug
            entries.append({
                "drug": drug,
                "channel": CHANNEL_MAP.get(cur, cur),
                "ic50_nm": ic50,
                "hill": hill,
            })
    # Deterministic order: by drug, then by a fixed channel ordering.
    chan_order = {c: i for i, c in enumerate(["IKr", "ICaL", "INa", "INaL", "IKs", "Ito", "IK1"])}
    entries.sort(key=lambda e: (e["drug"], chan_order.get(e["channel"], 99)))
    return entries


def build_reference():
    entries = build_entries()
    drugs = sorted({e["drug"] for e in entries})
    return {
        "schema": "harmonia.cipa_block_reference/v1",
        "description": "Published CiPA multichannel block-potency reference (IC50 nM, "
                       "Hill) for the 12 CiPA training drugs, used by `harmonia crosscheck` "
                       "to independently diff Harmonia's transcribed channel-block records. "
                       "A machine cross-check is NOT a human `verified` stamp (spec.md §9).",
        "source": SOURCE,
        "n_drugs": len(drugs),
        "n_entries": len(entries),
        "drugs": drugs,
        "entries": entries,
    }


# --------------------------------------------------------------------------- #
# Validation-drug reference (the 16 CiPA VALIDATION drugs).
#
# The FDA/CiPA repo tabulates only the 12 TRAINING drugs. The validation set's
# block potencies come from two published sources, each credited per entry:
#
#  * hERG/IKr — Ridder et al. 2020 (Toxicol Appl Pharmacol 394:114961), Table 1:
#    the Bayesian-hierarchical static hERG IC50 (ambient temp, multi-site). These
#    16 values were re-verified against the published PMC table (PMC7166077) when
#    this file was authored — they transcribe exactly.
#  * ICaL — Li et al. 2019 (Clin Pharmacol Ther; the `li-2019` citation),
#    Supplement S2 Table 4 (Manual dataset; the table prints IC50 in µM, converted
#    here to nM x1000). Independently cross-corroborated: these agree within ~5x
#    with Harmonia's own crumb-2016 ICaL values for the drugs that carry one.
#
# This table is used ONLY to corroborate Harmonia's records into
# `pending_human_review` (a cross-method provenance signal); it is deliberately
# NOT fed to the `harmonia crosscheck` transcription-error detector, whose tight
# tolerance assumes the same-lineage Li-2017 training table. INaL/INa were left
# out: the Li-2019 fits for those are very poorly identified (credible intervals
# spanning several orders of magnitude), so they are weak corroboration anchors.
# --------------------------------------------------------------------------- #
# drug -> hERG IC50 nM (Ridder 2020 Table 1, verified vs PMC7166077)
VALIDATION_HERG = {
    "astemizole": 19.0, "azimilide": 380.0, "clarithromycin": 150000.0,
    "clozapine": 1500.0, "disopyramide": 4700.0, "domperidone": 74.0,
    "droperidol": 118.0, "ibutilide": 11.0, "loratadine": 1300.0,
    "metoprolol": 110000.0, "nifedipine": 71000.0, "nitrendipine": 20000.0,
    "pimozide": 19.0, "risperidone": 451.0, "tamoxifen": 1700.0, "vandetanib": 394.0,
}
# drug -> (ICaL IC50 nM, Hill)  (Li 2019 S2 Table 4, Manual; µM x1000)
VALIDATION_ICAL = {
    "astemizole": (553.0, 1.2), "azimilide": (13200.0, 0.71),
    "clarithromycin": (38100.0, 0.88), "clozapine": (5490.0, 0.94),
    "disopyramide": (32900.0, 0.69), "domperidone": (73.6, 0.49),
    "droperidol": (3230.0, 1.2), "ibutilide": (37000.0, 0.86),
    "loratadine": (703.0, 0.56), "metoprolol": (3280000.0, 0.54),
    "nifedipine": (11.4, 0.67), "nitrendipine": (35.7, 0.5),
    "pimozide": (64.5, 0.45), "risperidone": (1470.0, 0.59),
    "tamoxifen": (5720.0, 0.76), "vandetanib": (6060.0, 0.72),
}
VALIDATION_SOURCE = {
    "IKr": {"citation": "ridder-2020",
            "source": "Ridder et al. 2020, Toxicol Appl Pharmacol 394:114961, Table 1 — "
                      "Bayesian-hierarchical static hERG IC50 (ambient temp); verified vs PMC7166077."},
    "ICaL": {"citation": "li-2019",
             "source": "Li et al. 2019, Clin Pharmacol Ther, Supplement S2 Table 4 (Manual "
                       "dataset; µM converted to nM x1000)."},
}


def build_validation_reference():
    entries = []
    for drug in sorted(VALIDATION_HERG):
        entries.append({"drug": drug, "channel": "IKr", "ic50_nm": VALIDATION_HERG[drug],
                        "hill": None, "citation": VALIDATION_SOURCE["IKr"]["citation"]})
    for drug in sorted(VALIDATION_ICAL):
        ic50, hill = VALIDATION_ICAL[drug]
        entries.append({"drug": drug, "channel": "ICaL", "ic50_nm": ic50, "hill": hill,
                        "citation": VALIDATION_SOURCE["ICaL"]["citation"]})
    drugs = sorted({e["drug"] for e in entries})
    return {
        "schema": "harmonia.cipa_validation_reference/v1",
        "description": "Published block-potency reference for the 16 CiPA VALIDATION drugs "
                       "(hERG: Ridder 2020 Table 1, verified; ICaL: Li 2019 S2 Table 4). Used "
                       "ONLY to corroborate records into pending_human_review — a cross-method "
                       "provenance signal, NOT a human `verified` stamp and NOT the "
                       "transcription-error cross-check (spec.md §9, spec v0.8.2).",
        "sources": VALIDATION_SOURCE,
        "n_drugs": len(drugs),
        "n_entries": len(entries),
        "drugs": drugs,
        "entries": entries,
    }


# --------------------------------------------------------------------------- #
# Drug-reference table: free EFTPC (nM) + CiPA consensus risk category, used to
# corroborate the drug_reference records into pending_human_review (spec v0.8.3).
#
#  * Training (12) — the FDA/CiPA reference implementation,
#    `AP_simulation/data/CiPA_training_drugs.csv` (`therapeutic` = free EFTPC nM;
#    `CiPA` = 2/1/0 -> high/intermediate/low). Independent of the `li-2017`
#    primary citation on those records; verified value-by-value when authored
#    (11/12 EFTPCs match exactly, terfenadine 9 vs 4 nM = 2.25x; all 12 categories
#    match).
#  * Validation (16) — Li et al. 2019 (`li-2019`), main Table 1: the CiPA
#    validation-set risk category and free EFTPC. The category is the stable CiPA
#    consensus (it also appears in Colatsky 2016); the EFTPC is the exposure used
#    in the validation simulations.
# --------------------------------------------------------------------------- #
# drug -> (free EFTPC nM, CiPA category)  — TRAINING, FDA/CiPA CiPA_training_drugs.csv
DRUGREF_TRAINING = {
    "quinidine": (3237.0, "high"), "bepridil": (33.0, "high"),
    "dofetilide": (2.0, "high"), "sotalol": (14690.0, "high"),
    "chlorpromazine": (38.0, "intermediate"), "cisapride": (2.6, "intermediate"),
    "terfenadine": (4.0, "intermediate"), "ondansetron": (139.0, "intermediate"),
    "diltiazem": (122.0, "low"), "mexiletine": (4129.0, "low"),
    "ranolazine": (1948.2, "low"), "verapamil": (81.0, "low"),
}
# drug -> (free EFTPC nM, CiPA category)  — VALIDATION, FDA/CiPA newCiPA.csv
# (therapeutic column + CiPA code), verified value-by-value against the raw file
# and cross-corroborated against Llopis-Lorente 2022 Table 1; categories also
# independently confirmed in Colatsky 2016. NB: ibutilide's published EFTPC is
# 100 nM here (FDA + Llopis agree) vs 0.52 nM in Harmonia's record — a 200x
# discrepancy that this table surfaces; ibutilide's record is left UNcorroborated.
DRUGREF_VALIDATION = {
    "astemizole": (0.26, "intermediate"), "azimilide": (70.0, "high"),
    "clarithromycin": (1206.0, "intermediate"), "clozapine": (71.0, "intermediate"),
    "disopyramide": (742.0, "high"), "domperidone": (19.0, "intermediate"),
    "droperidol": (6.33, "intermediate"), "ibutilide": (100.0, "high"),
    "loratadine": (0.45, "low"), "metoprolol": (1800.0, "low"),
    "nifedipine": (7.7, "low"), "nitrendipine": (3.02, "low"),
    "pimozide": (0.431, "intermediate"), "risperidone": (1.81, "intermediate"),
    "tamoxifen": (21.0, "low"), "vandetanib": (255.4, "high"),
}
DRUGREF_SOURCE = {
    "training": {"citation": "li-2017",
                 "source": "FDA/CiPA reference implementation, AP_simulation/data/"
                           "CiPA_training_drugs.csv (therapeutic = free EFTPC nM; "
                           "CiPA risk category). Verified against the file when authored."},
    "validation": {"citation": "li-2019",
                   "source": "FDA/CiPA reference implementation, AP_simulation/data/newCiPA.csv "
                             "(therapeutic = free EFTPC nM; CiPA code 2/1/0 = high/intermediate/low), "
                             "the simulation input behind Li et al. 2019. Verified vs the raw file; "
                             "EFTPC cross-corroborated against Llopis-Lorente 2022 Table 1; categories "
                             "also confirmed in Colatsky 2016."},
}


def build_drugref_reference():
    entries = []
    for drug, (eftpc, cat) in sorted(DRUGREF_TRAINING.items()):
        entries.append({"drug": drug, "eftpc_free_nm": eftpc, "cipa_risk_category": cat,
                        "set": "training", "citation": DRUGREF_SOURCE["training"]["citation"]})
    for drug, (eftpc, cat) in sorted(DRUGREF_VALIDATION.items()):
        entries.append({"drug": drug, "eftpc_free_nm": eftpc, "cipa_risk_category": cat,
                        "set": "validation", "citation": DRUGREF_SOURCE["validation"]["citation"]})
    drugs = sorted({e["drug"] for e in entries})
    return {
        "schema": "harmonia.cipa_drugref_reference/v1",
        "description": "Free EFTPC (nM) + CiPA consensus risk category per drug, used ONLY to "
                       "corroborate drug_reference records into pending_human_review (spec "
                       "v0.8.3). NOT a human `verified` stamp (spec.md §9).",
        "sources": DRUGREF_SOURCE,
        "n_drugs": len(drugs),
        "n_entries": len(entries),
        "drugs": drugs,
        "entries": entries,
    }


def write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ref = build_reference()
    out = REFERENCES / "cipa_block_reference.json"
    write_json(out, ref)
    print(f"wrote {ref['n_entries']} reference entries across {ref['n_drugs']} drugs to {out}")
    vref = build_validation_reference()
    vout = REFERENCES / "cipa_validation_reference.json"
    write_json(vout, vref)
    print(f"wrote {vref['n_entries']} validation reference entries across "
          f"{vref['n_drugs']} drugs to {vout}")
    dref = build_drugref_reference()
    dout = REFERENCES / "cipa_drugref_reference.json"
    write_json(dout, dref)
    print(f"wrote {dref['n_entries']} drug-reference entries across "
          f"{dref['n_drugs']} drugs to {dout}")


if __name__ == "__main__":
    main()
