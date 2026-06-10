"""CiPA dynamic-hERG binding kinetics (spec v0.6).

The published Li-2017 IKr-Markov drug-binding parameters for the 12 CiPA dynamic-fit
compounds, sourced from the FDA/CiPA repository, plus the reduced-kernel binding model
(opt-in via ``herg_dynamic="cipa"``). The data are the durable contribution; the model
is a Tier-C reduction that never touches the default classifier.
"""
import harmonia
from harmonia.export.reference import KernelParams, simulate_beats, CiPABinding

# the 12 CiPA compounds with published Milnes-protocol dynamic-binding fits
CIPA_DYNAMIC_DRUGS = {
    "dofetilide", "bepridil", "cisapride", "chlorpromazine", "diltiazem", "mexiletine",
    "ondansetron", "quinidine", "ranolazine", "sotalol", "terfenadine", "verapamil",
}
_FIELDS = ("Kmax", "Ku", "n", "halfmax", "Vhalf")


def _cipa(ds, drug):
    return ds[f"channel_block.{drug}.ikr"].cipa_binding


def test_cipa_data_present_for_the_12_and_absent_otherwise(ds):
    """Exactly the 12 CiPA dynamic-fit compounds carry cipa_binding; the rest don't
    (no fabricated kinetics for compounds without published dynamic data)."""
    have = {b.drug for b in ds.channel_blocks if b.channel == "IKr" and b.cipa_binding}
    assert have == CIPA_DYNAMIC_DRUGS
    for drug in CIPA_DYNAMIC_DRUGS:
        cb = _cipa(ds, drug)
        for f in _FIELDS:
            assert isinstance(cb[f], (int, float))
        assert cb["Kt"] == 3.5e-05                 # shared fixed trapping rate
        assert cb["citation"] == "li-2017"
        assert cb["validation_citation"] == "li-2019"


def test_cipa_records_ship_unverified(ds):
    """Sourced from the FDA/CiPA repo by the maintainer — never auto-promoted (§9)."""
    for drug in CIPA_DYNAMIC_DRUGS:
        assert ds[f"channel_block.{drug}.ikr"].review_status == "unverified"


def test_cipa_zero_drug_is_no_block(ds):
    """At zero concentration the binding ODE produces no block: the CiPA path equals
    the drug-free AP."""
    cb = _cipa(ds, "dofetilide")
    drug_free = simulate_beats(KernelParams(), n_beats=6).apd90
    zero = simulate_beats(KernelParams(), n_beats=6, herg=CiPABinding(
        conc_nm=0.0, kmax=cb["Kmax"], ku=cb["Ku"], n=cb["n"],
        halfmax=cb["halfmax"], vhalf=cb["Vhalf"])).apd90
    assert abs(drug_free - zero) < 0.5


def test_cipa_block_increases_with_concentration(ds):
    """More drug => more bound hERG (monotone in concentration)."""
    cb = _cipa(ds, "dofetilide")

    def bound(conc):
        return simulate_beats(KernelParams(), n_beats=20, herg=CiPABinding(
            conc_nm=conc, kmax=cb["Kmax"], ku=cb["Ku"], n=cb["n"],
            halfmax=cb["halfmax"], vhalf=cb["Vhalf"])).herg_bound_mean

    assert bound(2.0) < bound(10.0) < bound(50.0)


def test_cipa_trapping_phenotype(ds):
    """At MATCHED concentration, a trapped blocker (dofetilide, Vhalf~0) retains far
    more diastolic block than a washout blocker (verapamil, Vhalf<<0) — the published
    trapping phenotype, reproduced by the Vhalf-governed un-trapping rate."""
    conc = 50.0
    dofb = _cipa(ds, "dofetilide")
    verb = _cipa(ds, "verapamil")
    dof = simulate_beats(KernelParams(), n_beats=25, herg=CiPABinding(
        conc_nm=conc, kmax=dofb["Kmax"], ku=dofb["Ku"], n=dofb["n"],
        halfmax=dofb["halfmax"], vhalf=dofb["Vhalf"]))
    ver = simulate_beats(KernelParams(), n_beats=25, herg=CiPABinding(
        conc_nm=conc, kmax=verb["Kmax"], ku=verb["Ku"], n=verb["n"],
        halfmax=verb["halfmax"], vhalf=verb["Vhalf"]))
    assert dof.herg_bound_mean > ver.herg_bound_mean
    assert dof.apd90 > ver.apd90


def test_cipa_assess_optin_runs(ds):
    """The opt-in assess path runs end to end and reports a finite prolongation."""
    a = harmonia.assess(ds, "dofetilide", n_mc=0, herg_dynamic="cipa", n_beats_dynamic=20)
    assert a.herg_dynamic is True
    assert a.dapd90_pct == a.dapd90_pct   # finite (not NaN)


def test_default_path_unaffected_by_cipa_data(ds):
    """Adding cipa_binding to the dataset must not change the DEFAULT assessment:
    the default never reads cipa_binding, and herg_dynamic stays False."""
    a = harmonia.assess(ds, "dofetilide", n_mc=0)
    assert a.herg_dynamic is False
    assert a.classification == "high"          # unchanged headline call
    b = harmonia.assess(ds, "verapamil", n_mc=0)
    assert b.classification == "low"
