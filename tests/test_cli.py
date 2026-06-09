"""CLI smoke tests."""
from harmonia.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert "harmonia" in capsys.readouterr().out


def test_validate_ok():
    assert main(["validate"]) == 0


def test_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "VERIFIED: 0/" in out
    assert "drugs (28)" in out


def test_simulate(capsys):
    assert main(["simulate", "dofetilide", "--mc", "12"]) == 0
    assert "classification-flip frequency" in capsys.readouterr().out


def test_flip(capsys):
    assert main(["flip", "verapamil", "--mc", "12"]) == 0
    assert "flip-view" in capsys.readouterr().out


def test_sensitivity(capsys):
    assert main(["sensitivity", "dofetilide", "--mc", "8"]) == 0
    out = capsys.readouterr().out
    assert "flip-sensitivity" in out
    assert "dominant uncertainty driver" in out


def test_combo(capsys):
    assert main(["combo", "terfenadine", "ondansetron", "--mc", "12"]) == 0
    out = capsys.readouterr().out
    assert "combination = terfenadine + ondansetron" in out
    assert "classification-flip frequency" in out


def test_population(capsys):
    assert main(["population", "sotalol", "--n", "20"]) == 0
    out = capsys.readouterr().out
    assert "NOT FOR PREDICTION" in out or "hypothesis" in out.lower()


def test_performance(capsys):
    assert main(["performance", "--set", "training"]) == 0
    assert "accuracy" in capsys.readouterr().out


def test_export_cipa_stdout(capsys):
    assert main(["export", "--format", "cipa"]) == 0
    assert "drug,channel,ic50_nM" in capsys.readouterr().out


def test_export_all(tmp_path):
    assert main(["export", "--all", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "tables" / "cipa_inputs.csv").exists()
    assert list((tmp_path / "omex").glob("*.omex"))
