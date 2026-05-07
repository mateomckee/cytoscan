import subprocess
from pathlib import Path

import cytoscan

def test_version_string():
    assert cytoscan.__version__
    assert isinstance(cytoscan.__version__, str)

def test_cli_version_command():
    result = subprocess.run(
        ["cytoscan", "version"], capture_output=True, text=True, check=True
    )
    assert "cytoscan" in result.stdout.lower()

def test_cli_help():
    result = subprocess.run(
        ["cytoscan", "--help"], capture_output=True, text=True, check=True
    )
    for cmd in ("run", "version"):
        assert cmd in result.stdout
    # global flags
    assert "--verbose" in result.stdout
    assert "--quiet" in result.stdout
    assert "--log-file" in result.stdout

def test_cli_run_scaffolds_directory(tmp_path: Path):
    """`cytoscan run` scaffolds the dir before doing anything else.
    With no frames present, it exits non-zero — but the scaffolding side-effects must still happen."""
    exp_dir = tmp_path / "exp"
    subprocess.run(["cytoscan", "run", str(exp_dir)], capture_output=True)

    assert (exp_dir / "config.yaml").is_file()
    assert (exp_dir / "input" / "brightfield").is_dir()
    assert (exp_dir / "input" / "fluorescent").is_dir()
    assert (exp_dir / "input" / "mixed").is_dir()

def test_cli_run_scaffold_idempotent(tmp_path: Path):
    """Running scaffold twice must not error or wipe the config."""
    exp_dir = tmp_path / "exp"
    subprocess.run(["cytoscan", "run", str(exp_dir)], capture_output=True)
    subprocess.run(["cytoscan", "run", str(exp_dir)], capture_output=True)
    assert (exp_dir / "config.yaml").is_file()

"""the bundled default.yaml must parse as valid config."""
def test_default_template_is_loadable():
    from importlib.resources import files
    import yaml

    text = files("cytoscan").joinpath("templates/default.yaml").read_text()
    data = yaml.safe_load(text)
    assert isinstance(data, dict)


