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
    for cmd in ("init", "run", "validate", "version"):
        assert cmd in result.stdout

def test_cli_init_scaffolds_directory(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    subprocess.run(["cytoscan", "init", str(exp_dir)], check=True)

    assert (exp_dir / "config.yaml").is_file()
    assert (exp_dir / "input" / "brightfield").is_dir()
    assert (exp_dir / "input" / "fluorescent").is_dir()
    assert (exp_dir / "input" / "mixed").is_dir()

def test_cli_init_idempotent(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    subprocess.run(["cytoscan", "init", str(exp_dir)], check=True)
    # second invocation must not error or wipe the config.
    subprocess.run(["cytoscan", "init", str(exp_dir)], check=True)
    assert (exp_dir / "config.yaml").is_file()

"""the bundled default.yaml must parse as valid config."""
def test_default_template_is_loadable():
    from importlib.resources import files
    import yaml

    text = files("cytoscan").joinpath("templates/default.yaml").read_text()
    data = yaml.safe_load(text)
    assert isinstance(data, dict)


