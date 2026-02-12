from typer.testing import CliRunner
from dopingflow.cli import app

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dopingflow" in result.output
