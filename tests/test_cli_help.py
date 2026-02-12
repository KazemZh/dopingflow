import subprocess


def test_cli_help():
    result = subprocess.run(["dopingflow", "--help"], capture_output=True)
    assert result.returncode == 0
    assert b"dopingflow" in result.stdout
