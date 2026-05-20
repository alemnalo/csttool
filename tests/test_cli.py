import pytest
from unittest.mock import patch, MagicMock
from csttool.cli import main

def test_cli_version(capsys):
    """Test that --version flag works."""
    with patch('sys.argv', ['csttool', '--version']):
        # expect SystemExit
        with pytest.raises(SystemExit):
            main()
    
    captured = capsys.readouterr()
    assert "csttool" in captured.out or "csttool" in captured.err

def test_cli_help(capsys):
    """Test that --help flag works."""
    with patch('sys.argv', ['csttool', '--help']):
        with pytest.raises(SystemExit):
            main()
            
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "usage:" in captured.err

@patch('csttool.cli.cmd_check')
def test_cli_check_command(mock_cmd_check):
    """Test that 'check' command calls the correct function."""
    with patch('sys.argv', ['csttool', 'check']):
        main()
        mock_cmd_check.assert_called_once()


@patch('csttool.cli.cmd_doctor')
def test_cli_doctor_command(mock_cmd_doctor):
    """Test that 'doctor' command calls the correct function."""
    with patch('sys.argv', ['csttool', 'doctor']):
        main()
        mock_cmd_doctor.assert_called_once()


def test_cli_doctor_runs(capsys):
    """Smoke-test: doctor command runs without raising and prints header."""
    with patch('sys.argv', ['csttool', 'doctor']):
        main()
    captured = capsys.readouterr()
    assert "csttool doctor" in captured.out
    assert "Python packages" in captured.out


def test_cli_check_deprecated_redirects(capsys):
    """check command should print a deprecation notice and run doctor."""
    with patch('sys.argv', ['csttool', 'check']):
        main()
    captured = capsys.readouterr()
    assert "deprecated" in captured.err
    assert "csttool doctor" in captured.out
