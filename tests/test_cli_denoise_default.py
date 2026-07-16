"""
The denoise method every CLI command resolves to by default.

MPPCA was chosen as the default denoiser (07 Decisions D8, superseding nlmeans) because it
estimates its own noise level and so needs neither a receiver-coil count nor an assumption
about the noise distribution — retiring audit findings AU2 (`rician=False`) and AU25 (coil
count) from the default path, since both live only in the nlmeans branch of `denoise()`.

That promotion was previously applied to the library function's signature default alone. Every
CLI parser still passed an explicit `default="nlmeans"`, which overrode it, so in practice no
CLI run ever used MPPCA and the CHANGELOG's claim that the default had changed was false.

These tests assert the value the CLI actually resolves and hands to the command — not the
library signature, which is what looked correct while the tool did the opposite.
"""

from unittest.mock import MagicMock, patch

import pytest

import csttool.cli as cli
from csttool.defaults import DEFAULT_DENOISE_METHOD


# (command, function attribute patched on csttool.cli, minimal args to reach dispatch)
CLI_COMMANDS = [
    ("preprocess", "cmd_preprocess", ["--nifti", "in.nii.gz", "--out", "out"]),
    ("run", "cmd_run", ["--nifti", "in.nii.gz", "--out", "out"]),
    ("batch", "cmd_batch", ["--manifest", "m.tsv", "--out", "out"]),
]


def _resolved_args(command, func_attr, argv):
    """Run main() far enough to capture the Namespace the command is dispatched with."""
    captured = MagicMock()
    with patch.object(cli, func_attr, captured):
        with patch("sys.argv", ["csttool", command] + argv):
            cli.main()

    assert captured.called, f"'{command}' never dispatched; argv may be incomplete"
    return captured.call_args[0][0]


def test_default_is_mppca():
    """The shared default is MPPCA - the constant every other site must read."""
    assert DEFAULT_DENOISE_METHOD == "mppca"


@pytest.mark.parametrize("command,func_attr,argv", CLI_COMMANDS)
def test_command_defaults_to_mppca(command, func_attr, argv):
    """Each command resolves --denoise-method to MPPCA when the user does not pass one."""
    args = _resolved_args(command, func_attr, argv)

    assert args.denoise_method == "mppca", (
        f"'csttool {command}' defaults to '{args.denoise_method}', not mppca"
    )


@pytest.mark.parametrize("command,func_attr,argv", CLI_COMMANDS)
def test_explicit_choice_still_wins(command, func_attr, argv):
    """Promoting the default must not stop a user from asking for nlmeans."""
    args = _resolved_args(command, func_attr, argv + ["--denoise-method", "nlmeans"])

    assert args.denoise_method == "nlmeans"


@pytest.mark.parametrize("command", [c[0] for c in CLI_COMMANDS])
def test_help_advertises_the_real_default(command, capsys):
    """The advertised default must match the resolved one; they were out of step before."""
    with patch("sys.argv", ["csttool", command, "--help"]):
        with pytest.raises(SystemExit):
            cli.main()

    help_text = " ".join(capsys.readouterr().out.split())

    assert f"default: {DEFAULT_DENOISE_METHOD}" in help_text, (
        f"'csttool {command} --help' does not advertise {DEFAULT_DENOISE_METHOD} as the default"
    )
