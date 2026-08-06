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


class TestResolveV1Path:
    """`csttool metrics` finds the V1 product for the streamline-vs-tensor panel.

    The lookup has to be conservative: guessing wrong would feed the panel a
    volume that is not a world-frame eigenvector field, and the angles it
    reported would be silently meaningless rather than absent.
    """

    def _args(self, **kwargs):
        import argparse
        return argparse.Namespace(**kwargs)

    def test_explicit_v1_wins(self, tmp_path):
        from csttool.cli.commands.metrics import _resolve_v1_path
        explicit = tmp_path / "explicit_v1.nii.gz"
        explicit.touch()
        sibling = tmp_path / "sub_v1.nii.gz"
        sibling.touch()
        fa = tmp_path / "sub_fa.nii.gz"
        fa.touch()
        assert _resolve_v1_path(self._args(v1=explicit, fa=fa)) == explicit

    def test_falls_back_to_the_sibling_of_the_fa_map(self, tmp_path):
        from csttool.cli.commands.metrics import _resolve_v1_path
        fa = tmp_path / "sub-x_fa.nii.gz"
        fa.touch()
        sibling = tmp_path / "sub-x_v1.nii.gz"
        sibling.touch()
        assert _resolve_v1_path(self._args(v1=None, fa=fa)) == sibling

    def test_returns_none_when_the_sibling_is_absent(self, tmp_path):
        from csttool.cli.commands.metrics import _resolve_v1_path
        fa = tmp_path / "sub-x_fa.nii.gz"
        fa.touch()
        assert _resolve_v1_path(self._args(v1=None, fa=fa)) is None

    def test_does_not_guess_from_a_non_standard_fa_name(self, tmp_path):
        """A BIDS-named FA map has no predictable V1 sibling; don't invent one."""
        from csttool.cli.commands.metrics import _resolve_v1_path
        fa = tmp_path / "sub-x_space-orig_model-DTI_param-FA_dwimap.nii.gz"
        fa.touch()
        (tmp_path / "sub-x_v1.nii.gz").touch()
        assert _resolve_v1_path(self._args(v1=None, fa=fa)) is None

    def test_missing_explicit_path_falls_through(self, tmp_path):
        from csttool.cli.commands.metrics import _resolve_v1_path
        fa = tmp_path / "sub-x_fa.nii.gz"
        fa.touch()
        sibling = tmp_path / "sub-x_v1.nii.gz"
        sibling.touch()
        args = self._args(v1=tmp_path / "does_not_exist.nii.gz", fa=fa)
        assert _resolve_v1_path(args) == sibling

    def test_no_fa_and_no_v1(self):
        from csttool.cli.commands.metrics import _resolve_v1_path
        assert _resolve_v1_path(self._args(v1=None, fa=None)) is None


class TestResolveDensityPath:
    """`csttool metrics` finds the CST density product for QC-1.

    ``csttool run`` passes ``--density`` explicitly; the fallback only serves a
    hand-run metrics stage, and like the V1 lookup it refuses to guess rather
    than feeding the panel a volume that is not a density map.
    """

    def _args(self, **kwargs):
        import argparse
        return argparse.Namespace(**kwargs)

    def _extraction_tree(self, tmp_path, with_density=True):
        trk = tmp_path / "extraction" / "trk"
        trk.mkdir(parents=True)
        cst_left = trk / "sub-x_cst_left.trk"
        cst_left.touch()
        scalar_maps = tmp_path / "extraction" / "scalar_maps"
        scalar_maps.mkdir()
        density = scalar_maps / "sub-x_cst_density.nii.gz"
        if with_density:
            density.touch()
        return cst_left, density

    def test_explicit_density_wins(self, tmp_path):
        from csttool.cli.commands.metrics import _resolve_density_path
        cst_left, sibling = self._extraction_tree(tmp_path)
        explicit = tmp_path / "explicit_density.nii.gz"
        explicit.touch()
        args = self._args(density=explicit, cst_left=cst_left)
        assert _resolve_density_path(args) == explicit

    def test_falls_back_to_the_extraction_scalar_maps(self, tmp_path):
        from csttool.cli.commands.metrics import _resolve_density_path
        cst_left, density = self._extraction_tree(tmp_path)
        args = self._args(density=None, cst_left=cst_left)
        assert _resolve_density_path(args) == density

    def test_returns_none_when_the_product_is_absent(self, tmp_path):
        from csttool.cli.commands.metrics import _resolve_density_path
        cst_left, _ = self._extraction_tree(tmp_path, with_density=False)
        args = self._args(density=None, cst_left=cst_left)
        assert _resolve_density_path(args) is None

    def test_returns_none_without_a_tractogram_to_anchor_on(self):
        from csttool.cli.commands.metrics import _resolve_density_path
        assert _resolve_density_path(self._args(density=None, cst_left=None)) is None
