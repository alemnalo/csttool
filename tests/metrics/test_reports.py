"""
Tests for report generation functions with metadata support.
"""
import pytest
import json
import tempfile
from pathlib import Path

from csttool.metrics.modules.reports import save_json_report


@pytest.fixture
def sample_comparison():
    """Create a minimal comparison dict for testing."""
    return {
        'left': {
            'morphology': {
                'n_streamlines': 100,
                'mean_length': 85.0,
                'std_length': 12.0,
                'min_length': 40.0,
                'max_length': 130.0,
                'tract_volume': 12000.0
            },
            'fa': {'mean': 0.45, 'std': 0.08}
        },
        'right': {
            'morphology': {
                'n_streamlines': 110,
                'mean_length': 88.0,
                'std_length': 11.0,
                'min_length': 42.0,
                'max_length': 135.0,
                'tract_volume': 13000.0
            },
            'fa': {'mean': 0.47, 'std': 0.07}
        },
        'asymmetry': {
            'volume': {'laterality_index': -0.04},
            'streamline_count': {'laterality_index': -0.05},
            'mean_length': {'laterality_index': -0.02},
            'fa': {'laterality_index': -0.02}
        }
    }


class TestSaveJsonReport:
    """Tests for save_json_report function."""
    
    def test_json_report_includes_version(self, sample_comparison):
        """Test that JSON report includes csttool version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject"
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            assert 'csttool_version' in report
            assert report['csttool_version'] is not None
    
    def test_json_report_includes_acquisition_metadata(self, sample_comparison):
        """Test that JSON report includes acquisition metadata when provided."""
        metadata = {
            'acquisition': {
                'protocol': 'Multi-shell',
                'b_values': [0, 1000, 2000],
                'n_directions': 64,
                'resolution': [2.0, 2.0, 2.0]
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject",
                metadata=metadata
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            assert 'acquisition' in report
            assert report['acquisition']['protocol'] == 'Multi-shell'
            assert report['acquisition']['b_values'] == [0, 1000, 2000]
            assert report['acquisition']['n_directions'] == 64
    
    def test_json_report_includes_processing_metadata(self, sample_comparison):
        """Test that JSON report includes processing metadata when provided."""
        metadata = {
            'processing': {
                'denoising_method': 'patch2self',
                'gibbs_correction': True,
                'motion_correction': False,
                'tracking_method': 'Deterministic (DTI)',
                'roi_approach': 'Atlas-to-Subject (HO)',
                'whole_brain_streamlines': 500000,
                'extraction_method': 'passthrough'
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject",
                metadata=metadata
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            assert 'processing' in report
            assert report['processing']['denoising_method'] == 'patch2self'
            assert report['processing']['whole_brain_streamlines'] == 500000
    
    def test_json_report_includes_qc_thresholds(self, sample_comparison):
        """Test that JSON report includes QC thresholds when provided."""
        metadata = {
            'qc_thresholds': {
                'fa_threshold': 0.15,
                'min_length': 30.0,
                'max_length': 200.0
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject",
                metadata=metadata
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            assert 'qc_thresholds' in report
            assert report['qc_thresholds']['fa_threshold'] == 0.15
            assert report['qc_thresholds']['min_length'] == 30.0
    
    def test_json_report_backward_compatible(self, sample_comparison):
        """Test that JSON report works without metadata (backward compatible)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, 
                tmpdir, 
                "test_subject"
            )
            
            with open(json_path) as f:
                report = json.load(f)
            
            # Should have empty dicts for metadata sections
            assert 'acquisition' in report
            assert 'processing' in report
            assert 'qc_thresholds' in report
            assert report['acquisition'] == {}
            assert report['processing'] == {}
            assert report['qc_thresholds'] == {}
            
            # Core fields should still be present
            assert 'subject_id' in report
            assert 'processing_date' in report
            assert 'metrics' in report


def test_html_report_backward_compatibility_old_data():
    """Test HTML report handles old data without median/min/max fields."""
    from csttool.metrics.modules.reports import save_html_report
    import tempfile

    # Old data structure without median, min, max
    comparison = {
        'left': {
            'morphology': {
                'n_streamlines': 100,
                'tract_volume': 1000.0,
                'mean_length': 50.0,
                'std_length': 5.0,
                'min_length': 40.0,
                'max_length': 60.0
            },
            'fa': {'mean': 0.45, 'std': 0.08}  # No median, min, max!
        },
        'right': {
            'morphology': {
                'n_streamlines': 95,
                'tract_volume': 950.0,
                'mean_length': 49.0,
                'std_length': 4.5,
                'min_length': 41.0,
                'max_length': 58.0
            },
            'fa': {'mean': 0.44, 'std': 0.07}  # No median, min, max!
        },
        'asymmetry': {
            'volume': {'laterality_index': 0.026},
            'streamline_count': {'laterality_index': 0.026},
            'mean_length': {'laterality_index': 0.010},
            'fa': {'laterality_index': 0.011}
        }
    }

    viz_paths = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not crash with old data
        html_path = save_html_report(
            comparison,
            viz_paths,
            tmpdir,
            "test_subject",
            version="0.3.0",
            space="Native"
        )

        assert html_path.exists()

        # Verify HTML contains expected content
        html_content = html_path.read_text()
        assert 'test_subject' in html_content
        assert 'FA' in html_content
        # Should have fallback median values (using mean as fallback)
        assert '0.45' in html_content  # mean value should be present


# ---------------------------------------------------------------------------
# Provenance in reports
# ---------------------------------------------------------------------------

@pytest.fixture
def full_provenance():
    """A realistic provenance dict matching get_provenance_dict() output."""
    return {
        'git_commit': 'abc123def456',
        'python_version': '3.11.4 (main, Jul  5 2023, 13:45:01) [GCC 12.2.0]',
        'command_line': ['csttool', 'run', '--nifti', '/home/user/data.nii.gz'],
        'dependencies': {
            'numpy': '1.26.4',
            'scipy': '1.13.0',
            'dipy': '1.9.0',
            'nibabel': '5.2.1',
            'nilearn': '0.10.4',
            'dicom2nifti': '2.4.6',
            'pydicom': '2.4.4',
            'matplotlib': '3.8.4',
            'scikit-learn': '1.5.0',
            'h5py': '3.11.0',
            'cython': '3.0.10',
        },
        'platform': 'Linux-6.1.0-18-amd64-x86_64-with-glibc2.36',
        'machine': 'x86_64',
        'processor': 'x86_64',
        'hardware': {
            'cpu_model': 'Intel(R) Core(TM) i7-13700K',
            'cpu_count': 16,
            'total_ram_gb': 31.1,
            'gpu': ['NVIDIA GeForce RTX 4060'],
        },
        'thread_env': {
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': None,
            'VECLIB_MAXIMUM_THREADS': None,
            'DIPY_NUM_THREADS': None,
        },
    }


class TestProvenanceInJsonReport:
    """Tests for provenance in JSON report."""

    def test_provenance_written_to_json(self, sample_comparison, full_provenance):
        """Provenance key is present in *_bilateral_metrics.json."""
        metadata = {'provenance': full_provenance}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(
                sample_comparison, tmpdir, "test_subject", metadata=metadata
            )
            with open(json_path) as f:
                report = json.load(f)

            assert 'provenance' in report
            assert report['provenance']['hardware']['cpu_model'] == 'Intel(R) Core(TM) i7-13700K'
            assert report['provenance']['dependencies']['dipy'] == '1.9.0'
            # Command line and git commit are preserved in JSON
            assert report['provenance']['command_line'] == full_provenance['command_line']
            assert report['provenance']['git_commit'] == 'abc123def456'

    def test_provenance_empty_when_not_provided(self, sample_comparison):
        """Provenance key is an empty dict when metadata has no provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_json_report(sample_comparison, tmpdir, "test_subject")
            with open(json_path) as f:
                report = json.load(f)

            assert 'provenance' in report
            assert report['provenance'] == {}


class TestProvenanceInHtmlReport:
    """Tests for provenance in HTML report."""

    def test_html_contains_reproducibility_section(self, sample_comparison, full_provenance):
        """Generated HTML contains the Reproducibility section."""
        from csttool.metrics.modules.reports import save_html_report

        metadata = {'provenance': full_provenance}
        viz_paths = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = save_html_report(
                sample_comparison, viz_paths, tmpdir, "test_subject",
                metadata=metadata,
            )
            html_content = html_path.read_text()

            assert 'Reproducibility' in html_content
            assert 'Intel(R) Core(TM) i7-13700K' in html_content
            assert '16 logical cores' in html_content
            assert '31.1 GB RAM' in html_content
            assert 'NVIDIA GeForce RTX 4060' in html_content
            # Thread limits are summarised, not printed one raw variable per line.
            assert 'OMP=1' in html_content
            assert 'OMP_NUM_THREADS' not in html_content
            # Fixed dependency subset, with the projects' own capitalisation.
            assert 'NumPy 1.26.4' in html_content
            # Git commit must NOT appear in the HTML
            assert 'abc123def456' not in html_content
            # Raw command line must NOT appear in the HTML
            assert '/home/user/data.nii.gz' not in html_content

    def test_html_without_provenance_does_not_break(self, sample_comparison):
        """HTML report renders without provenance metadata."""
        from csttool.metrics.modules.reports import save_html_report

        viz_paths = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = save_html_report(
                sample_comparison, viz_paths, tmpdir, "test_subject",
            )
            html_content = html_path.read_text()

            assert html_path.exists()
            # Should still contain the section heading with fallback message
            assert 'Reproducibility' in html_content
            assert 'not recorded' in html_content

    def test_html_with_partial_provenance_does_not_break(self, sample_comparison):
        """HTML report renders when provenance has missing keys."""
        from csttool.metrics.modules.reports import save_html_report

        partial = {
            'python_version': '3.10.12',
            'platform': 'Linux',
            'machine': 'x86_64',
        }
        metadata = {'provenance': partial}
        viz_paths = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = save_html_report(
                sample_comparison, viz_paths, tmpdir, "test_subject",
                metadata=metadata,
            )
            html_content = html_path.read_text()

            assert 'Reproducibility' in html_content
            assert '3.10.12' in html_content
            # Missing keys render as N/A
            assert 'N/A' in html_content

    def test_html_with_none_hardware_values(self, sample_comparison):
        """HTML report handles None / missing hardware sub-fields."""
        from csttool.metrics.modules.reports import save_html_report

        sparse_prov = {
            'python_version': '3.11.0',
            'platform': 'Linux',
            'machine': 'x86_64',
            'hardware': {
                'cpu_model': None,
                'cpu_count': 8,
                'total_ram_gb': None,
                'gpu': None,
            },
            'thread_env': {},
            'dependencies': {},
        }
        metadata = {'provenance': sparse_prov}
        viz_paths = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = save_html_report(
                sample_comparison, viz_paths, tmpdir, "test_subject",
                metadata=metadata,
            )
            html_content = html_path.read_text()

            assert 'Reproducibility' in html_content
            # Should not crash; N/A placeholders for missing values
            assert 'N/A' in html_content


class TestPdfWithProvenance:
    """Tests for PDF generation with provenance section."""

    def test_pdf_renders_with_provenance(self, sample_comparison, full_provenance):
        """PDF generation succeeds on a report containing the provenance section."""
        pytest.importorskip("weasyprint")

        from csttool.metrics.modules.reports import save_html_report, html_to_pdf

        metadata = {'provenance': full_provenance}
        viz_paths = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = save_html_report(
                sample_comparison, viz_paths, tmpdir, "test_subject",
                metadata=metadata,
            )
            pdf_path = Path(tmpdir) / "test_subject_report.pdf"
            result = html_to_pdf(html_path, pdf_path)

            assert result is not None
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0
