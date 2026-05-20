import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from csttool import __version__

# Minimum dcm2niix release date (encoded as integer YYYYMMDD)
_DCM2NIIX_MIN = 20220720

_CORE_PACKAGES = [
    ("numpy",        "numpy"),
    ("scipy",        "scipy"),
    ("cython",       "Cython"),
    ("dipy",         "dipy"),
    ("nibabel",      "nibabel"),
    ("nilearn",      "nilearn"),
    ("pydicom",      "pydicom"),
    ("dicom2nifti",  "dicom2nifti"),
    ("platformdirs", "platformdirs"),
    ("jinja2",       "jinja2"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(label: str, detail: str = "") -> None:
    line = f"  ✓ {label}"
    if detail:
        line += f": {detail}"
    print(line)


def _warn(label: str, detail: str = "", hint: str = "") -> None:
    line = f"  ○ {label}"
    if detail:
        line += f": {detail}"
    print(line)
    if hint:
        for h in hint.splitlines():
            print(f"    {h}")


def _fail(label: str, detail: str = "", hint: str = "") -> None:
    line = f"  ✗ {label}"
    if detail:
        line += f": {detail}"
    print(line)
    if hint:
        for h in hint.splitlines():
            print(f"    {h}")


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_packages() -> bool:
    _section("Python packages")
    all_ok = True
    for pkg_name, import_name in _CORE_PACKAGES:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            _ok(pkg_name, ver)
        except ImportError:
            _fail(pkg_name, "not found", f"pip install {pkg_name}")
            all_ok = False
    return all_ok


def _check_dcm2niix() -> bool:
    _section("External tools")
    path = shutil.which("dcm2niix")
    if path is None:
        _fail(
            "dcm2niix",
            "not found on PATH",
            "conda install -c conda-forge dcm2niix\n"
            "or download from https://github.com/rordenlab/dcm2niix/releases\n"
            "Fallback: dicom2nifti will be used (no BIDS JSON sidecars)",
        )
        return False

    try:
        result = subprocess.run(
            [path, "-v"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        raw = result.stdout + result.stderr
    except Exception as exc:
        _warn("dcm2niix", f"found at {path} but could not query version ({exc})")
        return True

    match = re.search(r"v\d+\.\d+\.(\d{8})", raw)
    if not match:
        _warn("dcm2niix", f"found at {path} but could not parse version")
        return True

    ver_date = int(match.group(1))
    ver_str = match.group(0)
    if ver_date >= _DCM2NIIX_MIN:
        _ok("dcm2niix", f"{ver_str}  [{path}]")
        return True
    else:
        _fail(
            "dcm2niix",
            f"{ver_str} is older than required >= 1.0.{_DCM2NIIX_MIN}",
            "conda install -c conda-forge 'dcm2niix>=1.0.20220720'",
        )
        return False


def _check_atlas() -> bool:
    _section("Atlas data")
    try:
        from csttool.data import is_data_installed, get_user_data_dir
        data_dir = get_user_data_dir()
        if is_data_installed():
            _ok("FMRIB58_FA + Harvard-Oxford atlases", str(data_dir))
            return True
        else:
            _fail(
                "Atlas data",
                "not installed",
                "csttool fetch-data --accept-fsl-license",
            )
            return False
    except Exception as exc:
        _fail("Atlas data", f"error checking ({exc})")
        return False


def _check_system() -> bool:
    _section("System")
    all_ok = True

    try:
        from platformdirs import user_data_dir, user_cache_dir
        for label, dirfn in [("User data dir", user_data_dir), ("Cache dir", user_cache_dir)]:
            d = Path(dirfn("csttool"))
            try:
                d.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=d, delete=True):
                    pass
                _ok(label, str(d))
            except OSError as exc:
                _fail(label, f"not writable ({exc})")
                all_ok = False
    except ImportError:
        _fail("platformdirs", "not installed")
        all_ok = False

    cpu_count = os.cpu_count() or 1
    omp = os.environ.get("OMP_NUM_THREADS")
    omp_note = f"OMP_NUM_THREADS={omp}" if omp else "OMP_NUM_THREADS not set (dipy uses all cores)"
    _ok(f"CPUs: {cpu_count}", omp_note)

    return all_ok


def _check_report_backend() -> bool:
    _section("Report backend")
    _ok("HTML (jinja2)", "available")  # jinja2 is core; already checked above

    try:
        from weasyprint import HTML as _HTML
    except ImportError:
        _plat = platform.system()
        if _plat == "Windows":
            hint = "conda install -c conda-forge weasyprint pango cairo gdk-pixbuf"
        else:
            hint = "pip install 'csttool[reports]'"
        _warn("PDF (weasyprint)", "not installed — HTML report still produced", hint)
        return True  # optional: never fails doctor

    # weasyprint imported — verify native libs load by rendering a tiny PDF
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            _HTML(string="<p>csttool doctor smoke test</p>").write_pdf(tmp.name)
        import weasyprint
        _ok("PDF (weasyprint)", getattr(weasyprint, "__version__", "installed"))
    except OSError as exc:
        _plat = platform.system()
        if _plat == "Windows":
            hint = "conda install -c conda-forge weasyprint pango cairo gdk-pixbuf"
        else:
            hint = (
                "Install native libs:\n"
                "  Debian/Ubuntu: sudo apt install libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0\n"
                "  macOS:         brew install pango cairo gdk-pixbuf"
            )
        _warn(
            "PDF (weasyprint)",
            f"installed but native libs could not load — {exc}",
            hint,
        )

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> bool:
    """Run comprehensive environment diagnostics. Returns True if pipeline-ready."""
    print("=" * 60)
    print("csttool doctor")
    print("=" * 60)
    print(f"  csttool:  {__version__}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    packages_ok  = _check_packages()
    dcm2niix_ok  = _check_dcm2niix()
    atlas_ok     = _check_atlas()
    system_ok    = _check_system()
    _check_report_backend()   # always optional — never blocks summary

    print()
    print("=" * 60)

    pipeline_ready = packages_ok and system_ok
    if pipeline_ready:
        if not atlas_ok:
            print("  ⚠  Packages OK — run 'csttool fetch-data --accept-fsl-license' before extract/metrics")
        elif not dcm2niix_ok:
            print("  ⚠  Packages OK — dcm2niix missing/outdated; DICOM import uses dicom2nifti fallback")
        else:
            print("  ✓  Environment ready")
    else:
        print("  ✗  Issues found — see above")

    return pipeline_ready
