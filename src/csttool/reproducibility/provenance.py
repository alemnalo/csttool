"""
Provenance tracking: Capture git hash, Python version, dependency versions,
hardware details, and environment variables relevant to reproducibility.

Handles missing git gracefully (e.g., in installed packages or shallow clones).
"""

import os
import platform
import subprocess
import sys


def get_git_commit_hash() -> str | None:
    """Get git commit hash, handling missing git gracefully.

    Tries multiple approaches:
    1. Run ``git rev-parse HEAD``
    2. Check environment variable ``GITHUB_SHA`` (CI environments)
    3. Return None if git unavailable or not in git repo

    Returns:
        Git commit hash as string, or None if unavailable
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if "GITHUB_SHA" in os.environ:
        return os.environ["GITHUB_SHA"]

    return None


def get_python_version() -> str:
    """Get Python version string."""
    return sys.version


def get_dependency_versions() -> dict:
    """Get versions of key dependencies.

    Includes the packages originally reported (numpy, scipy, dipy, nibabel)
    plus the five the audits identified as missing from provenance (AU8):
    nilearn, dicom2nifti, pydicom, matplotlib, scikit-learn, h5py.

    Returns:
        Dict mapping package name to version string
    """
    _pkg_names = [
        "numpy", "scipy", "dipy", "nibabel",
        "nilearn", "dicom2nifti", "pydicom", "matplotlib",
        "scikit-learn", "h5py", "cython",
    ]
    versions = {}
    for name in _pkg_names:
        try:
            pkg = __import__(name)
            versions[name] = pkg.__version__
        except (ImportError, AttributeError):
            versions[name] = "not installed"
    return versions


def get_hardware_info() -> dict:
    """Collect hardware details relevant to reproducibility.

    Returns a dict with cpu_model, cpu_count, total_ram_gb, and gpu info.
    All keys are present; unavailable information is None.
    """
    info = {
        "cpu_count": os.cpu_count(),
        "cpu_model": None,
        "total_ram_gb": None,
        "gpu": None,
    }

    # --- CPU model ---
    try:
        if sys.platform == "linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        elif sys.platform == "darwin":
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                info["cpu_model"] = r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # --- RAM ---
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal"):
                        # value is in kB
                        kb = int(line.split(":")[1].strip().split()[0])
                        info["total_ram_gb"] = round(kb / 1024 / 1024, 1)
                        break
        elif sys.platform == "darwin":
            r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                info["total_ram_gb"] = round(int(r.stdout.strip()) / 1024 ** 3, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # --- GPU ---
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            info["gpu"] = [g.strip() for g in r.stdout.strip().split("\n") if g.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


def get_thread_env() -> dict:
    """Capture environment variables that control BLAS / OpenMP threading.

    These are common sources of non-reproducibility in numerical pipelines.
    """
    env_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "DIPY_NUM_THREADS",
    ]
    return {v: os.environ.get(v, None) for v in env_vars}


def get_provenance_dict() -> dict:
    """Get complete provenance information for a pipeline run.

    Returns a dict with the following top-level keys:

    - **git_commit** (str or None)
    - **python_version** (str)
    - **command_line** (list[str]) — ``sys.argv`` at call time
    - **dependencies** (dict[str, str])
    - **platform** (str) — ``platform.platform()``
    - **machine** (str) — ``platform.machine()``
    - **processor** (str)
    - **hardware** (dict) — cpu_model, cpu_count, total_ram_gb, gpu
    - **thread_env** (dict) — threading environment variables
    """
    return {
        "git_commit": get_git_commit_hash(),
        "python_version": get_python_version(),
        "command_line": sys.argv,
        "dependencies": get_dependency_versions(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "hardware": get_hardware_info(),
        "thread_env": get_thread_env(),
    }