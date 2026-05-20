
import argparse
import sys
import tomllib
from pathlib import Path
from csttool import __version__
from .doctor import cmd_doctor

# Map package names to import names (for cases where they differ)
IMPORT_NAME_MAP = {
    "dicom2nifti": "dicom2nifti",
    "jinja2": "jinja2",
    "cython": "cython",
}

# Dependencies that are functionally optional (won't fail check if missing)
OPTIONAL_DEPS = {"weasyprint"}


def _get_project_dependencies() -> list[str]:
    """Load required dependencies from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent.parent.parent.parent / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    return config["project"]["dependencies"]


def _extract_package_name(dep_spec: str) -> str:
    """Extract package name from dependency spec (e.g., 'numpy>=1.0' -> 'numpy')."""
    # Remove version specifiers and extras
    name = dep_spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0].split(";")[0].strip()
    return name


def _get_import_name(package_name: str) -> str:
    """Get the import name for a package (handles special cases)."""
    return IMPORT_NAME_MAP.get(package_name, package_name)


def _check_dependency(package_name: str, import_name: str | None = None, optional: bool = False) -> bool:
    """
    Check if a dependency is available and print status.

    Returns True if available, False otherwise.
    """
    if import_name is None:
        import_name = _get_import_name(package_name)

    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        symbol = "✓"
        print(f"{symbol} {package_name}: {version}")
        return True
    except ImportError:
        symbol = "○" if optional else "✗"
        status = "not found (optional)" if optional else "NOT FOUND"
        print(f"{symbol} {package_name}: {status}")
        return not optional


def _discover_csttool_modules() -> list[str]:
    """Discover all csttool submodules (directories with __init__.py)."""
    csttool_path = Path(__file__).parent.parent.parent
    modules = []

    for item in csttool_path.iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            # Skip cli module (checked separately) and __pycache__
            if item.name not in ("cli", "__pycache__"):
                modules.append(item.name)

    return sorted(modules)


def _check_csttool_modules() -> bool:
    """Check if all discovered csttool modules can be imported."""
    modules = _discover_csttool_modules()
    all_ok = True

    for module_name in modules:
        try:
            __import__(f"csttool.{module_name}")
            print(f"✓ csttool.{module_name}: available")
        except ImportError as e:
            print(f"✗ csttool.{module_name}: NOT FOUND ({e})")
            all_ok = False

    return all_ok


def cmd_check(args: argparse.Namespace) -> bool:
    """Deprecated: redirects to cmd_doctor."""
    print("Note: 'csttool check' is deprecated — use 'csttool doctor' instead.", file=sys.stderr)
    return cmd_doctor(args)
