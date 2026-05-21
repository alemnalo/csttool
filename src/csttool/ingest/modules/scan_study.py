"""
scan_study.py

Scan a study directory to discover all DICOM series.

This module provides functionality to:
- Identify DICOM-containing directories
- Extract basic series information
- Build a catalog of available series for user selection
"""

from pathlib import Path
from typing import List, Dict, Optional
import os


def is_dicom_file(filepath: Path) -> bool:
    """
    Check if a file is a valid DICOM file.
    
    Checks both file extension and DICOM magic number.
    
    Parameters
    ----------
    filepath : Path
        Path to file to check
        
    Returns
    -------
    bool
        True if file appears to be a valid DICOM
    """
    # Check extension first (fast path)
    if filepath.suffix.lower() in ['.dcm', '.dicom']:
        return True
    
    # Check for DICOM magic number at byte 128
    # DICOM files have "DICM" at bytes 128-131
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            magic = f.read(4)
            return magic == b'DICM'
    except (IOError, OSError):
        return False


def is_dicom_directory(directory: Path) -> bool:
    """
    Check whether a directory contains DICOM files.
    
    Parameters
    ----------
    directory : Path
        Path to directory to check
        
    Returns
    -------
    bool
        True if directory contains at least one DICOM file
    """
    directory = Path(directory)
    
    if not directory.is_dir():
        return False
    
    # Check for .dcm files first (fast); use set to avoid double-counting on
    # case-insensitive filesystems (Windows).
    dcm_files = set(directory.glob("*.dcm")) | set(directory.glob("*.DCM"))
    if dcm_files:
        return True
    
    # Check first few files for DICOM magic number
    for f in list(directory.iterdir())[:10]:
        if f.is_file() and not f.name.startswith('.'):
            if is_dicom_file(f):
                return True
    
    return False


def count_dicom_files(directory: Path) -> int:
    """
    Count the number of DICOM files in a directory.
    
    Parameters
    ----------
    directory : Path
        Path to directory
        
    Returns
    -------
    int
        Number of DICOM files found
    """
    directory = Path(directory)
    
    # Count .dcm files; use set to avoid double-counting on case-insensitive
    # filesystems (Windows).
    dcm_count = len(set(directory.glob("*.dcm")) | set(directory.glob("*.DCM")))

    if dcm_count > 0:
        return dcm_count
    
    # Count files with DICOM magic number
    count = 0
    for f in directory.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            if is_dicom_file(f):
                count += 1
    
    return count


def scan_study(
    study_dir: Path,
    recursive: bool = True,
    verbose: bool = True
) -> List[Dict]:
    """
    Scan a study directory to discover all DICOM series.
    
    Parameters
    ----------
    study_dir : Path
        Path to study directory (may contain multiple series subdirectories)
    recursive : bool, optional
        Search subdirectories for DICOM files. Default is True.
    verbose : bool, optional
        Print progress information. Default is True.
        
    Returns
    -------
    series_list : List[Dict]
        List of discovered series, each containing:
        - path: Path to series directory
        - name: Directory name
        - n_files: Number of DICOM files
        - depth: Directory depth relative to study_dir
        
    Examples
    --------
    >>> series = scan_study(Path("~/anom"))
    >>> for s in series:
    ...     print(f"{s['name']}: {s['n_files']} files")
    """
    study_dir = Path(study_dir).expanduser().resolve()
    
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")
    
    if not study_dir.is_dir():
        raise ValueError(f"Not a directory: {study_dir}")
    
    if verbose:
        print(f"  → Scanning study directory: {study_dir}")
    
    series_list = []
    
    # Check if study_dir itself contains DICOMs (single series case)
    if is_dicom_directory(study_dir):
        n_files = count_dicom_files(study_dir)
        series_list.append({
            'path': study_dir,
            'name': study_dir.name,
            'n_files': n_files,
            'depth': 0
        })
        if verbose:
            print(f"    • Found series at root: {study_dir.name} ({n_files} files)")
    
    # Scan subdirectories
    if recursive:
        for item in sorted(study_dir.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                _scan_recursive(item, series_list, study_dir, verbose, max_depth=3)
    
    if verbose:
        print(f"\n  ✓ Found {len(series_list)} DICOM series")
    
    return series_list


def _scan_recursive(
    directory: Path,
    series_list: List[Dict],
    root_dir: Path,
    verbose: bool,
    max_depth: int,
    current_depth: int = 1
):
    """
    Recursively scan for DICOM series.
    
    Parameters
    ----------
    directory : Path
        Current directory to scan
    series_list : List[Dict]
        List to append found series to (modified in place)
    root_dir : Path
        Original study root for depth calculation
    verbose : bool
        Print progress information
    max_depth : int
        Maximum recursion depth
    current_depth : int
        Current recursion depth
    """
    if current_depth > max_depth:
        return
    
    if is_dicom_directory(directory):
        n_files = count_dicom_files(directory)
        relative_path = directory.relative_to(root_dir)
        
        series_list.append({
            'path': directory,
            'name': directory.name,
            'relative_path': str(relative_path),
            'n_files': n_files,
            'depth': current_depth
        })
        
        if verbose:
            indent = "  " * (current_depth + 1)
            print(f"{indent}• Found series: {directory.name} ({n_files} files)")
    else:
        # Continue searching subdirectories
        for item in sorted(directory.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                _scan_recursive(
                    item, series_list, root_dir, verbose, 
                    max_depth, current_depth + 1
                )


def filter_series_by_file_count(
    series_list: List[Dict],
    min_files: int = 10
) -> List[Dict]:
    """
    Filter series list to only include those with sufficient files.
    
    Series with very few files are likely derived images or localizers.
    
    Parameters
    ----------
    series_list : List[Dict]
        List of series from scan_study()
    min_files : int
        Minimum number of files to include. Default is 10.
        
    Returns
    -------
    filtered : List[Dict]
        Filtered series list
    """
    return [s for s in series_list if s['n_files'] >= min_files]


def print_series_summary(series_list: List[Dict], verbose: bool = True) -> None:
    """
    Print a formatted summary of discovered series.

    Parameters
    ----------
    series_list : List[Dict]
        List of series from scan_study()
    verbose : bool, optional
        Print verbose details. Default is True.
    """
    if not series_list:
        print("  ⚠️ No DICOM series found")
        return

    print("\n" + "=" * 60)
    print("DISCOVERED DICOM SERIES")
    print("=" * 60)

    if verbose:
        print(f"{'#':<4} {'Series Name':<45} {'Files':<10}")
        print("-" * 60)

        for i, series in enumerate(series_list, 1):
            print(f"{i:<4} {series['name']:<45} {series['n_files']:<10}")
    else:
        for i, series in enumerate(series_list, 1):
            print(f"  {i}. {series['name']} ({series['n_files']} files)")

    print("=" * 60)