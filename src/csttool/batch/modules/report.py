import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Standard headers aligned with single-subject report
METRICS_HEADERS = [
    # Batch metadata
    "subject_id", "session_id", "status", "error_category", 
    "duration_seconds", "error",
    
    # Morphology
    "left_n_streamlines", "right_n_streamlines",
    "left_mean_length_mm", "right_mean_length_mm",
    "left_tract_volume_mm3", "right_tract_volume_mm3",
    "volume_laterality_index", "streamline_count_laterality_index"
]

# Add headers for all scalars and regions dynamically to ensure consistency
_SCALARS = ['fa', 'md', 'rd', 'ad']
_REGIONS = ['pontine', 'plic', 'precentral']

for scalar in _SCALARS:
    METRICS_HEADERS.extend([
        f"left_{scalar}_mean", f"left_{scalar}_std",
        f"right_{scalar}_mean", f"right_{scalar}_std",
        f"{scalar}_laterality_index",
        # Length-biased counterpart of the headline mean, for bias auditing (AU10):
        f"left_{scalar}_mean_point_weighted",
        f"right_{scalar}_mean_point_weighted",
    ])
    for region in _REGIONS:
        METRICS_HEADERS.extend([
            f"left_{scalar}_{region}",
            f"right_{scalar}_{region}",
            f"{scalar}_{region}_laterality_index"
        ])

logger = logging.getLogger(__name__)

def generate_batch_reports(results: List[Any], output_dir: Path):
    """
    Orchestrates the generation of all batch-level reports.
    
    Args:
        results: List of SubjectResult objects
        output_dir: Path to the batch output root
    """
    write_batch_metrics_csv(results, output_dir)
    write_batch_summary_json(results, output_dir)

def write_batch_metrics_csv(results: List[Any], output_dir: Path):
    """
    Writes a CSV aggregate of all subject metrics.
    Numeric cells are empty for failed or skipped subjects.
    """
    csv_path = output_dir / "batch_metrics.csv"
    
    rows = []
    for res in results:
        # Standard fields always populated
        row = {
            "subject_id": res.subject_id,
            "session_id": res.session_id or "",
            "status": res.status,
            "error_category": res.error_category or "",
            "duration_seconds": round(res.duration_seconds, 1),
            "error": res.error or ""
        }
        
        # If success, attempt to extract metrics from the per-subject JSON
        if res.status == "success":
            try:
                # Construct path to the subject's JSON report
                subj_path = output_dir / res.subject_id
                if res.session_id:
                    subj_path = subj_path / res.session_id
                
                metrics_file = subj_path / "metrics" / f"{res.subject_id}_bilateral_metrics.json"
                
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        m = data.get("metrics", {})
                        
                        if 'left' in m and 'right' in m:
                            left = m['left']
                            right = m['right']
                            asym = m['asymmetry']
                            
                            # Morphology
                            row.update({
                                'left_n_streamlines': left['morphology']['n_streamlines'],
                                'right_n_streamlines': right['morphology']['n_streamlines'],
                                'left_mean_length_mm': left['morphology']['mean_length'],
                                'right_mean_length_mm': right['morphology']['mean_length'],
                                'left_tract_volume_mm3': left['morphology']['tract_volume'],
                                'right_tract_volume_mm3': right['morphology']['tract_volume'],
                                'volume_laterality_index': asym['volume']['laterality_index'],
                                'streamline_count_laterality_index': asym['streamline_count']['laterality_index'],
                            })
                            
                            # Scalars (Global)
                            for scalar in _SCALARS:
                                if scalar in left:
                                    row.update({
                                        f'left_{scalar}_mean': left[scalar]['mean'],
                                        f'left_{scalar}_std': left[scalar]['std'],
                                        f'right_{scalar}_mean': right[scalar]['mean'],
                                        f'right_{scalar}_std': right[scalar]['std'],
                                        f'{scalar}_laterality_index': asym[scalar]['laterality_index'],
                                        f'left_{scalar}_mean_point_weighted': left[scalar].get('mean_point_weighted'),
                                        f'right_{scalar}_mean_point_weighted': right[scalar].get('mean_point_weighted'),
                                    })
                                    
                                    # Localized Metrics
                                    if 'pontine' in left[scalar]: # check if localized metrics exist
                                        for region in _REGIONS:
                                            # Values
                                            l_val = left[scalar].get(region, 0.0)
                                            r_val = right[scalar].get(region, 0.0)
                                            row[f'left_{scalar}_{region}'] = l_val
                                            row[f'right_{scalar}_{region}'] = r_val
                                            
                                            # LI for region
                                            li_key = f'{scalar}_{region}'
                                            if li_key in asym:
                                                row[f'{scalar}_{region}_laterality_index'] = asym[li_key]['laterality_index']

            except Exception as e:
                logger.warning(f"Could not extract metrics for {res.subject_id}: {e}")
                
        rows.append(row)
        
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_HEADERS, restval="")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"✓ Batch metrics CSV: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to write batch_metrics.csv: {e}")

def write_batch_summary_json(results: List[Any], output_dir: Path):
    """
    Writes a versioned JSON summary of the entire batch run.
    Contains summary counters and individual status records.
    """
    summary_path = output_dir / "batch_summary.json"
    
    counts = {
        "total": len(results),
        "success": sum(1 for r in results if r.status == "success"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "skipped": sum(1 for r in results if r.status == "skipped")
    }
    
    summary = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "summary": counts,
        "results": [
            {
                "subject_id": r.subject_id,
                "session_id": r.session_id,
                "status": r.status,
                "error_category": r.error_category,
                "duration_seconds": round(r.duration_seconds, 1),
                "error": r.error
            } for r in results
        ]
    }
    
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Batch summary JSON: {summary_path}")
    except Exception as e:
        logger.error(f"Failed to write batch_summary.json: {e}")
