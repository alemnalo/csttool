import os
import sys
import time
import signal
import json
import shutil
import hashlib
import logging
import argparse
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime

from .modules.locking import acquire_subject_lock, release_lock
from .modules.logging_setup import setup_subject_logger

try:
    from csttool import __version__
except ImportError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

# Error Categories
class ErrorCategory:
    INPUT_MISSING = "INPUT_MISSING"
    VALIDATION = "VALIDATION"
    PIPELINE_FAILED = "PIPELINE_FAILED"
    TIMEOUT = "TIMEOUT"
    SYSTEM = "SYSTEM"

@dataclass
class SubjectSpec:
    """Specification for a single subject/session to iterate over."""
    subject_id: str
    session_id: Optional[str]
    input_path: Path
    input_type: Literal["nifti", "dicom"]
    bval_path: Optional[Path] = None
    bvec_path: Optional[Path] = None
    json_path: Optional[Path] = None
    series_uid: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchConfig:
    """Configuration for the entire batch run."""
    out: Path
    force: bool = False
    timeout_minutes: int = 120
    keep_work: bool = False
    # Pipeline overrides
    denoise_method: str = "nlmeans"
    preprocessing: bool = True
    generate_pdf: bool = False
    # ... any other arguments that cmd_run accepts

@dataclass
class SubjectResult:
    """Result of a single subject/session processing attempt."""
    subject_id: str
    session_id: Optional[str]
    status: Literal["success", "failed", "skipped"]
    error_category: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    log_path: Optional[Path] = None
    outputs: Optional[Dict[str, str]] = None

def compute_config_hash(config: BatchConfig) -> str:
    """
    Computes a stable hash of the batch configuration.
    Used to detect if settings have changed since a previous run.
    """
    # Exclude non-config fields like 'force' or 'keep_work' if they don't affect output data
    # Actually, almost everything in BatchConfig affects output except force/keep_work/timeout.
    data = asdict(config)
    excluded = {'force', 'keep_work', 'timeout_minutes'}
    filtered = {k: str(v) for k, v in data.items() if k not in excluded}
    
    config_json = json.dumps(filtered, sort_keys=True)
    return hashlib.sha256(config_json.encode('utf-8')).hexdigest()

def is_subject_completed(subject: SubjectSpec, output_dir: Path, expected_hash: str) -> bool:
    """Checks if a subject has already been successfully completed with the same config."""
    done_file = output_dir / "_done.json"
    if not done_file.exists():
        return False
        
    try:
        with open(done_file, 'r') as f:
            data = json.load(f)
            return data.get("config_hash") == expected_hash
    except Exception:
        return False

def promote_outputs(work_dir: Path, final_dir: Path, done_marker: Dict[str, Any]) -> None:
    """
    Atomically promote outputs from _work/ to final location.
    The _done.json file is written LAST to ensure atomicity.
    """
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Rename output files/directories
    # We move them from _work/ to final_dir
    for item in work_dir.iterdir():
        if item.name == "_done.json": # Reserved
            continue
        
        target = final_dir / item.name
        # If target exists (e.g. from a previous partial success), remove it
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        
        os.rename(item, target)
        
    # 2. Write _done.json atomically (write to .tmp then rename)
    done_file = final_dir / "_done.json"
    tmp_done = final_dir / "_done.json.tmp"
    
    with open(tmp_done, 'w') as f:
        json.dump(done_marker, f, indent=2)
        
    os.rename(tmp_done, done_file)

def _get_subject_output_dir(base_out: Path, sub_id: str, ses_id: Optional[str]) -> Path:
    """Returns the final output directory for a subject/session."""
    path = base_out / sub_id
    if ses_id:
        path = path / ses_id
    return path

# Global for signal handling
_current_child: Optional[multiprocessing.Process] = None

def run_subject_with_timeout(
    subject: SubjectSpec, 
    config: BatchConfig,
    log_path: Path
) -> SubjectResult:
    """
    Runs a single subject in an isolated child process with a timeout.
    """
    global _current_child
    
    queue = multiprocessing.Queue()
    _current_child = multiprocessing.Process(
        target=_run_subject_worker, 
        args=(subject, config, log_path, queue)
    )
    
    start_time = time.time()
    _current_child.start()
    
    # Wait for completion or timeout
    _current_child.join(timeout=config.timeout_minutes * 60)
    
    if _current_child.is_alive():
        # Timeout occurred
        logger.warning(f"Timeout: Subject {subject.subject_id} exceeded {config.timeout_minutes}m")
        _current_child.terminate()
        _current_child.join(timeout=30) # Grace period
        if _current_child.is_alive():
            _current_child.kill()
            _current_child.join()
            
        _current_child = None
        return SubjectResult(
            subject_id=subject.subject_id,
            session_id=subject.session_id,
            status="failed",
            error_category=ErrorCategory.TIMEOUT,
            duration_seconds=time.time() - start_time,
            error=f"Processing exceeded {config.timeout_minutes} minute limit",
            log_path=log_path
        )
    
    # Check if process exited normally
    exit_code = _current_child.exitcode
    _current_child = None
    
    if exit_code != 0:
        return SubjectResult(
            subject_id=subject.subject_id,
            session_id=subject.session_id,
            status="failed",
            error_category=ErrorCategory.SYSTEM,
            duration_seconds=time.time() - start_time,
            error=f"Child process exited with code {exit_code}",
            log_path=log_path
        )
        
    # Get result from queue
    try:
        # Wait a bit for the queue to populate
        return queue.get(timeout=5)
    except Exception as e:
        return SubjectResult(
            subject_id=subject.subject_id,
            session_id=subject.session_id,
            status="failed",
            error_category=ErrorCategory.SYSTEM,
            duration_seconds=time.time() - start_time,
            error=f"Failed to retrieve result from worker: {e}",
            log_path=log_path
        )

def _run_subject_worker(
    subject: SubjectSpec, 
    config: BatchConfig, 
    log_path: Path, 
    queue: multiprocessing.Queue
) -> None:
    """
    Entry point for child process.
    Sets up logging and runs the actual pipeline.
    """
    # 1. Setup isolated logger
    subj_logger = setup_subject_logger(
        subject.subject_id, subject.session_id, log_path, verbose=True
    )
    
    start_time = time.time()
    
    try:
        # 2. Prepare output directory
        base_out = _get_subject_output_dir(config.out, subject.subject_id, subject.session_id)
        work_dir = base_out / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Create arg namespace for cmd_run proxy
        # We'll use a simplified version of cmd_run logic directly
        # to avoid argparse dependency inside the worker.
        from csttool.cli.commands.run import cmd_run
        
        # Merge global options with per-subject options
        merged_options = asdict(config)
        merged_options.update(subject.options)
        
        # Construct Namespace compatible with cmd_run
        args = argparse.Namespace()
        for k, v in merged_options.items():
            setattr(args, k, v)
            
        # Ensure correct input paths are set
        setattr(args, 'subject_id', subject.subject_id)
        setattr(args, 'session_id', subject.session_id)
        setattr(args, 'out', work_dir) 
        
        if subject.input_type == "nifti":
            setattr(args, 'nifti', subject.input_path)
            setattr(args, 'dicom', None)
        else:
            setattr(args, 'dicom', subject.input_path)
            setattr(args, 'nifti', None)
            setattr(args, 'series', None)
            setattr(args, 'series_uid', subject.series_uid)
            
        # 4. Execute pipeline
        # cmd_run prints to stdout, which console handler of subj_logger catches
        # if verbose is set. Actually cmd_run uses print(), which doesn't go through logging.
        # We might need to redirect stdout if we want it in the structured log.
        # For v1, we assume SubjectLogger's StreamHandler handles plain prints if they are redirected.
        # sys.stdout = StreamLogger(subj_logger, logging.INFO) # optional
        
        cmd_run(args)
        
        # 5. Success
        duration = time.time() - start_time
        queue.put(SubjectResult(
            subject_id=subject.subject_id,
            session_id=subject.session_id,
            status="success",
            duration_seconds=duration,
            log_path=log_path
        ))
        
    except Exception as e:
        subj_logger.error(f"Pipeline failed: {e}", exc_info=True)
        queue.put(SubjectResult(
            subject_id=subject.subject_id,
            session_id=subject.session_id,
            status="failed",
            error_category=ErrorCategory.PIPELINE_FAILED,
            error=str(e),
            duration_seconds=time.time() - start_time,
            log_path=log_path
        ))

def run_batch(
    subjects: List[SubjectSpec],
    config: BatchConfig,
    verbose: bool = False
) -> List[SubjectResult]:
    """
    Main batch processing orchestrator.
    Handles resume logic, per-subject locks, and output promotion.
    """
    config_hash = compute_config_hash(config)
    results = []

    # Setup signal handling for the main orchestrator
    def _sig_handler(signum, frame):
        logger.warning(f"Batch interrupted by signal {signum}. Shutting down...")
        # Close the current child if it exists
        if _current_child and _current_child.is_alive():
            _current_child.terminate()
        sys.exit(1)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    logger.info(f"Starting batch processing for {len(subjects)} subjects")
    logger.info(f"Output directory: {config.out}")
    if verbose:
        logger.info(f"Config hash: {config_hash[:8]}")

    for subj_spec in subjects:
        final_dir = _get_subject_output_dir(config.out, subj_spec.subject_id, subj_spec.session_id)
        
        # 1. Check if already completed
        if not config.force and is_subject_completed(subj_spec, final_dir, config_hash):
            if verbose:
                logger.info(f"Skipping {subj_spec.subject_id}: Already completed")
            results.append(SubjectResult(
                subject_id=subj_spec.subject_id,
                session_id=subj_spec.session_id,
                status="skipped"
            ))
            continue
            
        # 2. Acquire subject lock
        try:
            lock = acquire_subject_lock(final_dir)
        except Exception as e:
            logger.error(f"Could not acquire lock for {subj_spec.subject_id}: {e}")
            results.append(SubjectResult(
                subject_id=subj_spec.subject_id,
                session_id=subj_spec.session_id,
                status="failed",
                error_category=ErrorCategory.SYSTEM,
                error=f"Locking failed: {e}"
            ))
            continue
            
        try:
            # 3. Setup log path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"{subj_spec.subject_id}"
            if subj_spec.session_id:
                log_name += f"_{subj_spec.session_id}"
            log_path = final_dir / "logs" / f"{log_name}_{timestamp}.log"
            
            # 4. Run subject in isolated process
            res = run_subject_with_timeout(subj_spec, config, log_path)
            
            # 5. Handle output promotion or failure move
            work_dir = final_dir / "_work"
            if res.status == "success":
                logger.info(f"✓ {subj_spec.subject_id}: Success in {res.duration_seconds:.1f}s")
                done_marker = {
                    "csttool_version": __version__,
                    "config_hash": config_hash,
                    "completed_at": datetime.now().isoformat(),
                    "duration_seconds": res.duration_seconds,
                    "outputs": {}
                }
                promote_outputs(work_dir, final_dir, done_marker)
                if not config.keep_work:
                    if work_dir.exists():
                        shutil.rmtree(work_dir)
            else:
                logger.error(f"✗ {subj_spec.subject_id}: Failed ({res.error_category})")
                failed_dir = final_dir / "_failed_attempts" / f"{timestamp}_{res.error_category}"
                failed_dir.mkdir(parents=True, exist_ok=True)
                if work_dir.exists():
                    os.rename(work_dir, failed_dir / "_work")
                    
            results.append(res)
            
        finally:
            release_lock(lock)
            
    return results
