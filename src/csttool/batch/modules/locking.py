import os
import json
import socket
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


class LockError(Exception):
    """Raised when a lock cannot be acquired."""
    pass


@dataclass
class Lock:
    """Represents an acquired filesystem lock."""
    lock_file: Path
    _filelock: object  # filelock.FileLock instance


def _acquire_lock(lock_path: Path, name: str = "Batch") -> Lock:
    """Acquire an advisory filesystem lock using filelock (cross-platform)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fl = FileLock(str(lock_path) + ".filelock")
    try:
        fl.acquire(timeout=0)
    except Timeout:
        try:
            metadata = json.loads(lock_path.read_text())
            info = (f"held by PID {metadata.get('pid')} "
                    f"on {metadata.get('hostname')} "
                    f"since {metadata.get('started_at')}")
        except Exception:
            info = "held by another process"
        raise LockError(f"{name} lock {lock_path} {info}")

    try:
        metadata = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": datetime.now().isoformat(),
        }
        lock_path.write_text(json.dumps(metadata, indent=2))
    except Exception as exc:
        logger.warning(f"Failed to write metadata to lock file {lock_path}: {exc}")

    return Lock(lock_file=lock_path, _filelock=fl)


def acquire_batch_lock(output_dir: Path) -> Lock:
    """Acquire global batch lock in output root."""
    return _acquire_lock(output_dir / "batch.lock", name="Batch")


def acquire_subject_lock(subject_dir: Path) -> Lock:
    """Acquire per-subject lock in subject output directory."""
    return _acquire_lock(subject_dir / ".lock", name="Subject")


def release_lock(lock: Optional[Lock]) -> None:
    """Release the lock and remove the lock file."""
    if lock is None:
        return
    try:
        lock._filelock.release()
        lock.lock_file.unlink(missing_ok=True)
        Path(str(lock.lock_file) + ".filelock").unlink(missing_ok=True)
    except Exception as exc:
        logger.error(f"Error releasing lock {lock.lock_file}: {exc}")
