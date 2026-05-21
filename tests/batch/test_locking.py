import pytest
import os
from pathlib import Path
from csttool.batch.modules.locking import acquire_batch_lock, acquire_subject_lock, release_lock, LockError

def test_batch_lock_flow(tmp_path):
    # 1. Acquire lock
    lock = acquire_batch_lock(tmp_path)
    assert lock.lock_file.exists()
    assert lock.lock_file.name == "batch.lock"
    
    # 2. Try to acquire again - should fail
    with pytest.raises(LockError) as exc:
        acquire_batch_lock(tmp_path)
    assert "Batch lock" in str(exc.value)
    
    # 3. Release lock
    release_lock(lock)
    
    # 4. Acquire again - should succeed
    lock2 = acquire_batch_lock(tmp_path)
    assert lock2._filelock is not None
    release_lock(lock2)

def test_subject_lock_flow(tmp_path):
    sub_dir = tmp_path / "sub-01"
    sub_dir.mkdir()
    
    # 1. Acquire subject lock
    lock = acquire_subject_lock(sub_dir)
    assert lock.lock_file.exists()
    assert lock.lock_file.parent == sub_dir
    assert lock.lock_file.name == ".lock"
    
    # 2. Try to acquire again - should fail
    with pytest.raises(LockError) as exc:
        acquire_subject_lock(sub_dir)
    assert "Subject lock" in str(exc.value)
    
    # 3. Release
    release_lock(lock)
    
    # 4. Success after release
    lock2 = acquire_subject_lock(sub_dir)
    release_lock(lock2)

def test_lock_metadata_content(tmp_path):
    lock = acquire_batch_lock(tmp_path)
    content = lock.lock_file.read_text()
    import json
    data = json.loads(content)
    assert "pid" in data
    assert "hostname" in data
    assert "started_at" in data
    release_lock(lock)
