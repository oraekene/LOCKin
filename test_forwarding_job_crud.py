import os
import json
import shutil
import uuid
import pytest

# Ensure persistence_helpers picks up the test DATA_DIR
def _set_data_dir(tmp_path):
    test_dir = str(tmp_path / "data")
    os.environ["DATA_DIR"] = test_dir
    return test_dir

def test_create_load_update_delete_forwarding_job(tmp_path):
    test_dir = _set_data_dir(tmp_path)

    import persistence_helpers as ph

    user_id = 9999

    # --- create ---
    job = {
        "job_name": "unit-test-job",
        "filters": {"keywords": ["alpha", "beta"]},
        "dedup_window_seconds": 30,
        "cooldown_seconds_per_asset": 3600
    }
    job_id = ph.save_forwarding_job(user_id, job)
    assert job_id, "save_forwarding_job should return an id"

    # user file should exist and contain forwarding_jobs with our id
    user_obj = ph.read_user_file(user_id)
    assert isinstance(user_obj, dict)
    fw_jobs = user_obj.get("forwarding_jobs", [])
    assert any(j.get("id") == job_id for j in fw_jobs), "Saved job metadata not found in user file"

    # load by id
    loaded = ph.load_forwarding_job(user_id, job_id)
    assert loaded and loaded.get("id") == job_id

    # --- update ---
    loaded["keywords"] = ["gamma", "delta"]
    # ensure id preserved
    loaded["id"] = job_id
    returned = ph.save_forwarding_job(user_id, loaded)
    # returned may be id or truthy; ensure truthy
    assert returned

    # reload and check change (either metadata or full saved file)
    reloaded = ph.load_forwarding_job(user_id, job_id)
    assert reloaded is not None
    # depending on implementation, keywords may be in the saved full config; try to find them
    if "keywords" in reloaded:
        assert reloaded["keywords"] == ["gamma", "delta"]
    else:
        # try to find full job file in user_backtests
        backtests_dir = os.path.join(test_dir, "user_backtests", str(user_id))
        found_keywords = False
        if os.path.exists(backtests_dir):
            for fn in os.listdir(backtests_dir):
                path = os.path.join(backtests_dir, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        txt = f.read()
                    if "unit-test-job" in txt and ("gamma" in txt or "delta" in txt):
                        found_keywords = True
                        break
                except Exception:
                    continue
        assert found_keywords or ("keywords" in reloaded), "Updated keywords not persisted"

    # --- delete ---
    ok = ph.delete_forwarding_job(user_id, job_id)
    assert ok, "delete_forwarding_job should return True"

    final = ph.read_user_file(user_id)
    assert not any(j.get("id") == job_id for j in final.get("forwarding_jobs", [])), "Job metadata still present after delete"

    # cleanup (should be handled by tmp_path but be safe)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
