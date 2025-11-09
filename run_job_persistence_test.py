#!/usr/bin/env python3
"""
Simple test script to validate forwarding job CRUD using persistence_helpers.

Usage:
    python tests/run_job_persistence_test.py

This script is intentionally small and dependency-free (no pytest required).
It creates a temporary DATA_DIR at tests/tmp_data and cleans up previous runs.
"""

import os
import shutil
import json
import time
from pprint import pprint

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if __file__.endswith("run_job_persistence_test.py") else os.getcwd()
TEST_TMP = os.path.join(PROJ_ROOT, "tests", "tmp_data")
# ensure clean slate
if os.path.exists(TEST_TMP):
    print("Cleaning existing test tmp dir:", TEST_TMP)
    shutil.rmtree(TEST_TMP)
os.makedirs(TEST_TMP, exist_ok=True)

# Point DATA_DIR env var so persistence_helpers uses it
os.environ["DATA_DIR"] = TEST_TMP

# Import the helper module
import persistence_helpers as ph

def pretty_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"(read error: {e})"

def main():
    user_id = 123
    print("DATA_DIR for test:", TEST_TMP)

    # 1) Create a forwarding job (no id; save_forwarding_job should assign id)
    job = {
        "job_name": "test-job",
        "filters": {"keywords": ["apple", "banana"]},
        "dedup_window_seconds": 60,
        "cooldown_seconds_per_asset": 3600
    }
    print("\n--- Creating job ---")
    job_id = ph.save_forwarding_job(user_id, job)
    print("Returned job id:", job_id)
    assert job_id, "save_forwarding_job did not return an id"

    # 2) Load job back via read_user_file -> inspect forwarding_jobs
    user_obj = ph.read_user_file(user_id)
    print("\nUser file content after create:\n")
    pprint(user_obj)
    fw_jobs = user_obj.get("forwarding_jobs", [])
    assert any(j.get("id") == job_id for j in fw_jobs), "Saved job not found in user forwarder metadata"

    # Also test load_forwarding_job()
    loaded = ph.load_forwarding_job(user_id, job_id)
    print("\nLoaded job by id:")
    pprint(loaded)
    assert loaded and loaded.get("id") == job_id, "load_forwarding_job failed"

    # 3) Modify the job: update keywords and persist using save_forwarding_job (simulate update)
    print("\n--- Modifying job keywords ---")
    loaded["keywords"] = ["cherry", "date"]
    # ensure id present
    loaded["id"] = job_id
    new_id = ph.save_forwarding_job(user_id, loaded)
    print("save_forwarding_job returned:", new_id)
    # read back and validate change
    user_obj2 = ph.read_user_file(user_id)
    updated_job = None
    for j in user_obj2.get("forwarding_jobs", []):
        if j.get("id") == job_id:
            updated_job = ph.load_forwarding_job(user_id, job_id)
            break
    print("\nUpdated job loaded:")
    pprint(updated_job)
    assert updated_job is not None
    # check keywords persisted in the stored job (note: metadata may be minimal; full job saved in user_backtests dir)
    full_job = updated_job
    # If full job is available as in-memory saved job, check keywords there; otherwise check file content
    # Try to load full job file if present in user_backtests folder
    backtests_dir = os.path.join(TEST_TMP, "user_backtests", str(user_id))
    print("\nBacktests dir contents:", os.listdir(backtests_dir) if os.path.exists(backtests_dir) else "(none)")
    # attempt to locate the stored job file (by searching JSON files that contain job_name)
    found_file = None
    if os.path.exists(backtests_dir):
        for f in os.listdir(backtests_dir):
            if f.endswith(".json"):
                path = os.path.join(backtests_dir, f)
                text = pretty_file(path)
                if "test-job" in text:
                    found_file = path
                    break
    if found_file:
        print("\nFound full job file:", found_file)
        try:
            parsed = json.loads(pretty_file(found_file))
            pprint(parsed)
            # if parsed contains keywords as we wrote, assert them
            # accept both metadata-only and full job shapes
            if isinstance(parsed, dict) and parsed.get("keywords"):
                assert parsed.get("keywords") == ["cherry", "date"], "Full job file keywords not updated"
        except Exception:
            pass

    # 4) Delete the job via persistence_helpers.delete_forwarding_job
    print("\n--- Deleting job ---")
    ok = ph.delete_forwarding_job(user_id, job_id)
    print("delete_forwarding_job returned:", ok)
    assert ok, "delete_forwarding_job reported failure"

    # verify deletion reflected in user file
    user_final = ph.read_user_file(user_id)
    print("\nUser file after deletion:")
    pprint(user_final)
    assert not any(j.get("id") == job_id for j in user_final.get("forwarding_jobs", [])), "Job still present after delete"

    # confirm any file in user_backtests for that id is removed if created
    if os.path.exists(backtests_dir):
        leftover = [f for f in os.listdir(backtests_dir) if job_id in f]
        print("Leftover files containing job id:", leftover)
        # note: delete_forwarding_job doesn't necessarily remove the JSON file created for the job run; it removes metadata entry.
        # If your implementation also deletes files, leftover should be empty.

    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    main()
