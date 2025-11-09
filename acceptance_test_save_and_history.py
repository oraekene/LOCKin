#!/usr/bin/env python3
"""
Acceptance test: Save as named config + append run history

This script tests:
  - persistence_helpers.save_user_backtest_config
  - persistence_helpers.list_user_backtest_configs
  - persistence_helpers.load_user_backtest_config
  - persistence_helpers.append_backtest_history
  - persistence_helpers.list_backtest_history

It uses an isolated DATA_DIR (./data_acceptance_test) so it won't touch production data.
"""

import os
import shutil
import json
import sys
import uuid
from pathlib import Path

# Make sure the repo root (.) is on sys.path so imports work when running from repo root.
repo_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(repo_root))

# Override DATA_DIR for the test (change if you prefer another location)
TEST_DATA_DIR = os.environ.get("DATA_DIR", str(repo_root / "data_acceptance_test"))
os.environ["DATA_DIR"] = TEST_DATA_DIR

# Import persistence helpers
try:
    import persistence_helpers as ph
except Exception as e:
    print("ERROR: Could not import persistence_helpers:", e)
    raise

def cleanup_test_dir():
    """Remove the test data dir if present (useful to run repeatedly)."""
    if os.path.exists(TEST_DATA_DIR):
        print("Cleaning up existing test data dir:", TEST_DATA_DIR)
        shutil.rmtree(TEST_DATA_DIR)

def ensure_test_dir():
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    print("Using DATA_DIR =", TEST_DATA_DIR)

def run_acceptance_flow():
    user_id = 123456789  # test user id (no relation to real users)
    print("Test user id:", user_id)

    # 1) Create a sample draft/config
    sample_cfg = {
        "name": "acceptance-test-config-" + uuid.uuid4().hex[:6],
        "owner_user_id": user_id,
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "source_chat_ids": [111111, 222222],
        "time_range": {"type": "lookback", "days": 7},
        "min_market_cap": 1000000,
        "min_confidence": 0.2,
        "max_signals": 250,
        "position_sizing": {"mode": "percent_of_portfolio", "value": 5.0},
        "keywords": ["token", "airdrops"],
        "cashtags": ["$ABC", "$XYZ"],
        "exit_rules": []
    }

    # 2) Save config
    print("\nSaving sample config...")
    cfg_id = ph.save_user_backtest_config(user_id, sample_cfg)
    print("Saved config id:", cfg_id)

    # 3) List configs and confirm presence
    configs = ph.list_user_backtest_configs(user_id)
    print("\nListed saved configs (count={}):".format(len(configs)))
    for c in configs:
        print(" - id:", c.get("id"), "name:", c.get("name"))

    # Assert our saved config is present
    assert any(c.get("id") == cfg_id for c in configs), "Saved config not found in list_user_backtest_configs"

    # 4) Load the saved config
    loaded = ph.load_user_backtest_config(user_id, cfg_id)
    print("\nLoaded config (id={}):".format(cfg_id))
    print(json.dumps(loaded, indent=2))

    assert loaded is not None and loaded.get("id") == cfg_id, "Loaded config mismatch"

    # 5) Append a backtest run to history
    print("\nAppending a run to history...")
    run_meta = {
        "name": f"run-for-{cfg_id}",
        "metadata": {"summary": {"collected": 123, "pnl": 12.34}}
    }
    run_id = ph.append_backtest_history(user_id, run_meta, result_path=f"results/{cfg_id}/run.json")
    print("Appended run id:", run_id)

    # 6) List backtest history and verify
    history = ph.list_backtest_history(user_id)
    print("\nBacktest history (count={}):".format(len(history)))
    for r in history:
        print(" - run id:", r.get("id"), "name:", r.get("name"), "timestamp:", r.get("timestamp"))

    assert any(r.get("id") == run_id for r in history), "Appended run not found in history"

    # 7) Load run specifically
    loaded_run = ph.get_backtest_run(user_id, run_id)
    print("\nLoaded run (id={}):".format(run_id))
    print(json.dumps(loaded_run, indent=2))

    assert loaded_run is not None and loaded_run.get("id") == run_id, "Loaded run mismatch"

    # 8) Summary - show the path to the user file for manual inspection
    user_file_path = os.path.join(TEST_DATA_DIR, f"user_{user_id}.dat")
    print("\nUser file path (for manual inspection):", user_file_path)
    if os.path.exists(user_file_path):
        print("User file content preview (first 400 chars):")
        with open(user_file_path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read(400)
            print(text)
    else:
        print("User file not present (this may be because persistence uses a CryptoManager; check your configuration).")

    print("\nAcceptance test completed successfully.")

if __name__ == "__main__":
    # Start clean
    cleanup_test_dir()
    ensure_test_dir()
    run_acceptance_flow()
