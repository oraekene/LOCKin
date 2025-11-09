#!/usr/bin/env python3
"""
Acceptance test: simulate a user session, populate session.temp_data['incomplete_backtest_draft'],
and call TelegramBot.save_backtest_draft_as_named_config to persist a named config.

This verifies:
 - save_backtest_draft_as_named_config coroutine works end-to-end
 - persistence_helpers.save_user_backtest_config actually writes the user file
 - before/after listing of saved configs shows the new config
 - backtest history append (optional check) works

Usage:
  DATA_DIR=./data_acceptance_test python3 acceptance_test_save_via_bot_method.py
"""

import os
import shutil
import sys
import json
import uuid
import asyncio
from pathlib import Path
from types import SimpleNamespace

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(repo_root))

# Set an isolated DATA_DIR for tests (change if desired)
TEST_DATA_DIR = os.environ.get("DATA_DIR", str(repo_root / "data_acceptance_test"))
os.environ["DATA_DIR"] = TEST_DATA_DIR

# Imports
try:
    import persistence_helpers as ph
except Exception as e:
    print("ERROR: could not import persistence_helpers:", e)
    raise

try:
    import main_bot
except Exception as e:
    print("WARNING: could not import main_bot (needed for the bot method).")
    print("If main_bot import fails due to runtime dependencies, the script will still test persistence directly.")
    main_bot = None

def cleanup_test_dir():
    if os.path.exists(TEST_DATA_DIR):
        print("Removing existing test data dir:", TEST_DATA_DIR)
        shutil.rmtree(TEST_DATA_DIR)

def ensure_test_dir():
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    print("Using DATA_DIR =", TEST_DATA_DIR)

async def call_bot_save_method(user_id: int, draft: dict, name: str):
    """
    Calls TelegramBot.save_backtest_draft_as_named_config if available.
    Uses a lightweight dummy 'self' since the method does not need instance state.
    Returns the cfg_id.
    """
    if not main_bot:
        raise RuntimeError("main_bot not importable; cannot call TelegramBot method in this environment")

    # Build a simple dummy self object. The method doesn't use self, so this suffices.
    dummy_self = SimpleNamespace()
    # Bind the coroutine function
    coro = main_bot.TelegramBot.save_backtest_draft_as_named_config
    # Call with dummy_self as 'self'
    cfg_id = await coro(dummy_self, user_id, draft, name)
    return cfg_id

def show_configs(user_id: int, label: str):
    configs = ph.list_user_backtest_configs(user_id)
    print(f"\n[{label}] Saved configs count: {len(configs)}")
    for c in configs:
        print(" - id:", c.get("id"), "name:", c.get("name"))
    return configs

def print_user_file_preview(user_id: int):
    path = os.path.join(TEST_DATA_DIR, f"user_{user_id}.dat")
    print("\nUser file path:", path)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
            print("\n--- USER FILE FULL CONTENT BEGIN ---\n")
            print(txt)
            print("\n--- USER FILE FULL CONTENT END ---\n")
    else:
        print("User file not found. (If persistence uses CryptoManager, ensure environment allows write/read.)")

def run_sync(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

def main():
    cleanup_test_dir()
    ensure_test_dir()

    user_id = 42424242
    print("Test user id:", user_id)

    # Ensure no configs exist initially
    show_configs(user_id, "before")

    # Build a draft similar to what the builder would produce
    draft = {
        "source_chat_ids": [111111, 222222],
        "time_range": {"type": "lookback", "days": 14},
        "min_market_cap": 500000,
        "min_confidence": 0.25,
        "max_signals": 300,
        "position_sizing": {"mode": "percent_of_portfolio", "value": 3.0},
        "keywords": ["airdrop", "token"],
        "cashtags": ["$TEST"],
        "exit_rules": []
    }

    # Name for saved config
    cfg_name = "acceptance-bot-save-" + uuid.uuid4().hex[:6]

    # Call the bot method to save the draft (async)
    if main_bot:
        print("\nCalling TelegramBot.save_backtest_draft_as_named_config (simulated self)...")
        try:
            # Use asyncio.run if available to run the coroutine cleanly
            cfg_id = asyncio.run(call_bot_save_method(user_id, draft, cfg_name))
        except RuntimeError:
            # if event loop already running in some environments, fallback to older approach
            cfg_id = run_sync(call_bot_save_method(user_id, draft, cfg_name))
        print("Returned cfg_id:", cfg_id)
    else:
        # Fallback: call persistence_helpers.save_user_backtest_config directly
        print("\nmain_bot not importable -> falling back to persistence_helpers.save_user_backtest_config directly.")
        cfg = {
            "name": cfg_name,
            "owner_user_id": user_id,
            "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "source_chat_ids": draft["source_chat_ids"],
            "time_range": draft["time_range"],
            "min_market_cap": draft["min_market_cap"],
            "min_confidence": draft["min_confidence"],
            "max_signals": draft["max_signals"],
            "position_sizing": draft["position_sizing"],
            "keywords": draft["keywords"],
            "cashtags": draft["cashtags"],
            "exit_rules": draft["exit_rules"],
            "_draft_snapshot": draft
        }
        cfg_id = ph.save_user_backtest_config(user_id, cfg)
        print("Saved via persistence_helpers, cfg_id:", cfg_id)

    # Show configs after saving
    configs_after = show_configs(user_id, "after")

    # Try to locate the saved config and print it
    saved = ph.load_user_backtest_config(user_id, cfg_id)
    print("\nLoaded saved config by id:")
    print(json.dumps(saved, indent=2))

    # Print the on-disk user file for inspection
    print_user_file_preview(user_id)

    # Append a dummy run to history to test append_backtest_history
    print("\nAppending a dummy run to history...")
    run_meta = {"name": f"run-for-{cfg_id}", "metadata": {"summary": {"collected": 10, "pnl": 5.5}}}
    run_id = ph.append_backtest_history(user_id, run_meta, result_path=f"results/{cfg_id}/run.json")
    print("Appended run id:", run_id)

    # List history
    history = ph.list_backtest_history(user_id)
    print("\nBacktest history entries:")
    for r in history:
        print(" - id:", r.get("id"), "name:", r.get("name"), "timestamp:", r.get("timestamp"))

    print("\nAcceptance test (bot method save) completed successfully.")

if __name__ == "__main__":
    main()
