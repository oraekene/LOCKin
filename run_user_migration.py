#!/usr/bin/env python3
"""
scripts/run_user_migration.py

Usage:
    DATA_DIR=./data python scripts/run_user_migration.py

This script:
  1) executes scripts/backup_user_data.sh (best-effort)
  2) loads each user_{id}.dat, runs migrate_user_file_if_needed, and writes the normalized file back.
"""
import os
import subprocess
import sys
import json
import glob
from pathlib import Path

# Attempt to import persistence_helpers from repo path
try:
    import persistence_helpers as ph
except Exception as e:
    # if running from scripts dir, add repository root to path
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    import persistence_helpers as ph

def run_backup_script():
    script = os.path.join(os.path.dirname(__file__), "backup_user_data.sh")
    if os.path.exists(script) and os.access(script, os.X_OK):
        print("Running backup script:", script)
        subprocess.check_call([script], env=os.environ)
    elif os.path.exists(script):
        print("Making backup script executable and running:", script)
        os.chmod(script, 0o750)
        subprocess.check_call([script], env=os.environ)
    else:
        print("Backup script not found at", script, "- skipping automatic backup. Run manually before migrating.")
        return

def migrate_all_users(data_dir):
    pattern = os.path.join(data_dir, "user_*.dat")
    files = glob.glob(pattern)
    if not files:
        print("No user_*.dat files found in", data_dir)
        return
    for f in files:
        try:
            # attempt to parse uid
            base = os.path.basename(f)
            uid = int(base.split("_", 1)[1].split(".dat", 1)[0])
        except Exception:
            print("Skipping non-conforming filename:", f)
            continue
        print("Migrating user:", uid)
        try:
            ud = ph.read_user_file(uid)
        except Exception as e:
            print("  -> Failed to read user file:", e)
            continue
        migrated = ph.migrate_user_file_if_needed(ud)
        try:
            ph.write_user_file(uid, migrated)
            print("  -> migrated and saved:", f)
        except Exception as e:
            print("  -> Failed to write migrated file:", e)

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIR") or ph._get_data_dir()
    print("Using DATA_DIR =", data_dir)
    try:
        run_backup_script()
    except subprocess.CalledProcessError as e:
        print("Backup script failed (non-zero exit). Aborting migration.", e)
        sys.exit(2)
    migrate_all_users(data_dir)
    print("Migration pass complete.")
