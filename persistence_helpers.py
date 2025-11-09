"""
persistence_helpers.py

Filesystem-backed persistence helpers for:
 - saved backtest configs
 - backtest history
 - forwarding jobs
 - migration helper for user_{id}.dat

Design notes:
 - Tries to reuse main_bot.CryptoManager and main_bot.DATA_DIR when available to preserve encrypted user files.
 - Falls back to plain JSON files if CryptoManager is not available (useful for tests).
 - Writes to DATA_DIR atomically using tempfile + os.replace.
 - Uses UUIDv4 ids for saved configs and runs.
 - Keeps metadata arrays in the user_{id}.dat file under keys:
     - saved_backtests
     - backtest_history
     - forwarding_jobs
     - scheduled_backtests
"""

import os
import json
import tempfile
import uuid
import shutil
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# Defaults
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
USER_FILE_TEMPLATE = "user_{user_id}.dat"
USER_BACKTESTS_DIRNAME = "user_backtests"

# Helper: try to import main_bot module to reuse its CryptoManager / DATA_DIR if present
def _get_main_module():
    try:
        import main_bot as mb
        return mb
    except Exception:
        return None

def _get_data_dir() -> str:
    mb = _get_main_module()
    if mb is not None and hasattr(mb, "DATA_DIR") and mb.DATA_DIR:
        return mb.DATA_DIR
    return DEFAULT_DATA_DIR

def _ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

# ---------- low-level read/write for user_{id}.dat (supports optional CryptoManager) ----------
def read_user_file(user_id: int) -> Dict[str, Any]:
    """
    Read and return the parsed user object for user_{id}.dat.
    If the file does not exist or is corrupt, returns {}.
    Attempts to use main_bot.CryptoManager.decrypt if available.
    """
    DATA_DIR = _get_data_dir()
    _ensure_dirs(DATA_DIR)
    user_file = os.path.join(DATA_DIR, USER_FILE_TEMPLATE.format(user_id=user_id))
    if not os.path.exists(user_file):
        return {}

    mb = _get_main_module()
    CryptoManager = getattr(mb, "CryptoManager", None) if mb is not None else None

    try:
        with open(user_file, "rb") as f:
            raw = f.read()
        # If CryptoManager present, try to decrypt
        if CryptoManager:
            # support both class/static and instance styles
            if hasattr(CryptoManager, "decrypt"):
                text = CryptoManager.decrypt(raw)
            else:
                # fallback: CryptoManager is a callable
                text = CryptoManager(raw)
        else:
            # assume plaintext JSON stored
            text = raw.decode("utf-8")
        return json.loads(text) if text else {}
    except Exception:
        # On any failure, return empty dict so migration/appending doesn't crash
        return {}

def write_user_file(user_id: int, obj: Dict[str, Any]) -> None:
    """
    Atomically write the user object back to user_{id}.dat.
    Uses CryptoManager.encrypt if available, otherwise writes plaintext JSON.
    """
    DATA_DIR = _get_data_dir()
    _ensure_dirs(DATA_DIR)
    user_file = os.path.join(DATA_DIR, USER_FILE_TEMPLATE.format(user_id=user_id))

    mb = _get_main_module()
    CryptoManager = getattr(mb, "CryptoManager", None) if mb is not None else None

    plaintext = json.dumps(obj, indent=2, sort_keys=False)
    # prepare bytes
    if CryptoManager:
        if hasattr(CryptoManager, "encrypt"):
            ciphertext = CryptoManager.encrypt(plaintext)
        else:
            ciphertext = CryptoManager(plaintext)
        # ensure bytes
        if isinstance(ciphertext, str):
            data = ciphertext.encode("utf-8")
        else:
            data = ciphertext
    else:
        data = plaintext.encode("utf-8")

    # atomic write
    fd, tmp_path = tempfile.mkstemp(prefix=f"user_{user_id}_", suffix=".tmp", dir=DATA_DIR)
    try:
        with os.fdopen(fd, "wb") as tmpf:
            tmpf.write(data)
        os.replace(tmp_path, user_file)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# ---------- migration helper ----------
# ----- REPLACEMENT: migrate_user_file_if_needed -----
def migrate_user_file_if_needed(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the user_data dict contains the keys we expect and normalise legacy items.

    - Adds missing top-level keys with defaults:
        saved_backtests: []
        backtest_history: []
        forwarding_jobs: []
        scheduled_backtests: []
    - For each forwarding job ensure 'last_triggered_by_asset' exists (dict).
    - For legacy exit rules (no 'id') create a stable deterministic id
      as sha256(json.dumps(rule, sort_keys=True)).hexdigest() and add a name if missing.
    - This function returns the updated dict (does NOT persist files).
    """
    import hashlib
    import json as _json

    user_data = user_data or {}
    # Top-level lists
    user_data.setdefault("saved_backtests", [])
    user_data.setdefault("backtest_history", [])
    user_data.setdefault("forwarding_jobs", [])
    user_data.setdefault("scheduled_backtests", [])

    # Ensure forwarding_jobs entries have default fields
    for job in user_data.get("forwarding_jobs", []):
        if not isinstance(job, dict):
            continue
        job.setdefault("dedup_window_seconds", 60)
        # default cooldown is 24h as per plan
        job.setdefault("cooldown_seconds_per_asset", 86400)
        job.setdefault("last_triggered_by_asset", {})

    # Normalize saved exit-rules if present under any legacy key
    # Two common legacy places:
    #  - user_data.get("exit_rules", [])
    #  - inside forwarding_jobs[*].get("exit_rules", [])
    def _ensure_rule_ids_and_names(rule):
        """
        Ensure rule is a dict with id and name. For legacy anonymous rules,
        generate deterministic id from sha256 of rule JSON.
        """
        if not isinstance(rule, dict):
            return rule
        if rule.get("id"):
            return rule
        # produce deterministic id
        payload = _json.dumps(rule, sort_keys=True, separators=(",", ":"))
        rule_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        rule["id"] = rule_id
        if not rule.get("name"):
            # short fallback name
            rule["name"] = f"legacy_rule_{rule_id[:8]}"
        return rule

    # Top-level legacy exit_rules
    if "exit_rules" in user_data and isinstance(user_data["exit_rules"], list):
        new_list = []
        for r in user_data["exit_rules"]:
            new_list.append(_ensure_rule_ids_and_names(r))
        user_data["exit_rules"] = new_list

    # Scan forwarding_jobs for exit-rules embedded
    for job in user_data.get("forwarding_jobs", []):
        ers = job.get("exit_rules") or job.get("exit_rule_ids") or []
        if isinstance(ers, list) and ers:
            normalized = []
            for e in ers:
                if isinstance(e, dict):
                    normalized.append(_ensure_rule_ids_and_names(e))
                else:
                    # if it's already an id or string, keep as-is
                    normalized.append(e)
            # prefer storing as exit_rule_ids (ids only) if all are dicts with ids
            ids = []
            for e in normalized:
                if isinstance(e, dict) and e.get("id"):
                    ids.append(e["id"])
            if ids:
                job["exit_rule_ids"] = ids
            else:
                job["exit_rules_inline"] = normalized

    # Preserve schema compatibility for older keys:
    # e.g., if user_data had 'incomplete_backtest_draft' keep it, but ensure
    # that any draft uses the new draft schema keys minimally
    draft = user_data.get("incomplete_backtest_draft")
    if isinstance(draft, dict):
        # ensure minimal keys exist to avoid builder crashes
        draft.setdefault("step", "source_chats")
        draft.setdefault("draft", {})
        d = draft["draft"]
        d.setdefault("source_chat_ids", [])
        d.setdefault("time_range", {"mode": "lookback", "lookback_days": 7})
        d.setdefault("position_sizing", {"mode": "percent_of_portfolio", "value": 5.0, "total_trading_amount": 100.0, "freeze_portfolio": False})
        d.setdefault("fees", {"fee_pct": 0.2, "slippage_pct": 0.0})
        d.setdefault("max_signals", 500)
        # reassign normalized draft
        user_data["incomplete_backtest_draft"] = draft

    return user_data
# ----- END REPLACEMENT -----

# ---------- helpers for user_backtests dir ----------
def _user_backtests_dir(user_id: int) -> str:
    DATA_DIR = _get_data_dir()
    path = os.path.join(DATA_DIR, USER_BACKTESTS_DIRNAME, str(user_id))
    _ensure_dirs(path)
    return path

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

# ---------- API functions requested by the plan ----------

def save_user_backtest_config(user_id: int, config_name: str, config_dict: Dict[str, Any]) -> str:
    """
    Save full config to DATA_DIR/user_backtests/<user_id>/<id>.json
    Also append compact metadata into user_{id}.dat.saved_backtests.
    Returns generated id (uuid4 as str).
    """
    uid = int(user_id)
    cfg_id = str(uuid.uuid4())
    backtests_dir = _user_backtests_dir(uid)
    cfg_path = os.path.join(backtests_dir, f"{cfg_id}.json")
    # write full config file (plaintext JSON)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # update user metadata
    user_obj = read_user_file(uid)
    user_obj = migrate_user_file_if_needed(user_obj)
    meta = {
        "id": cfg_id,
        "name": config_name or None,
        "created_at": _now_iso(),
        "config_path": cfg_path,
        "summary": {
            "min_market_cap": config_dict.get("market_filters", {}).get("min_market_cap", None),
            "created_by": "UI"
        }
    }
    user_obj.setdefault("saved_backtests", [])
    user_obj["saved_backtests"].append(meta)
    write_user_file(uid, user_obj)
    return cfg_id

def list_user_backtest_configs(user_id: int) -> List[Dict[str, Any]]:
    """
    Return list of saved_backtests metadata sorted by created_at desc.
    """
    uid = int(user_id)
    user_obj = read_user_file(uid)
    user_obj = migrate_user_file_if_needed(user_obj)
    items = list(user_obj.get("saved_backtests", []))
    # sort robustly (missing created_at placed last)
    def key_fn(x):
        return x.get("created_at") or ""
    items.sort(key=key_fn, reverse=True)
    return items

def load_user_backtest_config(user_id: int, config_id: str) -> Dict[str, Any]:
    """
    Load and return the full config dict saved under user_backtests/<user_id>/<id>.json
    If not found returns {}.
    """
    uid = int(user_id)
    backtests_dir = _user_backtests_dir(uid)
    cfg_path = os.path.join(backtests_dir, f"{config_id}.json")
    if not os.path.exists(cfg_path):
        # try to locate via user metadata as fallback
        user_obj = read_user_file(uid)
        for meta in user_obj.get("saved_backtests", []):
            if meta.get("id") == config_id and meta.get("config_path"):
                cfg_path = meta.get("config_path")
                break
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def delete_user_backtest_config(user_id: int, config_id: str) -> bool:
    """
    Delete the config file and remove metadata entry. Returns True if deleted/removed.
    """
    uid = int(user_id)
    success = False
    backtests_dir = _user_backtests_dir(uid)
    cfg_path = os.path.join(backtests_dir, f"{config_id}.json")
    if os.path.exists(cfg_path):
        try:
            os.remove(cfg_path)
            success = True
        except Exception:
            success = False

    # remove from metadata
    user_obj = read_user_file(uid)
    user_obj = migrate_user_file_if_needed(user_obj)
    before = len(user_obj.get("saved_backtests", []))
    user_obj["saved_backtests"] = [m for m in user_obj.get("saved_backtests", []) if m.get("id") != config_id]
    after = len(user_obj["saved_backtests"])
    if after != before:
        write_user_file(uid, user_obj)
        success = True
    return success

def append_backtest_history(user_id: int, run_metadata: Dict[str, Any], result_path: Optional[str] = None) -> str:
    """
    Appends a run metadata entry to user_{id}.dat.backtest_history.
    If result_path is None, writes the full result JSON into user_backtests/<user_id>/<run_id>.json and points result_path to that file.
    Returns run_id.
    """
    uid = int(user_id)
    run_id = str(uuid.uuid4())
    run_meta = {
        "id": run_id,
        "name": run_metadata.get("name", None),
        "timestamp": _now_iso(),
        "metadata": run_metadata.get("metadata", run_metadata),
        "result_path": None
    }

    if result_path:
        run_meta["result_path"] = result_path
    else:
        # write the full metadata (or provided result) to a file
        backtests_dir = _user_backtests_dir(uid)
        path = os.path.join(backtests_dir, f"{run_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(run_metadata, f, indent=2)
            run_meta["result_path"] = path
        except Exception:
            run_meta["result_path"] = None

    user_obj = read_user_file(uid)
    user_obj = migrate_user_file_if_needed(user_obj)
    user_obj.setdefault("backtest_history", [])
    user_obj["backtest_history"].append(run_meta)
    write_user_file(uid, user_obj)
    return run_id

def list_backtest_history(user_id: int, sort_by: str = 'timestamp', page: int = 1, per_page: int = 8) -> Dict[str, Any]:
    """
    Returns dict: { total: int, page: int, per_page: int, items: [...] }
    Items are the user_obj['backtest_history'] sorted by sort_by (supported 'timestamp').
    """
    uid = int(user_id)
    user_obj = read_user_file(uid)
    user_obj = migrate_user_file_if_needed(user_obj)
    items = list(user_obj.get("backtest_history", []))
    if sort_by == 'timestamp':
        items.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    total = len(items)
    # pagination bounds
    page = max(1, int(page or 1))
    per_page = max(1, int(per_page or 8))
    start = (page - 1) * per_page
    end = start + per_page
    slice_items = items[start:end]
    return {"total": total, "page": page, "per_page": per_page, "items": slice_items}

# ---------- forwarding job persistence ----------
def save_forwarding_job(user_id: int, job_dict: Dict[str, Any]) -> str:
    """
    Create or update a forwarding job in user_{id}.dat.forwarding_jobs.
    If job_dict contains 'id' then update; otherwise assign an id.
    Ensures job has last_triggered_by_asset map.
    Returns job_id.
    """
    uid = int(user_id)
    job = dict(job_dict)
    job_id = job.get("id") or str(uuid.uuid4())
    job["id"] = job_id
    # default values
    job.setdefault("dedup_window_seconds", int(job.get("dedup_window_seconds", 60)))
    job.setdefault("cooldown_seconds_per_asset", int(job.get("cooldown_seconds_per_asset", 86400)))
    job.setdefault("last_triggered_by_asset", job.get("last_triggered_by_asset") or {})
    user_obj = read_user_file(uid)
    user_obj = migrate_user_file_if_needed(user_obj)
    jobs = user_obj.get("forwarding_jobs", [])
    # replace if exists
    replaced = False
    for i, j in enumerate(jobs):
        if j.get("id") == job_id:
            jobs[i] = job
            replaced = True
            break
    if not replaced:
        jobs.append(job)
    user_obj["forwarding_jobs"] = jobs
    write_user_file(uid, user_obj)
    return job_id

def load_forwarding_job(user_id: int, job_id: str) -> Optional[Dict[str, Any]]:
    uid = int(user_id)
    user_obj = read_user_file(uid)
    user_obj = migrate_user_file_if_needed(user_obj)
    for j in user_obj.get("forwarding_jobs", []):
        if j.get("id") == job_id:
            # ensure structure
            j.setdefault("last_triggered_by_asset", {})
            return j
    return None

# ---------- generic utility ----------
def delete_forwarding_job(user_id: int, job_id: str) -> bool:
    uid = int(user_id)
    user_obj = read_user_file(uid)
    user_obj = migrate_user_file_if_needed(user_obj)
    before = len(user_obj.get("forwarding_jobs", []))
    user_obj["forwarding_jobs"] = [j for j in user_obj.get("forwarding_jobs", []) if j.get("id") != job_id]
    after = len(user_obj["forwarding_jobs"])
    if after != before:
        write_user_file(uid, user_obj)
        return True
    return False
