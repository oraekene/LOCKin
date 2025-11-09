# backtester.py
"""
Backtester utilities wired to ExitRuleEngine.
Supports:
 - Multiple/partial exits
 - ATR rules (via ohlc_series)
 - Atomic trade log persistence with CryptoManager
"""

import time, os, asyncio, re, tempfile, logging, json, importlib, importlib.util, uuid, math, statistics, inspect
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Coroutine, Tuple
from exit_rules import ExitRuleEngine, parse_rule_spec
# from backtester import collect_historical_signals
from dataclasses import dataclass, asdict
from types import ModuleType
from error_handler3 import ErrorHandler, BacktestError, MarketAPIError, DemoTradingError, PersistenceError

# ------------------------------------------------------------
# Demo trader concurrency / rate-limit controls
DEMO_MARKET_API_CONCURRENCY = 8  # number of concurrent MarketAPI price fetches during exit checks
# default interval for scheduled demo reports (seconds). Set to 3600 (1 hour) or smaller for testing. every hour seems too frequent though
REPORTING_INTERVAL_SECONDS_DEFAULT = 3600
# maximum number of concurrently active demo trades allowed per user (default)
DEMO_MAX_ACTIVE_TRADES_PER_USER = 5
# ------------------------------------------------------------

# Type aliases for clarity
SignalDict = Dict[str, Any]
ResolveResult = Dict[str, Any]

def _make_rules_from_specs(specs: List[Dict[str, Any]]):
    return [parse_rule_spec(s) for s in specs]

def _now_seconds():
    return int(time.time())

async def collect_historical_signals(
        user_id: int,
        config: Any,
        forwarder: Optional[Any] = None,
        data_dir: Optional[str] = None,
        find_cashtag_fn: Optional[Callable[[str], List[str]]] = None,
        find_eth_contract_fn: Optional[Callable[[str], Optional[str]]] = None,
        find_solana_contract_fn: Optional[Callable[[str], Optional[str]]] = None,
        max_signals: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, Optional[str]], Any]] = None
) -> List[Dict[str, Any]]:
    """
    Collect historical signals from a user's archived messages.

    Args:
      - user_id: user id whose archive to scan
      - config: object or dict-like with attributes/keys:
          - config_id (optional)
          - source_chat_ids (optional list)
          - keywords (optional list)
          - cashtags (optional list)
          - custom_patterns (optional list of regexes)
          - time_range: object with start_date and end_date (both datetimes) OR dict with same keys
          - match_logic: "AND" or "OR" (default OR)
          - contracts / detect_contracts (bool)
      - forwarder: optional object with .bot_instance.load_user_data(user_id) async method (preferred)
      - data_dir: fallback DATA_DIR path if forwarder not available (only used when files are plain JSON or NoopCrypto used)
      - find_*_fn: optional detector helpers; if omitted we try to import from main_bot and fallback to regex
      - max_signals: cap collected signals (memory safety)
      - progress_callback: optional callable or coroutine called as progress_callback(collected_count, scanned_count, last_identifier)

    Returns:
      - list of signal dicts with keys:
        {"type", "identifier", "detected_at" (datetime), "source_chat_id", "source_job_id", "original_message", "confidence_score"}
    """

    # Helper to normalize incoming config (accept dict or object)
    def _get_attr(obj, name, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    # Build job id for grouping
    job_id = _get_attr(config, "config_id", "backtest")

    # Try to import main_bot helpers if not provided (safe, non-fatal)
    if find_cashtag_fn is None or find_eth_contract_fn is None or find_solana_contract_fn is None:
        try:
            # Attempt to import from your main bot file. If names differ, caller should pass functions explicitly.
            from main_bot import _find_cashtag as _mb_find_cashtag, _find_ethereum_contract as _mb_find_eth, _find_solana_contract as _mb_find_sol
            if find_cashtag_fn is None:
                find_cashtag_fn = _mb_find_cashtag
            if find_eth_contract_fn is None:
                find_eth_contract_fn = _mb_find_eth
            if find_solana_contract_fn is None:
                find_solana_contract_fn = _mb_find_sol
        except Exception:
            # ignore; we'll rely on regex fallback below
            pass

    # Regex fallbacks (conservative)
    _cashtag_re = re.compile(r"\$([A-Za-z0-9_]{1,32})")
    _eth_re = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
    # Solana regex is tricky; use a conservative length and charset as fallback
    _sol_re = re.compile(r"\b[A-Za-z0-9]{32,44}\b")

    # 1) Load messages: prefer forwarder.bot_instance.load_user_data
    messages = []
    try:
        if forwarder and getattr(forwarder, "bot_instance", None):
            load_fn = getattr(forwarder.bot_instance, "load_user_data", None)
            if load_fn:
                user_data = await load_fn(user_id)
                # Prioritize keys commonly used in your app
                messages = user_data.get("historical_messages") or user_data.get("messages") or []
    except Exception as e:
        logging.debug(f"[collector] forwarder.load_user_data failed: {e}")

    # 2) fallback: read file from data_dir if messages empty
    if not messages:
        try:
            # fallback if provided by caller or environment
            data_dir = data_dir or os.environ.get("DATA_DIR")
            if data_dir:
                path = os.path.join(data_dir, f"user_{user_id}.dat")
                if os.path.exists(path):
                    raw = open(path, "rb").read()
                    # try to decode as utf-8 JSON (works if NoopCrypto used in dev)
                    try:
                        text = raw.decode("utf-8")
                        ud = json.loads(text)
                        messages = ud.get("historical_messages") or ud.get("messages") or []
                    except Exception:
                        # If encrypted, we don't attempt decryption here (decryption belongs to app context)
                        messages = []
        except Exception as e:
            logging.debug(f"[collector] fallback file read failed: {e}")

    if not messages:
        logging.info(f"[collector] no messages found for user {user_id}; returning []")
        return []

    # Transformer to produce the returned signal dict
    def _mk_signal(t, identifier, dt, chat_id, job_id_local, text, conf):
        return {
            "type": t,
            "identifier": identifier,
            "detected_at": dt,
            "source_chat_id": chat_id,
            "source_job_id": job_id_local,
            "original_message": text,
            "confidence_score": conf
        }

    # Normalize filters
    source_chats = set(_get_attr(config, "source_chat_ids", []) or []) or None
    keywords = [k.lower() for k in (_get_attr(config, "keywords", []) or [])]
    cashtag_whitelist = [c.lower() for c in (_get_attr(config, "cashtags", []) or [])]
    custom_patterns = _get_attr(config, "custom_patterns", []) or []
    contract_required = bool(_get_attr(config, "contracts", False))
    match_logic = (_get_attr(config, "match_logic", "OR") or "OR").upper()
    time_range = _get_attr(config, "time_range", None)
    # time_range expected to have start_date and end_date as datetimes (or dict keys)
    tr_start = None
    tr_end = None
    if time_range:
        if isinstance(time_range, dict):
            tr_start = time_range.get("start_date")
            tr_end = time_range.get("end_date")
        else:
            tr_start = getattr(time_range, "start_date", None)
            tr_end = getattr(time_range, "end_date", None)

    collected: List[Dict[str, Any]] = []
    seen = set()  # de-dup as (type, identifier, minute_bucket)
    scanned = 0

    for msg in messages:
        scanned += 1
        try:
            # Normalize date
            raw_date = msg.get("date") if isinstance(msg, dict) else getattr(msg, "date", None)
            if raw_date is None:
                continue

            if isinstance(raw_date, str):
                try:
                    msg_time = datetime.fromisoformat(raw_date)
                except Exception:
                    try:
                        msg_time = datetime.utcfromtimestamp(int(raw_date))
                    except Exception:
                        continue
            elif isinstance(raw_date, (int, float)):
                msg_time = datetime.utcfromtimestamp(int(raw_date))
            elif isinstance(raw_date, datetime):
                msg_time = raw_date
            else:
                continue

            # time range filter
            if tr_start and tr_end:
                if msg_time < tr_start or msg_time > tr_end:
                    continue

            # chat filter
            chat_id = msg.get("chat_id") if isinstance(msg, dict) else getattr(msg, "chat_id", None)
            if source_chats is not None and chat_id not in source_chats:
                continue

            text = (msg.get("text") if isinstance(msg, dict) else getattr(msg, "text", "")) or ""
            text_str = str(text)

            found_any = False

            # 1) custom patterns
            if custom_patterns:
                if match_logic == "AND":
                    if all(re.search(p, text_str, re.IGNORECASE) for p in custom_patterns):
                        ident = ";".join(custom_patterns)
                        sig = _mk_signal("custom_pattern", ident, msg_time, chat_id, job_id, text_str, 0.85)
                        collected.append(sig)
                        found_any = True
                else:
                    for p in custom_patterns:
                        if re.search(p, text_str, re.IGNORECASE):
                            sig = _mk_signal("custom_pattern", p, msg_time, chat_id, job_id, text_str, 0.85)
                            collected.append(sig)
                            found_any = True

            # 2) keywords
            if keywords:
                if match_logic == "AND":
                    if all(k in text_str.lower() for k in keywords):
                        identifier = ",".join(keywords)
                        key = ("keyword", identifier, msg_time.replace(second=0, microsecond=0))
                        if key not in seen:
                            collected.append(_mk_signal("keyword", identifier, msg_time, chat_id, job_id, text_str, 0.7))
                            seen.add(key)
                            found_any = True
                else:
                    for k in keywords:
                        if k in text_str.lower():
                            identifier = k
                            key = ("keyword", identifier, msg_time.replace(second=0, microsecond=0))
                            if key not in seen:
                                collected.append(_mk_signal("keyword", identifier, msg_time, chat_id, job_id, text_str, 0.7))
                                seen.add(key)
                                found_any = True

            # 3) cashtags
            tags = []
            if find_cashtag_fn:
                try:
                    tags = find_cashtag_fn(text_str) or []
                except Exception:
                    tags = []
            else:
                tags = [m.group(1) for m in _cashtag_re.finditer(text_str)]

            for tag in tags:
                if cashtag_whitelist and tag.lower() not in cashtag_whitelist:
                    continue
                key = ("cashtag", tag.lower(), msg_time.replace(second=0, microsecond=0))
                if key not in seen:
                    collected.append(_mk_signal("cashtag", tag, msg_time, chat_id, job_id, text_str, 0.9))
                    seen.add(key)
                    found_any = True

            # 4) contracts: prefer explicit helpers, else regex fallback
            contract = None
            if find_eth_contract_fn:
                try:
                    contract = find_eth_contract_fn(text_str)
                except Exception:
                    contract = None
            if not contract and find_solana_contract_fn:
                try:
                    contract = find_solana_contract_fn(text_str)
                except Exception:
                    contract = None
            if not contract:
                m = _eth_re.search(text_str)
                if m:
                    contract = m.group(0)
                else:
                    # conservative solana detector - may generate false positives; prefer helper
                    m2 = _sol_re.search(text_str)
                    if m2:
                        candidate = m2.group(0)
                        # ignore if obviously not a token (very short, all digits, etc)
                        if len(candidate) >= 32 and not candidate.isdigit():
                            contract = candidate

            if contract:
                key = ("contract", contract.lower(), msg_time.replace(second=0, microsecond=0))
                if key not in seen:
                    collected.append(_mk_signal("contract", contract, msg_time, chat_id, job_id, text_str, 0.95))
                    seen.add(key)
                    found_any = True

            # progress callback: allow coroutine or sync callable
            if progress_callback:
                try:
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(len(collected), scanned, (collected[-1]["identifier"] if collected else None))
                    else:
                        # allow sync callable (don't block await)
                        progress_callback(len(collected), scanned, (collected[-1]["identifier"] if collected else None))
                except Exception:
                    # don't fail the scan for progress callback errors
                    logging.debug("[collector] progress_callback error", exc_info=True)

            # cap memory if requested
            if max_signals and len(collected) >= max_signals:
                logging.info(f"[collector] reached max_signals={max_signals}; stopping early")
                break

        except Exception:
            logging.debug("[collector] message processing error", exc_info=True)
            continue

    logging.info(f"[collector] collected={len(collected)} scanned={scanned} for user={user_id}")
    return collected


def collect_historical_signals_sync(
        user_id: int,
        config: Any,
        forwarder: Optional[Any] = None,
        data_dir: Optional[str] = None,
        find_cashtag_fn: Optional[Callable[[str], List[str]]] = None,
        find_eth_contract_fn: Optional[Callable[[str], Optional[str]]] = None,
        find_solana_contract_fn: Optional[Callable[[str], Optional[str]]] = None,
        max_signals: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, Optional[str]], Any]] = None
) -> List[Dict[str, Any]]:
    """
    Sync wrapper for CLI/tests. Runs the async collector and returns the list.
    """
    return asyncio.run(collect_historical_signals(
        user_id=user_id,
        config=config,
        forwarder=forwarder,
        data_dir=data_dir,
        find_cashtag_fn=find_cashtag_fn,
        find_eth_contract_fn=find_eth_contract_fn,
        find_solana_contract_fn=find_solana_contract_fn,
        max_signals=max_signals,
        progress_callback=progress_callback
    ))
# ---------- END: collect_historical_signals ----------


# ---------------------------------------------------------------------
# Core simulation: multiple partial exits
# ---------------------------------------------------------------------
def simulate_trade_with_partial_exits(
    entry_price: float,
    price_series: List[float],
    entry_time: Optional[float] = None,
    exit_rule_specs: Optional[List[Dict[str, Any]]] = None,
    tick_interval_seconds: int = 60,
    ohlc_series: Optional[List[Dict[str, float]]] = None,
    initial_position_size: float = 1.0
) -> Dict[str, Any]:
    """
    Simulate a single trade with multiple/partial exits.
    """
    entry_time = float(entry_time or _now_seconds())
    specs = exit_rule_specs or []
    rules = _make_rules_from_specs(specs)

    engine = ExitRuleEngine(entry_price=entry_price, entry_time=entry_time)
    engine.set_rules(rules)
    if ohlc_series:
        engine.set_ohlc_series(ohlc_series)

    ts = entry_time
    ticks_processed = 0
    exit_events = []
    remaining_size = float(initial_position_size)

    for p in price_series:
        ts += tick_interval_seconds
        ticks_processed += 1
        engine.on_price_tick(price=p, ts=ts)

        while True:
            dec = engine.evaluate()
            if not dec:
                break

            exit_size_pct = getattr(dec, "exit_size_pct", None)
            if exit_size_pct is None:
                exit_size_pct = 100.0  # assume full close if not specified

            remove_fraction = min(remaining_size, exit_size_pct / 100.0)
            removed_size = remove_fraction

            exit_events.append({
                "exit_price": dec.exit_price,
                "exit_time": dec.exit_time,
                "reason": dec.reason,
                "profit_pct": dec.profit_pct,
                "info": dec.info,
                "exit_size_pct": exit_size_pct,
                "removed_size": removed_size,
            })

            remaining_size = max(0.0, remaining_size - removed_size)

            if remaining_size <= 0.0 or engine.closed:
                break

        if remaining_size <= 0.0:
            break

    return {
        "entry_price": entry_price,
        "entry_time": entry_time,
        "exit_events": exit_events,
        "ticks_processed": ticks_processed,
        "last_price": engine.last_price,
        "remaining_size": remaining_size,
        "runtime_info": {
            "peak_price": engine.peak_price,
            "trough_price": engine.trough_price
        }
    }

# ---------------------------------------------------------------------
# Wrapper for single-exit mode
# ---------------------------------------------------------------------
def simulate_trade_with_exit(
    entry_price: float,
    price_series: List[float],
    entry_time: Optional[float] = None,
    exit_rule_specs: Optional[List[Dict[str, Any]]] = None,
    tick_interval_seconds: int = 60,
    ohlc_series: Optional[List[Dict[str, float]]] = None
) -> Dict[str, Any]:
    res = simulate_trade_with_partial_exits(entry_price, price_series, entry_time, exit_rule_specs, tick_interval_seconds, ohlc_series, initial_position_size=1.0)
    first = res["exit_events"][0] if res["exit_events"] else None
    return {
        "entry_price": res["entry_price"],
        "entry_time": res["entry_time"],
        "exit_decision": first,
        "ticks_processed": res["ticks_processed"],
        "last_price": res["last_price"],
        "runtime_info": res["runtime_info"]
    }

# ---------------------------------------------------------------------
# Trade log persistence
# ---------------------------------------------------------------------
def persist_trade_log(user_id: int,
                      trade_log: Dict[str, Any],
                      data_dir: Optional[str] = None,
                      crypto_manager: Optional[Any] = None,
                      repo_root: Optional[str] = None) -> None:
    """
    Persist trade_log to user's encrypted data file atomically.

    Dependency-injected version:
      - data_dir: path to directory where user files live (was DATA_DIR)
      - crypto_manager: object providing .encrypt(plaintext: str) -> bytes and
                        .decrypt(ciphertext: bytes) -> str methods OR
                        a class/module with those methods as statics.

    Backwards-compatible behavior:
      - If crypto_manager or data_dir is None, the function will attempt to
        dynamically load main_bot(fixed).py from repo_root (original behavior).
        However, prefer injecting these dependencies in production.

    Notes:
      - This function writes atomically using tempfile + os.replace.
      - Raises RuntimeError/FileNotFoundError if fallback dynamic import fails.
    """
    # Prefer injected dependencies
    if data_dir is not None and crypto_manager is not None:
        DATA_DIR = data_dir
        CryptoManager = crypto_manager
    else:
        # Fallback (backwards compatible) -- try to dynamically import as before
        repo_root = repo_root or os.getcwd()
        main_bot_path = os.path.join(repo_root, "main_bot(fixed).py")

        if not os.path.exists(main_bot_path):
            raise FileNotFoundError(f"main_bot(fixed).py not found in {repo_root}")

        spec = importlib.util.spec_from_file_location("main_bot_fixed", main_bot_path)
        main_bot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_bot)

        DATA_DIR = getattr(main_bot, "DATA_DIR", os.path.join(repo_root, "data"))
        CryptoManager = getattr(main_bot, "CryptoManager", None)
        if CryptoManager is None:
            raise RuntimeError("CryptoManager not found in main_bot(fixed).py")

    # Ensure data dir exists
    os.makedirs(DATA_DIR, exist_ok=True)
    user_file = os.path.join(DATA_DIR, f"user_{user_id}.dat")

    # Read existing data (if any)
    data = {}
    if os.path.exists(user_file):
        try:
            with open(user_file, "rb") as f:
                enc = f.read()
            # crypto_manager might be a class with staticmethods, or an instance
            if hasattr(CryptoManager, "decrypt"):
                text = CryptoManager.decrypt(enc)
            else:
                # attempt call as function-like fallback
                text = CryptoManager(enc)
            data = json.loads(text)
        except Exception:
            # If decryption/parsing fails, continue with empty data (we still append new logs)
            data = {}

    # Append trade log list
    data.setdefault("trade_logs", [])
    data["trade_logs"].append(trade_log)

    plaintext = json.dumps(data, indent=2)

    # Use encrypt method provided; keep compatible with class/static or instance
    if hasattr(CryptoManager, "encrypt"):
        ciphertext = CryptoManager.encrypt(plaintext)
    else:
        # fallback attempt (rare)
        ciphertext = CryptoManager(plaintext)

    # Atomic write using tempfile + os.replace in target DATA_DIR
    fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, prefix=f"user_{user_id}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmpf:
            # ensure bytes
            if isinstance(ciphertext, str):
                tmpf.write(ciphertext.encode("utf-8"))
            else:
                tmpf.write(ciphertext)
        os.replace(tmp_path, user_file)
    finally:
        # Cleanup leftover temp file in case of failure
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

class SimpleTTLCache:
    """Tiny in-memory TTL cache for (key -> (value, expiry_ts)). Not thread-safe; OK for single-process bot."""
    def __init__(self):
        self._store: Dict[Any, Tuple[Any, float]] = {}

    def get(self, key):
        v = self._store.get(key)
        if not v:
            return None
        value, expiry = v
        if time.time() > expiry:
            del self._store[key]
            return None
        return value

    def set(self, key, value, ttl: int):
        self._store[key] = (value, time.time() + ttl)

    def clear(self):
        self._store.clear()

# ---------- START: MarketAPIAdapter-aware resolver (solana + evm detection) ----------

def _detect_chain_for_identifier(identifier: str) -> Optional[str]:
    """
    Return 'evm' | 'solana' | None (meaning symbol) based on identifier shape.
    - evm: startswith '0x' and 40+ hex chars (common ETH/ERC20).
    - solana: long alphanumeric (32-44) and not all digits.
    - None: treat as symbol/cashtag.
    """
    if not identifier:
        return None
    s = identifier.strip()
    # EVM / 0x hex contract
    if s.startswith("0x") and len(s) >= 42:
        try:
            int(s[2:42], 16)
            return "evm"
        except Exception:
            pass
    # Solana-style (base58-ish) — conservative detection: length and charset
    if 32 <= len(s) <= 44 and re.fullmatch(r"[A-Za-z0-9]+", s) and not s.isdigit():
        # If it looks like an EVM hex without 0x, still prefer solana only if not hex-ish
        # Prefer solana for these long base58-like ids
        return "solana"
    return None


async def _default_resolve_price_with_market_api(
        market_api: Any,
        identifier: str,
        at_timestamp: Optional[int] = None,
        timeout: int = 8
) -> Tuple[Optional[float], Optional[str]]:
    """
    Adapter-aware price resolver for your MarketAPIAdapter.

    - Detects whether identifier is EVM contract (0x...), Solana address, or a symbol/cashtag.
    - Calls market_api.get_price(chain=..., symbol=..., contract=..., vs_currency='usd')
      and returns (price_or_None, provider_or_error_str).
    - Uses asyncio.wait_for to bound latency.
    """
    if market_api is None:
        return None, "no_market_api"

    ident = (identifier or "").strip()
    if not ident:
        return None, "empty_identifier"

    # Determine chain hint
    detected_chain = _detect_chain_for_identifier(ident)

    async def _call(chain_hint: Optional[str], symbol: Optional[str], contract: Optional[str]):
        try:
            coro = market_api.get_price(chain=chain_hint, symbol=symbol, contract=contract, vs_currency="usd")
            val = await asyncio.wait_for(coro, timeout=timeout)
            if val is None:
                return None, f"{getattr(market_api, '__class__', type(market_api)).__name__}:no_data"
            return float(val), getattr(market_api, "__class__", type(market_api)).__name__
        except asyncio.TimeoutError:
            return None, f"{getattr(market_api, '__class__', type(market_api)).__name__}:timeout"
        except Exception as e:
            return None, f"{getattr(market_api, '__class__', type(market_api)).__name__}:exception:{type(e).__name__}"

    # If we think this is a contract-like id, call with the detected chain first.
    if detected_chain == "evm":
        price, provider = await _call(chain_hint="evm", symbol=None, contract=ident)
        if price is not None:
            return price, provider
        # fallback to generic
        price, provider = await _call(chain_hint="any", symbol=None, contract=ident)
        return price, provider

    if detected_chain == "solana":
        price, provider = await _call(chain_hint="solana", symbol=None, contract=ident)
        if price is not None:
            return price, provider
        # fallback to any/other providers
        price, provider = await _call(chain_hint="any", symbol=None, contract=ident)
        return price, provider

    # Otherwise treat as symbol/cashtag: strip leading $ if present
    if ident.startswith("$"):
        ident = ident[1:]

    # Try symbol lookup across providers with 'any' chain preference
    price, provider = await _call(chain_hint="any", symbol=ident, contract=None)
    if price is not None:
        return price, provider

    # Try explicit evm symbol and solana symbol as fallbacks (some providers prefer chain hint)
    price, provider = await _call(chain_hint="evm", symbol=ident, contract=None)
    if price is not None:
        return price, provider
    price, provider = await _call(chain_hint="solana", symbol=ident, contract=None)
    if price is not None:
        return price, provider

    # Last resort: treat identifier as a contract on 'any'
    price, provider = await _call(chain_hint="any", symbol=None, contract=identifier)
    return price, provider
# ---------- END: MarketAPIAdapter-aware resolver ----------

async def validate_and_resolve_signals(
        signals: List[SignalDict],
        resolve_price_fn: Optional[Callable[[str, Optional[int]], Coroutine[Any, Any, Tuple[Optional[float], Optional[str]]]]] = None,
        market_api: Optional[Any] = None,
        cache: Optional[SimpleTTLCache] = None,
        cache_ttl: int = 300,
        concurrency: int = 6,
        retries: int = 2,
        retry_backoff_base: float = 0.5,
        request_timeout: int = 6
) -> List[ResolveResult]:
    """
    Validate and resolve a list of raw signals into price-resolved signals for backtesting.
    - signals: list of signal dicts (output of collect_historical_signals)
    - resolve_price_fn: optional async callable(symbol, at_ts) -> (price_or_none, provider_or_error)
                        If provided, used directly. If not, `market_api` is required.
    - market_api: fallback provider object (must have get_price or get_bulk_prices)
    - cache: optional SimpleTTLCache instance
    - cache_ttl: seconds to cache price responses
    - concurrency: max concurrent resolve tasks
    - retries: number of retries on transient failures (total attempts = 1 + retries)
    - retry_backoff_base: base seconds for exponential backoff
    - request_timeout: timeout seconds for each resolve attempt

    Returns list of dicts: original signal fields plus:
      - resolved: bool
      - resolved_price: float | None
      - resolved_at_ts: int | None (unix secs)
      - resolved_provider: str | None
      - resolve_error: str | None (if not resolved)
      - attempts: int
    """
    if not signals:
        return []

    if cache is None:
        cache = SimpleTTLCache()

    # If resolve_price_fn not provided, build from market_api
    if resolve_price_fn is None:
        if market_api is None:
            raise ValueError("Either resolve_price_fn or market_api must be provided")
        async def _resolver(sym: str, at_ts: Optional[int] = None):
            return await _default_resolve_price_with_market_api(market_api, sym, at_timestamp=at_ts, timeout=request_timeout)
        resolve_price_fn = _resolver

    sem = asyncio.Semaphore(concurrency)
    results: List[ResolveResult] = []

    async def _resolve_single(sig: SignalDict) -> ResolveResult:
        # Build canonical key for caching. Use identifier (e.g., cashtag or contract).
        identifier = sig.get("identifier")
        if identifier is None:
            return {**sig, "resolved": False, "resolved_price": None, "resolved_at_ts": None,
                    "resolved_provider": None, "resolve_error": "no_identifier", "attempts": 0}

        # try to determine timestamp to query price at (prefer detected_at)
        detected_at = sig.get("detected_at")
        at_ts = None
        if isinstance(detected_at, (int, float)):
            at_ts = int(detected_at)
        else:
            try:
                # assume datetime-like
                at_ts = int(detected_at.timestamp())
            except Exception:
                at_ts = None

        cache_key = (identifier, at_ts)
        cached = cache.get(cache_key)
        if cached is not None:
            price, provider = cached
            return {**sig, "resolved": True, "resolved_price": price, "resolved_at_ts": int(time.time()),
                    "resolved_provider": provider, "resolve_error": None, "attempts": 0}

        attempts = 0
        last_error = None
        price = None
        provider = None

        # Acquire semaphore to limit concurrency
        async with sem:
            for attempt in range(1, retries + 2):  # 1 initial + retries
                attempts = attempt
                try:
                    p, prov = await resolve_price_fn(identifier, at_ts)
                    # p can be None (no data) or float
                    if p is not None:
                        price = float(p)
                        provider = prov
                        # cache it
                        try:
                            cache.set(cache_key, (price, provider), cache_ttl)
                        except Exception:
                            pass
                        last_error = None
                        break
                    else:
                        last_error = prov or "no_data"
                except Exception as e:
                    last_error = f"exception:{type(e).__name__}"
                # backoff before retrying
                if attempt <= retries + 1:
                    delay = retry_backoff_base * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

        resolved = price is not None
        return {**sig,
                "resolved": resolved,
                "resolved_price": price,
                "resolved_at_ts": int(time.time()) if resolved else None,
                "resolved_provider": provider,
                "resolve_error": None if resolved else last_error,
                "attempts": attempts
                }

    # Kick off tasks concurrently but preserve input order in results
    tasks = [asyncio.create_task(_resolve_single(s)) for s in signals]
    # gather preserves order
    resolved_list = await asyncio.gather(*tasks, return_exceptions=False)
    return resolved_list

def validate_and_resolve_signals_sync(
        signals: List[SignalDict],
        resolve_price_fn: Optional[Callable[[str, Optional[int]], Coroutine[Any, Any, Tuple[Optional[float], Optional[str]]]]] = None,
        market_api: Optional[Any] = None,
        cache: Optional[SimpleTTLCache] = None,
        cache_ttl: int = 300,
        concurrency: int = 6,
        retries: int = 2,
        retry_backoff_base: float = 0.5,
        request_timeout: int = 6
) -> List[ResolveResult]:
    """Sync wrapper for tests/CLI."""
    return asyncio.run(validate_and_resolve_signals(
        signals=signals,
        resolve_price_fn=resolve_price_fn,
        market_api=market_api,
        cache=cache,
        cache_ttl=cache_ttl,
        concurrency=concurrency,
        retries=retries,
        retry_backoff_base=retry_backoff_base,
        request_timeout=request_timeout
    ))
# ---------- END: validate_and_resolve_signals (T6.3) ----------

# ---------- START: fetch_price_series_for_signals ----------
async def fetch_price_series_for_signals(resolved_signals: List[ResolveResult],
                                         market_api: Any,
                                         lookahead_seconds: int = 86400,
                                         history_interval: str = "60",
                                         concurrency: int = 4
                                         ) -> List[Dict[str, Any]]:
    """
    For each resolved signal that has resolved=True and resolved_at_ts, fetch a forward-looking price series.
    - lookahead_seconds: how many seconds of history after the signal timestamp to fetch (default 1 day)
    - history_interval: resolution for history provider (if supported) - provider-specific
    Returns a new list of signals where each signal dict includes "price_series": List[float] or None.
    """

    # simple concurrency limit
    sem = asyncio.Semaphore(concurrency)

    async def _fetch_for(sig):
        if not sig.get("resolved"):
            sig["price_series"] = None
            return sig
        at_ts = sig.get("resolved_at_ts") or None
        # fallback to detected_at
        if not at_ts:
            da = sig.get("detected_at")
            try:
                at_ts = int(da.timestamp())
            except Exception:
                at_ts = None
        if not at_ts:
            sig["price_series"] = None
            return sig

        start_ts = int(at_ts)
        end_ts = int(at_ts + lookahead_seconds)

        async with sem:
            try:
                # market_api should implement get_price_history(chain, symbol, contract, start_ts, end_ts, interval)
                # prefer contract or symbol lookup
                contract = None
                symbol = None
                idf = sig.get("identifier")
                # detect chain hint like earlier (copying detection)
                chain_hint = None
                if isinstance(idf, str) and idf.startswith("0x"):
                    chain_hint = "evm"
                    contract = idf
                elif isinstance(idf, str) and (len(idf) >= 32 and idf.isalnum()):
                    chain_hint = "solana"
                    contract = idf
                else:
                    symbol = idf.lstrip("$") if isinstance(idf, str) else None

                data = await market_api.get_price_history(chain=chain_hint or "any", symbol=symbol, contract=contract, start_ts=start_ts, end_ts=end_ts, interval=history_interval)
                if data:
                    # data expected as list[ (ts, price) ] — we convert to list of prices for simulator
                    try:
                        price_series = [float(p[1]) for p in data]
                        sig["price_series"] = price_series
                        sig["price_series_ts"] = [int(p[0]) for p in data]
                    except Exception:
                        # accept provider-specific formats
                        sig["price_series"] = [float(v) for v in (data or []) if isinstance(v, (int, float))]
                else:
                    sig["price_series"] = None
            except Exception:
                sig["price_series"] = None
        return sig

    # schedule all fetches
    tasks = [asyncio.create_task(_fetch_for(s)) for s in resolved_signals]
    out = await asyncio.gather(*tasks)
    return out
# ---------- END: fetch_price_series_for_signals ----------

# ---------- START: run_backtest_for_user (orchestrator) ----------
async def _attempt_call_existing_simulator(resolved_signals: List[Dict[str, Any]], config: Any) -> Any:
    """
    Try to find an existing simulation/backtest function inside this module or imported modules.
    Looks for common names and calls the first matching one with (resolved_signals, config).
    If none exists, falls back to a minimal internal simulator (simple P&L per-signal).

    This merged version preserves the original behavior (series lookup via resolved_with_series,
    _simulate_with_rules, and original result fields), and keeps the dedup & cooldown improvements.
    """
    # common candidate function names your repo might already have
    candidates = [
        "simulate_positions_from_resolved_signals",
        "run_simulation",
        "run_backtest_simulation",
        "simulate_backtest",
        "simulate_trades"
    ]

    # 1) Look in current module globals
    g = globals()
    for name in candidates:
        fn = g.get(name)
        if fn and callable(fn):
            logging.info(f"[backtester] using existing simulator function: {name}")
            # support async or sync functions
            try:
                if inspect.iscoroutinefunction(fn):
                    return await fn(resolved_signals, config)
                else:
                    # run sync function in executor to avoid blocking
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(None, lambda: fn(resolved_signals, config))
            except Exception:
                # if a candidate fails, continue to next candidate
                pass

    # 2) Try to import common backtest runner modules (best-effort, non-fatal)
    try:
        import backtest_runner  # common name
        for name in candidates:
            fn = getattr(backtest_runner, name, None)
            if fn and callable(fn):
                logging.info(f"[backtester] using external backtest_runner.{name}")
                if inspect.iscoroutinefunction(fn):
                    return await fn(resolved_signals, config)
                else:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(None, lambda: fn(resolved_signals, config))
    except Exception:
        pass

    # 3) Fallback minimal simulation: use parsed exit rules (if present) and price series when available
    logging.info("[backtester] no existing simulator found; using fallback minimal simulator (rule-aware)")

    # Helper: read configured parsed rules (UI parser output) or convert ExitRule objects if present
    parsed_rules = None
    if isinstance(config, dict):
        parsed_rules = config.get("exit_rules") or config.get("exit_rules_raw") or None
        # if strict converted ExitRule objects exist, try to extract simple actionable specs
        if not parsed_rules and "exit_rule_objs" in config and config.get("exit_rule_objs"):
            # Best-effort: map object attributes to simple spec dicts
            prs = []
            for obj in config.get("exit_rule_objs", []):
                try:
                    # object may be dataclass-like with attributes type/value/unit/multiplier/size_pct
                    typ = getattr(obj, "type", None) or getattr(obj, "rule_type", None)
                    if not typ:
                        continue
                    if typ in ("take_profit", "tp"):
                        val = getattr(obj, "value", None) or getattr(obj, "threshold", None)
                        unit = getattr(obj, "unit", None) or ("multiple" if getattr(obj, "multiple_mode", False) else "percent")
                        prs.append({"type": "take_profit", "value": float(val or 0), "unit": unit})
                    elif typ in ("stop_loss", "sl"):
                        val = getattr(obj, "value", None) or getattr(obj, "loss_pct", None)
                        prs.append({"type": "stop_loss", "value": float(val or 0)})
                    elif typ in ("percent_of_portfolio", "partial"):
                        size = getattr(obj, "size_pct", None) or getattr(obj, "percent", None)
                        val = getattr(obj, "value", None) or getattr(obj, "exit_threshold_pct", None)
                        prs.append({"type": "partial", "percent": float(size or 0), "exit_multiple": float((val/100.0)+1.0) if val is not None else None})
                    else:
                        # generic fallback
                        prs.append({"type": "raw", "text": str(obj)})
                except Exception:
                    continue
            if prs:
                parsed_rules = prs

    # Helper: get the per-signal price series (best-effort): resolved_with_series may be present in outer scope
    # Try to locate series by a common identifier key
    def _get_series_for_signal(signal):
        # look inside resolved_with_series variable if present
        try:
            for item in (resolved_with_series or []):
                # signals may share an 'identifier' or symbol key
                if item.get("identifier") and signal.get("identifier") and item.get("identifier") == signal.get("identifier"):
                    return item.get("price_series") or item.get("series") or item.get("prices")
                # also try matching by symbol/name if available
                if item.get("symbol") and signal.get("symbol") and item.get("symbol") == signal.get("symbol"):
                    return item.get("price_series") or item.get("series") or item.get("prices")
        except Exception:
            pass
        # fallback: maybe the signal itself contains a 'series' or 'price_series' key
        return signal.get("price_series") or signal.get("series") or signal.get("prices")

    # Utility: evaluate a single signal against simple rules using its series
    def _simulate_with_rules(entry_price, series, rules):
        """
        series: iterable of price numbers (in chronological order after entry)
        rules: list of parsed rule dicts (type,value,unit,percent,multiplier,exit_multiple)
        returns: (exit_price, exit_index, exit_reason)
        """
        if not series:
            # no ticks -> assume immediate +10% exit as fallback
            try:
                return (float(entry_price) * 1.10, 0, "fallback_fixed")
            except Exception:
                return (entry_price, 0, "fallback_fixed")

        # normalize series to list of floats
        prices = []
        for p in series:
            try:
                prices.append(float(p))
            except Exception:
                continue
        if not prices:
            try:
                return (float(entry_price) * 1.10, 0, "fallback_fixed")
            except Exception:
                return (entry_price, 0, "fallback_fixed")

        for idx, price in enumerate(prices, start=1):
            # compute metrics relative to entry
            try:
                multiple = price / entry_price if entry_price != 0 else float("inf")
            except Exception:
                multiple = float("inf")
            try:
                profit_pct = (price - entry_price) / entry_price * 100.0 if entry_price != 0 else float("inf")
            except Exception:
                profit_pct = float("inf")

            # check stop loss first (close early on loss)
            for r in rules or []:
                if r.get("type") in ("stop_loss", "sl"):
                    try:
                        sl_pct = float(r.get("value", 0))
                        if profit_pct <= -abs(sl_pct):
                            return (price, idx, f"stop_loss_{sl_pct}%")
                    except Exception:
                        continue

            # check take profit / partials
            for r in rules or []:
                if r.get("type") == "take_profit":
                    unit = r.get("unit", "multiple")
                    try:
                        if unit == "multiple":
                            target_mult = float(r.get("value", 1.0))
                            if multiple >= target_mult:
                                return (price, idx, f"take_profit_{target_mult}x")
                        else:
                            target_pct = float(r.get("value", 0.0))
                            if profit_pct >= target_pct:
                                return (price, idx, f"take_profit_{target_pct}%")
                    except Exception:
                        continue
                if r.get("type") == "partial":
                    # treat partial as close of 'percent' at 'exit_multiple' threshold if present
                    try:
                        pct = float(r.get("percent", 0))
                        exit_mult = r.get("exit_multiple")
                        if exit_mult is not None:
                            exit_mult = float(exit_mult)
                            if multiple >= exit_mult:
                                # return partial close (we return the price and reason; size handling recorded separately)
                                return (price, idx, f"partial_{pct}%@{exit_mult}x")
                    except Exception:
                        continue
            # continue scanning ticks until a rule satisfied

        # no rule triggered: default final exit at last price
        final_price = prices[-1]
        return (final_price, len(prices), "no_rule_triggered")

    # Run simulation across signals
    results = {"positions": [], "summary": {"total_signals": len(resolved_signals), "closed": 0, "pnl": 0.0}}

    # Minimal internal simulator: iterate resolved signals, find entry, run through price series, apply rules
    for s in resolved_signals:
        if not s.get("resolved"):
            continue

        # ---------- dedup & cooldown (minimal run-local implementation) ----------
        try:
            now_ts = int(time.time())
            # normalized identifier (lowercase)
            identifier_lower = str(s.get("resolved_symbol") or s.get("identifier") or s.get("symbol") or s.get("contract") or "").lower()
            job_id_local = s.get("source_job_id") or "backtest"
            # dedup window (seconds) applied across the whole run
            dedup_window = int((config.get("dedup_window_seconds") if isinstance(config, dict) else None) or 0)
            # cooldown per asset (seconds) applied per job_id + identifier
            cooldown_seconds = int((config.get("cooldown_seconds_per_asset") if isinstance(config, dict) else None) or 0)
        except Exception:
            dedup_window = 0
            cooldown_seconds = 0
            now_ts = int(time.time())
            identifier_lower = str(s.get("identifier") or s.get("symbol") or "").lower()
            job_id_local = s.get("source_job_id") or "backtest"

        # initialize run-local maps if not present (attached to the function object)
        if not hasattr(_attempt_call_existing_simulator, '_seen_identifiers'):
            setattr(_attempt_call_existing_simulator, '_seen_identifiers', {})
        if not hasattr(_attempt_call_existing_simulator, '_last_triggered'):
            setattr(_attempt_call_existing_simulator, '_last_triggered', {})
        seen_identifiers = getattr(_attempt_call_existing_simulator, '_seen_identifiers')
        last_triggered = getattr(_attempt_call_existing_simulator, '_last_triggered')

        # dedup check
        if dedup_window:
            last = seen_identifiers.get(identifier_lower)
            if last and (now_ts - int(last)) < dedup_window:
                # skip this signal as duplicate within dedup window
                continue
            seen_identifiers[identifier_lower] = now_ts

        # cooldown per asset check
        if cooldown_seconds:
            key = f"{job_id_local}:{identifier_lower}"
            last = last_triggered.get(key)
            if last and (now_ts - int(last)) < cooldown_seconds:
                # skip due to cooldown
                continue
            last_triggered[key] = now_ts
        # ---------- end dedup & cooldown ----------

        entry = s.get("resolved_price") or s.get("entry_price")
        if entry is None:
            # can't simulate without an entry price
            continue

        # try to find a tick series
        series = _get_series_for_signal(s)
        exit_price, exit_idx, reason = _simulate_with_rules(entry, series, parsed_rules)

        try:
            pnl = (exit_price - entry) / entry if entry != 0 else 0.0
        except Exception:
            pnl = 0.0

        # preserve original result keys and metadata so callers remain compatible
        results["positions"].append({
            "identifier": s.get("identifier") or s.get("id") or s.get("symbol"),
            "symbol": s.get("symbol"),
            "entry_price": float(entry),
            "exit_price": float(exit_price),
            "pnl": pnl,
            "exit_index": int(exit_idx),
            "exit_reason": reason,
            "resolved_provider": s.get("resolved_provider"),
            "detected_at": s.get("detected_at")
        })
        results["summary"]["closed"] += 1
        results["summary"]["pnl"] += pnl

    return results



async def run_backtest_for_user(user_id: int,
                                config: Any,
                                forwarder: Optional[Any] = None,
                                market_api: Optional[Any] = None,
                                max_signals: Optional[int] = 1000,
                                progress_callback: Optional[Any] = None
                                ) -> Dict[str, Any]:
    """
    End-to-end backtest runner for a user.
    Steps:
      1) collect historical signals
      2) resolve prices using market_api (MarketAPIAdapter) via validate_and_resolve_signals
      3) call existing simulation function if present, else run fallback minimal simulation
    Returns a dict with raw_count, resolved_count, results
    """
    logging.info(f"[backtester] run_backtest_for_user user_id={user_id} config={getattr(config,'config_id', None)}")

    # 1) collect signals
    raw_signals = await collect_historical_signals(
        user_id=user_id,
        config=config,
        forwarder=forwarder,
        data_dir=None,
        find_cashtag_fn=None,
        find_eth_contract_fn=None,
        find_solana_contract_fn=None,
        max_signals=max_signals,
        progress_callback=progress_callback
    )

    # 2) resolve prices
    if market_api is None:
        # try to get from forwarder.bot_instance.market_adapter
        try:
            market_api = getattr(forwarder.bot_instance, "market_adapter", None) if forwarder and getattr(forwarder, "bot_instance", None) else None
        except Exception:
            market_api = None

    if market_api is None:
        logging.warning("[backtester] market_api not provided and not available via forwarder; returning raw signals only")
        return {"error": "no_market_api", "raw_signals_count": len(raw_signals), "raw_signals": raw_signals}

    # Use a fresh per-run cache (or you can pass a shared cache)
    cache = SimpleTTLCache()
    resolved_signals = await validate_and_resolve_signals(
        signals=raw_signals,
        market_api=market_api,
        cache=cache,
        cache_ttl=300,
        concurrency=8,
        retries=2,
        request_timeout=8
    )
    
    resolved_with_series = await fetch_price_series_for_signals(
        resolved_signals, 
        market_api=market_api, 
        lookahead_seconds=86400, 
        concurrency=6
        )
    # then pass resolved_with_series to simulator or to _attempt_call_existing_simulator
    
    # -------------------------
    # STRICT conversion: convert parsed UI rules -> ExitRule objects
    # (inserted here, after resolved_with_series and before running simulator)
    # -------------------------
    try:
        # Try to import the strict converter from the same module if present,
        # otherwise fall back to backtester2 (or backtester_mod if you use that name).
        try:
            # Prefer a local function if you added adapter to this same module
            convert_strict = convert_parsed_rules_to_exitrules_strict  # noqa: F821
            backtester_mod_ref = None
        except NameError:
            convert_strict = None
            backtester_mod_ref = None

        if convert_strict is None:
            # try importing from this module (backtester) first
            try:
                from backtester import convert_parsed_rules_to_exitrules_strict as convert_strict  # type: ignore
                backtester_mod_ref = __import__("backtester")
            except Exception:
                # fallback to backtester2 if present
                try:
                    from backtester2 import convert_parsed_rules_to_exitrules_strict as convert_strict  # type: ignore
                    backtester_mod_ref = __import__("backtester2")
                except Exception:
                    convert_strict = None
                    backtester_mod_ref = None

        # Only attempt strict conversion if parsed rules exist in config
        parsed_rules = None
        if config and isinstance(config, dict):
            parsed_rules = config.get("exit_rules") or config.get("exit_rules_raw") or None
        elif hasattr(config, "get"):
            parsed_rules = config.get("exit_rules") or config.get("exit_rules_raw") or None

        if parsed_rules and convert_strict:
            try:
                exit_rule_objs = convert_strict(parsed_rules)
                # attach converted ExitRule dataclass instances to config for simulator use
                if isinstance(config, dict):
                    config['exit_rule_objs'] = exit_rule_objs
                else:
                    # if config is some object, set an attribute
                    try:
                        setattr(config, 'exit_rule_objs', exit_rule_objs)
                    except Exception:
                        # fallback: attach to a dictionary entry if possible
                        try:
                            config['exit_rule_objs'] = exit_rule_objs  # type: ignore
                        except Exception:
                            pass
            except ValueError as conv_err:
                # strict conversion failed -> return an error dict so caller (UI) can handle gracefully
                logging.warning(f"[backtester] strict exit-rule conversion failed: {conv_err}")
                return {
                    "error": "invalid_exit_rules",
                    "message": str(conv_err),
                    "raw_signals_count": len(raw_signals),
                    "resolved_signals_count": len([s for s in resolved_signals if s.get("resolved")]),
                    "resolved_signals": resolved_signals
                }
    except Exception as exc:
        # If something unexpected happens during import/conversion, log and continue without strict rules
        logging.exception("[backtester] unexpected error during strict exit-rule conversion - continuing without exit_rule_objs")

    # 3) run simulation (try existing function or fallback)
    results = await _attempt_call_existing_simulator(resolved_signals, config)

    return {
        "raw_signals_count": len(raw_signals),
        "resolved_signals_count": len([s for s in resolved_signals if s.get("resolved")]),
        "resolved_signals": resolved_signals,
        "results": results
    }
# ---------- END: run_backtest_for_user ----------

# ---------- START: DemoTrader (returns real TradeLog objects) ----------
# Insert this block AFTER the marker "# ---------- END: run_backtest_for_user ----------"
# This variant will attempt to construct and return your project's TradeLog dataclass
# (from main_bot.py). If TradeLog cannot be imported, it falls back to returning a dict.
# (ExitRuleEngine and parse_rule_spec are already used elsewhere in this module)
# Ensure json, os, logging, time are available in this module (they are at top of backtester.py)
            
# ---------------------------------------------------------------------
# Add this block to the end of backtester2.py (append after existing classes)
# New analytics utilities: PerformanceAnalytics with calculate_all helper.
# ---------------------------------------------------------------------

@dataclass
class ReturnsMetrics:
    total_return: float
    avg_return: float
    win_rate: float
    profit_factor: float
    total_trades: int

@dataclass
class RiskMetrics:
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_trade_return: float
    std_return: float

class PerformanceAnalytics:
    """
    Lightweight performance analytics utilities.
    Input: a list of trade-like objects (dicts or dataclass instances) with
           'entry_price' and 'exit_price' and optional 'entry_time'/'exit_time'.
    Output: dict containing returns_metrics and risk_metrics.
    """

    @staticmethod
    def _get_trade_prices(trade: Any) -> Optional[tuple]:
        """Normalize trade record to (entry_price, exit_price). Returns None if invalid."""
        # Support dicts and objects with attributes
        try:
            if isinstance(trade, dict):
                ep = trade.get("entry_price")
                xp = trade.get("exit_price")
            else:
                ep = getattr(trade, "entry_price", None)
                xp = getattr(trade, "exit_price", None)
            if ep is None or xp is None:
                return None
            ep = float(ep)
            xp = float(xp)
            if ep <= 0:
                return None
            return (ep, xp)
        except Exception:
            return None

    # Replace existing _calc_risk_metrics method with this in the PerformanceAnalytics class
    @staticmethod
    def _calc_risk_metrics(
        returns: Optional[List[float]] = None,
        trades: Optional[List[Any]] = None,
        trading_periods_per_year: Optional[int] = None,
        returns_are_percent: bool = False
    ) -> RiskMetrics:
        """
        Compute simple risk metrics:
         - Sharpe ratio approximated as mean/std * sqrt(N) where N is number of periods.
           If `trading_periods_per_year` is provided, attempt to annualize using that.
         - Max drawdown computed on equity curve constructed from sequential trade returns.

        Inputs (one of `returns` or `trades` must be provided):
          - returns: list[float] of returns. If returns_are_percent is True, values are interpreted as percentages (e.g., 12.5 -> 12.5%).
                     If returns_are_percent is False, values are interpreted as decimals (e.g., 0.125 -> 12.5%).
          - trades: list of trade-like objects/dicts. The method will attempt to read in this order:
                    'raw_pnl_percent', 'pnl_percent', 'pct_return'. Values are expected as percentages (e.g., 12.5 -> 12.5%).
          - trading_periods_per_year: optional int for annualization.
          - returns_are_percent: only used when `returns` is provided.

        Returns:
          RiskMetrics with fields:
            - sharpe_ratio (float)
            - max_drawdown_pct (float)    # e.g., 35.2 for 35.2%
            - avg_trade_return (float)    # percent (e.g., 12.5 for 12.5%)
            - std_return (float)          # percent
        """
        # Build a list of returns in **decimal** form (e.g., 0.125 for 12.5%)
        returns_decimal: List[float] = []

        # 1) Prefer explicit `returns` list if provided
        if returns is not None:
            for r in returns:
                try:
                    if r is None:
                        continue
                    # Interpret depending on flag
                    r_float = float(r)
                    if returns_are_percent:
                        r_decimal = r_float / 100.0
                    else:
                        r_decimal = r_float
                    returns_decimal.append(r_decimal)
                except Exception:
                    # skip malformed entries
                    continue

        # 2) If no returns list, attempt to extract from trade objects
        elif trades is not None:
            for t in trades:
                try:
                    # Support both dict-like and attr-like trade objects
                    val = None
                    for attr in ("raw_pnl_percent", "pnl_percent", "pct_return"):
                        if isinstance(t, dict) and attr in t:
                            val = t[attr]
                            break
                        elif hasattr(t, attr):
                            val = getattr(t, attr)
                            break
                    if val is None:
                        continue
                    # trades typically store percent values (e.g., 12.5)
                    r_decimal = float(val) / 100.0
                    returns_decimal.append(r_decimal)
                except Exception:
                    continue
        else:
            # Neither returns nor trades provided
            return RiskMetrics(
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                avg_trade_return=0.0,
                std_return=0.0
            )

        n = len(returns_decimal)
        if n == 0:
            return RiskMetrics(
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                avg_trade_return=0.0,
                std_return=0.0
            )

        # mean and std (on decimals)
        avg = statistics.mean(returns_decimal)
        # Use population std if only one sample, else sample std
        std = statistics.pstdev(returns_decimal) if n == 1 else statistics.stdev(returns_decimal)

        # Sharpe-like: use sqrt(n) scaling if no periods-per-year provided
        if std > 0:
            try:
                if trading_periods_per_year and trading_periods_per_year > 0:
                    sharpe = (avg / std) * math.sqrt(trading_periods_per_year)
                else:
                    sharpe = (avg / std) * math.sqrt(n)
            except Exception:
                sharpe = 0.0
        else:
            # std == 0 => no dispersion; returning 0.0 avoids inf propagation
            sharpe = 0.0

        # Equity curve from trade returns (start at 1.0, apply sequential returns)
        equity: List[float] = []
        cum = 1.0
        for r in returns_decimal:
            cum = cum * (1.0 + r)
            equity.append(cum)

        # compute max drawdown
        peak = -float("inf")
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak if peak > 0 else 0.0
            if drawdown > max_dd:
                max_dd = drawdown

        max_drawdown_pct = max_dd * 100.0  # percent

        # Return avg/std as percent to match previous behaviour
        avg_trade_return = avg * 100.0
        std_return = std * 100.0

        return RiskMetrics(
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown_pct,
            avg_trade_return=avg_trade_return,
            std_return=std_return
        )

    @staticmethod
    def _calc_risk_metrics(returns: List[float], trading_periods_per_year: Optional[int] = None) -> RiskMetrics:
        """
        Compute simple risk metrics:
         - Sharpe ratio approximated as mean/std * sqrt(N) where N is number of periods.
           Note: since trade returns are per-trade, we use sqrt(n) as a proxy if no period mapping is provided.
         - Max drawdown computed on equity curve constructed from sequential trade returns.
        """
        n = len(returns)
        if n == 0:
            return RiskMetrics(sharpe_ratio=0.0, max_drawdown_pct=0.0, avg_trade_return=0.0, std_return=0.0)

        # mean and std (sample std)
        avg = statistics.mean(returns)
        std = statistics.pstdev(returns) if n == 1 else statistics.stdev(returns)

        # Sharpe-like: use sqrt(n) scaling if no periods-per-year provided
        try:
            if trading_periods_per_year and trading_periods_per_year > 0:
                # Interpret returns as per-period returns; annualize
                sharpe = (avg / std) * math.sqrt(trading_periods_per_year) if std > 0 else float("inf")
            else:
                sharpe = (avg / std) * math.sqrt(n) if std > 0 else float("inf")
        except Exception:
            sharpe = 0.0

        # Equity curve from trade returns (start at 1.0, apply sequential returns)
        equity = []
        cum = 1.0
        for r in returns:
            cum = cum * (1.0 + r)
            equity.append(cum)

        # compute max drawdown
        peak = -float("inf")
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak if peak > 0 else 0.0
            if drawdown > max_dd:
                max_dd = drawdown

        max_drawdown_pct = max_dd * 100.0  # percent

        avg_trade_return = avg * 100.0  # percent
        std_return = std * 100.0

        return RiskMetrics(
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown_pct,
            avg_trade_return=avg_trade_return,
            std_return=std_return
        )

    @staticmethod
    def calculate_all(trades: List[Any], trading_periods_per_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Primary entry point used by the system.
        trades: list of trade dicts or objects (must have entry_price & exit_price).
        Returns a dict:
          {
            "returns_metrics": ReturnsMetrics(...),
            "risk_metrics": RiskMetrics(...),
            "raw_returns": [...],  # decimal returns per trade
          }
        """
        # Convert trades -> per-trade decimal returns
        raw_returns: List[float] = []
        for t in trades or []:
            prices = PerformanceAnalytics._get_trade_prices(t)
            if prices is None:
                continue
            ep, xp = prices
            # trade return as decimal: (exit - entry) / entry
            r = (xp - ep) / ep
            raw_returns.append(r)

        returns_metrics = PerformanceAnalytics._calc_returns_metrics(raw_returns)
        risk_metrics = PerformanceAnalytics._calc_risk_metrics(raw_returns, trading_periods_per_year=trading_periods_per_year)

        return {
            "returns_metrics": returns_metrics,
            "risk_metrics": risk_metrics,
            "raw_returns": raw_returns
        }


    # ---------------------
    # Public: open and evaluate
    # ---------------------
    # Replace the existing process_live_signal definition in backtester3.py with this version.

    async def process_live_signal(self, user_id: int, signal: Any, entry_price: Optional[float] = None, position_size_pct: Optional[float] = None):
        """
        Open a demo trade and return a TradeLog instance when available.
        Enforces per-user resource limit (max_active_trades_per_user) atomically.
        Centralized error handling: call ErrorHandler on exceptions and re-raise.
        """
        # Prepare small extra context for error handler
        extra_ctx = {"phase": "process_live_signal", "user_id": user_id}
        if isinstance(signal, dict):
            extra_ctx["signal_id"] = signal.get("identifier") or signal.get("resolved_symbol") or signal.get("symbol")
        else:
            extra_ctx["signal_id"] = getattr(signal, "identifier", None)

        try:
            # --- ORIGINAL LOGIC START (unchanged, just kept inside try) ---
            price = entry_price
            symbol = None
            if isinstance(signal, dict):
                symbol = signal.get("resolved_symbol") or signal.get("symbol") or signal.get("identifier")
            else:
                symbol = getattr(signal, "resolved_symbol", None) or getattr(signal, "symbol", None) or getattr(signal, "identifier", None)

            if price is None and self.market_adapter and symbol:
                try:
                    p = await self.market_adapter.get_current_price(symbol)
                    price = p.get("price") if isinstance(p, dict) else p
                except asyncio.TimeoutError as te:
                    # explicit timeout -> MarketAPIError
                    raise MarketAPIError(f"Timeout when contacting market provider for {symbol}: {te}", provider=getattr(self.market_adapter, "provider_name", None)) from te
                except Exception as e:
                    # try to map known HTTP client libs
                    # aiohttp
                    try:
                        import aiohttp
                        HTTPClientBase = aiohttp.ClientError
                    except Exception:
                        HTTPClientBase = Exception

                    # requests
                    try:
                        import requests
                        RequestsBase = requests.RequestException
                    except Exception:
                        RequestsBase = HTTPClientBase

                    # httpx
                    try:
                        import httpx
                        HTTPXBase = httpx.HTTPError
                    except Exception:
                        HTTPXBase = HTTPClientBase

                    if isinstance(e, (HTTPClientBase, RequestsBase, HTTPXBase)):
                        raise MarketAPIError(f"Market adapter network error for {symbol}: {e}", provider=getattr(self.market_adapter, "provider_name", None)) from e
                    # fallback: wrap generic exceptions as MarketAPIError so ErrorHandler can handle uniformly
                    raise MarketAPIError(f"Unexpected market adapter error resolving {symbol}: {e}", provider=getattr(self.market_adapter, "provider_name", None)) from e


            if price is None:
                raise RuntimeError("DemoTrader: unable to resolve entry price for signal")

            trade_id = str(uuid.uuid4())
            entry_time_iso = datetime.utcnow().isoformat()
            payload = {
                "trade_id": trade_id,
                "user_id": user_id,
                "symbol": symbol,
                "entry_time": entry_time_iso,
                "entry_price": float(price),
                "position_size_percent": float(position_size_pct or (signal.get("position_size_percent") if isinstance(signal, dict) else getattr(signal, "position_size_percent", 1.0))),
                "entry_market_cap": (signal.get("market_cap_at_detection") if isinstance(signal, dict) else getattr(signal, "market_cap_at_detection", None)),
                "is_demo": True,
                "fees_percent": float(signal.get("fees_percent", 0.0) if isinstance(signal, dict) else getattr(signal, "fees_percent", 0.0)),
                "slippage_percent": float(signal.get("slippage_percent", 0.0) if isinstance(signal, dict) else getattr(signal, "slippage_percent", 0.0)),
                "metadata": {k: (signal.get(k) if isinstance(signal, dict) else getattr(signal, k, None)) for k in ("source_chat_id","original_message") if (signal.get(k) if isinstance(signal, dict) else getattr(signal, k, None)) is not None}
            }

            # 2) Enforce per-user active trade limit atomically.
            async with self._lock:
                # current active count for user
                try:
                    current_count = 0
                    for tobj in self._active_trades.values():
                        try:
                            uid = tobj.get("user_id") if isinstance(tobj, dict) else getattr(tobj, "user_id", None)
                            if uid == user_id:
                                current_count += 1
                        except Exception:
                            continue
                except Exception:
                    current_count = 0

                if self.max_active_trades_per_user is not None and current_count >= int(self.max_active_trades_per_user):
                    # enforce rejection policy: raise to caller
                    err_msg = f"DemoTrader: max active demo trades reached for user {user_id} (limit={self.max_active_trades_per_user})"
                    logging.warning(err_msg)
                    raise RuntimeError(err_msg)

                # reserve slot with a placeholder
                self._active_trades[trade_id] = {"trade_id": trade_id, "user_id": user_id, "symbol": symbol, "entry_time": entry_time_iso, "entry_price": float(price), "is_demo": True}

            # 3) Build trade_obj (outside lock to avoid heavy work inside lock)
            trade_obj = None
            if self.trade_factory:
                try:
                    trade_obj = self.trade_factory(**payload) if callable(self.trade_factory) else self.trade_factory(payload)
                except Exception:
                    logging.exception("DemoTrader: explicit trade_factory failed; falling back to TradeLog import")
                    trade_obj = None

            if trade_obj is None:
                try:
                    # Attempt to construct TradeLog from main_bot if available
                    try:
                        from main_bot import TradeLog, Signal as MBSignal
                    except Exception:
                        # robust import fallback
                        spec = importlib.util.spec_from_file_location("main_bot", os.path.join(os.path.dirname(__file__), "main_bot.py"))
                        mm = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mm)
                        TradeLog = getattr(mm, "TradeLog")
                        MBSignal = getattr(mm, "Signal", None)

                    signal_obj = None
                    if MBSignal is not None:
                        if isinstance(signal, dict):
                            try:
                                signal_obj = MBSignal(
                                    type=signal.get("type"),
                                    identifier=signal.get("identifier") or signal.get("resolved_symbol") or signal.get("symbol"),
                                    detected_at=(signal.get("detected_at") if isinstance(signal.get("detected_at"), datetime) else (datetime.fromisoformat(signal.get("detected_at")) if signal.get("detected_at") else datetime.utcnow())),
                                    source_chat_id=signal.get("source_chat_id", -1),
                                    source_job_id=signal.get("source_job_id", "live_demo"),
                                    original_message=signal.get("original_message", ""),
                                    confidence_score=float(signal.get("confidence_score", 0.5)),
                                    resolved_symbol=signal.get("resolved_symbol") or signal.get("symbol")
                                )
                            except Exception:
                                signal_obj = None
                        else:
                            # try to coerce
                            try:
                                signal_obj = MBSignal(
                                    type=getattr(signal, "type", "keyword"),
                                    identifier=getattr(signal, "identifier", getattr(signal, "resolved_symbol", None)),
                                    detected_at=getattr(signal, "detected_at", datetime.utcnow()),
                                    source_chat_id=getattr(signal, "source_chat_id", -1),
                                    source_job_id=getattr(signal, "source_job_id", "live_demo"),
                                    original_message=getattr(signal, "original_message", ""),
                                    confidence_score=float(getattr(signal, "confidence_score", 0.5))
                                )
                            except Exception:
                                signal_obj = None

                    try:
                        entry_time_dt = datetime.fromisoformat(payload["entry_time"])
                    except Exception:
                        entry_time_dt = datetime.utcnow()

                    trade_obj = TradeLog(
                        trade_id=payload["trade_id"],
                        signal=signal_obj,
                        entry_time=entry_time_dt,
                        entry_price=float(payload["entry_price"]),
                        entry_market_cap=payload.get("entry_market_cap"),
                        exit_time=None,
                        exit_price=None,
                        exit_reason=None,
                        raw_pnl_percent=None,
                        adjusted_pnl_percent=None,
                        hold_duration_seconds=None,
                        position_size_percent=float(payload["position_size_percent"]),
                        portfolio_value_at_entry=payload.get("portfolio_value_at_entry"),
                        absolute_pnl=None,
                        is_demo=bool(payload.get("is_demo", True)),
                        fees_percent=float(payload.get("fees_percent", 0.0)),
                        slippage_percent=float(payload.get("slippage_percent", 0.0))
                    )
                except Exception:
                    logging.exception("DemoTrader: failed to construct TradeLog, falling back to dict payload")
                    trade_obj = payload

            # 4) Replace placeholder in _active_trades with the real trade_obj under lock
            async with self._lock:
                # sanity check: if placeholder missing, add again (defensive)
                if trade_id not in self._active_trades:
                    self._active_trades[trade_id] = trade_obj
                else:
                    self._active_trades[trade_id] = trade_obj

            # 5) Persist the trade (best-effort)
            try:
                await self._persist_trade(user_id, trade_obj)
            except Exception:
                logging.exception("DemoTrader._persist_trade failed during process_live_signal")

            logging.info("DemoTrader opened demo trade %s for user %s entry_price=%s", trade_id, user_id, price)
            return trade_obj
            # --- ORIGINAL LOGIC END ---
        except Exception as e:
            # Interpret & map known runtime issues to domain exceptions, then call ErrorHandler
            exc_text = str(e)
            if "unable to resolve entry price" in exc_text.lower() or "market adapter error" in exc_text.lower():
                wrapped = MarketAPIError(exc_text, provider=getattr(self.market_adapter, "provider_name", None) if getattr(self, "market_adapter", None) else None)
            elif "max active demo trades" in exc_text.lower():
                wrapped = DemoTradingError(exc_text)
            else:
                wrapped = DemoTradingError(exc_text)

            # Choose a bot-like object if forwarder is available
            bot_obj = None
            try:
                if getattr(self, "forwarder", None) and getattr(self.forwarder, "bot", None):
                    bot_obj = self.forwarder.bot
            except Exception:
                bot_obj = None

            # Best-effort notify user + log
            await ErrorHandler.handle_error(None, wrapped, bot=bot_obj, user_id=user_id, extra=extra_ctx, notify_user_for_internal_errors=True)

            # Re-raise so callers maintain existing semantics (up to caller to catch)
            raise

    async def evaluate_exits(self, trade: Any, current_price: float) -> Dict[str, Any]:
        """
        Evaluate exit rules for the trade. Uses ExitRuleEngine. Returns a dict:
        {"should_close": bool, "reason": str, "exit_price": float}
        """
        try:
            if isinstance(trade, dict):
                entry_price = float(trade.get("entry_price", 0))
                et = trade.get("entry_time")
                if isinstance(et, str):
                    try:
                        entry_time_ts = datetime.fromisoformat(et).timestamp()
                    except Exception:
                        entry_time_ts = time.time()
                elif isinstance(et, datetime):
                    entry_time_ts = et.timestamp()
                else:
                    entry_time_ts = time.time()
            else:
                entry_price = float(getattr(trade, "entry_price", 0))
                et = getattr(trade, "entry_time", None)
                entry_time_ts = et.timestamp() if isinstance(et, datetime) else time.time()

            engine = ExitRuleEngine(entry_price=entry_price, entry_time=entry_time_ts)

            # Try to load user-specific exit rules from user file; best-effort non-fatal
            rules = []
            try:
                user_id = trade.get("user_id") if isinstance(trade, dict) else getattr(trade, "user_id", 0)
                user_file = os.path.join(self.data_dir, f"user_{user_id}.dat")
                if os.path.exists(user_file):
                    crypto = self.crypto_manager_factory(user_id)
                    j = json.loads(crypto.decrypt(open(user_file, "rb").read()))
                    perf = j.get("performance_settings", {})
                    raw_rules = perf.get("exit_rules", []) or []
                    rules = [parse_rule_spec(r) for r in raw_rules]
            except Exception:
                pass

            if rules:
                engine.set_rules(rules)
            else:
                engine.set_rules([parse_rule_spec({"type":"stop_loss","value":-0.2}), parse_rule_spec({"type":"take_profit","value":1.0,"multiple_mode": True})])

            engine.on_price_tick(current_price, time.time())
            dec = engine.evaluate()
            if dec:
                return {"should_close": True, "reason": getattr(dec, "reason", "rule"), "exit_price": getattr(dec, "exit_price", current_price)}
            return {"should_close": False}
        except Exception:
            logging.exception("DemoTrader.evaluate_exits error")
            return {"should_close": False}
            
    async def check_demo_exits(self) -> None:
        """
        Concurrently check exit rules for all active demo trades.
        - Bounded concurrency controlled by DEMO_MARKET_API_CONCURRENCY.
        - For each active trade, attempts to fetch current price via self.market_adapter.get_current_price(symbol)
          (supports either float or dict {'price': ...}).
        - Calls self.evaluate_exits(trade, current_price) and if should_exit, executes and notifies.
        """
        try:
            # Snapshot to avoid holding lock while network IO happens
            async with self._lock:
                trades_snapshot = list(self._active_trades.items())  # list of (trade_id, trade_obj)

            if not trades_snapshot:
                return

            sem = asyncio.Semaphore(DEMO_MARKET_API_CONCURRENCY)
            tasks = []

            async def _fetch_eval_and_handle(tid: str, trade_obj: Any):
                # get symbol
                try:
                    symbol = trade_obj.get("symbol") if isinstance(trade_obj, dict) else getattr(trade_obj, "symbol", None)
                except Exception:
                    symbol = None

                # fetch price guarded by semaphore
                async with sem:
                    current_price = None
                    try:
                        if symbol and getattr(self, "market_adapter", None):
                            p = await self.market_adapter.get_current_price(symbol)
                            current_price = p.get("price") if isinstance(p, dict) else p
                    except Exception as e:
                        logging.debug("check_demo_exits: market_adapter error for %s: %s", symbol, e)

                if current_price is None:
                    # nothing to evaluate this tick
                    return

                try:
                    decision = await self.evaluate_exits(trade_obj, current_price)
                except Exception as e:
                    logging.exception("check_demo_exits: evaluate_exits failed for %s: %s", symbol, e)
                    return

                if decision and decision.get("should_close", decision.get("should_exit", False)):
                    # perform the exit
                    try:
                        await self._execute_demo_exit(trade_obj, decision)
                    except Exception as e:
                        logging.exception("check_demo_exits: _execute_demo_exit failed for %s: %s", symbol, e)

                    # attempt notification (best-effort)
                    try:
                        # user_id extraction (support both dict and dataclass)
                        user_id = trade_obj.get("user_id") if isinstance(trade_obj, dict) else getattr(trade_obj, "user_id", None)
                        if user_id is None:
                            # fallback to 0 if no user id stored
                            user_id = 0
                        await self._notify_demo_exit(user_id, trade_obj, decision)
                    except Exception as e:
                        logging.debug("check_demo_exits: notify failed for %s: %s", symbol, e)

            for trade_id, trade_obj in trades_snapshot:
                tasks.append(asyncio.create_task(_fetch_eval_and_handle(trade_id, trade_obj)))

            if tasks:
                # Await all tasks but don't let any single failure bubble up
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        logging.debug("check_demo_exits: task raised: %s", r)

        except Exception:
            logging.exception("Unhandled error in check_demo_exits")
            
    async def _reporting_loop(self):
        """
        Background loop that periodically calls send_scheduled_reports().
        Runs until self._running is False or task cancelled.
        """
        try:
            while self._running:
                try:
                    await self.send_scheduled_reports()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logging.exception("DemoTrader._reporting_loop: send_scheduled_reports failed")
                # sleep using the configured reporting interval
                await asyncio.sleep(self.reporting_interval_seconds)
        except asyncio.CancelledError:
            logging.info("DemoTrader reporting loop cancelled")
        except Exception:
            logging.exception("DemoTrader reporting loop fatal error")


# -----------------------------
# REPLACE the existing function:
# def generate_demo_report_for_user(self, user_id: int) -> Dict[str, Any]:
# in backtester4.py (DemoTrader class)
# -----------------------------
def generate_demo_report_for_user(self, user_id: int) -> Dict[str, Any]:
    """
    Build a summary report (dict) for a given demo user by scanning persisted demo trades
    and in-memory active trades. Returns a dict with summary fields and a 'text' field ready
    for sending. This version includes a 'Portfolio heat' analysis obtained from
    self.portfolio_heat_check(...) and attaches it to the result as 'portfolio_heat'.
    """
    try:
        report = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "total_trades_persisted": 0,
            "active_trades_in_memory": 0,
            "closed_trades_count": 0,
            "average_raw_pnl_percent": None,
            "total_absolute_pnl": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "open_symbols": [],
            # Portfolio heat will be attached below
            "portfolio_heat": None,
            # 'text' will contain the formatted message for sending
            "text": "",
        }

        # --- 1) load persisted trades and in-memory active trades ---
        persisted = {}
        try:
            persisted = self._load_persisted_trades_for_user(user_id)
        except Exception:
            logging.exception("generate_demo_report_for_user: failed to load persisted trades for user %s", user_id)

        # persisted is dict trade_id -> trade_dict
        persisted_trades = list(persisted.values())
        report["total_trades_persisted"] = len(persisted_trades)

        # active trades in memory are stored in self._active_trades as trade_id -> trade_dict
        in_memory = [t for t in self._active_trades.values() if (isinstance(t, dict) and t.get("user_id") == user_id) or (hasattr(t, "user_id") and getattr(t, "user_id") == user_id)]
        report["active_trades_in_memory"] = len(in_memory)

        # --- 2) stats from persisted trades (best effort) ---
        pnls = []
        abs_pnls = []
        winning = 0
        losing = 0
        open_symbols = set()

        # inspect persisted trades first (closed trades will be there)
        for t in persisted_trades:
            try:
                # raw_pnl_percent may be string or number
                raw_pnl = None
                if isinstance(t, dict):
                    raw_pnl = t.get("raw_pnl_percent") or t.get("pnl_percent") or t.get("raw_return_percent")
                else:
                    raw_pnl = getattr(t, "raw_pnl_percent", None) or getattr(t, "pnl_percent", None)
                if raw_pnl is not None:
                    pnls.append(float(raw_pnl))
                abs_p = t.get("absolute_pnl") if isinstance(t, dict) else getattr(t, "absolute_pnl", None)
                if abs_p is not None:
                    try:
                        abs_pnls.append(float(abs_p))
                    except Exception:
                        pass

                # count wins / losses
                if raw_pnl is not None:
                    try:
                        if float(raw_pnl) > 0:
                            winning += 1
                        else:
                            losing += 1
                    except Exception:
                        pass

                # collect symbol for message
                sym = None
                if isinstance(t, dict):
                    sym = t.get("resolved_symbol") or t.get("symbol") or t.get("pair")
                else:
                    sym = getattr(t, "resolved_symbol", None) or getattr(t, "symbol", None)
                if sym:
                    open_symbols.add(sym)
            except Exception:
                logging.debug("generate_demo_report_for_user: ignoring corrupted persisted trade for user %s", user_id)

        # also include active in-memory trades (these are open)
        for t in in_memory:
            try:
                # treat as open, collect symbol
                sym = t.get("resolved_symbol") if isinstance(t, dict) else getattr(t, "resolved_symbol", None)
                if not sym:
                    sym = t.get("symbol") if isinstance(t, dict) else getattr(t, "symbol", None)
                if sym:
                    open_symbols.add(sym)

                raw_pnl = t.get("raw_pnl_percent") if isinstance(t, dict) else getattr(t, "raw_pnl_percent", None)
                if raw_pnl is not None:
                    pnls.append(float(raw_pnl))
                abs_p = t.get("absolute_pnl") if isinstance(t, dict) else getattr(t, "absolute_pnl", None)
                if abs_p is not None:
                    abs_pnls.append(float(abs_p))
            except Exception:
                logging.debug("generate_demo_report_for_user: error processing in-memory trade for user %s", user_id)

        # metrics
        report["open_symbols"] = sorted(list(open_symbols))
        report["winning_trades"] = winning
        report["losing_trades"] = losing
        report["closed_trades_count"] = report["total_trades_persisted"]  # persisted are mostly closed
        report["average_raw_pnl_percent"] = float(statistics.mean(pnls)) if pnls else None
        report["total_absolute_pnl"] = float(sum(abs_pnls)) if abs_pnls else 0.0

        # --- 3) Portfolio heat analysis (new) ---
        try:
            heat = self.portfolio_heat_check(user_id=user_id, top_n=5, exposure_threshold_pct=25.0)
            report["portfolio_heat"] = heat
        except Exception:
            logging.exception("generate_demo_report_for_user: portfolio_heat_check failed for user %s", user_id)
            report["portfolio_heat"] = None

        # --- 4) Build text for messaging ---
        lines = []
        lines.append(f"📊 Demo Portfolio Report — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}")
        lines.append(f"User: {user_id}")
        lines.append(f"Persisted trades: {report['total_trades_persisted']}, Active (in-memory): {report['active_trades_in_memory']}")
        if report["average_raw_pnl_percent"] is not None:
            lines.append(f"Avg PnL (%): {report['average_raw_pnl_percent']:.2f}")
        lines.append(f"Total absolute PnL: {report['total_absolute_pnl']:.2f}")
        lines.append(f"Winning trades: {report['winning_trades']} — Losing trades: {report['losing_trades']}")

        # Insert portfolio heat summary (if available)
        ph = report.get("portfolio_heat")
        if ph:
            lines.append("")  # blank line
            lines.append("🔥 Portfolio heat (risk concentration):")
            # top exposures
            top = ph.get("top_exposures", [])
            if top:
                for i, item in enumerate(top, start=1):
                    sym = item.get("symbol")
                    exposure = item.get("exposure_value")
                    pct = item.get("exposure_pct")
                    # exposure_value may be None; format defensively
                    ev = f"${exposure:,.2f}" if isinstance(exposure, (int, float)) else "n/a"
                    pct_str = f"{pct:.2f}%" if isinstance(pct, (int, float)) else "n/a"
                    lines.append(f"{i}. {sym} — {ev} ({pct_str} of portfolio)")
            else:
                lines.append("No measurable exposures (insufficient trade fields).")

            # concentration flag / advice
            if ph.get("high_concentration", False):
                lines.append(f"⚠️ High concentration detected: {ph.get('max_exposure_pct', 0):.2f}% on {ph.get('max_exposure_symbol')}. Consider rebalancing.")
            else:
                lines.append("✅ No single-symbol concentration above threshold.")

            # short recommendations (if any)
            recs = ph.get("recommendations", [])
            for r in recs:
                lines.append(f"- {r}")

        # Finalize text
        report_text = "\n".join(lines)
        report["text"] = report_text

        return report

    except Exception:
        logging.exception("generate_demo_report_for_user: top-level failure for user %s", user_id)
        # return minimal fallback
        return {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "text": "⚠️ Failed to build report. Check server logs.",
            "portfolio_heat": None,
        }

    # -----------------------------
    # ADD this new helper method in DemoTrader (immediately after the replaced function)
    # -----------------------------
    def portfolio_heat_check(self, user_id: int, top_n: int = 5, exposure_threshold_pct: float = 25.0) -> Dict[str, Any]:
        """
        Analyze risk concentration across an active demo portfolio for `user_id`.

        Returns a dict:
            {
                "top_exposures": [ {"symbol": str, "exposure_value": float|None, "exposure_pct": float|None}, ... ],
                "total_exposure_value": float|None,
                "max_exposure_symbol": str|None,
                "max_exposure_pct": float|None,
                "high_concentration": bool,
                "sample_size": int,
                "recommendations": [str, ...]
            }

        The method is defensive: it tries multiple heuristics to compute notional exposure for each trade:
        - uses explicit fields if present: 'notional','position_value','absolute_notional'
        - falls back to entry_price * quantity if 'entry_price' and 'quantity' are present
        - falls back to percentage-of-portfolio if 'position_size'/'position_fraction' + 'portfolio_value_at_entry' exist
        - if none available, counts positions per-symbol (no USD exposure)
        """
        try:
            from collections import defaultdict, Counter

            # gather trades: persisted + in-memory (best effort)
            persisted = {}
            try:
                persisted = self._load_persisted_trades_for_user(user_id)
            except Exception:
                logging.debug("portfolio_heat_check: could not load persisted trades for user %s", user_id)

            persisted_trades = list(persisted.values())
            in_memory = [t for t in self._active_trades.values() if (isinstance(t, dict) and t.get("user_id") == user_id) or (hasattr(t, "user_id") and getattr(t, "user_id") == user_id)]

            # helper to extract a symbol and possible exposure value from a trade dict/object
            def _extract_symbol_and_exposure(t):
                # symbol resolution
                sym = None
                if isinstance(t, dict):
                    sym = t.get("resolved_symbol") or t.get("symbol") or t.get("pair")
                else:
                    sym = getattr(t, "resolved_symbol", None) or getattr(t, "symbol", None) or getattr(t, "pair", None)

                # attempt to compute exposure value (USD)
                exposure = None
                try:
                    if isinstance(t, dict):
                        # explicit notional-like names
                        for k in ("notional", "position_value", "absolute_notional", "position_usd", "position_value_usd"):
                            if k in t and t[k] is not None:
                                try:
                                    exposure = float(t[k])
                                    break
                                except Exception:
                                    exposure = None
                        # entry_price * quantity
                        if exposure is None and t.get("entry_price") and (t.get("quantity") or t.get("amount")):
                            try:
                                q = t.get("quantity") or t.get("amount")
                                exposure = float(t.get("entry_price")) * float(q)
                            except Exception:
                                exposure = None
                        # position fraction * portfolio value
                        if exposure is None and (t.get("position_size") or t.get("position_fraction") or t.get("position_pct")) and t.get("portfolio_value_at_entry"):
                            try:
                                frac = t.get("position_size") or t.get("position_fraction") or t.get("position_pct")
                                # if pct expressed as percent >1, convert
                                frac_f = float(frac)
                                if frac_f > 1:
                                    frac_f = frac_f / 100.0
                                exposure = float(t.get("portfolio_value_at_entry")) * frac_f
                            except Exception:
                                exposure = None
                    else:
                        # object-like access
                        for attr in ("notional", "position_value", "absolute_notional", "position_usd"):
                            if hasattr(t, attr):
                                v = getattr(t, attr)
                                if v is not None:
                                    try:
                                        exposure = float(v)
                                        break
                                    except Exception:
                                        exposure = None
                        if exposure is None and getattr(t, "entry_price", None) and getattr(t, "quantity", None):
                            try:
                                exposure = float(getattr(t, "entry_price")) * float(getattr(t, "quantity"))
                            except Exception:
                                exposure = None
                        if exposure is None and getattr(t, "portfolio_value_at_entry", None) and (getattr(t, "position_size", None) or getattr(t, "position_fraction", None)):
                            try:
                                frac = getattr(t, "position_size", None) or getattr(t, "position_fraction", None)
                                f = float(frac)
                                if f > 1:
                                    f = f / 100.0
                                exposure = float(getattr(t, "portfolio_value_at_entry")) * f
                            except Exception:
                                exposure = None
                except Exception:
                    exposure = None

                return sym, exposure

            # build sums
            exposure_by_symbol = defaultdict(float)
            missing_exposure_count = 0
            position_count_by_symbol = Counter()
            total_explicit_exposure = 0.0
            sample_size = 0

            # aggregate persisted + in-memory
            for t in persisted_trades + in_memory:
                try:
                    sym, exposure = _extract_symbol_and_exposure(t)
                    sample_size += 1
                    if sym:
                        position_count_by_symbol[sym] += 1
                        if exposure is not None:
                            exposure_by_symbol[sym] += abs(float(exposure))
                            total_explicit_exposure += abs(float(exposure))
                        else:
                            missing_exposure_count += 1
                    else:
                        missing_exposure_count += 1
                except Exception:
                    logging.debug("portfolio_heat_check: error extracting exposure for user %s", user_id)

            # If we have no explicit exposure numbers, we'll fallback to counting positions only.
            top_exposures = []
            recommendations = []
            max_exposure_pct = None
            max_exposure_symbol = None
            total_exposure_value = total_explicit_exposure if total_explicit_exposure > 0 else None

            if total_explicit_exposure and total_explicit_exposure > 0.0:
                # compute percent exposures
                for sym, val in sorted(exposure_by_symbol.items(), key=lambda x: x[1], reverse=True)[:top_n]:
                    pct = (val / total_explicit_exposure) * 100.0 if total_explicit_exposure > 0 else None
                    top_exposures.append({
                        "symbol": sym,
                        "exposure_value": float(val),
                        "exposure_pct": float(pct) if pct is not None else None
                    })
                    if max_exposure_pct is None or (pct is not None and pct > max_exposure_pct):
                        max_exposure_pct = float(pct) if pct is not None else max_exposure_pct
                        max_exposure_symbol = sym

                high_concentration = (max_exposure_pct is not None and max_exposure_pct >= float(exposure_threshold_pct))
                if high_concentration:
                    recommendations.append(f"Reduce exposure to {max_exposure_symbol} (currently {max_exposure_pct:.2f}% of portfolio).")
                # also suggest diversification if only a few symbols dominate
                if len(exposure_by_symbol) <= 2 and len(exposure_by_symbol) > 0:
                    recommendations.append("Portfolio is concentrated in very few symbols — consider diversifying.")
            else:
                # no explicit notional exposures available: fallback to counts
                total_positions = sum(position_count_by_symbol.values())
                if total_positions == 0:
                    # nothing to analyze
                    return {
                        "top_exposures": [],
                        "total_exposure_value": None,
                        "max_exposure_symbol": None,
                        "max_exposure_pct": None,
                        "high_concentration": False,
                        "sample_size": sample_size,
                        "recommendations": ["No measurable exposures found (trade records missing notional/quantity fields)."]
                    }
                # use relative counts as a proxy
                top_counts = position_count_by_symbol.most_common(top_n)
                for sym, cnt in top_counts:
                    pct = (cnt / total_positions) * 100.0
                    top_exposures.append({
                        "symbol": sym,
                        "exposure_value": None,
                        "exposure_pct": float(pct)
                    })
                    if max_exposure_pct is None or pct > max_exposure_pct:
                        max_exposure_pct = float(pct)
                        max_exposure_symbol = sym
                high_concentration = (max_exposure_pct is not None and max_exposure_pct >= float(exposure_threshold_pct))
                if high_concentration:
                    recommendations.append(f"Symbol {max_exposure_symbol} represents {max_exposure_pct:.2f}% of positions (by count). Consider rebalancing.")
                else:
                    recommendations.append("No single symbol dominates position count.")

            return {
                "top_exposures": top_exposures,
                "total_exposure_value": float(total_exposure_value) if total_exposure_value is not None else None,
                "max_exposure_symbol": max_exposure_symbol,
                "max_exposure_pct": float(max_exposure_pct) if max_exposure_pct is not None else None,
                "high_concentration": bool(high_concentration),
                "sample_size": int(sample_size),
                "missing_exposure_count": int(missing_exposure_count),
                "recommendations": recommendations,
            }

        except Exception:
            logging.exception("portfolio_heat_check: unexpected failure for user %s", user_id)
            return {
                "top_exposures": [],
                "total_exposure_value": None,
                "max_exposure_symbol": None,
                "max_exposure_pct": None,
                "high_concentration": False,
                "sample_size": 0,
                "missing_exposure_count": 0,
                "recommendations": ["Error computing portfolio heat — see server logs."],
            }

    # -----------------------------
    # ADD these Portfolio Rebalancer methods in DemoTrader (backtester4.py)
    # Place immediately AFTER portfolio_heat_check(...) in DemoTrader
    # -----------------------------
    import asyncio
    from datetime import datetime, timedelta

    def schedule_rebalance(self, user_id: int, interval_days: int = 7) -> dict:
        """
        Schedule a periodic rebalance reminder for user_id.
        Stores entry in self._scheduled_rebalances (dict).
        Returns summary dict with next_run timestamp.
        NOTE: This only schedules an in-memory reminder. Persist if you need persistence across restarts.
        """
        if not hasattr(self, "_scheduled_rebalances"):
            self._scheduled_rebalances = {}

        interval_days = int(max(1, interval_days))
        next_run = datetime.utcnow() + timedelta(days=interval_days)
        self._scheduled_rebalances[user_id] = {
            "interval_days": interval_days,
            "next_run": next_run,
            "enabled": True,
            "last_run": None,
        }
        return {
            "status": "scheduled",
            "user_id": user_id,
            "interval_days": interval_days,
            "next_run": next_run.isoformat()
        }
        
    def persist_schedules(self, path: str):
        try:
            import json
            if hasattr(self, "_scheduled_rebalances"):
                with open(path, "w") as f:
                    # convert datetimes to isoformat
                    serial = {}
                    for uid, meta in self._scheduled_rebalances.items():
                        serial[uid] = {
                            **{k: v for k, v in meta.items() if k != "next_run" and k != "last_run"},
                            "next_run": meta.get("next_run").isoformat() if isinstance(meta.get("next_run"), datetime) else meta.get("next_run"),
                            "last_run": meta.get("last_run").isoformat() if isinstance(meta.get("last_run"), datetime) else meta.get("last_run"),
                        }
                    json.dump(serial, f)
        except Exception:
            logging.exception("persist_schedules failed")

    def load_schedules(self, path: str):
        try:
            import json
            if not os.path.exists(path):
                return
            with open(path, "r") as f:
                loaded = json.load(f)
            self._scheduled_rebalances = {}
            for uid_str, meta in loaded.items():
                try:
                    uid = int(uid_str)
                except Exception:
                    uid = uid_str
                self._scheduled_rebalances[uid] = {
                    **meta,
                    "next_run": datetime.fromisoformat(meta.get("next_run")) if meta.get("next_run") else datetime.utcnow(),
                    "last_run": datetime.fromisoformat(meta.get("last_run")) if meta.get("last_run") else None,
                }
        except Exception:
            logging.exception("load_schedules failed")


    def cancel_rebalance(self, user_id: int) -> dict:
        """
        Cancel scheduled rebalance for user_id. Safe if nothing existed.
        """
        if hasattr(self, "_scheduled_rebalances") and user_id in self._scheduled_rebalances:
            del self._scheduled_rebalances[user_id]
            return {"status": "cancelled", "user_id": user_id}
        return {"status": "not_found", "user_id": user_id}

    def run_rebalance_for_user(self, user_id: int, exposure_threshold_pct: float = 25.0, dry_run: bool = True) -> dict:
        """
        Analyze the user's portfolio and create rebalancing recommendations.
        - Calls portfolio_heat_check(user_id)
        - For any top_exposure above exposure_threshold_pct, suggests a 'reduce' recommendation.
        - Returns a dict with:
            - portfolio_heat (raw)
            - recommendations: list of suggested actions (symbol, current_pct, suggested_target_pct, rationale)
            - dry_run: bool
        This method does NOT execute trades.
        """
        try:
            heat = self.portfolio_heat_check(user_id=user_id, top_n=10, exposure_threshold_pct=exposure_threshold_pct)
            recs = []

            top_expos = heat.get("top_exposures", [])
            total_value = heat.get("total_exposure_value", None)

            # If explicit exposure values exist, generate USD suggestions; else use counts/proxies
            for item in top_expos:
                sym = item.get("symbol")
                pct = item.get("exposure_pct")
                val = item.get("exposure_value")
                if pct is None:
                    # cannot compute percent; skip or suggest manual review
                    recs.append({
                        "symbol": sym,
                        "action": "review",
                        "reason": "Insufficient exposure data (no USD notional).",
                    })
                    continue

                if pct >= float(exposure_threshold_pct):
                    # target: reduce to half the threshold or equal-weight among top-N (simple heuristic)
                    suggested_target_pct = float(exposure_threshold_pct) / 2.0
                    recs.append({
                        "symbol": sym,
                        "action": "reduce",
                        "current_pct": float(pct),
                        "suggested_target_pct": suggested_target_pct,
                        "suggested_reduction_pct": float(pct) - suggested_target_pct,
                        "estimated_notional_reduction": (val - (total_value * (suggested_target_pct / 100.0))) if (val is not None and total_value) else None,
                        "reason": f"High concentration ({pct:.2f}% ≥ {exposure_threshold_pct}%)."
                    })
                else:
                    recs.append({
                        "symbol": sym,
                        "action": "ok",
                        "current_pct": float(pct),
                        "reason": "Exposure within threshold."
                    })

            # If top_expos empty and counts exist, propose rebalancing by counts
            if not top_expos:
                # use fallback: propose diversifying if few symbols
                if heat.get("sample_size", 0) > 0:
                    recs.append({
                        "action": "review_counts",
                        "reason": "No explicit USD exposures found; review positions by count and consider diversification."
                    })
                else:
                    recs.append({
                        "action": "no_positions",
                        "reason": "No positions found for this user."
                    })

            result = {
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "portfolio_heat": heat,
                "recommendations": recs,
                "dry_run": bool(dry_run)
            }

            return result

        except Exception:
            logging.exception("run_rebalance_for_user: failed for user %s", user_id)
            return {
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": "internal_error",
                "recommendations": [],
                "dry_run": bool(dry_run)
            }

    async def _rebalance_scheduler_loop(self, poll_interval_seconds: int = 3600):
        """
        Internal async loop that checks scheduled rebalances and triggers reminders.
        To use: await demo_trader._start_rebalance_scheduler() (or call start_rebalance_scheduler()).
        This loop is resilient: it checks self._scheduled_rebalances and calls send_rebalance_reminder
        when next_run <= now. It updates last_run and next_run after triggering.
        """
        try:
            if not hasattr(self, "_scheduled_rebalances"):
                self._scheduled_rebalances = {}

            while True:
                try:
                    now = datetime.utcnow()
                    for uid, meta in list(self._scheduled_rebalances.items()):
                        if not meta.get("enabled", True):
                            continue
                        next_run = meta.get("next_run")
                        if isinstance(next_run, str):
                            # deserialize if string
                            try:
                                next_run = datetime.fromisoformat(next_run)
                            except Exception:
                                next_run = datetime.utcnow()
                        if next_run <= now:
                            # run analysis and send reminder
                            try:
                                res = self.run_rebalance_for_user(uid, exposure_threshold_pct=meta.get("exposure_threshold_pct", 25.0), dry_run=True)
                                # send the reminder via best-effort channel
                                try:
                                    await self.send_rebalance_reminder(uid, res)
                                except TypeError:
                                    # send_rebalance_reminder may be sync; call directly
                                    self.send_rebalance_reminder(uid, res)
                                # update schedule
                                meta["last_run"] = now
                                meta["next_run"] = now + timedelta(days=int(meta.get("interval_days", 7)))
                                self._scheduled_rebalances[uid] = meta
                            except Exception:
                                logging.exception("Rebalance loop: failed to run rebalancer for user %s", uid)
                    await asyncio.sleep(int(poll_interval_seconds))
                except asyncio.CancelledError:
                    logging.info("Rebalance scheduler loop cancelled")
                    return
                except Exception:
                    logging.exception("Rebalance scheduler loop encountered error; continuing")
                    await asyncio.sleep(int(poll_interval_seconds))
        except Exception:
            logging.exception("_rebalance_scheduler_loop: top-level failure")

    def start_rebalance_scheduler(self, poll_interval_seconds: int = 3600):
        """
        Public helper to start the background scheduler (non-blocking).
        Call this once on app startup if you want automatic reminders.
        Returns asyncio.Task or None if it couldn't be started.
        NOTE: Your process must run an asyncio event loop for this to work.
        """
        try:
            loop = asyncio.get_event_loop()
            # ensure only one task
            if hasattr(self, "_rebalance_task") and not self._rebalance_task.done():
                return {"status": "already_running"}
            task = loop.create_task(self._rebalance_scheduler_loop(poll_interval_seconds=poll_interval_seconds))
            self._rebalance_task = task
            return {"status": "started", "task": task}
        except Exception:
            logging.exception("start_rebalance_scheduler failed")
            return {"status": "failed", "reason": "no_event_loop"}

    def stop_rebalance_scheduler(self):
        """
        Stop background scheduler task if running.
        """
        try:
            if hasattr(self, "_rebalance_task"):
                self._rebalance_task.cancel()
                return {"status": "cancelling"}
            return {"status": "not_running"}
        except Exception:
            logging.exception("stop_rebalance_scheduler error")
            return {"status": "error"}

    # -----------------------------
    # ADD this helper in DemoTrader (backtester4.py)
    # Place immediately AFTER the rebalancer methods above
    # -----------------------------
    def send_rebalance_reminder(self, user_id: int, rebal_result: dict):
        """
        Compose a human-readable reminder message from rebal_result and either:
          - send it via self.notify_user(user_id, text) if that helper exists
          - or self.bot.send_message / self.bot.api_send if a bot instance is attached
          - otherwise, return the message dict for external sender to use.
        """
        try:
            # compose text
            lines = []
            lines.append(f"🔔 Portfolio Rebalance Reminder — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}")
            recs = rebal_result.get("recommendations", [])
            if not recs:
                lines.append("No recommendations found. No action required.")
            else:
                lines.append("Recommendations:")
                for r in recs:
                    if r.get("action") == "reduce":
                        lines.append(f"- Reduce {r.get('symbol')}: currently {r.get('current_pct', 0):.2f}%, target {r.get('suggested_target_pct', 0):.2f}% — reason: {r.get('reason')}")
                    elif r.get("action") == "ok":
                        lines.append(f"- {r.get('symbol')}: OK ({r.get('current_pct', 0):.2f}%)")
                    elif r.get("action") == "review":
                        lines.append(f"- {r.get('symbol')}: Review (insufficient data).")
                    else:
                        lines.append(f"- {r.get('action')}: {r.get('reason')}")

            text = "\n".join(lines)

            # Best-effort delivery:
            # 1) prefer notify_user (if your app defines it)
            if hasattr(self, "notify_user") and callable(getattr(self, "notify_user")):
                try:
                    return self.notify_user(user_id, text)
                except Exception:
                    logging.exception("send_rebalance_reminder: notify_user failed")

            # 2) try bot reference if present (common pattern in your repo)
            bot = getattr(self, "bot", None)
            if bot is not None:
                try:
                    # adjust depending on your bot API: common helpers are .send_message or .send
                    if hasattr(bot, "send_message"):
                        return bot.send_message(user_id, text)
                    if hasattr(bot, "send"):
                        return bot.send(user_id, text)
                except Exception:
                    logging.exception("send_rebalance_reminder: bot send failed")

            # 3) fallback: return message dict so caller can send it
            return {"user_id": user_id, "text": text, "payload": rebal_result}

        except Exception:
            logging.exception("send_rebalance_reminder: top-level failure")
            return {"user_id": user_id, "text": "Failed to compose rebalance reminder (see logs)."}


    async def send_scheduled_reports(self) -> None:
        """
        Iterate over user demo folders in data_dir and send the generated report to each user.
        Uses forwarder.bot.send_message(user_id, text) if available; otherwise logs the report text.
        """
        try:
            # find user demo folders: user_{id}_demo
            for entry in os.listdir(self.data_dir):
                if not entry.startswith("user_") or not entry.endswith("_demo"):
                    continue
                try:
                    # extract user id
                    parts = entry.split("_")
                    # expect format: user_<id>_demo
                    if len(parts) < 3:
                        continue
                    user_id_part = parts[1]
                    try:
                        user_id = int(user_id_part)
                    except Exception:
                        continue

                    # generate report
                    report = await self.generate_demo_report_for_user(user_id)
                    text = report.get("text") if isinstance(report, dict) else str(report)

                    # send via bot if available
                    try:
                        if getattr(self, "forwarder", None) and getattr(self.forwarder, "bot", None):
                            await self.forwarder.bot.send_message(user_id, text)
                        else:
                            logging.info("Scheduled Report for user %s:\n%s", user_id, text)
                    except Exception:
                        logging.exception("send_scheduled_reports: failed to send report for user %s", user_id)

                except Exception:
                    logging.exception("send_scheduled_reports: iteration error for entry %s", entry)
        except Exception:
            logging.exception("send_scheduled_reports: top-level error")


    async def _execute_demo_exit(self, trade_obj: Any, decision: Dict[str, Any]) -> None:
        """
        Execute or partially execute an exit for a given trade object.
        - trade_obj: dict or dataclass-like (TradeLog)
        - decision: dict with keys: should_exit/should_close, exit_price, reason, partial (bool), percent (0-100)
        Behavior:
         - update exit fields on trade_obj
         - compute raw_pnl_percent and absolute_pnl where possible
         - persist via self._persist_trade(user_id, trade_obj)
         - remove from self._active_trades if fully closed
        """
        try:
            exit_price = decision.get("exit_price")
            if exit_price is None:
                logging.debug("_execute_demo_exit: no exit_price; skipping")
                return

            partial = bool(decision.get("partial", False))
            percent = float(decision.get("percent", 100.0)) if decision.get("percent") is not None else (50.0 if partial else 100.0)
            percent = max(0.0, min(100.0, percent))
            exit_fraction = percent / 100.0

            # extract entry price and user id
            entry_price = None
            user_id = None
            trade_id = None

            if isinstance(trade_obj, dict):
                entry_price = float(trade_obj.get("entry_price", 0.0) or 0.0)
                user_id = trade_obj.get("user_id", 0)
                trade_id = trade_obj.get("trade_id")
            else:
                entry_price = float(getattr(trade_obj, "entry_price", 0.0) or 0.0)
                user_id = getattr(trade_obj, "user_id", 0)
                trade_id = getattr(trade_obj, "trade_id", None)

            # compute pnl (entry -> exit)
            raw_pnl = None
            try:
                if entry_price and entry_price > 0:
                    raw_pnl = (float(exit_price) - entry_price) / entry_price
            except Exception:
                raw_pnl = None

            # compute closed notional and adjust storing fields (best-effort)
            # We don't have notional in this schema; treat percent as informational and set exit fields.
            now_iso = datetime.utcnow().isoformat()

            if isinstance(trade_obj, dict):
                trade_obj["exit_price"] = float(exit_price)
                trade_obj["exit_time"] = now_iso
                trade_obj["exit_reason"] = str(decision.get("reason", "auto_exit"))
                trade_obj["raw_pnl_percent"] = (raw_pnl * 100.0) if (raw_pnl is not None) else None
                # store partial metadata if partial
                if partial:
                    trade_obj.setdefault("partial_exits", [])
                    trade_obj["partial_exits"].append({"percent": percent, "exit_price": exit_price, "time": now_iso})
            else:
                setattr(trade_obj, "exit_price", float(exit_price))
                setattr(trade_obj, "exit_time", datetime.utcnow())
                setattr(trade_obj, "exit_reason", str(decision.get("reason", "auto_exit")))
                try:
                    setattr(trade_obj, "raw_pnl_percent", raw_pnl * 100.0 if raw_pnl is not None else None)
                except Exception:
                    pass
                if partial:
                    # attach list if not present
                    existing = getattr(trade_obj, "partial_exits", None)
                    if existing is None:
                        try:
                            setattr(trade_obj, "partial_exits", [{"percent": percent, "exit_price": exit_price, "time": now_iso}])
                        except Exception:
                            pass
                    else:
                        try:
                            existing.append({"percent": percent, "exit_price": exit_price, "time": now_iso})
                        except Exception:
                            pass

            # persist updated trade
            try:
                await self._persist_trade(user_id, trade_obj)
            except Exception:
                logging.exception("_execute_demo_exit: persist failed")

            # remove fully closed trades from active list
            if percent >= 99.999:
                try:
                    async with self._lock:
                        # ensure trade_id extraction works for both shapes
                        key = trade_id or (trade_obj.get("trade_id") if isinstance(trade_obj, dict) else getattr(trade_obj, "trade_id", None))
                        if key and key in self._active_trades:
                            self._active_trades.pop(key, None)
                except Exception:
                    logging.exception("_execute_demo_exit: failed to remove from _active_trades")

            logging.info("_execute_demo_exit: executed exit for trade_id=%s user=%s reason=%s pct=%.2f exit_price=%s",
                         trade_id, user_id, decision.get("reason"), percent, exit_price)

        except Exception:
            logging.exception("Unhandled error in _execute_demo_exit")

    async def _notify_demo_exit(self, user_id: int, trade_obj: Any, decision: Dict[str, Any]) -> None:
        """
        Send a short notification about demo exit. Best-effort:
         - If self.forwarder.bot exists (Telethon/pyrogram wrapper) attempt to send message.
         - Otherwise just log.
        """
        try:
            # Build textual summary
            exit_price = decision.get("exit_price")
            reason = decision.get("reason", "exit")
            pct = decision.get("percent", 100.0)
            raw_pnl = None
            if isinstance(trade_obj, dict):
                raw_pnl = trade_obj.get("raw_pnl_percent")
                sym = trade_obj.get("symbol")
            else:
                raw_pnl = getattr(trade_obj, "raw_pnl_percent", None)
                sym = getattr(trade_obj, "symbol", None)

            pnl_text = f"{raw_pnl:.2f}%" if (raw_pnl is not None) else "N/A"
            text = (
                f"Demo Trade Exit\n"
                f"User: {user_id}\n"
                f"Symbol: {sym}\n"
                f"Exit Price: {exit_price}\n"
                f"Percent Closed: {pct}\n"
                f"PnL: {pnl_text}\n"
                f"Reason: {reason}\n"
            )

            # Try to send via forwarder.bot (if available)
            try:
                if getattr(self, "forwarder", None) and getattr(self.forwarder, "bot", None):
                    # many bot wrappers expect int chat_id or username; here we use user_id
                    await self.forwarder.bot.send_message(user_id, text)
                    return
            except Exception:
                logging.debug("_notify_demo_exit: bot.send_message failed; falling back to log")

            logging.info("_notify_demo_exit: %s", text)
        except Exception:
            logging.exception("Unhandled error in _notify_demo_exit")

# ---------- END: DemoTrader (returns real TradeLog objects) ----------

# -----------------------------
# Position sizing utilities
# Place this block AFTER class PerformanceAnalytics (in backtester4.py)
# -----------------------------
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics

@dataclass
class PositionSizingResult:
    # Final recommendations (fractions, e.g., 0.02 == 2%)
    recommended_size_safest: float
    recommended_size_blended: float

    # Individual model outputs (fractions)
    kelly_fraction: float
    volatility_target_size: float
    ev_proportional_size: float

    # Supporting fields
    confidence: str                 # "low" | "medium" | "high"
    sample_size: int
    expected_value_percent: float   # percent (e.g., 2.5 means 2.5%)
    volatility_percent: float       # percent (e.g., 12.5 means 12.5%)

    def to_currency(self, portfolio_value: float, use_blended: bool = True) -> float:
        """Convert recommended fraction to currency amount. Default uses blended recommendation."""
        frac = self.recommended_size_blended if use_blended else self.recommended_size_safest
        return float(portfolio_value) * float(frac)

class PositionSizingCalculator:
    """
    Multi-model position sizing calculator.
    Methods accept historical trade records (dicts or objects with .raw_pnl_percent)
    and a performance_settings dict (with min/max position size, ev_proportional_k, risk_tolerance).
    Returns PositionSizingResult with both blended and safest recommendations.
    """

    @staticmethod
    def _get_raw_pnl_percent(trade: Any) -> Optional[float]:
        """Support dict or object shaped trades. Return percent (e.g., 12.5) or None."""
        try:
            if isinstance(trade, dict):
                return float(trade.get("raw_pnl_percent") or trade.get("pnl_percent") or trade.get("raw_return_percent"))
            else:
                return float(getattr(trade, "raw_pnl_percent", None) or getattr(trade, "pnl_percent", None) or getattr(trade, "raw_return_percent", None))
        except Exception:
            return None

    @staticmethod
    def _calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Raw Kelly fraction. Inputs: win_rate (0-1), avg_win (fraction), avg_loss (fraction)."""
        try:
            if avg_win == 0:
                return 0.0
            if avg_loss == 0:
                return 1.0
            win_loss_ratio = avg_win / avg_loss
            kelly_fraction = win_rate - ((1.0 - win_rate) / win_loss_ratio)
            return float(kelly_fraction)
        except Exception:
            return 0.0

    @staticmethod
    def calculate_position_size(trades: List[Any], performance_settings: Dict[str, Any]) -> PositionSizingResult:
        """
        Core entry point.
        trades: list of trade objects or dicts (must include raw_pnl_percent or equivalent).
        performance_settings: dict, expected keys:
           - min_position_size (fraction, default 0.005)
           - max_position_size (fraction, default 0.10)
           - ev_proportional_k (float, default 1.0)
           - risk_tolerance ('conservative'|'moderate'|'aggressive', default 'moderate')
        Returns PositionSizingResult where sizes are fractions (0.02 == 2%).
        """
        # --- load settings ---
        min_size = float(performance_settings.get("min_position_size", 0.005))
        max_size = float(performance_settings.get("max_position_size", 0.10))
        k_value = float(performance_settings.get("ev_proportional_k", 1.0))
        risk_tolerance = performance_settings.get("risk_tolerance", "moderate")

        # --- normalize trade returns into fractional returns (e.g., 0.12 for 12%) ---
        raw_pct = [PositionSizingCalculator._get_raw_pnl_percent(t) for t in trades]
        completed = [p for p in raw_pct if p is not None]
        sample_size = len(completed)

        if sample_size == 0:
            # fallback conservative defaults
            return PositionSizingResult(
                recommended_size_safest=min_size,
                recommended_size_blended=min_size,
                kelly_fraction=0.0,
                volatility_target_size=min_size,
                ev_proportional_size=0.0,
                confidence="low",
                sample_size=0,
                expected_value_percent=0.0,
                volatility_percent=0.0
            )

        # convert percent to fraction for calculations (e.g., 12.5 -> 0.125)
        returns_frac = [p / 100.0 for p in completed]

        # basic metrics
        win_count = len([r for r in returns_frac if r > 0])
        win_rate = float(win_count) / max(1, sample_size)
        wins = [r for r in returns_frac if r > 0]
        losses = [r for r in returns_frac if r < 0]

        avg_win = float(statistics.mean(wins)) if wins else 0.0
        avg_loss = float(abs(statistics.mean(losses))) if losses else 0.0
        expected_value = (win_rate * (avg_win * 100.0)) - ((1.0 - win_rate) * (avg_loss * 100.0))  # percent form
        volatility = float(statistics.stdev(returns_frac)) if len(returns_frac) > 1 else 0.0

        # --- Model A: Half-Kelly (growth-focused) ---
        kelly_fraction = PositionSizingCalculator._calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        safe_kelly_size = max(0.0, kelly_fraction / 2.0)

        # --- Model B: Volatility-targeting (risk-focused) ---
        risk_multipliers = {"conservative": 0.5, "moderate": 1.0, "aggressive": 1.5}
        multiplier = risk_multipliers.get(risk_tolerance, 1.0)
        target_risk_per_trade = 0.02 * multiplier  # default 2% target risk scaled by tolerance
        # Avoid division by zero: use a small floor for volatility
        vol_floor = max(1e-6, volatility)
        volatility_target_size = target_risk_per_trade / (2.0 * vol_floor) if vol_floor > 0 else min_size

        # --- Model C: EV-proportional (edge-focused) ---
        # ev_frac is fractional expected value per trade (e.g., 0.025 for 2.5%)
        ev_frac = max(-1.0, expected_value / 100.0)
        ev_proportional_size = (k_value * ev_frac / (avg_loss if avg_loss > 0 else 1.0)) if avg_loss > 0 else 0.0

        # clean negative values
        model_results = [max(0.0, safe_kelly_size), max(0.0, volatility_target_size), max(0.0, ev_proportional_size)]

        # --- Compose final recommendations ---
        recommended_size_safest = min(model_results) if model_results else min_size
        recommended_size_blended = statistics.mean(model_results) if model_results else min_size

        # clamp to user's allowed range
        recommended_size_safest = max(min_size, min(max_size, recommended_size_safest))
        recommended_size_blended = max(min_size, min(max_size, recommended_size_blended))

        # confidence heuristic
        if sample_size >= 50 and volatility < 0.15:
            confidence = "high"
        elif sample_size >= 20:
            confidence = "medium"
        else:
            confidence = "low"

        return PositionSizingResult(
            recommended_size_safest=float(recommended_size_safest),
            recommended_size_blended=float(recommended_size_blended),
            kelly_fraction=float(kelly_fraction),
            volatility_target_size=float(volatility_target_size),
            ev_proportional_size=float(ev_proportional_size),
            confidence=confidence,
            sample_size=sample_size,
            expected_value_percent=float(expected_value),
            volatility_percent=float(volatility * 100.0)
        )

class DemoTrader:
    """
    Demo (paper) trading engine that returns TradeLog instances when possible.
    - data_dir: where demo trades are persisted
    - crypto_manager_factory(user_id) -> crypto manager (encrypt/decrypt)
    - market_adapter: adapter with async get_current_price(symbol) -> {'price': float} or float
    - scheduler_interval_seconds: how often the internal scheduler checks exits (not used for process_live_signal)
    """
    def __init__(
        self,
        data_dir: str,
        crypto_manager_factory: Callable[[int], Any],
        market_adapter: Optional[Any] = None,
        scheduler_interval_seconds: int = 5,
        reporting_interval_seconds: Optional[int] = None,
        trade_factory: Optional[Callable[..., Any]] = None,
        max_active_trades_per_user: Optional[int] = None,
        bot: Optional[Any] = None,
    ):
        """
        DemoTrader constructor (updated for scheduled reporting).
        - reporting_interval_seconds: how often to generate/send reports (seconds). If None, uses REPORTING_INTERVAL_SECONDS_DEFAULT.
        - All other params unchanged from previous design.
        """
        self.data_dir = data_dir
        self.crypto_manager_factory = crypto_manager_factory
        self.market_adapter = market_adapter
        self.scheduler_interval_seconds = scheduler_interval_seconds
        # reporting
        self.reporting_interval_seconds = int(reporting_interval_seconds or REPORTING_INTERVAL_SECONDS_DEFAULT)
        self._report_task: Optional[asyncio.Task] = None

        # optional explicit factory to create TradeLog objects; if None, DemoTrader will try to instantiate TradeLog or fallback
        self.trade_factory = trade_factory
        self.bot = bot # Save bot reference for ErrorHandler fallback (may be None) 
        # Resource limiting
        self.max_active_trades_per_user = int(max_active_trades_per_user or DEMO_MAX_ACTIVE_TRADES_PER_USER)

        # runtime state
        self._active_trades: Dict[str, Any] = {} # trade_id -> trade_obj
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # forwarder may be injected by main_bot; used for message sending
        self.forwarder = None

    async def _count_active_trades_for_user(self, user_id: int) -> int:
        """
        Return the number of currently active demo trades for the given user.
        Uses the demo's lock to take a consistent snapshot.
        """
        async with self._lock:
            count = 0
            for tobj in self._active_trades.values():
                try:
                    uid = tobj.get("user_id") if isinstance(tobj, dict) else getattr(tobj, "user_id", None)
                    if uid == user_id:
                        count += 1
                except Exception:
                    # be defensive; skip malformed entries
                    continue
            return count


    def set_max_active_trades_per_user(self, n: int):
        """
        Runtime setter for per-user active trade limit. Non-async since it's a simple assignment.
        """
        try:
            self.max_active_trades_per_user = int(n)
            logging.info("DemoTrader: max_active_trades_per_user set to %d", self.max_active_trades_per_user)
        except Exception:
            logging.exception("DemoTrader: failed to set max_active_trades_per_user")


    # ---------------------
    # Internal helpers
    # ---------------------
    def _user_demo_folder(self, user_id: int) -> str:
        folder = os.path.join(self.data_dir, f"user_{user_id}_demo")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _trade_path(self, user_id: int, trade_id: str) -> str:
        return os.path.join(self._user_demo_folder(user_id), f"demo_trade_{trade_id}.json")

    async def _persist_trade(self, user_id: int, trade_obj: Any):
        try:
            # convert dataclass -> dict if needed
            if hasattr(trade_obj, "__dataclass_fields__"):
                payload = asdict(trade_obj)
            elif hasattr(trade_obj, "__dict__") and not isinstance(trade_obj, dict):
                payload = dict(trade_obj.__dict__)
            else:
                payload = dict(trade_obj)

            text = json.dumps(payload, default=str)
            crypto = self.crypto_manager_factory(user_id) if callable(self.crypto_manager_factory) else self.crypto_manager_factory
            ciphertext = crypto.encrypt(text)
            path = self._trade_path(user_id, payload.get("trade_id") or str(uuid.uuid4()))
            with open(path, "wb") as f:
                f.write(ciphertext)
        except Exception:
            logging.exception("DemoTrader._persist_trade failed")

    async def _load_persisted_trades_for_user(self, user_id: int) -> Dict[str, Any]:
        ret: Dict[str, Any] = {}
        folder = self._user_demo_folder(user_id)
        if not os.path.exists(folder):
            return ret
        for fn in os.listdir(folder):
            if not fn.startswith("demo_trade_") or not fn.endswith(".json"):
                continue
            path = os.path.join(folder, fn)
            try:
                crypto = self.crypto_manager_factory(user_id) if callable(self.crypto_manager_factory) else self.crypto_manager_factory
                raw = open(path, "rb").read()
                text = crypto.decrypt(raw)
                obj = json.loads(text)
                tid = obj.get("trade_id") or fn.split("demo_trade_")[-1].rsplit(".json", 1)[0]
                ret[tid] = obj
            except Exception:
                logging.exception("DemoTrader failed to load persisted trade %s", path)
        return ret

    # ---------------------
    # Lifecycle
    # ---------------------
    async def start(self):
        """Start the scheduler loop that evaluates exits and the reporting loop."""
        async with self._lock:
            if self._running:
                return
            self._running = True
            # main scheduler loop (exit checks)
            self._task = asyncio.create_task(self._scheduler_loop())
            # reporting loop (periodic)
            try:
                # create the reporting loop task only once
                if not getattr(self, "_report_task", None):
                    self._report_task = asyncio.create_task(self._reporting_loop())
            except Exception:
                logging.exception("DemoTrader.start: failed to create reporting task")
            logging.info("DemoTrader started (scheduler + reporting)")

    async def stop(self):
        """Stop the scheduler and reporting tasks and wait for them to finish."""
        async with self._lock:
            if not self._running:
                return
            self._running = False

            # cancel scheduler task
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None

            # cancel reporting task
            if getattr(self, "_report_task", None):
                try:
                    self._report_task.cancel()
                    await self._report_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logging.exception("DemoTrader.stop: error while awaiting _report_task")
                finally:
                    self._report_task = None

            logging.info("DemoTrader stopped")


    async def _scheduler_loop(self):
        """
        Main background loop: delegate to check_demo_exits() each tick.
        This keeps the loop simple and centralizes exit logic in check_demo_exits.
        """
        try:
            while self._running:
                try:
                    await self.check_demo_exits()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logging.exception("DemoTrader._scheduler_loop: check_demo_exits failed")
                await asyncio.sleep(self.scheduler_interval_seconds)
        except asyncio.CancelledError:
            logging.info("DemoTrader scheduler cancelled")
        except Exception:
            logging.exception("DemoTrader scheduler fatal error")

# ---------------------------------------------------------------------
# Adapter: convert parsed exit-rules (from parse_exit_rules) into
# engine-friendly "spec" dicts and ExitRule dataclass instances.
# Place/append this block to the end of backtester2.py
# ---------------------------------------------------------------------
from typing import List, Dict, Any, Optional

# import the rule converter already present in exit_rules.py
try:
    from exit_rules import parse_rule_spec, ExitRule
except Exception:
    # fallback to module import name differences
    from .exit_rules import parse_rule_spec, ExitRule  # type: ignore

def convert_parsed_rules_to_specs(parsed_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert the output of parse_exit_rules(...) into the 'spec' dicts expected by exit_rules.parse_rule_spec.
    - parsed_rules: list of dicts produced by parse_exit_rules (keys like 'type','value','unit','percent','multiplier','exit_multiple').
    Returns: list of spec dicts suitable for parse_rule_spec(spec).
    Unknown/unsupported parsed rule types are returned as {'type': 'raw', 'text': <...>} so callers can surface errors.
    """
    specs: List[Dict[str, Any]] = []
    for r in parsed_rules or []:
        typ = r.get("type")
        if not typ:
            specs.append({"type": "raw", "text": str(r)})
            continue

        # TAKE PROFIT
        if typ == "take_profit":
            unit = r.get("unit", "multiple")
            if unit == "multiple":
                # Keep multiple-mode true so engine can interpret
                specs.append({"type": "take_profit", "value": float(r.get("value", 0)), "multiple_mode": True})
            else:
                specs.append({"type": "take_profit", "value": float(r.get("value", 0)), "multiple_mode": False})
            continue

        # STOP LOSS
        if typ == "stop_loss":
            specs.append({"type": "stop_loss", "value": float(r.get("value", 0))})
            continue

        # ATR trailing
        if typ in ("atr_trailing", "atr"):
            specs.append({"type": "atr_trailing", "multiplier": float(r.get("multiplier", r.get("value", 1.0)))})
            continue

        # Partial exits (map to percent_of_portfolio advice)
        if typ == "partial":
            pct = float(r.get("percent") or r.get("size_pct") or 0.0)
            exit_multiple = r.get("exit_multiple")
            # convert 2x -> profit percent 100%
            if exit_multiple is not None:
                try:
                    exit_multiple_f = float(exit_multiple)
                    profit_pct_threshold = max(0.0, (exit_multiple_f - 1.0) * 100.0)
                except Exception:
                    profit_pct_threshold = 0.0
            else:
                profit_pct_threshold = float(r.get("value", 0.0))
            specs.append({
                "type": "percent_of_portfolio",
                "size_pct": float(pct),
                "value": float(profit_pct_threshold),
                "close_position": False
            })
            continue

        # time-decay take profit (if user provided such tokens; parse_exit_rules may not produce these yet)
        if typ in ("time_decay_take_profit",):
            specs.append({
                "type": "time_decay_take_profit",
                "initial_pct": r.get("initial_pct"),
                "final_pct": r.get("final_pct"),
                "decay_seconds": r.get("decay_seconds")
            })
            continue

        # percent-only like "200%" -> treat as take_profit percent
        if typ == "raw" and isinstance(r.get("text"), str):
            # try a last-ditch parse: if it's like "200%" map to take_profit percent
            text = r.get("text", "").strip()
            if text.endswith("%"):
                try:
                    v = float(text.rstrip("%").strip())
                    specs.append({"type": "take_profit", "value": v, "multiple_mode": False})
                    continue
                except Exception:
                    pass
        # fallback - keep raw so caller can handle
        specs.append({"type": "raw", "text": r})

    return specs


def convert_parsed_rules_to_exitrules(parsed_rules: List[Dict[str, Any]]) -> List[ExitRule]:
    """
    Convert parsed_rules -> spec dicts -> ExitRule dataclass instances using exit_rules.parse_rule_spec.
    Returns only ExitRule instances for supported spec types. Unknown types ('raw') are skipped.
    """
    specs = convert_parsed_rules_to_specs(parsed_rules)
    exit_rules: List[ExitRule] = []
    for s in specs:
        if not s:
            continue
        if s.get("type") == "raw":
            # skip raw entries: the caller/UI should surface these as invalid
            continue
        try:
            # parse_rule_spec will validate and construct ExitRule
            er = parse_rule_spec(s)
            exit_rules.append(er)
        except Exception:
            # If parse_rule_spec fails (invalid spec), skip it — caller should handle empties/errors.
            continue
    return exit_rules

# ------------------------
# Add to backtester2.py (after convert_parsed_rules_to_exitrules)
# Function name: convert_parsed_rules_to_exitrules_strict
# ------------------------
def convert_parsed_rules_to_exitrules_strict(parsed_rules):
    """
    Strict converter: if any parsed rule is unknown/raw or cannot be converted to an ExitRule,
    raise ValueError with a helpful message. Otherwise, return list[ExitRule].
    """
    # detect raw tokens early
    raws = [r for r in (parsed_rules or []) if r.get("type") == "raw"]
    if raws:
        texts = [r.get("text", str(r)) for r in raws]
        raise ValueError(f"Unrecognized exit rule tokens: {texts}. Please rephrase (examples: 'tp 2x', 'sl 10%').")

    # attempt conversion; if conversion fails for any spec, raise
    specs = convert_parsed_rules_to_specs(parsed_rules)
    exit_rules = []
    for s in specs:
        try:
            er = parse_rule_spec(s)
            exit_rules.append(er)
        except Exception as exc:
            raise ValueError(f"Failed to convert rule spec {s}: {exc}") from exc

    return exit_rules




