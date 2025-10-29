# @title FINAL WORKING VERSION OF TELEGRAM BOT MONOLITH VERSION
import os
import uvicorn
import time
import datetime
import asyncio
import nest_asyncio
import json
import fcntl
import logging
import re
import base64
import sys
import requests
import threading
import hmac
import hashlib
import uuid
import random
import string
import traceback
import inspect
import secrets
from telethon import TelegramClient, errors, events, Button
from telethon.tl.types import User
from telethon.errors import (SessionPasswordNeededError, PhoneCodeInvalidError,
                           ApiIdInvalidError, PhoneNumberInvalidError)

from pathlib import Path
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account
# --- START: Subscription System Imports & Config ---
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from starlette.responses import JSONResponse
from decimal import Decimal, InvalidOperation
from web3 import Web3
from typing import Optional, Tuple
# edit these


app = FastAPI()
# --- Environment-aware path configuration ---
def get_base_path():
    """Get base path based on environment"""
    env_path = os.getenv('TELEGRAM_BOT_BASE_PATH')
    if env_path:
        return env_path

    # Auto-detect environment
    if os.path.exists('/content/drive/MyDrive'):
        # Google Colab environment
        return '/content/drive/MyDrive/telegram_forwarder'
    else:
        # VPS/Local environment
        return '.'

# --- Configuration (with session_fix) ---
BASE_PATH = get_base_path()
CRYPTO_DIR = os.path.join(BASE_PATH, "crypto")
DATA_DIR = os.path.join(BASE_PATH, "data")
SESSIONS_DIR = os.path.join(BASE_PATH, "sessions")  # ADDED FROM session_fix.txt
os.makedirs(CRYPTO_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)  # ADDED FROM session_fix.txt

# MASTER CONTROL PANEL
# Set to True to enable all subscription checks, limits, and payment commands.
# Set to False to make all features available to all users for free.
SUBSCRIPTION_ENABLED = True

# PAYMENT PROVIDER API KEYS & MODE
# It is strongly recommended to load these from environment variables for security.
PAYSTACK_TEST_SECRET_KEY = os.getenv("PAYSTACK_TEST_SECRET_KEY")
PAYSTACK_LIVE_SECRET_KEY = os.getenv("PAYSTACK_LIVE_SECRET_KEY")
NOWPAYMENTS_API_KEY = os.getenv("NOWPAYMENTS_API_KEY")
NOWPAYMENTS_IPN_SECRET = os.getenv("NOWPAYMENTS_IPN_SECRET")

# Safe boolean parsing — prefers explicit env setting (default False)
def _parse_bool_env(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

PAYSTACK_TEST_MODE = _parse_bool_env(os.getenv("PAYSTACK_TEST_MODE"), default=False)


# Path for storing payments (uses existing DATA_DIR variable in the monolith)
PAYMENTS_STORE_PATH = os.path.join(DATA_DIR, "payments.json")
USER_PROFILES_PATH = os.getenv("USER_PROFILES_PATH", "data/user_profiles.json")
# challenge TTL seconds (10 minutes default)
LINK_CHALLENGE_TTL = int(os.getenv("LINK_CHALLENGE_TTL", "600"))
APP_NAME = os.getenv("APP_NAME", "TelegramForwarderBot")
TRACK_POLL_INTERVAL = int(os.getenv("TRACK_POLL_INTERVAL", "30"))  # seconds
# PRICING CONFIGURATION (in USD)
PRICES = {
    'monthly_usd': 13.0,
    'yearly_usd': 132.0,
}

# AFFILIATE COMMISSION RATES (System 1 - in USD)
COMMISSIONS = {
    'first_month_bonus': 3.0,
    'subsequent_month': 1.0,
    'yearly_new_subscriber': 14.0,
    'yearly_upgrade': 12.0,
    'minimum_payout_threshold': 10.0,
}

# WEBHOOK SERVER CONFIGURATION

# This must be your server's public IP or a domain pointing to it.
# For local testing, use a service like ngrok to get a temporary public URL.
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL")  # required in production; validate at startup
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8001"))


# FEATURE AND USAGE LIMITS FOR FREE VS. PAID USERS
# Use float('inf') to represent "unlimited".

USER_LIMITS = {
    'free': {
        'max_jobs': 5,
        'max_custom_pattern_jobs': 1,
        'max_sources_per_job': 2,
        'max_destinations_per_job': 1,
        'max_keywords_per_job': 3,
        'max_patterns_per_job': 3,
        'can_use_and_logic': False,
        # Enforces a minimum cooldown on forwarding jobs for free users.
        # '5' means any timer they set cannot be less than 5 minutes.
        'min_timer_minutes': 5,
    },
    'paid': {
        'max_jobs': float('inf'),
        'max_custom_pattern_jobs': float('inf'),
        'max_sources_per_job': float('inf'),
        'max_destinations_per_job': float('inf'),
        'max_keywords_per_job': float('inf'),
        'max_patterns_per_job': float('inf'),
        'can_use_and_logic': True,
        # Paid users have no minimum cooldown; they can set it to 0 for instant forwards.
        'min_timer_minutes': 0,
    }
}
# --- END: Subscription System Imports & Config ---

# CoinGecko base URL (public API). Can set COINGECKO_API_URL env to override.
COINGECKO_API_URL = os.getenv("COINGECKO_API_URL", "https://api.coingecko.com/api/v3")

# Map chain_id -> CoinGecko "platform id" for token price by contract.
# Defaults assume Base uses platform slug "base". If this is wrong for your environment
# override using COINGECKO_PLATFORM_BY_CHAIN env as JSON, e.g.
# COINGECKO_PLATFORM_BY_CHAIN='{"84532":"base","8453":"base","1":"ethereum","137":"polygon-pos"}'
_DEFAULT_CG_PLATFORM = {"84532": "base", "8453": "base", "1": "ethereum", "137": "polygon-pos"}
try:
    _env_map = os.getenv("COINGECKO_PLATFORM_BY_CHAIN")
    if _env_map:
        COINGECKO_PLATFORM_BY_CHAIN = json.loads(_env_map)
    else:
        COINGECKO_PLATFORM_BY_CHAIN = _DEFAULT_CG_PLATFORM
except Exception:
    COINGECKO_PLATFORM_BY_CHAIN = _DEFAULT_CG_PLATFORM

# Chain alias map (accept many common names)
_CHAIN_ALIAS = {
    "base": 8453,            # mainnet canonical
    "base-mainnet": 8453,
    "base-main": 8453,
    "base-sepolia": 84532,   # testnet
    "base-sep": 84532,
    "base-test": 84532,
    "sepolia-base": 84532,
    "ethereum": 1,
    "eth": 1,
    "polygon": 137,
    "matic": 137
}

# --- Enhanced logging configuration with timestamps ---
logging.basicConfig(
    format='[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'  # Add explicit timestamp format
)
nest_asyncio.apply()

# --- Load configuration from environment ---
def load_configuration():
    """Load configuration from environment variables"""
    # Try to load from .env file in base path
    env_file = os.path.join(BASE_PATH, 'creds.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        # Fallback to system environment
        load_dotenv()
        
    def parse_bool(val, default=False):
        if val is None:
            return default
        v = str(val).strip().lower()
        return v in ("1", "true", "yes", "y", "on")

    # Required environment variables
    MASTER_SECRET = os.getenv('MASTER_SECRET')
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    BOT_API_ID = os.getenv('BOT_API_ID')
    BOT_API_HASH = os.getenv('BOT_API_HASH')

    # Validation
    if not MASTER_SECRET:
        raise ValueError("MASTER_SECRET environment variable is required")
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN environment variable is required")
    if not BOT_API_ID:
        raise ValueError("BOT_API_ID environment variable is required")
    if not BOT_API_HASH:
        raise ValueError("BOT_API_HASH environment variable is required")

    try:
        BOT_API_ID = int(BOT_API_ID)
    except ValueError:
        raise ValueError("BOT_API_ID must be a valid integer")

    # ADDED: Demo mode specific variables (only if demo mode is on)
    DEMO_MODE = parse_bool(os.getenv("DEMO_MODE"), default=False)
    DEMO_API_ID = os.getenv("DEMO_API_ID")
    DEMO_API_HASH = os.getenv("DEMO_API_HASH")
    DEMO_PHONE_NUMBER = os.getenv("DEMO_PHONE_NUMBER")
    DEMO_SESSION_NAME = os.getenv("DEMO_SESSION_NAME", "demo_session")

        # Validate demo vars
        if not all([DEMO_API_ID, DEMO_API_HASH, DEMO_PHONE_NUMBER]):
            raise ValueError("DEMO_MODE is true, but DEMO_API_ID, DEMO_API_HASH, or DEMO_PHONE_NUMBER are missing.")
        try:
            DEMO_API_ID = int(DEMO_API_ID)
        except ValueError:
            raise ValueError("DEMO_API_ID must be a valid integer")

    return (MASTER_SECRET, BOT_TOKEN, BOT_API_ID, BOT_API_HASH, DEMO_MODE,
            DEMO_API_ID, DEMO_API_HASH, DEMO_PHONE_NUMBER, DEMO_SESSION_NAME)


# Load configuration
(MASTER_SECRET, BOT_TOKEN, BOT_API_ID, BOT_API_HASH, DEMO_MODE,
 DEMO_API_ID, DEMO_API_HASH, DEMO_PHONE_NUMBER, DEMO_SESSION_NAME) = load_configuration()


# --- SECURITY: Master secret-based encryption with per-user salts ---
class CryptoManager:
    """Handles encryption and decryption using master secret with per-user derivation."""
    def __init__(self, user_id: int):
        self.user_id = str(user_id)
        self.salt_path = Path(CRYPTO_DIR) / f"salt_{self.user_id}.key"
        self.salt = self._get_or_create_salt()
        self.key = self._derive_key()

    def _get_or_create_salt(self) -> bytes:
        if self.salt_path.exists():
            return self.salt_path.read_bytes()
        salt = os.urandom(16)
        self.salt_path.write_bytes(salt)
        logging.info(f"🔐 New salt file created for user {self.user_id}")
        return salt

    def _derive_key(self, iterations: int = 390_000) -> bytes:
        pwd = f"{MASTER_SECRET}-{self.user_id}".encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=iterations,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(pwd))

    def encrypt(self, data: str) -> bytes:
        f = Fernet(self.key)
        return f.encrypt(data.encode())

    def decrypt(self, encrypted_data: bytes) -> str:
        f = Fernet(self.key)
        try:
            return f.decrypt(encrypted_data).decode()
        except InvalidToken:
            raise ValueError("Decryption failed: invalid token or key.")
            
    # --- FastAPI webhook route handler for Coinbase Commerce ---
    @app.post("/api/payments/coinbase/webhook")
    async def coinbase_webhook(request: Request):
        """
        FastAPI POST endpoint that accepts Coinbase Commerce webhooks (raw bytes + signature header),
        verifies and processes them by calling coinbase_handle_webhook(...).

        Expects the Coinbase signature header (try common header names). Returns 200/202 quickly.
        """
        # read raw request body (required for signature verification)
        try:
            body_bytes = await request.body()
        except Exception as e:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"could not read request body: {e}"})

        # Coinbase signature header names vary in docs; accept common variants
        signature_header = None
        hdrs = request.headers
        for h in ("X-CC-Webhook-Signature", "X-CC-Webhook-Sig", "X-Cc-Webhook-Signature", "X-CC-Signature"):
            if h in hdrs:
                signature_header = hdrs[h]
                break
        # fallback to any header that contains 'coinbase' or 'cc' and 'sig'
        if not signature_header:
            for k, v in hdrs.items():
                kl = k.lower()
                if ("coinbase" in kl or "cc" in kl) and ("sig" in kl or "signature" in kl):
                    signature_header = v
                    break

        # Resolve send_message and admin notify callables from module-level bot_instance if available
        send_message_callable = None
        admin_notify_callable = None
        try:
            # if make_send_message_callable and bot_instance are available in module scope, use them
            send_message_callable = make_send_message_callable(bot_instance)  # noqa: F821
        except Exception:
            # leave None if not resolvable; coinbase_handle_webhook tolerates None
            send_message_callable = None

        try:
            admin_notify_callable = make_admin_notify_callable(bot_instance)  # noqa: F821
        except Exception:
            admin_notify_callable = None

        # call the core handler you already added in the monolith
        try:
            result = coinbase_handle_webhook(body_bytes, signature_header, None, send_message_callable=send_message_callable, admin_notify_callable=admin_notify_callable) # noqa: F821
        except Exception as e:
            # surface minimal info and return 500 so Coinbase can retry if needed
            try:
                if admin_notify_callable:
                    admin_notify_callable(f"coinbase_webhook handler exception: {e}")
            except Exception:
                pass
            return JSONResponse(status_code=500, content={"ok": False, "error": f"handler exception: {e}"})

        # Successful processing -> return 200
        return JSONResponse(status_code=200, content={"ok": True, "result": result})

# MODULE LEVEL            
def init_payments_db(_db_conn=None):
    """
    Backwards-compatible replacement for DB init.
    Ensures payments file exists and is initialized (idempotent).
    This keeps the old function name to avoid refactoring call sites.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PAYMENTS_STORE_PATH):
        empty = {"payments": [], "webhook_events": []}
        # atomic write with locking
        with open(PAYMENTS_STORE_PATH, "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(empty, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass
    return True
            
def _load_payments_store():
    """Return payments store dict (payments, webhook_events). Uses file lock for safety."""
    if not os.path.exists(PAYMENTS_STORE_PATH):
        init_payments_db()
    with open(PAYMENTS_STORE_PATH, "r") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except Exception:
                pass
    if not isinstance(data, dict):
        data = {"payments": [], "webhook_events": []}
    # normalize
    data.setdefault("payments", [])
    data.setdefault("webhook_events", [])
    return data

def _save_payments_store(data):
    """Save the payments store dict atomically with exclusive lock."""
    tmp_path = PAYMENTS_STORE_PATH + ".tmp"
    with open(tmp_path, "w") as tf:
        try:
            fcntl.flock(tf, fcntl.LOCK_EX)
            json.dump(data, tf, indent=2, default=str)
            tf.flush()
            os.fsync(tf.fileno())
        finally:
            try:
                fcntl.flock(tf, fcntl.LOCK_UN)
            except Exception:
                pass
    # Replace original file
    os.replace(tmp_path, PAYMENTS_STORE_PATH)
    
def _create_payment_record(record):
    """
    Insert a payment record dict into payments store and return the new record with id.
    """
    store = _load_payments_store()
    # create a stable id
    record_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow().isoformat() + "Z"
    record_full = {
        "id": record_id,
        "merchant_payment_id": record.get("merchant_payment_id"),
        "user_id": int(record.get("user_id")) if record.get("user_id") is not None else None,
        "amount_usd": float(record.get("amount_usd")) if record.get("amount_usd") is not None else None,
        "amount_token": float(record.get("amount_token")) if record.get("amount_token") is not None else None,
        "token_symbol": record.get("token_symbol", "USDC"),
        "token_chain": record.get("token_chain", "base"),
        "status": record.get("status", "created"),
        "hosted_url": record.get("hosted_url"),
        "onchain_tx_hash": record.get("onchain_tx_hash"),
        "reference": record.get("reference"),
        "metadata": record.get("metadata") or {},
        "created_at": record.get("created_at") or now,
        "updated_at": now
    }
    store["payments"].append(record_full)
    _save_payments_store(store)
    return record_full


def _update_payment_record_by_id(pid, updates: dict):
    """Update payment record by id, return updated record or None."""
    store = _load_payments_store()
    for p in store["payments"]:
        if p.get("id") == pid or p.get("reference") == pid or p.get("merchant_payment_id") == pid:
            p.update(updates)
            p["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            _save_payments_store(store)
            return p
    return None


def _find_payment_by_reference(reference):
    store = _load_payments_store()
    for p in store["payments"]:
        if p.get("reference") == reference:
            return p
    return None


def _find_payment_by_merchant_id(merchant_id):
    store = _load_payments_store()
    for p in store["payments"]:
        if p.get("merchant_payment_id") == merchant_id:
            return p
    return None

def _append_webhook_event(event_record):
    store = _load_payments_store()
    evt = {
        "id": str(uuid.uuid4()),
        "payment_id": event_record.get("payment_id"),
        "event_type": event_record.get("event_type"),
        "payload": event_record.get("payload"),
        "received_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    store["webhook_events"].append(evt)
    _save_payments_store(store)
    return evt
    
def handle_pay_command(user_id, amount_usd, db_conn, bot_instance):
    """
    High-level entry to start a payment flow for a user for `amount_usd`.
    This function will attempt the recommended Coinbase Commerce flow first and
    also send a direct/manual fallback instruction so users can pay directly if needed.

    Args:
        user_id: Telegram user id (int).
        amount_usd: numeric (float/Decimal-compatible).
        db_conn: psycopg2-like DB connection.
        bot_instance: your bot object (used to build send_message/admin callers).

    Returns:
        dict: {"ok": True, "coinbase": {...}, "direct": {...}} or {"ok": False, "error": "..."}
    """
    send_fn = make_send_message_callable(bot_instance)
    admin_notify = make_admin_notify_callable(bot_instance)

    # 1) try to create a Coinbase checkout (recommended)
    try:
        cb_result = coinbase_create_checkout(db_conn, user_id, amount_usd, currency="USDC", chain="base", metadata=None)
    except Exception as e:
        cb_result = {"ok": False, "error": f"exception: {e}\n{traceback.format_exc()}"}

    # If coinbase checkout created, send hosted URL to user
    try:
        if cb_result.get("ok"):
            charge = cb_result.get("charge") or {}
            hosted = charge.get("hosted_url") or (charge.get("data") or {}).get("hosted_url") or cb_result.get("charge", {}).get("hosted_url")
            reference = cb_result.get("reference")
            payment_id = cb_result.get("payment_id")
            msg = (
                f"✅ Payment initialized for *{amount_usd} USD*.\n\n"
                f"To complete payment, open this secure Coinbase Commerce page and follow the instructions (choose Base network / USDC):\n\n{hosted}\n\n"
                f"If you prefer to pay directly (manual transfer), reply with `/tx <tx_hash> {reference}` after you send the USDC.\n"
            )
            send_fn(user_id, msg)
        else:
            # If creation failed, tell user we'll provide direct fallback
            send_fn(user_id, "⚠️ Unable to create Coinbase checkout. Please use the direct/manual USDC transfer option below.")
    except Exception:
        # fail silently for notifications but continue to fallback
        try:
            admin_notify(f"handle_pay_command: failed sending hosted_url to user {user_id}: {cb_result}")
        except Exception:
            pass

    # 2) Always send the direct/manual fallback instructions (so user has an immediate alternative)
    try:
        direct = start_direct_payment_flow(db_conn, user_id, amount_usd, send_fn)
    except Exception as e:
        direct = {"ok": False, "error": str(e)}

    # If direct flow failed to message user, notify admin
    if not direct.get("ok"):
        try:
            admin_notify(f"Direct payment flow failed for user {user_id}: {direct.get('error')}")
        except Exception:
            pass

    return {"ok": True, "coinbase": cb_result, "direct": direct}
    
def register_basename_tx(private_key: str, label: str, duration: int = 31536000,
                         resolver: str | None = None, reverse_record: bool = True,
                         gas_params: dict | None = None) -> dict:
    """
    Register a basename on Base by calling the RegistrarController.register(RegisterRequest) payable method.

    Returns dict:
      {
        "ok": True|False,
        "tx_hash": "<0x...>" or None,
        "explorer_url": "<basescan url>" or None,
        "error": "<message>" or None,
        "price_wei": <int> (queried price)
      }

    Notes:
    - `label` is the human portion (e.g. "alice" for "alice.base").
    - `duration` is in seconds (default 1 year).
    - If resolver is None the function will use the well-known L2Resolver for the chain.
    - Uses connect_base_web3() helper from the monolith (must be present).
    """
    try:
        # lazy import local helpers (so this file doesn't require web3 at import-time)
        from eth_account import Account
        from web3 import Web3
    except Exception as e:
        return {"ok": False, "error": f"missing dependency web3/eth_account: {e}"}

    try:
        w3 = connect_base_web3()  # your existing helper
        if not w3:
            return {"ok": False, "error": "connect_base_web3() returned None"}
    except Exception as e:
        return {"ok": False, "error": f"connect_base_web3() error: {e}"}

    # Resolve chain and choose addresses for RegistrarController & L2Resolver
    try:
        chain_id = int(w3.eth.chain_id)
    except Exception:
        chain_id = None

    # Defaults (from base/basenames repo and BaseScan). These are authoritative addresses:
    # - Base Sepolia RegistrarController: 0x49ae3... (Sepolia)
    # - Base Mainnet RegistrarController: 0x4cCb0B... (Mainnet)
    REGISTRAR_BY_CHAIN = {
        84532: "0x49ae3cc2e3aa768b1e5654f5d3c6002144a59581",  # Base Sepolia (test)
        8453:  "0x4cCb0BB02FCABA27e82a56646E81d8c5bC4119a5",  # Base mainnet
    }
    L2RESOLVER_BY_CHAIN = {
        84532: "0x6533C94869D28fAA8dF77cc63f9e2b2D6Cf77eBA",  # Sepolia L2Resolver
        8453:  "0xC6d566A56A1aFf6508b41f6c90ff131615583BCD",  # Mainnet L2Resolver
    }

    registrar_addr = REGISTRAR_BY_CHAIN.get(chain_id) or REGISTRAR_BY_CHAIN.get(84532)
    default_resolver = L2RESOLVER_BY_CHAIN.get(chain_id) or L2RESOLVER_BY_CHAIN.get(84532)
    if resolver is None:
        resolver = Web3.toChecksumAddress(default_resolver)
    else:
        resolver = Web3.toChecksumAddress(resolver)

    registrar_addr = Web3.toChecksumAddress(registrar_addr)

    # Minimal ABI for the RegistrarController we need: registerPrice(string,uint256) and register(tuple) payable
    registrar_abi = [
        {
            "inputs": [
                {"internalType": "string", "name": "name", "type": "string"},
                {"internalType": "uint256", "name": "duration", "type": "uint256"}
            ],
            "name": "registerPrice",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [
                {
                    "components": [
                        {"internalType": "string", "name": "name", "type": "string"},
                        {"internalType": "address", "name": "owner", "type": "address"},
                        {"internalType": "uint256", "name": "duration", "type": "uint256"},
                        {"internalType": "address", "name": "resolver", "type": "address"},
                        {"internalType": "bytes[]", "name": "data", "type": "bytes[]"},
                        {"internalType": "bool", "name": "reverseRecord", "type": "bool"}
                    ],
                    "internalType": "struct RegistrarController.RegisterRequest",
                    "name": "request",
                    "type": "tuple"
                }
            ],
            "name": "register",
            "outputs": [],
            "stateMutability": "payable",
            "type": "function"
        }
    ]

    try:
        contract = w3.eth.contract(address=registrar_addr, abi=registrar_abi)
    except Exception as e:
        return {"ok": False, "error": f"failed to create registrar contract instance: {e}"}

    # Get price (registerPrice) in wei for requested duration
    try:
        price_wei = int(contract.functions.registerPrice(label, duration).call())
    except Exception as e:
        # If registerPrice call fails, return the error (might be invalid name or other)
        return {"ok": False, "error": f"failed to query registerPrice: {e}"}

    # Owner (address) derived from private key
    try:
        acct = Account.from_key(private_key)
        owner_addr = Web3.toChecksumAddress(acct.address)
    except Exception as e:
        return {"ok": False, "error": f"invalid private key: {e}"}

    # Prepare request struct as a Python tuple matching the ABI tuple ordering
    request_tuple = (label, owner_addr, int(duration), resolver, [], bool(reverse_record))

    # Build transaction
    try:
        nonce = w3.eth.get_transaction_count(owner_addr)
    except Exception:
        nonce = 0

    tx = contract.functions.register(request_tuple).buildTransaction({
        "from": owner_addr,
        "value": int(price_wei),
        "nonce": nonce,
        # gas/gasPrice will be set below (estimate or provided)
        "chainId": chain_id or 84532
    })

    # Estimate gas if not provided
    try:
        if gas_params and "gas" in gas_params:
            tx["gas"] = int(gas_params["gas"])
        else:
            estimated = w3.eth.estimate_gas(tx)
            tx["gas"] = int(estimated * 1.2)  # small cushion
    except Exception:
        # fallback gas ceiling
        tx.setdefault("gas", 500000)

    # gasPrice or maxFee settings
    try:
        if gas_params and "gasPrice" in gas_params:
            tx["gasPrice"] = int(gas_params["gasPrice"])
        else:
            # prefer EIP-1559 fields if present in node; fallback to legacy gas price
            try:
                tx["maxFeePerGas"] = w3.eth.generate_max_priority_fee  # harmless attempt
                # actual logic: if w3 supports fee history / EIP-1559, web3 will accept maxFeePerGas
                # but many nodes accept gasPrice too — we'll set gasPrice as gas_price for compatibility:
                tx["gasPrice"] = w3.eth.gas_price
            except Exception:
                tx["gasPrice"] = w3.eth.gas_price
    except Exception:
        tx.setdefault("gasPrice", w3.eth.gas_price if hasattr(w3.eth, "gas_price") else 0)

    # Sign & send
    try:
        signed = acct.sign_transaction(tx)
        raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_tx", None) or None
        if not raw:
            # try the modern attr name as fallback
            raw = signed.rawTransaction if hasattr(signed, "rawTransaction") else None
        if not raw:
            return {"ok": False, "error": "signed transaction object missing raw bytes (signing failed)"}

        tx_hash_bytes = w3.eth.send_raw_transaction(raw)
        tx_hash = tx_hash_bytes.hex() if isinstance(tx_hash_bytes, (bytes, bytearray)) else str(tx_hash_bytes)
    except Exception as e:
        return {"ok": False, "error": f"failed to sign/send transaction: {e}", "price_wei": price_wei}

    # Explorer URL
    explorer_base = "https://sepolia.basescan.org/tx" if (chain_id == 84532 or chain_id is None) else "https://basescan.org/tx"
    explorer_url = f"{explorer_base}/{tx_hash}"

    return {"ok": True, "tx_hash": tx_hash, "explorer_url": explorer_url, "price_wei": price_wei}

def handle_register_request(user_id: int, label: str, use_server_wallet: bool = True, bot_instance=None) -> dict:
    """
    High-level flow to handle a register-name request triggered by a user.
    - If use_server_wallet True, this will attempt to register using the server wallet private key (env var).
    - If False, it will return the required payable parameters so the user can sign in their wallet.

    Returns dict with:
      {
        "ok": True|False,
        "mode": "server"|"manual",
        "tx": {...} (if server mode: tx response from register_basename_tx),
        "price_wei": <int>,
        "instructions": "<string>" (if manual),
        "error": "<message>"
      }
    """
    # defensive imports
    from web3 import Web3
    import os

    # sanitize label
    label = label.strip().lower()
    if not label:
        return {"ok": False, "error": "empty label"}

    # ensure label is valid-ish (min len check) — registrar enforces more rules
    if len(label) < 3:
        return {"ok": False, "error": "name too short (min 3 chars)"}

    # If server flow selected: take encrypted key from env (or raw env) and use it
    if use_server_wallet:
        # support both SERVER_WALLET_PRIVATE_KEY (raw) and SERVER_WALLET_PRIVATE_KEY_ENC (encrypted)
        from os import environ
        enc_key = environ.get("SERVER_WALLET_PRIVATE_KEY_ENC")
        raw_key = environ.get("SERVER_WALLET_PRIVATE_KEY")

        priv = None
        if raw_key:
            priv = raw_key.strip()
        elif enc_key:
            # If you use a decrypt helper in your monolith (e.g. decrypt(encrypted_string)), call it.
            # We try to call a decrypt helper if present; otherwise return a clear error so admin can supply raw key.
            decrypt_fn = globals().get("decrypt")
            if callable(decrypt_fn):
                try:
                    priv = decrypt_fn(enc_key)
                except Exception as e:
                    return {"ok": False, "error": f"failed to decrypt server key: {e}"}
            else:
                return {"ok": False, "error": "SERVER_WALLET_PRIVATE_KEY_ENC present but no decrypt() helper found"}

        if not priv:
            return {"ok": False, "error": "no server private key found in env (SERVER_WALLET_PRIVATE_KEY or SERVER_WALLET_PRIVATE_KEY_ENC)"}

        # run the onchain tx
        try:
            tx_res = register_basename_tx(priv, label)
        except Exception as e:
            return {"ok": False, "error": f"register_basename_tx failed: {e}"}

        return {"ok": tx_res.get("ok", False), "mode": "server", "tx": tx_res, "price_wei": tx_res.get("price_wei")}
    else:
        # Manual (non-custodial) path: compute price and provide user with instructions to pay via their wallet
        try:
            w3 = connect_base_web3()
            chain_id = int(w3.eth.chain_id)
            REGISTRAR_BY_CHAIN = {
                84532: "0x49ae3cc2e3aa768b1e5654f5d3c6002144a59581",
                8453:  "0x4cCb0BB02FCABA27e82a56646E81d8c5bC4119a5",
            }
            registrar_addr = REGISTRAR_BY_CHAIN.get(chain_id) or REGISTRAR_BY_CHAIN.get(84532)
            registrar_addr = Web3.toChecksumAddress(registrar_addr)
            registrar_abi = [
                {
                    "inputs": [
                        {"internalType": "string", "name": "name", "type": "string"},
                        {"internalType": "uint256", "name": "duration", "type": "uint256"}
                    ],
                    "name": "registerPrice",
                    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            contract = w3.eth.contract(address=registrar_addr, abi=registrar_abi)
            duration = 31536000
            price_wei = int(contract.functions.registerPrice(label, duration).call())
        except Exception as e:
            return {"ok": False, "error": f"failed to compute price for manual flow: {e}"}

        # Build a friendly instruction the user can follow (they can call register(...) in a contract UI or craft a tx)
        explorer = "https://sepolia.basescan.org/address" if chain_id == 84532 else "https://basescan.org/address"
        instructions = (
            f"To register `{label}.base` from your own wallet:\n\n"
            f"1) Ensure you are on the Base network (Sepolia/test or main as appropriate).\n"
            f"2) Send a transaction to the RegistrarController contract: {registrar_addr}\n"
            f"   - Method: register(RegisterRequest request) (payable)\n"
            f"   - Request payload (example fields): name='{label}', owner=<your address>, duration=31536000, resolver=<recommended resolver address>\n"
            f"   - Total value (wei): {price_wei}\n\n"
            "Easiest: open the contract page in BaseScan (Write Contract) and call `register` with a RegisterRequest tuple; "
            "set `value` to the listed amount. Or use your wallet dapp UI to craft the same call.\n\n"
            f"Registrar contract page: {explorer}/{registrar_addr}\n\n"
            "If you'd like, you can ask the bot to perform the registration for you using the server wallet (admin mode)."
        )

        return {"ok": True, "mode": "manual", "instructions": instructions, "price_wei": price_wei}
        
# -------------------------
# User profiles (file-backed) and wallet-linking helpers
# Module-level (no indentation). Place after handle_register_request and before connect_base_web3.
# -------------------------
def init_user_profiles_db():
    """
    Ensure the file-backed user profiles JSON file exists and has a dict structure.
    Call once at startup (e.g. in main() after init_payments_db()).
    """
    d = os.path.dirname(USER_PROFILES_PATH) or "."
    if d and not os.path.exists(d):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
    if not os.path.exists(USER_PROFILES_PATH):
        try:
            with open(USER_PROFILES_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f)
        except Exception:
            # fail silently; caller can handle missing file later
            pass


def _load_user_profiles() -> dict:
    """
    Load and return the user profiles dict. Returns {} on error.
    Structure:
      {
        "<user_id>": {
           "linked_address": "0x...",
           "challenge": {"nonce": "...", "message": "...", "expires": 169xxx}
        },
        ...
      }
    """
    try:
        with open(USER_PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_user_profiles(profiles: dict) -> bool:
    try:
        tmp = USER_PROFILES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        os.replace(tmp, USER_PROFILES_PATH)
        return True
    except Exception:
        return False


def create_link_challenge(user_id: int) -> str:
    """
    Create a time-limited challenge message for user_id.
    Returns the message string that the user should sign with their wallet (personal_sign).
    Also stores the challenge in user_profiles.json.
    """
    profiles = _load_user_profiles()
    uid = str(user_id)
    nonce = secrets.token_urlsafe(18)  # random short token
    now = int(time.time())
    expires = now + LINK_CHALLENGE_TTL

    # message to be signed (human readable)
    message = (
        f"Link {APP_NAME} account (user {uid})\n"
        f"Nonce: {nonce}\n"
        f"Expires at (unix): {expires}\n\n"
        f"Purpose: Link your wallet address to your Telegram account for onchain actions.\n"
        f"Do not share your private key."
    )

    # store challenge
    p = profiles.get(uid, {})
    p["challenge"] = {"nonce": nonce, "message": message, "expires": expires}
    # preserve linked_address if present
    if "linked_address" in p:
        p["linked_address"] = p["linked_address"]
    profiles[uid] = p
    _save_user_profiles(profiles)
    return message


def verify_wallet_signature(user_id: int, address: str, signature: str) -> dict:
    """
    Verify an Ethereum signature for the stored challenge message.
    - address: the hex address the user claims (case-insensitive). Accepts 0x... form.
    - signature: hex signature string (0x-prefixed).
    Returns dict:
      {"ok": True, "address": "<checksum>", "reason": None}
      or {"ok": False, "reason": "..."}

    Uses eth_account to recover address from signed message (encode_defunct).
    """
    try:
        from eth_account.messages import encode_defunct
        from eth_account import Account
    except Exception as e:
        return {"ok": False, "reason": f"eth_account required: {e}"}

    profiles = _load_user_profiles()
    uid = str(user_id)
    p = profiles.get(uid)
    if not p or "challenge" not in p:
        return {"ok": False, "reason": "no active challenge found; request a new /link_wallet first"}

    challenge = p["challenge"]
    expires = int(challenge.get("expires", 0))
    now = int(time.time())
    if now > expires:
        # clear stale challenge
        p.pop("challenge", None)
        profiles[uid] = p
        _save_user_profiles(profiles)
        return {"ok": False, "reason": "challenge expired; request a new /link_wallet"}

    message = challenge.get("message")
    if not message:
        return {"ok": False, "reason": "challenge message missing (internal error)"}

    # normalize signature/address
    sig = signature.strip()
    if sig.startswith('"') and sig.endswith('"'):
        sig = sig[1:-1]
    if not sig.startswith("0x"):
        # try to tolerate bare hex
        sig = "0x" + sig

    try:
        encoded = encode_defunct(text=message)
        recovered = Account.recover_message(encoded, signature=sig)
    except Exception as e:
        return {"ok": False, "reason": f"signature verification failed: {e}"}

    # normalize checksum of recovered and provided address
    try:
        from web3 import Web3
        recovered_c = Web3.to_checksum_address(recovered)
        provided_c = Web3.to_checksum_address(address)
    except Exception:
        # fallback to lowercase compare
        recovered_c = recovered.lower()
        provided_c = address.lower()

    if recovered_c != provided_c:
        return {"ok": False, "reason": "signature does not match the provided address"}

    # success: bind address to user and clear challenge
    p["linked_address"] = recovered_c
    p.pop("challenge", None)
    profiles[uid] = p
    saved = _save_user_profiles(profiles)
    if not saved:
        return {"ok": False, "reason": "failed to persist profile (disk error)"}

    return {"ok": True, "address": recovered_c, "reason": None}
    
# -------------------------
# User profiles helpers: updated verify + helpers
# Place at module-level (no indentation) near the other profile helpers.
# -------------------------
def get_user_linked_address(user_id: int) -> Optional[str]:
    """
    Return the linked checksum address for the given user_id, or None if none.
    """
    try:
        profiles = _load_user_profiles()
        p = profiles.get(str(user_id), {})
        addr = p.get("linked_address")
        return addr
    except Exception:
        return None


def admin_list_links() -> dict:
    """
    Return a dict of all user_id -> linked_address pairs. Returns {} on error.
    Admin helper.
    """
    try:
        profiles = _load_user_profiles()
        out = {}
        for uid, data in profiles.items():
            if "linked_address" in data and data["linked_address"]:
                out[uid] = data["linked_address"]
        return out
    except Exception:
        return {}


def verify_wallet_signature(user_id: int, address: Optional[str], signature: str) -> dict:
    """
    Verify a signature for the stored challenge message.
    If `address` is None or empty, the function will infer the address from the signature.
    Returns dict:
      {"ok": True, "address": "<checksum>", "reason": None}
      or {"ok": False, "reason": "..."}

    NOTE: this is a safe replacement for the earlier verify_wallet_signature.
    """
    try:
        from eth_account.messages import encode_defunct
        from eth_account import Account
    except Exception as e:
        return {"ok": False, "reason": f"eth_account required: {e}"}

    try:
        from web3 import Web3
    except Exception:
        Web3 = None  # we'll fallback to lower-case comparison if web3 not available

    profiles = _load_user_profiles()
    uid = str(user_id)
    p = profiles.get(uid)
    if not p or "challenge" not in p:
        return {"ok": False, "reason": "no active challenge found; request a new /link_wallet first"}

    challenge = p["challenge"]
    expires = int(challenge.get("expires", 0))
    now = int(time.time())
    if now > expires:
        # clear stale challenge
        p.pop("challenge", None)
        profiles[uid] = p
        _save_user_profiles(profiles)
        return {"ok": False, "reason": "challenge expired; request a new /link_wallet"}

    message = challenge.get("message")
    if not message:
        return {"ok": False, "reason": "challenge message missing (internal error)"}

    # normalize signature
    sig = signature.strip()
    if sig.startswith('"') and sig.endswith('"'):
        sig = sig[1:-1]
    if not sig.startswith("0x"):
        sig = "0x" + sig

    # recover address from signature
    try:
        encoded = encode_defunct(text=message)
        recovered = Account.recover_message(encoded, signature=sig)
    except Exception as e:
        return {"ok": False, "reason": f"signature verification failed: {e}"}

    # normalize addresses
    try:
        if Web3:
            recovered_c = Web3.to_checksum_address(recovered)
        else:
            recovered_c = recovered.lower()
    except Exception:
        recovered_c = recovered.lower()

    provided_c = None
    if address and str(address).strip():
        try:
            if Web3:
                provided_c = Web3.to_checksum_address(address)
            else:
                provided_c = str(address).lower()
        except Exception:
            provided_c = str(address).lower()

    # If address provided, ensure it matches recovered
    if provided_c:
        if recovered_c != provided_c:
            return {"ok": False, "reason": "signature does not match the provided address"}
        final_addr = provided_c
    else:
        # infer and use the recovered address
        final_addr = recovered_c

    # success: bind address to user and clear challenge
    p["linked_address"] = final_addr
    p.pop("challenge", None)
    profiles[uid] = p
    saved = _save_user_profiles(profiles)
    if not saved:
        return {"ok": False, "reason": "failed to persist profile (disk error)"}

    return {"ok": True, "address": final_addr, "reason": None}

# -------------------------
# Tracked Base wallet helpers (basic)
# -------------------------


def add_tracked_base_wallet(user_id: int, wallet_address: str) -> dict:
    """Add wallet to user's tracked_wallets in user_profiles.json"""
    try:
        from web3 import Web3
    except Exception:
        return {"ok": False, "error": "web3 required"}

    profiles = _load_user_profiles()
    uid = str(user_id)
    p = profiles.get(uid, {})
    tracked = p.get("tracked_wallets", {})
    # normalize
    try:
        addr = Web3.toChecksumAddress(wallet_address)
    except Exception:
        return {"ok": False, "error": "invalid address"}

    if addr in tracked:
        return {"ok": False, "error": "already tracked"}
    # store metadata like last_seen_block or timestamp
    tracked[addr] = {"last_seen_block": None, "last_seen_time": None}
    p["tracked_wallets"] = tracked
    profiles[uid] = p
    saved = _save_user_profiles(profiles)
    return {"ok": saved, "address": addr} if saved else {"ok": False, "error": "failed to persist"}

def remove_tracked_base_wallet(user_id: int, wallet_address: str) -> dict:
    profiles = _load_user_profiles()
    uid = str(user_id)
    p = profiles.get(uid, {})
    tracked = p.get("tracked_wallets", {})
    addr = wallet_address
    try:
        from web3 import Web3
        addr = Web3.toChecksumAddress(wallet_address)
    except Exception:
        pass
    if addr not in tracked:
        return {"ok": False, "error": "not tracked"}
    tracked.pop(addr, None)
    p["tracked_wallets"] = tracked
    profiles[uid] = p
    saved = _save_user_profiles(profiles)
    return {"ok": saved, "address": addr} if saved else {"ok": False, "error": "failed to persist"}

def list_tracked_base_wallets(user_id: int) -> dict:
    profiles = _load_user_profiles()
    p = profiles.get(str(user_id), {})
    return {"ok": True, "tracked": p.get("tracked_wallets", {})}

async def wallet_poll_loop(poll_interval: int = TRACK_POLL_INTERVAL):
    """
    Background loop: scan tracked wallets for all users and check for new transfers.
    When new transfers are found, call process_tracked_wallet_event(user_id, address, tx) which
    should notify or trigger forwarding.
    """
    while True:
        try:
            profiles = _load_user_profiles()
            for uid, pdata in profiles.items():
                tracked = pdata.get("tracked_wallets", {})
                if not tracked:
                    continue
                for addr, meta in list(tracked.items()):
                    try:
                        # call helper get_token_transfer_logs(addr) that returns list of transfers,
                        # ensure your get_token_transfer_logs returns recent txs; it may accept args.
                        new_txs = []
                        try:
                            new_txs = get_token_transfer_logs(addr) or []
                        except Exception:
                            new_txs = []
                        # filter by last_seen_time or block if you saved it
                        last_seen_time = meta.get("last_seen_time") or 0
                        for tx in new_txs:
                            # tx should include 'timestamp' or 'blockNumber'
                            tx_time = tx.get("timestamp") or tx.get("time") or int(time.time())
                            if tx_time and tx_time <= last_seen_time:
                                continue
                            # process new tx
                            # attempt to call a process hook; fallback: notify owner/admin
                            try:
                                # try user-specified hook first
                                hook = globals().get("process_tracked_wallet_event")
                                if callable(hook):
                                    hook(int(uid), addr, tx)
                                else:
                                    # fallback notify owner via admin_notify or print
                                    admin_notify = globals().get("make_admin_notify_callable")
                                    if callable(admin_notify):
                                        notify_fn = admin_notify()
                                        notify_fn(f"Tracked wallet {addr} saw tx: {tx.get('hash') or tx.get('tx_hash')}")
                                    else:
                                        logging.info(f"[tracked-wallet] user {uid} {addr} tx: {tx}")
                            except Exception:
                                logging.exception("error processing tracked tx")
                            # update meta to avoid reprocessing
                            meta["last_seen_time"] = tx_time
                        # save updated meta
                        pdata["tracked_wallets"][addr] = meta
                        profiles[uid] = pdata
                    except Exception:
                        logging.exception("error scanning wallet")
            _save_user_profiles(profiles)
        except Exception:
            logging.exception("wallet poll loop error")
        await asyncio.sleep(poll_interval)

def decrypt(enc: str) -> str:
    return CryptoManager(0).decrypt(enc.encode() if isinstance(enc, str) else enc)

def connect_base_web3():
    """
    Build and return a web3.Web3 instance connected to BASE_RPC_URL from env.

    Returns:
        web3 (Web3): an instance of Web3 connected to the Base RPC url.

    Raises:
        RuntimeError if connection cannot be established.
    """
    
    try:
        from web3 import Web3, HTTPProvider  # local import so monolith can still run if unused
    except Exception as e:
        raise RuntimeError("web3.py is required for connect_base_web3. Install with `pip install web3`.") from e

    rpc_url = os.getenv("BASE_RPC_URL") or os.getenv("BASE_SEPOLIA_RPC") or os.getenv("BASE_RPC")
    if not rpc_url:
        raise RuntimeError("BASE_RPC_URL not set in environment. Set BASE_RPC_URL to your Base RPC (Alchemy/Grove/Infura).")

    w3 = Web3(HTTPProvider(rpc_url, request_kwargs={'timeout': 30}))
    # quick connection test
    if not w3.isConnected():
        raise RuntimeError(f"Unable to connect to Base RPC at {rpc_url!r}. Check network and RPC key.")
    return w3
    
# -----------------------
# market_api / multi-chain helpers (adds 'base' support)
# Module-level functions (place after connect_base_web3 and before resolve_basename)
# Dependencies: requests, web3
# -----------------------

def normalize_chain(chain: Optional[str|int]) -> dict:
    """
    Normalize a chain alias or id into a dict:
      {"id": <int>, "name": "<canonical>", "is_evm": True}
    Accepts: numeric chain id or strings like "base", "base-sepolia", "84532".
    """
    if chain is None:
        return {"id": None, "name": None, "is_evm": False}
    # numeric input
    try:
        if isinstance(chain, (int,)) or (isinstance(chain, str) and chain.isdigit()):
            cid = int(chain)
            # invert alias map
            for k, v in _CHAIN_ALIAS.items():
                if v == cid:
                    return {"id": cid, "name": k, "is_evm": True}
            return {"id": cid, "name": str(cid), "is_evm": True}
    except Exception:
        pass
    # string alias
    key = str(chain).strip().lower()
    if key in _CHAIN_ALIAS:
        return {"id": _CHAIN_ALIAS[key], "name": key, "is_evm": True}
    # fallback: if looks like 'base' or 'base-sepolia' map heuristically
    if "base" in key:
        # prefer sepolia token name if 'sepolia' appears, else mainnet
        if "sepolia" in key or "sep" in key:
            return {"id": 84532, "name": "base-sepolia", "is_evm": True}
        return {"id": 8453, "name": "base", "is_evm": True}
    return {"id": None, "name": key, "is_evm": False}

def get_web3_for_chain(chain: Optional[str|int]):
    """
    Return a Web3 instance for the provided chain alias or id.
    For 'base' chains this calls your existing connect_base_web3() helper.
    For other chains this tries env vars: ETHEREUM_RPC_URL, POLYGON_RPC_URL, or FALLBACK_RPC_URL.
    """
    norm = normalize_chain(chain)
    cid = norm.get("id")
    name = norm.get("name")

    # Base family
    if cid in (8453, 84532) or (isinstance(name, str) and name.startswith("base")):
        # use existing helper in this monolith (must exist)
        try:
            return connect_base_web3()
        except Exception as e:
            raise RuntimeError(f"connect_base_web3() failed: {e}")

    # Ethereum mainnet
    if cid == 1 or name in ("ethereum", "eth"):
        rpc = os.getenv("ETHEREUM_RPC_URL") or os.getenv("FALLBACK_RPC_URL")
        if not rpc:
            raise RuntimeError("ETHEREUM_RPC_URL not set")
        from web3 import Web3, HTTPProvider
        return Web3(HTTPProvider(rpc, request_kwargs={"timeout": 30}))

    # Polygon
    if cid == 137 or name in ("polygon", "matic"):
        rpc = os.getenv("POLYGON_RPC_URL") or os.getenv("FALLBACK_RPC_URL")
        if not rpc:
            raise RuntimeError("POLYGON_RPC_URL not set")
        from web3 import Web3, HTTPProvider
        return Web3(HTTPProvider(rpc, request_kwargs={"timeout": 30}))

    # fallback to generic RPC env
    rpc = os.getenv("FALLBACK_RPC_URL")
    if not rpc:
        raise RuntimeError("No RPC configured for chain and FALLBACK_RPC_URL is not set")
    from web3 import Web3, HTTPProvider
    return Web3(HTTPProvider(rpc, request_kwargs={"timeout": 30}))

def _erc20_decimals(web3, token_address: str) -> Optional[int]:
    """
    Return token decimals using a minimal ERC20 ABI. Returns None on error.
    """
    try:
        if not token_address:
            return None
        from web3 import Web3
        token = Web3.to_checksum_address(token_address)
        erc20_abi = [
            {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"type":"function"},
        ]
        contract = web3.eth.contract(address=token, abi=erc20_abi)
        return int(contract.functions.decimals().call())
    except Exception:
        return None

def get_token_price_usd(chain: Optional[str|int], token_address: Optional[str] = None,
                        token_symbol: Optional[str] = None) -> Optional[float]:
    """
    Try to get a USD price for token on the given chain.
    Strategy:
      1) If token_address provided and CoinGecko supports the platform, query:
         /simple/token_price/{platform}?contract_addresses={contract}&vs_currencies=usd
      2) If that fails and token_symbol provided, fallback to /simple/price by symbol (CoinGecko id mapping needed).
      3) Otherwise return None (caller can implement onchain DEX or Chainlink fallback).
    Notes:
      - COINGECKO_PLATFORM_BY_CHAIN can be overridden via env (JSON).
      - This function makes a network call to CoinGecko's public API by default.
    """
    # 1) use CoinGecko by contract (preferred)
    try:
        cid = normalize_chain(chain).get("id")
        platform = COINGECKO_PLATFORM_BY_CHAIN.get(str(cid)) or COINGECKO_PLATFORM_BY_CHAIN.get(str(chain)) or None
        if token_address and platform:
            url = f"{COINGECKO_API_URL}/simple/token_price/{platform}"
            params = {"contract_addresses": token_address, "vs_currencies": "usd"}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                j = resp.json()
                # The key is the lowercase contract address
                key = token_address.lower()
                if key in j and "usd" in j[key]:
                    try:
                        return float(j[key]["usd"])
                    except Exception:
                        pass
        # 2) fallback by token_symbol (best-effort)
        if token_symbol:
            # coin ids mapping is out-of-scope; use /simple/price if user provided coin id as symbol
            url2 = f"{COINGECKO_API_URL}/simple/price"
            params2 = {"ids": token_symbol, "vs_currencies": "usd"}
            resp2 = requests.get(url2, params=params2, timeout=10)
            if resp2.status_code == 200:
                j2 = resp2.json()
                if token_symbol in j2 and "usd" in j2[token_symbol]:
                    return float(j2[token_symbol]["usd"])
    except Exception:
        # swallow and fall-through to None
        pass

    # 3) No price found
    return None

def market_get_token_info(chain: Optional[str|int], token_address: Optional[str] = None,
                          token_symbol: Optional[str] = None) -> dict:
    """
    High-level combined info for a token on a chain:
      {"decimals": int|None, "price_usd": float|None, "symbol": str|None}
    This function tries CoinGecko for price and onchain for decimals.
    """
    out = {"decimals": None, "price_usd": None, "symbol": token_symbol}
    try:
        web3 = None
        try:
            web3 = get_web3_for_chain(chain)
        except Exception:
            web3 = None
        if web3 and token_address:
            out["decimals"] = _erc20_decimals(web3, token_address)
        price = get_token_price_usd(chain, token_address, token_symbol)
        out["price_usd"] = price
    except Exception:
        pass
    return out

def resolve_basename(name: str) -> str | None:
    """
    Resolve a Basename (e.g. "alice", "alice.base", or "alice.base.eth") -> checksum address on Base.
    Returns checksum address string or None if not found / zero-address.

    Depends on connect_base_web3() existing in the monolith and uses a minimal resolver ABI.
    You may override the resolver address with env var BASE_SEPOLIA_L2RESOLVER or BASE_L2RESOLVER.
    """
    
    try:
        from web3 import Web3
    except Exception as e:
        raise RuntimeError("web3.py required for resolve_basename") from e

    if not name or not isinstance(name, str):
        return None

    # normalize input to "label.base"
    n = name.strip().rstrip(".")
    if n.endswith(".base.eth"):
        n = n[: -len(".base.eth")]
    if n.endswith(".base"):
        label = n
    else:
        label = n
    full_name = label if label.endswith(".base") else f"{label}.base"

    # ENS namehash
    def namehash(name_str: str) -> bytes:
        node = b"\x00" * 32
        if not name_str:
            return node
        labels = name_str.split(".")
        # use a Web3 keccak when available
        # connect_base_web3 may not exist in this scope; we'll use local Web3.keccak for label hashing
        for label_part in reversed(labels):
            label_bytes = label_part.encode("utf-8")
            label_hash = Web3.keccak(label_bytes)
            node = Web3.keccak(node + label_hash)
        return node

    # minimal resolver ABI
    resolver_abi = [
        {
            "constant": True,
            "inputs": [{"name": "node", "type": "bytes32"}],
            "name": "addr",
            "outputs": [{"name": "ret", "type": "address"}],
            "payable": False,
            "stateMutability": "view",
            "type": "function",
        }
    ]

    # resolver address fallback (canonical Sepolia L2Resolver)
    default_resolver = os.getenv("BASE_SEPOLIA_L2RESOLVER", os.getenv("BASE_L2RESOLVER", "0x6533C94869D28fAA8dF77cc63f9e2b2D6Cf77eBA"))

    try:
        w3 = connect_base_web3()
    except Exception as e:
        raise RuntimeError("connect_base_web3() failed: " + str(e))

    # compute node (bytes32)
    node_bytes = namehash(full_name)

    try:
        resolver_addr = Web3.to_checksum_address(default_resolver)
    except Exception:
        resolver_addr = default_resolver

    try:
        resolver = w3.eth.contract(address=resolver_addr, abi=resolver_abi)
        # call resolver.addr(node) with bytes32
        resolved = resolver.functions.addr(node_bytes).call()
        if not resolved or int(resolved, 16) == 0:
            return None
        return Web3.to_checksum_address(resolved)
    except Exception:
        return None


def get_token_transfer_logs(web3, token_contract_address, to_address, from_block=None, to_block='latest'):
    """
    Query token Transfer logs for a given token contract where `to_address` is the recipient.

    Args:
        web3: a Web3 instance (returned by connect_base_web3()).
        token_contract_address: token contract address (string).
        to_address: recipient address to filter logs for (string).
        from_block: optional starting block number or 'earliest' / int.
        to_block: block range end (default 'latest').

    Returns:
        list of raw log dicts that appear to be Transfer events to `to_address`.
    """
    
    # normalize
    token_addr = Web3.toChecksumAddress(token_contract_address)
    to_addr_clean = to_address.lower().replace("0x", "")

    # Default lookback: if no from_block provided, use recent 5000 blocks (safe fallback)
    if from_block is None:
        try:
            latest = web3.eth.block_number
            # keep a reasonably sized window by default; larger networks may need adjustments
            from_block = max(0, latest - 5000)
        except Exception:
            from_block = 0

    params = {
        "fromBlock": from_block,
        "toBlock": to_block,
        "address": token_addr
    }

    try:
        logs = web3.eth.get_logs(params)
    except Exception as e:
        # bubble up with context
        raise RuntimeError(f"web3.eth.get_logs failed: {e}")

    # Topic[0] is Transfer event signature (we won't compute it here; instead filter by topics length and topic[2])
    filtered = []
    for log in logs:
        topics = log.get("topics", [])
        if len(topics) >= 3:
            # topics[2] is 'to' (indexed), it's a bytes-like (HexBytes)
            try:
                topic2_hex = topics[2].hex()  # 0x... hex string
            except Exception:
                # skip malformed
                continue
            # Compare final 40 hex chars to the address
            if topic2_hex[-40:].lower() == to_addr_clean:
                filtered.append(log)
    return filtered


def verify_tx_has_token_transfer(web3, token_contract_address, tx_hash, to_address, amount_expected, token_decimals=6):
    """
    Verify that a given transaction (tx_hash) includes an ERC-20 Transfer to `to_address`
    of exactly `amount_expected` tokens (amount_expected is expressed in token units, e.g., 12.5 USDC).

    Args:
        web3: Web3 instance.
        token_contract_address: token contract address string.
        tx_hash: transaction hash string.
        to_address: recipient address string.
        amount_expected: Decimal/float representing human token amount (e.g., 12.5 for USDC).
        token_decimals: token decimals (USDC typically 6).

    Returns:
        dict: {"ok": bool, "found": int_amount_or_None, "tx_hash": tx_hash, "matches": [matching_log_dicts...]}
    """
    

    try:
        receipt = web3.eth.get_transaction_receipt(tx_hash)
    except Exception as e:
        return {"ok": False, "error": f"could not fetch receipt: {e}", "tx_hash": tx_hash}

    # compute expected integer amount in token smallest units
    try:
        exp_dec = Decimal(str(amount_expected))
    except (InvalidOperation, TypeError):
        return {"ok": False, "error": "invalid amount_expected", "tx_hash": tx_hash}

    exp_units = int((exp_dec * (Decimal(10) ** token_decimals)).to_integral_value())

    matches = []
    token_addr_checksum = Web3.toChecksumAddress(token_contract_address)
    to_addr_clean = to_address.lower().replace("0x", "")

    for log in receipt.get("logs", []):
        # match token contract address
        log_addr = log.get("address", "").lower()
        if log_addr != token_addr_checksum.lower():
            continue
        topics = log.get("topics", [])
        if len(topics) < 3:
            continue
        # topics[0] should be Transfer signature; we just inspect topics[2] for 'to'
        topic2_hex = topics[2].hex()
        if topic2_hex[-40:].lower() != to_addr_clean:
            continue
        # data field is the value (32 bytes) encoded as hex
        data_hex = log.get("data", "")
        if not data_hex:
            continue
        try:
            value_int = int(data_hex, 16)
        except Exception:
            continue
        if value_int == exp_units:
            matches.append({
                "log_index": log.get("logIndex"),
                "value_int": value_int,
                "value_human": float(Decimal(value_int) / (Decimal(10) ** token_decimals)),
                "topic2": topic2_hex,
            })

    ok = len(matches) > 0
    return {"ok": ok, "found": matches, "tx_hash": tx_hash, "expected_units": exp_units}


def get_dev_wallet_details(user_id=None):
    """
    Return a dict with developer/server wallet details used in the direct-payment fallback.
    Reads env vars for addresses and token contract addresses.

    Args:
        user_id: optionally used to build a reference string (not persisted here).

    Returns:
        dict with keys:
            - server_wallet_address
            - usdc_contract_address
            - usdt_contract_address (optional)
            - example_reference (string)
            - instruction_text (string)
    """
    
    prefix = os.getenv("PAYMENT_REFERENCE_PREFIX", "BOTPAY-")
    server_addr = os.getenv("SERVER_WALLET_ADDRESS")
    usdc_addr = os.getenv("USDC_CONTRACT_ADDRESS") or os.getenv("USDC_CONTRACT_ADDRESS_BASE")
    usdt_addr = os.getenv("USDT_CONTRACT_ADDRESS") or os.getenv("USDT_CONTRACT_ADDRESS_BASE")

    # create a sample reference for the user to paste back
    
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    uid_part = str(user_id) if user_id is not None else "anon"
    reference = f"{prefix}{uid_part}-{rand}"

    instruction = (
        f"Send EXACTLY the requested amount of USDC (Base network) to this address:\n\n"
        f"{server_addr}\n\n"
        f"Token contract (USDC on Base): {usdc_addr}\n\n"
        f"IMPORTANT: include this reference in the tx memo or paste it here after sending: {reference}\n\n"
        "After sending, paste the transaction hash using the bot command: /tx <tx_hash> <reference>\n"
        "Example: /tx 0xabc123... " + reference
    )

    return {
        "server_wallet_address": server_addr,
        "usdc_contract_address": usdc_addr,
        "usdt_contract_address": usdt_addr,
        "example_reference": reference,
        "instruction_text": instruction
    }

def start_direct_payment_flow(db_conn, user_id, amount_usd, send_message_callable):
    """
    File-backed replacement. db_conn parameter ignored. Creates a payment record in payments.json
    and sends direct/manual instructions to user (uses make_send_message_callable wrapper for sending).
    """


    try:
        amt_dec = Decimal(str(amount_usd))
    except Exception:
        return {"ok": False, "error": "invalid amount_usd"}

    token_amount = float(amt_dec)
    prefix = os.getenv("PAYMENT_REFERENCE_PREFIX", "BOTPAY-")
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    reference = f"{prefix}{user_id}-{rand}"

    rec = {
        "merchant_payment_id": None,
        "user_id": int(user_id),
        "amount_usd": float(amt_dec),
        "amount_token": token_amount,
        "token_symbol": "USDC",
        "token_chain": "base",
        "status": "awaiting_onchain",
        "hosted_url": None,
        "reference": reference,
        "metadata": {"method": "direct_fallback"}
    }
    saved = _create_payment_record(rec)
    details = get_dev_wallet_details(user_id=user_id)

    message = (
        f"Please send EXACTLY *{token_amount} USDC* (Base network) to the address below.\n\n"
        f"Recipient: {details['server_wallet_address']}\n\n"
        f"USDC contract (Base): {details['usdc_contract_address']}\n\n"
        f"Reference to include (or paste with /tx after sending): {reference}\n\n"
        "IMPORTANT: Use the Base network. After sending, paste the transaction hash here like:\n"
        f"/tx <tx_hash> {reference}"
    )

    try:
        send_message_callable(user_id, message)
    except Exception as e:
        return {"ok": False, "error": f"failed to send message: {e}", "payment_id": saved.get("id"), "reference": reference}

    return {"ok": True, "payment_id": saved.get("id"), "reference": reference}

def verify_direct_payment(db_conn, user_id, reference, tx_hash, send_message_callable, token_contract_address=None, token_decimals=6):
    """
    File-backed replacement. db_conn ignored. Verifies tx via web3 and updates payments.json on success.
    """


    rec = _find_payment_by_reference(reference)
    if not rec:
        return {"ok": False, "reason": "no matching payment request found with that reference for this user"}

    if int(rec.get("user_id") or 0) != int(user_id):
        return {"ok": False, "reason": "reference does not belong to this user"}

    amount_token = rec.get("amount_token")
    server_addr = os.getenv("SERVER_WALLET_ADDRESS")
    if not server_addr:
        return {"ok": False, "reason": "SERVER_WALLET_ADDRESS not configured on server"}

    token_addr = token_contract_address or os.getenv("USDC_CONTRACT_ADDRESS") or os.getenv("USDC_CONTRACT_ADDRESS_BASE")
    if not token_addr:
        return {"ok": False, "reason": "USDC token contract address not configured (USDC_CONTRACT_ADDRESS)"}

    try:
        w3 = connect_base_web3()
    except Exception as e:
        return {"ok": False, "reason": f"web3 connect error: {e}"}

    try:
        v = verify_tx_has_token_transfer(w3, token_addr, tx_hash, server_addr, amount_expected=amount_token, token_decimals=token_decimals)
    except Exception as e:
        return {"ok": False, "reason": f"verification error: {e}"}

    if not v.get("ok"):
        reason = v.get("error") or "no matching token transfer found in tx"
        return {"ok": False, "reason": reason, "details": v}

    # Update file-backed payment record
    updated = _update_payment_record_by_id(rec.get("id"), {"status": "confirmed", "onchain_tx_hash": tx_hash})
    if not updated:
        return {"ok": False, "reason": "failed to update payment record"}

    # Notify user
    try:
        send_message_callable(user_id, f"Payment confirmed ✅ — tx: {tx_hash}\nReference: {reference}")
    except Exception:
        pass

    return {"ok": True, "payment_id": rec.get("id"), "tx_hash": tx_hash}

    
def coinbase_create_checkout(db_conn, user_id, amount_usd, currency="USDC", chain="base", metadata=None):
    """
    File-backed replacement. db_conn parameter is ignored for compatibility.
    Creates Coinbase charge, persists to payments.json, returns same result shape as before.
    """
    # Reuse earlier implementation but persist via _create_payment_record
    
    api_key = os.getenv("COINBASE_API_KEY")
    if not api_key:
        return {"ok": False, "error": "COINBASE_API_KEY not set"}

    prefix = os.getenv("PAYMENT_REFERENCE_PREFIX", "BOTPAY-")
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    reference = f"{prefix}{user_id}-{rand}"

    payload = {
        "name": "Bot Payment",
        "description": f"Payment for user {user_id}",
        "pricing_type": "fixed_price",
        "local_price": {"amount": str(amount_usd), "currency": "USD"},
        "metadata": metadata or {"user_id": user_id, "reference": reference, "chain": chain},
    }

    headers = {
        "X-CC-Api-Key": api_key,
        "X-CC-Version": "2018-03-22",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post("https://api.commerce.coinbase.com/charges", headers=headers, json=payload, timeout=30)
    except Exception as e:
        return {"ok": False, "error": f"request failed: {e}"}

    if resp.status_code >= 400:
        return {"ok": False, "error": f"coinbase create charge failed: {resp.status_code} {resp.text}"}

    try:
        j = resp.json()
    except Exception as e:
        return {"ok": False, "error": f"invalid json from coinbase: {e}"}

    charge = j.get("data") or j
    hosted_url = charge.get("hosted_url") or (charge.get("data") or {}).get("hosted_url")
    merchant_id = charge.get("id")

    # Persist to file-backed store
    rec = {
        "merchant_payment_id": merchant_id,
        "user_id": int(user_id),
        "amount_usd": float(amount_usd),
        "amount_token": float(amount_usd),
        "token_symbol": currency,
        "token_chain": chain,
        "status": "created",
        "hosted_url": hosted_url,
        "reference": reference,
        "metadata": charge.get("metadata") or payload.get("metadata")
    }
    saved = _create_payment_record(rec)
    return {"ok": True, "charge": charge, "payment_id": saved["id"], "reference": reference}

def coinbase_verify_signature(raw_body_bytes, signature_header):
    """
    Verify Coinbase Commerce webhook signature.

    Args:
        raw_body_bytes: raw request body bytes (exact bytes received).
        signature_header: the header value provided by Coinbase (commonly 'X-CC-Webhook-Signature').

    Returns:
        bool
    """
    
    secret = os.getenv("COINBASE_WEBHOOK_SECRET")
    if not secret:
        # failing open would be insecure — surface False
        return False

    # Coinbase typically provides a signature string. We'll accept a few header formats.
    sig = signature_header or ""
    # If header looks like "t=...,v1=..." (stripe-style), try to extract the last hex substring
    m = re.search(r"([0-9a-fA-F]{64,})", sig)
    if m:
        sig_value = m.group(1)
    else:
        sig_value = sig.strip()

    try:
        computed = hmac.new(secret.encode("utf-8"), raw_body_bytes, digestmod=hashlib.sha256).hexdigest()
    except Exception:
        return False

    # compare digests in constant time
    try:
        ok = hmac.compare_digest(computed, sig_value)
    except Exception:
        ok = False
    return ok


def _extract_tx_hash_from_payload(obj):
    """
    Helper: scan webhook payload dict for an onchain tx hash (0x...64hex).
    Returns first match or None.
    """
    
    hex_re = re.compile(r"0x[a-fA-F0-9]{64}")
    def scan(v):
        if isinstance(v, str):
            m = hex_re.search(v)
            if m:
                return m.group(0)
        elif isinstance(v, dict):
            for _, x in v.items():
                found = scan(x)
                if found:
                    return found
        elif isinstance(v, list):
            for item in v:
                found = scan(item)
                if found:
                    return found
        return None
    return scan(obj)

def coinbase_handle_webhook(request_body_bytes, signature_header, db_conn, send_message_callable=None, admin_notify_callable=None):
    """
    File-backed webhook processor. db_conn parameter retained for compatibility but ignored.
    Persists webhook events and updates payments.json.
    """
    
    ok_sig = coinbase_verify_signature(request_body_bytes, signature_header)
    if not ok_sig:
        return {"ok": False, "error": "invalid signature"}

    try:
        payload = json.loads(request_body_bytes)
    except Exception as e:
        return {"ok": False, "error": f"invalid json payload: {e}"}

    event_type = None
    data = None
    if isinstance(payload, dict) and "event" in payload and isinstance(payload["event"], dict):
        event_type = payload["event"].get("type")
        data = payload["event"].get("data")
    else:
        event_type = payload.get("type")
        data = payload.get("data") or payload

    if data is None:
        data = payload

    # Try to find merchant_payment_id or reference
    merchant_id = data.get("id") or data.get("code") or data.get("charge_id") or None
    payment_id = None
    if merchant_id:
        p = _find_payment_by_merchant_id(merchant_id)
        if p:
            payment_id = p.get("id")

    if not payment_id:
        meta_ref = None
        try:
            meta_ref = (data.get("metadata") or {}).get("reference")
        except Exception:
            meta_ref = None
        if meta_ref:
            p = _find_payment_by_reference(meta_ref)
            if p:
                payment_id = p.get("id")

    # persist webhook event
    try:
        _append_webhook_event({"payment_id": payment_id, "event_type": event_type, "payload": payload})
    except Exception:
        pass

    # Map status
    status = None
    if event_type in ("charge:confirmed", "charge:resolved", "charge:pending", "charge:failed", "charge:created"):
        if event_type in ("charge:confirmed", "charge:resolved"):
            status = "confirmed"
        elif event_type == "charge:pending":
            status = "pending"
        elif event_type == "charge:created":
            status = "created"
        elif event_type == "charge:failed":
            status = "failed"
    else:
        status = data.get("status") or None
        if status:
            status = status.lower()
            if status in ("completed", "confirmed"):
                status = "confirmed"
            elif status == "pending":
                status = "pending"

    tx_hash = _extract_tx_hash_from_payload(payload)

    updated = False
    if payment_id:
        try:
            _update_payment_record_by_id(payment_id, {"status": status, "onchain_tx_hash": tx_hash})
            updated = True
            # notify user if we can find user id in record
            rec = _load_payments_store()
            for p in rec.get("payments", []):
                if p.get("id") == payment_id:
                    uid = p.get("user_id")
                    if send_message_callable and uid:
                        try:
                            send_message_callable(uid, f"Payment status update: {status}. Tx: {tx_hash or 'n/a'}")
                        except Exception:
                            pass
                    break
        except Exception:
            pass

    # Fallback: attempt to match hosted_url in payload to find payment
    if (not updated) and (data.get("hosted_url") or (payload.get("data") or {}).get("hosted_url")):
        hosted = data.get("hosted_url") or (payload.get("data") or {}).get("hosted_url")
        # find matching payment
        store = _load_payments_store()
        for p in store["payments"]:
            if p.get("hosted_url") == hosted:
                _update_payment_record_by_id(p.get("id"), {"status": status, "onchain_tx_hash": tx_hash})
                updated = True
                if send_message_callable and p.get("user_id"):
                    try:
                        send_message_callable(p.get("user_id"), f"Payment status update: {status}. Tx: {tx_hash or 'n/a'}")
                    except Exception:
                        pass
                break

    if admin_notify_callable:
        try:
            admin_notify_callable(f"Coinbase webhook processed: event={event_type}, tx={tx_hash}, updated={updated}")
        except Exception:
            pass

    return {"ok": True, "event_type": event_type, "tx_hash": tx_hash, "updated": updated}

def reconcile_pending_payments(poll_interval_seconds=300, stop_event=None):
    """
    Background reconciliation loop.

    - Scans file-backed payments store for payments in statuses: 'created', 'pending', 'awaiting_onchain'
    - For Coinbase-created payments: calls Coinbase charges API to refresh status, updates records and notifies users.
    - For direct/manual 'awaiting_onchain' payments: scans recent token Transfer logs to find a matching transfer to SERVER_WALLET_ADDRESS,
      and marks payment confirmed with the discovered tx hash.

    Args:
        poll_interval_seconds: seconds between polls (default 300).
        stop_event: optional threading.Event you can set to stop the loop.
    """

    admin_notify = None
    send_fn = None
    try:
        admin_notify = make_admin_notify_callable(globals().get("bot_instance"))
    except Exception:
        admin_notify = lambda msg: print("[admin_notify]", msg)
    try:
        send_fn = make_send_message_callable(globals().get("bot_instance"))
    except Exception:
        send_fn = lambda uid, text: print(f"[send_fn] to={uid} text={text}")

    COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
    COINBASE_CHARGE_GET_URL = "https://api.commerce.coinbase.com/charges/{}"
    token_decimals = int(os.getenv("USDC_DECIMALS", "6"))
    server_addr = os.getenv("SERVER_WALLET_ADDRESS")
    token_addr = os.getenv("USDC_CONTRACT_ADDRESS") or os.getenv("USDC_CONTRACT_ADDRESS_BASE")

    while True:
        try:
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                break
            store = _load_payments_store()
            payments = list(store.get("payments", []))
            # Process Coinbase-created or pending charges
            for p in payments:
                status = (p.get("status") or "").lower()
                merchant_id = p.get("merchant_payment_id")
                reference = p.get("reference")
                user_id = p.get("user_id")
                # Coinbase flows
                if merchant_id and status in ("created", "pending"):
                    if not COINBASE_API_KEY:
                        continue
                    try:
                        headers = {
                            "X-CC-Api-Key": COINBASE_API_KEY,
                            "X-CC-Version": "2018-03-22"
                        }
                        url = COINBASE_CHARGE_GET_URL.format(merchant_id)
                        resp = requests.get(url, headers=headers, timeout=20)
                        if resp.status_code != 200:
                            continue
                        j = resp.json()
                        data = j.get("data") or j
                        # attempt to pull status from data.status or timeline
                        new_status = None
                        ds = data.get("timeline") or []
                        if ds and isinstance(ds, list):
                            # timeline last status
                            last = ds[-1]
                            new_status = (last.get("status") or "").lower()
                        if not new_status:
                            new_status = (data.get("status") or "").lower()
                        mapped = None
                        if new_status in ("COMPLETED".lower(), "completed", "confirmed"):
                            mapped = "confirmed"
                        elif new_status in ("PENDING".lower(), "pending"):
                            mapped = "pending"
                        elif new_status in ("FAILED".lower(), "failed"):
                            mapped = "failed"
                        # extract any tx hash from payload
                        tx_hash = _extract_tx_hash_from_payload(data) or None
                        if mapped and mapped != status:
                            _update_payment_record_by_id(p.get("id"), {"status": mapped, "onchain_tx_hash": tx_hash})
                            if send_fn and user_id:
                                try:
                                    send_fn(user_id, f"Payment status update: {mapped}. Tx: {tx_hash or 'n/a'}")
                                except Exception:
                                    pass
                            if admin_notify:
                                try:
                                    admin_notify(f"Reconcile: updated payment {reference} ({merchant_id}) to {mapped}")
                                except Exception:
                                    pass
                    except Exception as e:
                        # ignore single payment errors
                        if admin_notify:
                            try:
                                admin_notify(f"Reconcile coinbase error for {merchant_id or reference}: {e}")
                            except Exception:
                                pass
                        continue

                # Direct/manual onchain reconciliation: awaiting_onchain
                if status in ("awaiting_onchain", "created", "pending"):
                    # only attempt onchain find if we have token and server addr info and amount
                    if p.get("token_chain") == "base" and server_addr and token_addr and p.get("amount_token"):
                        try:
                            w3 = connect_base_web3()
                        except Exception as e:
                            if admin_notify:
                                try:
                                    admin_notify(f"Reconcile web3 connect error: {e}")
                                except Exception:
                                    pass
                            continue

                        # attempt to find logs to our server address with exact amount
                        try:
                            # search last N blocks by default
                            logs = get_token_transfer_logs(w3, token_addr, server_addr, from_block=None, to_block="latest")
                            # compute expected integer units
                            exp_units = None
                            try:
                                exp_units = int(Decimal(str(p.get("amount_token"))) * (Decimal(10) ** token_decimals))
                            except Exception:
                                exp_units = None

                            # scan logs for a matching value
                            matched_tx = None
                            for log in logs:
                                data_hex = log.get("data", "")
                                if not data_hex:
                                    continue
                                try:
                                    value_int = int(data_hex, 16)
                                except Exception:
                                    continue
                                if exp_units is not None and value_int == exp_units:
                                    # capture tx hash field (may be bytes or hexstr)
                                    txh = log.get("transactionHash") or log.get("transaction_hash") or log.get("txHash") or log.get("tx_hash")
                                    if not txh:
                                        # some web3 implementations include receipt-logs as dict with 'transactionHash' as HexBytes
                                        try:
                                            txh = log.get("transactionHash").hex()
                                        except Exception:
                                            txh = None
                                    if txh:
                                        matched_tx = txh
                                        break
                            if matched_tx:
                                _update_payment_record_by_id(p.get("id"), {"status": "confirmed", "onchain_tx_hash": matched_tx})
                                if send_fn and user_id:
                                    try:
                                        send_fn(user_id, f"Payment auto-confirmed ✅ — tx: {matched_tx}\nReference: {p.get('reference')}")
                                    except Exception:
                                        pass
                                if admin_notify:
                                    try:
                                        admin_notify(f"Auto-confirmed direct payment ref={p.get('reference')} tx={matched_tx}")
                                    except Exception:
                                        pass
                        except Exception as e:
                            # ignore per-payment errors
                            if admin_notify:
                                try:
                                    admin_notify(f"Reconcile get_token_transfer_logs error: {e}")
                                except Exception:
                                    pass
                            continue

            # done scanning payments
        except Exception as outer:
            # keep the loop alive but warn admin
            try:
                if admin_notify:
                    admin_notify(f"reconcile_pending_payments outer loop error: {outer}\n{traceback.format_exc()}")
            except Exception:
                pass

        # sleep until next poll
        try:
            time.sleep(poll_interval_seconds)
        except KeyboardInterrupt:
            break
        except Exception:
            # if stop_event is set we break next iteration
            continue
            
# REPLACED: do NOT start the reconcile thread at import time.
# Provide a factory so main() can start it after init_payments_db() is run.
def _create_reconcile_thread(poll_interval_seconds: int = 300):
    """
    Create (but do not start) the reconcile background thread and register
    the stop-event globally so main() can start/stop it safely.
    Returns the Thread object.
    """
    _reconcile_stop = threading.Event()
    t = threading.Thread(
        target=reconcile_pending_payments,
        kwargs={"poll_interval_seconds": poll_interval_seconds, "stop_event": _reconcile_stop},
        daemon=True,
    )
    globals()['_reconcile_stop_event'] = _reconcile_stop
    globals()['_reconcile_thread'] = t
    return t

def create_server_proof_transfer(to_address, amount_native=0.0001, record_as_payment=True, description="proof-transfer"):
    """
    Send a small native transfer from the server wallet to a target address.

    - Supports encrypted server key in SERVER_WALLET_PRIVATE_KEY_ENC (Fernet token string),
      falling back to SERVER_WALLET_PRIVATE_KEY plaintext.
    - Robust to web3.py API differences across versions:
        - uses w3.is_connected() (with fallback)
        - uses Web3.to_checksum_address()
        - extracts raw signed tx bytes from multiple possible SignedTransaction attributes
    - Optionally records a confirmed 'proof' payment into the file-backed payments store.

    Returns:
        {"ok": True, "tx_hash": tx_hash, "payment_id": payment_id_or_None}
        or {"ok": False, "error": "..."}
    """
    try:
        from web3 import Web3, HTTPProvider
    except Exception as e:
        return {"ok": False, "error": "web3.py is required. Install with: pip install web3", "exc": str(e)}

    # RPC
    rpc = os.getenv("BASE_RPC_URL")
    if not rpc:
        return {"ok": False, "error": "BASE_RPC_URL env var not set"}

    try:
        w3 = Web3(HTTPProvider(rpc, request_kwargs={"timeout": 30}))
    except Exception as e:
        return {"ok": False, "error": f"failed to create Web3 provider: {e}"}

    # connection check (tolerant)
    try:
        connected = w3.is_connected()
    except Exception:
        try:
            connected = w3.isConnected()
        except Exception:
            connected = False
    if not connected:
        return {"ok": False, "error": f"cannot connect to RPC at {rpc}"}

    # Load private key: prefer encrypted env var
    priv = None
    enc_env = os.getenv("SERVER_WALLET_PRIVATE_KEY_ENC")
    if enc_env:
        try:
            crypto = CryptoManager(0)
            priv = crypto.decrypt(enc_env.encode("utf-8"))
        except Exception as e:
            return {"ok": False, "error": f"failed to decrypt SERVER_WALLET_PRIVATE_KEY_ENC: {e}", "trace": traceback.format_exc()}
    else:
        priv = os.getenv("SERVER_WALLET_PRIVATE_KEY")

    if not priv:
        return {"ok": False, "error": "server private key not found in env (SERVER_WALLET_PRIVATE_KEY_ENC or SERVER_WALLET_PRIVATE_KEY)"}

    # Build account
    try:
        acct = w3.eth.account.from_key(priv)
        from_addr = acct.address
    except Exception as e:
        return {"ok": False, "error": f"invalid server private key: {e}"}

    # Chain id
    try:
        chain_id = int(os.getenv("BASE_CHAIN_ID", os.getenv("BASE_CHAIN", os.getenv("CHAIN_ID", 84532))))
    except Exception:
        chain_id = 84532

    # Nonce, gas price
    try:
        nonce = w3.eth.get_transaction_count(from_addr)
    except Exception as e:
        return {"ok": False, "error": f"could not fetch nonce for {from_addr}: {e}"}

    try:
        gas_price = w3.eth.gas_price
    except Exception:
        gas_price = int(10 * (10 ** 9))

    # Compute value
    try:
        value = int(float(amount_native) * (10 ** 18))
    except Exception:
        return {"ok": False, "error": "invalid amount_native"}

    # to checksum address (tolerant)
    try:
        to_chks = Web3.to_checksum_address(to_address)
    except Exception:
        try:
            to_chks = Web3.to_checksum_address(to_address.lower())
        except Exception:
            # last resort: use input (may fail)
            to_chks = to_address

    tx = {
        "to": to_chks,
        "value": value,
        "gas": 21000,
        "gasPrice": gas_price,
        "nonce": nonce,
        "chainId": int(chain_id)
    }

    # Sign
    try:
        signed = w3.eth.account.sign_transaction(tx, private_key=priv)
    except Exception as e:
        return {"ok": False, "error": f"failed to sign transaction: {e}", "trace": traceback.format_exc()}

    # Robustly extract raw signed tx bytes across web3.py versions
    raw_tx = None
    attrs_to_try = ("rawTransaction", "raw_transaction", "rawTx", "raw_signed_tx", "raw_signed_transaction", "rawSignedTransaction", "raw")
    for a in attrs_to_try:
        raw_tx = getattr(signed, a, None)
        if raw_tx:
            break

    # If attribute returned hex string, convert to bytes
    if isinstance(raw_tx, str):
        h = raw_tx[2:] if raw_tx.startswith("0x") else raw_tx
        try:
            raw_tx = bytes.fromhex(h)
        except Exception:
            # leave as-is and let send_raw_transaction fail if unsupported
            pass

    if not raw_tx:
        # helpful debug message but return error to avoid silent failure
        try:
            attrs = [k for k in dir(signed) if not k.startswith("_")]
        except Exception:
            attrs = []
        return {"ok": False, "error": "could not find raw signed transaction bytes on SignedTransaction object", "signed_attrs": attrs}

    # Send
    try:
        tx_hash = w3.eth.send_raw_transaction(raw_tx).hex()
    except Exception as e:
        return {"ok": False, "error": f"tx submission failed: {e}", "trace": traceback.format_exc()}

    payment_id = None
    if record_as_payment:
        try:
            rec = {
                "merchant_payment_id": None,
                "user_id": None,
                "amount_usd": None,
                "amount_token": None,
                "token_symbol": "ETH",
                "token_chain": "base",
                "status": "confirmed",
                "hosted_url": None,
                "reference": f"PROOF-{int(time.time())}",
                "metadata": {"description": description, "tx": tx_hash},
                "onchain_tx_hash": tx_hash
            }
            saved = _create_payment_record(rec)
            payment_id = saved.get("id")
        except Exception:
            payment_id = None

    # Optionally wait for a short receipt (non-blocking behavior chosen here)
    try:
        # small wait for inclusion (best-effort)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    except Exception:
        receipt = None

    return {"ok": True, "tx_hash": tx_hash, "payment_id": payment_id, "receipt": None if not receipt else {"status": getattr(receipt, "status", None), "blockNumber": getattr(receipt, "blockNumber", None)}}
        
def cmd_pay_via_coinbase(user_id, amount_usd, db_conn, bot_instance):
    """
    Explicitly start the Coinbase Commerce path only.
    Useful if you have a UI choice that invokes Coinbase as the selected path.
    Returns the coinbase_create_checkout result.
    """
    send_fn = make_send_message_callable(bot_instance)
    try:
        res = coinbase_create_checkout(db_conn, user_id, amount_usd, currency="USDC", chain="base", metadata=None)
    except Exception as e:
        res = {"ok": False, "error": f"exception creating coinbase checkout: {e}"}
    if res.get("ok"):
        charge = res.get("charge") or {}
        hosted = charge.get("hosted_url") or (charge.get("data") or {}).get("hosted_url")
        msg = f"Please complete payment on Coinbase Commerce page:\n\n{hosted}"
        try:
            send_fn(user_id, msg)
        except Exception:
            pass
    else:
        try:
            send_fn(user_id, f"Could not create checkout: {res.get('error')}")
        except Exception:
            pass
    return res


def cmd_pay_direct(user_id, amount_usd, db_conn, bot_instance):
    """
    Explicitly start the direct/manual server-wallet fallback flow.
    Calls start_direct_payment_flow and sends the instruction to the user.
    """
    send_fn = make_send_message_callable(bot_instance)
    try:
        result = start_direct_payment_flow(db_conn, user_id, amount_usd, send_fn)
    except Exception as e:
        result = {"ok": False, "error": f"exception: {e}"}
    if not result.get("ok"):
        try:
            send_fn(user_id, f"Error starting direct payment: {result.get('error')}")
        except Exception:
            pass
    return result
    
def forward_to_base_account(dest_str: str, amount_native: float, job_id: str | None = None, user_id: int | None = None, description: str | None = None) -> dict:
    """
    Send a native Base transfer to dest_str (which can be a checksum address, a 0x hex,
    or a basename like 'alice.base'). Uses create_server_proof_transfer(...) for the actual send.

    Returns:
      {"ok": True|False, "tx_hash": str|null, "explorer_url": str|null, "error": str|null}
    """
    try:
        from web3 import Web3
    except Exception as e:
        return {"ok": False, "error": f"web3/eth-account required: {e}"}

    # 1) Resolve basename -> address if necessary
    to_address = None
    try:
        s = str(dest_str).strip()
        # prefix support: onchain:alice.base or onchain:0x...
        if s.lower().startswith("onchain:"):
            s = s.split(":", 1)[1].strip()

        # if looks like basename (endswith .base) -> resolve
        if s.lower().endswith(".base"):
            try:
                resolved = resolve_basename(s)  # returns address or None
                if not resolved:
                    return {"ok": False, "error": f"basename {s} could not be resolved"}
                to_address = resolved
            except Exception as e:
                return {"ok": False, "error": f"basename resolution failed: {e}"}
        # if looks like a hex address -> checksum it
        elif Web3.isAddress(s) or (s.startswith("0x") and len(s) == 42):
            try:
                to_address = Web3.toChecksumAddress(s)
            except Exception:
                to_address = s
        else:
            # Not a known format
            return {"ok": False, "error": "destination not recognized as .base name or 0x address"}

        if not to_address:
            return {"ok": False, "error": "could not determine a target address"}

    except Exception as e:
        return {"ok": False, "error": f"destination parsing error: {e}"}

    # 2) Amount validation
    try:
        amount_native = float(amount_native)
        if amount_native <= 0:
            return {"ok": False, "error": "onchain amount must be > 0"}
    except Exception:
        return {"ok": False, "error": "invalid onchain amount"}

    # 3) Call the existing helper that sends from server wallet
    try:
        # create_server_proof_transfer returns {"ok":True,"tx_hash":..., ...} style
        desc = description or (f"forward-job:{job_id}" if job_id else "forward-job")
        tx_result = create_server_proof_transfer(to_address, amount_native, record_as_payment=True, description=desc)
    except Exception as e:
        return {"ok": False, "error": f"failed to send onchain tx: {e}"}

    if not tx_result or not tx_result.get("ok"):
        return {"ok": False, "error": tx_result.get("error", "unknown error"), "tx_hash": tx_result.get("tx_hash")}

    # 4) Build explorer URL (use chain id from connect_base_web3)
    explorer = None
    try:
        w3 = connect_base_web3()
        chain_id = getattr(w3.eth, "chain_id", None) or None
        if chain_id == 84532:
            explorer = f"https://sepolia.basescan.org/tx/{tx_result.get('tx_hash')}"
        else:
            explorer = f"https://basescan.org/tx/{tx_result.get('tx_hash')}"
    except Exception:
        explorer = None

    # 5) Optionally record payment record in payments store (use _create_payment_record if available)
    try:
        rec = {
            "merchant_payment_id": f"forward-{job_id or 'unknown'}",
            "user_id": int(user_id) if user_id else None,
            "amount_usd": None,
            "amount_native": float(amount_native),
            "chain": "base",
            "to_address": to_address,
            "tx_hash": tx_result.get("tx_hash"),
            "note": f"forward job {job_id}"
        }
        if globals().get("_create_payment_record"):
            _create_payment_record(rec)
    except Exception:
        # non-fatal
        pass

    return {"ok": True, "tx_hash": tx_result.get("tx_hash"), "explorer_url": explorer}

def forward_to_trader_endpoint(endpoint_url: str, text: str, job_id: str | None = None, user_id: int | None = None, extra: dict | None = None) -> dict:
    """
    POST the text payload to a trading endpoint (webhook). Returns {"ok":True,"status_code":200,...} or {"ok":False,"error":...}.
    """
    try:
        import requests, json
    except Exception as e:
        return {"ok": False, "error": f"requests required: {e}"}

    payload = {
        "job_id": job_id,
        "user_id": user_id,
        "text": text,
        "extra": extra or {}
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(endpoint_url, json=payload, headers=headers, timeout=10)
        return {"ok": True, "status_code": resp.status_code, "text": resp.text, "url": endpoint_url}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def cmd_submit_tx(user_id, tx_hash, reference, *args, **kwargs):
    """
    Adapter wrapper that calls verify_direct_payment in file-backed mode.
    Keeps the same function name for handler compatibility.
    """
    # expects globals to provide send function and payments store
    return verify_direct_payment(None, user_id, reference, tx_hash, make_send_message_callable(globals().get('bot_instance')))

def cmd_pay_status(user_id, reference, *args, **kwargs):
    """
    Adapter: send the payment status to the user using file-backed store.
    """
    send_fn = make_send_message_callable(globals().get('bot_instance'))
    p = _find_payment_by_reference(reference)
    if not p:
        send_fn(user_id, f"No payment found for reference {reference}.")
        return {"ok": False, "error": "not_found"}
    msg = f"Payment {reference} (id={p.get('id')}) status: *{p.get('status')}*."
    if p.get('onchain_tx_hash'):
        msg += f"\nTx: {p.get('onchain_tx_hash')}"
    msg += f"\nAmount (USD): {p.get('amount_usd')}"
    send_fn(user_id, msg)
    return {"ok": True, "id": p.get('id'), "status": p.get('status'), "tx_hash": p.get('onchain_tx_hash')}

def admin_list_proofs(bot_instance, _db_conn, limit=20):
    """
    File-backed admin helper: fetch recent confirmed payments from payments.json and notify admin.
    _db_conn param ignored (kept for compatibility).
    """
    admin_notify = make_admin_notify_callable(bot_instance)
    store = _load_payments_store()
    confirmed = [p for p in store.get("payments", []) if p.get("status") == "confirmed"]
    confirmed.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
    rows = confirmed[:limit]
    if not rows:
        try:
            admin_notify("No confirmed payments found.")
        except Exception:
            pass
        return {"ok": True, "rows": []}

    lines = []
    out = []
    for p in rows:
        lines.append(f"id={p.get('id')} user={p.get('user_id')} ref={p.get('reference')} amt={p.get('amount_usd')} {p.get('token_symbol')}/{p.get('token_chain')} tx={p.get('onchain_tx_hash') or 'n/a'} updated={p.get('updated_at')}")
        out.append({
            "id": p.get("id"), "user_id": p.get("user_id"), "reference": p.get("reference"),
            "amount_usd": p.get("amount_usd"), "token": p.get("token_symbol"), "chain": p.get("token_chain"),
            "tx": p.get("onchain_tx_hash"), "updated_at": p.get("updated_at")
        })
    try:
        admin_notify("Recent confirmed payments:\n\n" + "\n".join(lines))
    except Exception:
        pass
    return {"ok": True, "rows": out}

def make_send_message_callable(bot_instance):
    """
    Return a function send_message_callable(user_id, text) that adapts to your bot instance.

    Behavior:
    - Tries common method names in order: `send_message`, `sendMessage`, `send`.
    - If the chosen method is an async coroutine, the returned callable will run it via `asyncio.get_event_loop().create_task`
      (fire-and-forget). If no event loop is running, it will run it with `asyncio.run`.
    - If the bot provides a `bot_instance.send_text`-style helper, it will still try the common names first.
    - Does not raise on send failure; it returns False on failure and True on success.

    Usage:
        send_message = make_send_message_callable(my_bot)
        send_message(12345678, "Hello!")  # returns True/False

    Note: This wrapper intentionally tolerates multiple bot implementations and keeps the bot-call site minimal.
    """

    # discover a send method on the bot instance
    candidate_names = ["send_message", "sendMessage", "send", "send_text", "send_text_message", "sendMessageAsync"]
    send_method = None
    for name in candidate_names:
        if hasattr(bot_instance, name):
            send_method = getattr(bot_instance, name)
            break

    # If bot has an attribute `bot` (common for wrappers) try that inner object too
    if send_method is None and hasattr(bot_instance, "bot"):
        inner = getattr(bot_instance, "bot")
        for name in candidate_names:
            if hasattr(inner, name):
                send_method = getattr(inner, name)
                bot_instance = inner  # prefer inner object for calls
                break

    # If still None, fall back to a generic "post to telegram API" if the instance has token + http client.
    fallback_http_post = None
    if send_method is None:
        # Attempt to compose a minimal HTTP POST sender via token + chat_id if possible
        token = getattr(bot_instance, "token", None) or getattr(bot_instance, "api_key", None)
        if token:
            import requests
            def _fallback_http_send(chat_id, text):
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                try:
                    r = requests.post(url, json={"chat_id": int(chat_id), "text": text})
                    return r.status_code == 200
                except Exception:
                    return False
            fallback_http_post = _fallback_http_send

    # If still nothing, create a no-op sender that logs to stdout
    if send_method is None and fallback_http_post is None:
        def _noop_send(chat_id, text):
            try:
                print(f"[bot-send-missing] to={chat_id} text={text}")
            except Exception:
                pass
            return False
        return _noop_send

    # Determine if send_method is coroutine
    is_coroutine = False
    if send_method is not None:
        is_coroutine = inspect.iscoroutinefunction(send_method)

    def _sync_wrapper(chat_id, text):
        try:
            if send_method is not None:
                # try common parameter orders: (chat_id, text) or (text, chat_id) or (chat_id, {"text":...})
                try:
                    # Preferred signature: send_message(chat_id, text, **kwargs)
                    send_method(int(chat_id), text)
                    return True
                except TypeError:
                    # try send_method(text, chat_id)
                    try:
                        send_method(text, int(chat_id))
                        return True
                    except Exception:
                        # try passing a single object argument (some wrappers expect dict)
                        try:
                            send_method({"chat_id": int(chat_id), "text": text})
                            return True
                        except Exception:
                            pass
                except Exception:
                    # other runtime error
                    return False
            elif fallback_http_post:
                return fallback_http_post(chat_id, text)
        except Exception:
            return False
        return False

    async def _async_call_and_ignore(coro):
        try:
            await coro
        except Exception:
            pass

    def _send_message(chat_id, text):
        """
        The callable returned to the codebase. Synchronous API: returns True/False.
        If underlying method is async, it schedules fire-and-forget and returns True if scheduled.
        """
        try:
            if send_method is not None:
                if is_coroutine:
                    # schedule coroutine without awaiting (fire-and-forget)
                    loop = None
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None

                    coro = send_method(int(chat_id), text)
                    if loop:
                        # schedule it
                        try:
                            loop.create_task(_async_call_and_ignore(coro))
                            return True
                        except Exception:
                            # fallback to run until complete
                            try:
                                asyncio.run(coro)
                                return True
                            except Exception:
                                return False
                    else:
                        # no running loop, run directly
                        try:
                            asyncio.run(coro)
                            return True
                        except Exception:
                            return False
                else:
                    return _sync_wrapper(chat_id, text)
            elif fallback_http_post:
                return fallback_http_post(chat_id, text)
            else:
                # no-op fallback
                try:
                    print(f"[bot-send] to={chat_id} text={text}")
                    return False
                except Exception:
                    return False
        except Exception:
            return False

    return _send_message

def make_admin_notify_callable(bot_instance, admin_chat_id=None):
    """
    Return an admin notifier callable admin_notify_callable(text) which notifies an admin channel/chat.

    - If admin_chat_id is provided, the returned function will send to that chat id.
    - Otherwise it attempts to read a sensible default from common attributes on bot_instance:
        - bot_instance.admin_chat_id
        - bot_instance.ADMIN_CHAT_ID
        - env var BOT_ADMIN_CHAT_ID
    - Uses the same send-message method detection from make_send_message_callable.

    Usage:
        admin_notify = make_admin_notify_callable(my_bot, admin_chat_id=12345678)
        admin_notify("Important event happened")

    Returns:
        function(text) -> True/False
    """

    # detect admin chat id if not provided
    admin_id = admin_chat_id
    if admin_id is None:
        admin_id = getattr(bot_instance, "admin_chat_id", None) or getattr(bot_instance, "ADMIN_CHAT_ID", None) or os.getenv("BOT_ADMIN_CHAT_ID")

    # reuse the send_message wrapper
    send_fn = make_send_message_callable(bot_instance)

    def _admin_notify(text):
        if not admin_id:
            # fallback to printing to stdout for visibility
            try:
                print("[admin-notify-missing] " + str(text))
            except Exception:
                pass
            return False
        try:
            return send_fn(admin_id, text)
        except Exception:
            try:
                print("[admin-notify-failed] " + str(text))
            except Exception:
                pass
            return False

    return _admin_notify

class UserSession:
    """Manages user session state during bot interaction."""
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.state = "idle"
        self.state_history = []
        self.temp_data = {}
        self.forwarder = None
        self.pending_job = {}
        self.modifying_job_index = None
        self.is_admin = False
        # --- NEW PAGINATION ATTRIBUTES ---
        self.pagination_page = 0
        self.pagination_data = []
        self.pagination_type = None
        self.last_paginated_message_id = None

    def set_state(self, new_state):
        """Sets a new state and records the old one in history."""
        if self.state != new_state: # Avoid duplicate states in history
            self.state_history.append(self.state)
        self.state = new_state

    def go_back(self):
        """Goes back to the previous state from history, if available."""
        if self.state_history:
            self.state = self.state_history.pop()
            return self.state
        # Fallback to idle if there's no history
        self.state = "idle"
        return self.state

class TelegramForwarder:
    """Manages all forwarding logic and state for a SINGLE user account."""
    # MODIFIED based on session_fix.txt and message_processing_integration.txt
    def __init__(self, api_id, api_hash, phone_number, user_id, bot_instance=None, is_demo_forwarder=False):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.user_id = user_id
        self.bot_instance = bot_instance  # Reference to bot for user notifications

        # MODIFIED to handle demo mode session file
        if is_demo_forwarder:
            # DEMO_SESSION_NAME is a global loaded from config
            self.session_file = os.path.join(SESSIONS_DIR, DEMO_SESSION_NAME)
        else:
            # Original logic
            session_name = f"session_{user_id}_{phone_number.replace('+', '')}"
            self.session_file = os.path.join(SESSIONS_DIR, session_name)

        self.client = TelegramClient(self.session_file, api_id, api_hash)
        self.jobs = []
        self.chat_cache = {}
        self.last_forwarded = {}
        self.is_authorized = False
        self.saved_patterns = []

        # --- THIS IS THE NEW EVENT HANDLER ---
        # It's registered the moment the object is created.
        @self.client.on(events.NewMessage())
        async def _message_handler(event):
            """This function is now automatically called by Telethon on a new message."""
            message = event.message

            # Check if this message is from a source channel for any active job
            for job in self.jobs:
                if message.chat_id in job.get('source_ids', []):
                    # If it's a match, process it immediately.
                    # The existing _process_message method can be reused perfectly.
                    await self._process_message(message, job)

    # Add this new method to the TelegramForwarder class
    async def start(self):
        """Connects the client and runs it until disconnected."""
        logging.info(f"Starting event-driven listener for {self.phone_number}...")
        # The 'run_until_disconnected' call will block here,
        # continuously listening for events and triggering the handler.
        await self.client.start()
        await self.client.run_until_disconnected()
        logging.info(f"Event-driven listener for {self.phone_number} has stopped.")

    # Add this new method to the TelegramForwarder class
    async def stop(self):
        """Gracefully disconnects the client."""
        if self.client.is_connected():
            logging.info(f"Stopping event-driven listener for {self.phone_number}...")
            # This will cause 'run_until_disconnected' in the start() method to complete.
            await self.client.disconnect()

    async def connect_and_authorize(self, phone_code=None, password=None):
        try:
            if not self.client.is_connected():
                await self.client.connect()

            if not await self.client.is_user_authorized():
                if phone_code:
                    await self.client.sign_in(self.phone_number, phone_code)
                else:
                    await self.client.send_code_request(self.phone_number)
                    return "code_requested"

            await self._build_chat_cache()
            self.is_authorized = True
            return "authorized"

        except SessionPasswordNeededError:
            if password:
                try:
                    await self.client.sign_in(password=password)
                    await self._build_chat_cache()
                    self.is_authorized = True
                    return "authorized"
                except errors.PasswordHashInvalidError:
                    return "invalid_password"
            return "2fa_required"
        except (ApiIdInvalidError, ValueError):
            return "invalid_api"
        except PhoneNumberInvalidError:
            return "invalid_phone"
        except PhoneCodeInvalidError:
            return "invalid_code"
        except errors.PhoneCodeExpiredError:
            logging.warning(f"Phone code expired for {self.phone_number}")
            return "code_expired"
        except errors.FloodWaitError as e:
            logging.warning(f"Flood wait error: {e.seconds} seconds")
            return f"flood_wait_{e.seconds}"
        except Exception as e:
            logging.error(f"Unexpected connection error for {self.phone_number}: {e}")
            error_str = str(e).lower()
            if "confirmation code has expired" in error_str or "expired" in error_str:
                return "code_expired"
            elif "flood" in error_str:
                return "flood_wait"
            return "error"

    # --- FIX #3: APPLIED ---
    # This entire method is replaced to fix the caching bug.
    async def _build_chat_cache(self):
        """
        Builds a cache of chat IDs to rich, composite titles, including both names and handles where available.
        """
        logging.info(f"Building chat cache for {self.phone_number}...")
        self.chat_cache.clear()

        async for dialog in self.client.iter_dialogs():
            # --- START OF NEW, COMPOSITE TITLE LOGIC ---

            entity = getattr(dialog, 'entity', None)
            # If for some reason there's no entity, we cannot get detailed info.
            # Fallback to the basic dialog name or the ID itself.
            if not entity:
                self.chat_cache[dialog.id] = dialog.name or f"Unknown Chat ({dialog.id})"
                continue

            primary_name = ""
            handle = getattr(entity, 'username', None)

            # 1. Determine the primary, human-readable name.
            if isinstance(entity, User):
                first = getattr(entity, 'first_name', '')
                last = getattr(entity, 'last_name', '')
                primary_name = f"{first} {last}".strip()
            elif hasattr(entity, 'title'): # This covers Channel, Chat, etc.
                primary_name = getattr(entity, 'title', '')

            # 2. Compose the final display title based on what we found.
            final_display_title = primary_name

            # If a handle exists, append it in parentheses for maximum clarity.
            if handle:
                # If a primary name also exists, combine them.
                if final_display_title:
                    final_display_title = f"{primary_name} (@{handle})"
                # Otherwise, the handle is the primary identifier.
                else:
                    final_display_title = f"@{handle}"

            # 3. Final failsafe if all attempts to find a name failed.
            if not final_display_title:
                final_display_title = f"Unknown Chat ({dialog.id})"

            # --- END OF NEW, COMPOSITE TITLE LOGIC ---

            self.chat_cache[dialog.id] = final_display_title

        logging.info(f"Chat cache for {self.phone_number} built with {len(self.chat_cache)} entries.")


    def get_chats_list(self):
        return [(chat_id, title) for chat_id, title in self.chat_cache.items()]

    async def _get_entity_details(self, identifier):
        try:
            # Try parsing as a numeric ID first
            numeric_id = int(identifier)
            # It's a valid integer, now check cache
            if numeric_id in self.chat_cache:
                return (numeric_id, self.chat_cache[numeric_id])
            # If not in cache, let get_entity handle it to fetch the name
        except (ValueError, TypeError):
            # Not a numeric ID, treat as string username/link
            pass

        try:
            entity = await self.client.get_entity(identifier)
            title = getattr(entity, 'title', getattr(entity, 'first_name', f"User {entity.id}"))
            if entity.id not in self.chat_cache:
                self.chat_cache[entity.id] = title
            return (entity.id, title)
        except Exception as e:
            logging.error(f"Could not find entity '{identifier}': {e}")
            raise ValueError(f"Could not find chat/user: {identifier}")

    # _persist_offsets is removed as saving is handled by the more general save_user_data

    # REPLACED based on message_processing_integration.txt
    async def _process_message(self, message, job):
        """Process message with enhanced error tracking and user notifications"""
        if not message.text:
            return

        forward_type = job.get('type')
        job_results = None
        match_logic = job.get('match_logic', 'OR')
        is_hl_dest = job.get('destination_ids') == ['hyperliquid_account']

        if forward_type == 'keywords':
            keywords = job.get('keywords', [])
            match = (not keywords) or \
                    (match_logic == 'AND' and all(k.lower() in message.text.lower() for k in keywords)) or \
                    (match_logic != 'AND' and any(k.lower() in message.text.lower() for k in keywords))

            if match:
                if is_hl_dest:
                    await self._execute_hyperliquid_trade(message, job)
                else:
                    job_results = await self._check_and_forward(message.text, f"keywords_{job.get('source_ids',[0])[0]}", job)

        elif forward_type == 'custom_pattern':
            patterns = job.get('patterns', [])
            if not patterns: return

            try:
                match = (match_logic == 'AND' and all(re.search(p, message.text) for p in patterns)) or \
                        (match_logic != 'AND' and any(re.search(p, message.text) for p in patterns))

                if match:
                    if is_hl_dest:
                        await self._execute_hyperliquid_trade(message, job)
                    else:
                        key = "&".join(patterns)
                        job_results = await self._check_and_forward(message.text, key, job)
            except re.error as e:
                logging.warning(f"Invalid regex in job for user {self.user_id}: {e}")

        elif forward_type == 'solana':
            if contract := self._find_solana_contract(message.text):
                if is_hl_dest:
                    # Create a temporary message object to pass the contract as text
                    temp_message = type('TempMessage', (), {'text': contract})()
                    await self._execute_hyperliquid_trade(temp_message, job)
                else:
                    job_results = await self._check_and_forward(contract, contract, job)

        elif forward_type == 'ethereum':
            if contract := self._find_ethereum_contract(message.text):
                if is_hl_dest:
                    temp_message = type('TempMessage', (), {'text': contract})()
                    await self._execute_hyperliquid_trade(temp_message, job)
                else:
                    job_results = await self._check_and_forward(contract, contract, job)

        elif forward_type == 'cashtags':
            found_tags = self._find_cashtag(message.text)
            specified_tags = [st.lower() for st in job.get('cashtags', [])]
            for tag in found_tags:
                if not specified_tags or tag.lower() in specified_tags:
                    if is_hl_dest:
                        temp_message = type('TempMessage', (), {'text': tag})()
                        await self._execute_hyperliquid_trade(temp_message, job)
                        break # Only process the first matched tag for HL
                    else:
                        tag_results = await self._check_and_forward(tag, tag, job)
                        if tag_results and not job_results:
                            job_results = tag_results

        # Notify user of any forwarding issues (if there's a bot reference and it wasn't a HL trade)
        if job_results and job_results.get('failed', 0) > 0 and hasattr(self, 'bot_instance'):
            await self.bot_instance.notify_user_of_forwarding_errors(self.user_id, job_results)

    # REPLACED based on enhanced_forwarding.txt
    async def _check_and_forward(self, text, key, job):
        """Forward message to all destinations with comprehensive error reporting"""
        if not self._can_forward(key, job['type'], self._parse_timer(job.get('timer', ''))):
            return

        destination_ids = job.get('destination_ids', [])
        if not destination_ids and not job.get('onchain_destinations'):
            logging.warning(f"[{self.phone_number}] No destination IDs found for job type '{job['type']}'")
            return

        # Track results for all destinations
        results = []
        successful_forwards = 0
        failed_forwards = 0

        logging.info(f"[{self.phone_number}] Starting forward operation to {len(destination_ids)} destinations")
        failed_details = []

        # --- existing Telegram forwards (unchanged) ---
        for dest_id in destination_ids:
            try:
                # existing send method; it may be self._send_message or similar
                sent = await self._send_message(dest_id, text) if hasattr(self, "_send_message") else await self.send_message(dest_id, text)
                ok = bool(sent)
            except Exception as e:
                ok = False
                logging.exception(f"[{self.phone_number}] error sending to {dest_id}: {e}")

            results.append({"dest": dest_id, "ok": ok})
            if ok:
                successful_forwards += 1
            else:
                failed_forwards += 1
                failed_details.append(str(dest_id))
                
        # --- NEW: forward text to trader endpoints (if any) ---
        trader_endpoints = job.get('trader_endpoints', []) or []
        if trader_endpoints:
            for url in trader_endpoints:
                try:
                    res = await loop.run_in_executor(None, forward_to_trader_endpoint, url, text, job.get('id'), self.user_id, {"source": "telegram_forward"})
                except Exception as e:
                    res = {"ok": False, "error": str(e)}
                if res.get("ok"):
                    results.append({"dest": url, "ok": True, "resp": res})
                    successful_forwards += 1
                else:
                    results.append({"dest": url, "ok": False, "error": res.get("error")})
                    failed_forwards += 1
                    failed_details.append(f"url:{res.get('error')}")

        # --- NEW: handle onchain destinations if present ---
        onchain_dests = job.get('onchain_destinations', []) or []
        onchain_transfer_enabled = bool(job.get('onchain_transfer', False)) or bool(job.get('onchain_amount'))
        onchain_amount = job.get('onchain_amount') or job.get('amount_native') or 0
        try:
            onchain_amount = float(onchain_amount) if onchain_amount else 0.0
        except Exception:
            onchain_amount = 0.0

        if onchain_dests and onchain_transfer_enabled and onchain_amount > 0:
            logging.info(f"[{self.phone_number}] Executing onchain forwards to {len(onchain_dests)} destinations")
            
            loop = asyncio.get_running_loop()
            for dest in onchain_dests:
                try:
                    # Offload the synchronous transfer helper to a thread to avoid blocking.
                    res = await loop.run_in_executor(None, forward_to_base_account, dest, onchain_amount, job.get('id'), self.user_id, f"forward:{job.get('type')}")
                except Exception as e:
                    res = {"ok": False, "error": str(e)}

                if res.get("ok"):
                    successful_forwards += 1
                    results.append({"dest": dest, "ok": True, "tx_hash": res.get("tx_hash"), "explorer": res.get("explorer_url")})
                else:
                    failed_forwards += 1
                    results.append({"dest": dest, "ok": False, "error": res.get("error")})
                    failed_details.append(f"{dest}:{res.get('error')}")
        else:
            # If there are onchain destinations but transfer not enabled, skip onchain sends.
            if onchain_dests and not onchain_transfer_enabled:
                logging.info(f"[{self.phone_number}] Found onchain destinations but onchain_transfer not enabled; skipping onchain sends.")

        # Log if many failed
        if failed_forwards > 0:
            logging.error(f"[{self.phone_number}] ❌ Failed to forward to {failed_forwards} destinations: {', '.join(failed_details)}")

        # Update forward time only if at least one destination succeeded
        if successful_forwards > 0:
            self._update_forward_time(key, job['type'])

        return {
            'total': len(destination_ids) + len(onchain_dests),
            'successful': successful_forwards,
            'failed': failed_forwards,
            'results': results
        }

    def _can_forward(self, key, type, timer):
        return not self.last_forwarded.get((type, key)) or (datetime.datetime.now() - self.last_forwarded[(type, key)]) >= timer

    def _update_forward_time(self, key, type):
        self.last_forwarded[(type, key)] = datetime.datetime.now()

    def _parse_timer(self, timer_str):
        if timer_str and len(parts := timer_str.lower().split()) == 2 and parts[0].isdigit():
            val, unit = int(parts[0]), parts[1]
            if "minute" in unit:
                return datetime.timedelta(minutes=val)
            elif "hour" in unit:
                return datetime.timedelta(hours=val)
        return datetime.timedelta()

    def _find_solana_contract(self, text):
        if match := re.search(r"[1-9A-HJ-NP-Za-km-z]{32,44}", text):
            return match.group(0) if 32 <= len(match.group(0)) <= 44 else None

    def _find_ethereum_contract(self, text):
        if match := re.search(r"0x[a-fA-F0-9]{40}", text):
            return match.group(0)

    def _find_cashtag(self, text):
        return re.findall(r"\$[A-Za-z0-9_]{1,16}", text)

    # REPLACED based on enhanced_forwarding.txt
    async def _send_message(self, dest_id, text):
        """Send message to destination with detailed error tracking"""
        dest_title = self.chat_cache.get(dest_id, f"ID {dest_id}")

        try:
            logging.info(f"[{self.phone_number}] Attempting to forward to '{dest_title}' ({dest_id})...")
            await self.client.send_message(dest_id, text)
            logging.info(f"[{self.phone_number}] ✅ Successfully forwarded to '{dest_title}' ({dest_id})")
            return {'success': True, 'destination': dest_title, 'dest_id': dest_id, 'error': None}

        except errors.UserIsBlockedError:
            error_msg = "Bot is blocked by user/chat"
            logging.error(f"[{self.phone_number}] ❌ Failed to forward to '{dest_title}' ({dest_id}): {error_msg}")
            return {'success': False, 'destination': dest_title, 'dest_id': dest_id, 'error': error_msg, 'error_type': 'blocked'}

        except errors.ChatWriteForbiddenError:
            error_msg = "No permission to write in this chat"
            logging.error(f"[{self.phone_number}] ❌ Failed to forward to '{dest_title}' ({dest_id}): {error_msg}")
            return {'success': False, 'destination': dest_title, 'dest_id': dest_id, 'error': error_msg, 'error_type': 'forbidden'}

        except errors.ChatAdminRequiredError:
            error_msg = "Admin privileges required to send messages"
            logging.error(f"[{self.phone_number}] ❌ Failed to forward to '{dest_title}' ({dest_id}): {error_msg}")
            return {'success': False, 'destination': dest_title, 'dest_id': dest_id, 'error': error_msg, 'error_type': 'admin_required'}

        except errors.UserNotParticipantError:
            error_msg = "Account is not a member of this chat"
            logging.error(f"[{self.phone_number}] ❌ Failed to forward to '{dest_title}' ({dest_id}): {error_msg}")
            return {'success': False, 'destination': dest_title, 'dest_id': dest_id, 'error': error_msg, 'error_type': 'not_member'}

        except errors.PeerIdInvalidError:
            error_msg = "Invalid chat ID or chat not accessible"
            logging.error(f"[{self.phone_number}] ❌ Failed to forward to '{dest_title}' ({dest_id}): {error_msg}")
            return {'success': False, 'destination': dest_title, 'dest_id': dest_id, 'error': error_msg, 'error_type': 'invalid_id'}

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(f"[{self.phone_number}] ❌ Failed to forward to '{dest_title}' ({dest_id}): {error_msg}")
            return {'success': False, 'destination': dest_title, 'dest_id': dest_id, 'error': error_msg, 'error_type': 'unknown'}

    async def _execute_hyperliquid_trade(self, message, job):
        """
        Parses a message, loads user's HL profile, and executes a trade.
        """
        user_id = self.user_id
        logging.info(f"[{self.phone_number}] Triggered Hyperliquid trade for user {user_id}.")

        try:
            # 1. Load user data and profile
            user_data = await self.bot_instance.load_user_data(user_id)
            if not user_data or not user_data.get('hl_api_key') or not user_data.get('hl_api_secret_encrypted'):
                await self.bot_instance.bot.send_message(user_id, "❌ **Trade Failed:** Hyperliquid account is not connected.")
                return

            profile = user_data.get('hl_trade_profile', {})
            required_keys = ['side', 'type', 'size_usd', 'leverage']
            if not all(key in profile for key in required_keys):
                await self.bot_instance.bot.send_message(user_id, "❌ **Trade Failed:** Your trade defaults are incomplete. Please set them in 'Manage Hyperliquid'.")
                return

            # 2. Decrypt the private key
            crypto = CryptoManager(user_id)
            private_key = crypto.decrypt(user_data['hl_api_secret_encrypted'].encode('utf-8'))
            account = Account.from_key(private_key)

            # 3. Parse the symbol from the message
            symbol = message.text.strip().upper().lstrip('$')
            if not symbol:
                logging.warning(f"[{self.phone_number}] Could not parse symbol from message: '{message.text}'")
                return

            # 4. Initialize SDK and get market info
            info = Info(constants.MAINNET_API_URL, skip_ws=True)
            exchange = Exchange(account, constants.MAINNET_API_URL)
            all_mids = info.all_mids()

            if symbol not in all_mids:
                await self.bot_instance.bot.send_message(user_id, f"❌ **Trade Failed:** The symbol `{symbol}` is not recognized by Hyperliquid.")
                return

            # 5. Calculate order size
            market_price = float(all_mids[symbol])
            order_size_in_asset = profile['size_usd'] / market_price

            meta = info.meta()
            asset_info = next((a for a in meta['universe'] if a['name'] == symbol), None)
            if not asset_info:
                await self.bot_instance.bot.send_message(user_id, f"❌ **Trade Failed:** Could not retrieve metadata for `{symbol}`.")
                return

            sz_decimals = asset_info['szDecimals']
            rounded_size = round(order_size_in_asset, sz_decimals)

            # 6. Execute the order
            is_buy = (profile['side'] == 'buy')
            leverage = profile['leverage']
            order_result = None

            if profile['type'] == 'market':
                slippage = 0.01
                order_result = exchange.market_open(symbol, is_buy, rounded_size, market_price * (1 + (slippage if is_buy else -slippage)), leverage)

            elif profile['type'] == 'limit':
                limit_rule = profile.get('limit_rule', {})
                if not limit_rule:
                    await self.bot_instance.bot.send_message(user_id, "❌ **Trade Failed:** No limit rule is set for limit orders.")
                    return

                # --- NEW: Limit Order Price Calculation ---
                l2_snapshot = info.l2_snapshot(symbol)
                best_bid = float(l2_snapshot['levels'][0][0]['px'])
                best_ask = float(l2_snapshot['levels'][1][0]['px'])

                limit_price = 0
                if limit_rule.get('mode') == 'percent_offset':
                    offset = float(limit_rule.get('offset', 0)) / 100
                    if is_buy:
                        limit_price = best_bid * (1 - offset) # Place below best bid
                    else:
                        limit_price = best_ask * (1 + offset) # Place above best ask

                elif limit_rule.get('mode') == 'fixed_offset':
                    offset = float(limit_rule.get('offset', 0))
                    if is_buy:
                        limit_price = best_bid - offset
                    else:
                        limit_price = best_ask + offset
                else:
                    await self.bot_instance.bot.send_message(user_id, f"❌ **Trade Failed:** Invalid limit rule mode '{limit_rule.get('mode')}'.")
                    return

                order_result = exchange.order(symbol, is_buy, rounded_size, limit_price, {"limit": {"tif": "Gtc"}})


            # 7. Notify user of the result
            if order_result and 'status' in order_result and order_result['status'] == 'ok':
                status = order_result['response']['data']['statuses'][0]
                if 'filled' in status:
                    fill_info = status['filled']
                    summary_msg = (
                        f"✅ **Hyperliquid Order Filled!**\n\n"
                        f"**Action:** `{profile['side'].upper()}`\n"
                        f"**Symbol:** `{fill_info['coin']}`\n"
                        f"**Amount:** `{fill_info['sz']}` {fill_info['coin']}\n"
                        f"**Avg. Price:** `${fill_info['avgPx']}`\n"
                        f"**Total Value:** `${float(fill_info['sz']) * float(fill_info['avgPx']):.2f} USD`"
                    )
                    await self.bot_instance.bot.send_message(user_id, summary_msg)
                else:
                    await self.bot_instance.bot.send_message(user_id, f"✅ **Hyperliquid Order Placed:** {status}")
            else:
                error_msg = order_result.get('response', 'Unknown error')
                await self.bot_instance.bot.send_message(user_id, f"❌ **Trade Failed:** Could not place order for `{symbol}`.\n**Reason:** `{error_msg}`")

        except Exception as e:
            logging.error(f"FATAL: Hyperliquid trade execution failed for user {user_id}: {e}", exc_info=True)
            await self.bot_instance.bot.send_message(user_id, "❌ **Critical Error:** A system error occurred while trying to place your trade. Please check the bot logs.")

    # ADDED from enhanced_forwarding.txt
    async def test_job_destinations(self, job):
        """Test all destinations in a job and return detailed results"""
        destination_ids = job.get('destination_ids', [])
        test_message = "gm"

        results = []
        for dest_id in destination_ids:
            result = await self._send_message(dest_id, test_message)
            results.append(result)

        return results
        
    # -----------------------------------------------------------
    # NEW: Auto-forward onchain events into in-memory forwarders
    # -----------------------------------------------------------
    async def process_tracked_wallet_event(self, user_id: int, event_info: dict):
        """
        Called by your polling engine whenever a tracked Base wallet receives a relevant transaction.
        We forward to the same destinations selected in normal Telegram job flows.
        """
        fwd = self.user_forwarders.get(user_id)
        if not fwd or not fwd.is_authorized:
            return

        tx_hash = event_info.get("hash")
        text = f"🔔 Onchain wallet activity detected\nTX: {tx_hash}"

        # Reuse _check_and_forward just like Telegram inbound forwarding
        await self._check_and_forward(
            text=text,
            key=str(tx_hash),   # dedupe key using tx hash
            job=None            # handled job-by-job by underlying logic
        )

class TelegramBot:
    """Main bot class that handles user interactions."""
    # MODIFIED based on session_fix.txt and user_notification_system.txt
    def __init__(self):
        # FIXED PATH from session_fix.txt
        bot_session_path = os.path.join(SESSIONS_DIR, 'bot_session')
        self.bot = TelegramClient(bot_session_path, BOT_API_ID, BOT_API_HASH)

        self.user_sessions = {}
        self.user_forwarders = {}
        # ADDED from user_notification_system.txt
        self.last_error_summary = {}   # Prevent spam of error notifications

        # ADDED for Demo Mode
        self.demo_mode = DEMO_MODE
        self.demo_forwarder = None
        self.forwarder_tasks = {} # <--- THIS LINE IS ADDED

        # --- START OF MODIFICATIONS AND ADDITIONS ---
        self.webhook_app = app
        self.setup_webhooks()
        # --- END OF MODIFICATIONS AND ADDITIONS ---

        # Register pagination handlers (FIXED)
        self.bot.add_event_handler(
            self.handle_user_chats_pagination,
            events.CallbackQuery(pattern=b'page_user_chats_(.*)')
        )
        self.bot.add_event_handler(
            self.handle_admin_chats_pagination,
            events.CallbackQuery(pattern=b'page_admin_chats_(.*)')
        )
        self.bot.add_event_handler(
            self.handle_page_close,
            events.CallbackQuery(pattern=b'page_close')
        )

    #
    # ... inside the TelegramBot class ...

    # Add this new method to the TelegramBot class
    async def start_forwarder_session(self, user_id, forwarder):
        """Creates and manages a background task for a user's forwarder."""
        if user_id in self.forwarder_tasks:
            logging.warning(f"Task for user {user_id} already exists. Stopping it before starting a new one.")
            await self.stop_forwarder_session(user_id)

        # asyncio.create_task runs the forwarder's start() method in the background
        task = asyncio.create_task(forwarder.start())
        self.forwarder_tasks[user_id] = task
        logging.info(f"Background listener task created for user {user_id}.")

    # Add this new method to the TelegramBot class
    async def stop_forwarder_session(self, user_id):
        """Stops and cleans up a user's forwarder task."""
        if user_id in self.forwarder_tasks:
            task = self.forwarder_tasks[user_id]
            forwarder = self.user_forwarders.get(user_id)

            # Gracefully stop the forwarder's client
            if forwarder:
                await forwarder.stop()

            # Cancel the asyncio task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logging.info(f"Background listener task for user {user_id} was successfully cancelled.")

            del self.forwarder_tasks[user_id]

    # In your TelegramBot class, add these callback
    async def handle_user_chats_pagination(self, event):
        try:
            page_index = int(event.data.decode().split('_')[-1])
            await self.list_user_chats(event, page_index=page_index)
            # DON'T call event.answer() here - it's already called in show_paginated_list
        except (ValueError, IndexError) as e:
            logging.error(f"Error handling user chats pagination: {e}")
            await event.answer("Invalid page request")  # Only answer on error

    async def handle_admin_chats_pagination(self, event):
        try:
            page_index = int(event.data.decode().split('_')[-1])
            await self.admin_view_all_chats(event, page_index=page_index)
            # DON'T call event.answer() here - it's already called in show_paginated_list
        except (ValueError, IndexError) as e:
            logging.error(f"Error handling admin chats pagination: {e}")
            await event.answer("Invalid page request")  # Only answer on error

    async def handle_page_close(self, event):
        """Handle close button for paginated lists"""
        try:
            await event.edit("Menu closed.", buttons=None)
            await event.answer()
        except Exception as e:
            logging.error(f"Error closing paginated menu: {e}")
            await event.answer()

    # --- START OF MODIFICATIONS AND ADDITIONS ---
    # Phase 4: Backend Webhook Integration
    def setup_webhooks(self):
        @self.webhook_app.post("/webhook/paystack")
        async def handle_paystack_webhook(request: Request):
            body = await request.body()
            signature = request.headers.get('x-paystack-signature')

            secret = PAYSTACK_TEST_SECRET_KEY if PAYSTACK_TEST_MODE else PAYSTACK_LIVE_SECRET_KEY

            # Verify the webhook signature
            if not hmac.compare_digest(
                hmac.new(secret.encode(), body, hashlib.sha512).hexdigest(),
                signature
            ):
                return JSONResponse(content={"status": "error", "message": "Invalid signature"}, status_code=400)

            event_data = await request.json()

            if event_data['event'] == 'charge.success':
                user_id = event_data['data']['metadata']['user_id']
                amount = event_data['data']['amount'] / 100

                # Determine plan and duration
                if amount == PRICES['monthly_usd']:
                    plan_duration_days = 30
                    plan_type = 'Monthly'
                else:
                    plan_duration_days = 365
                    plan_type = 'Yearly'

                await self.activate_subscription(user_id, plan_duration_days, "paystack", plan_type)

            return JSONResponse(content={"status": "success"}, status_code=200)

        @self.webhook_app.post("/webhook/nowpayments")
        async def handle_nowpayments_webhook(request: Request, x_nowpayments_sig: str = Header(None)):
            body = await request.body()

            # Verify the IPN signature
            if NOWPAYMENTS_IPN_SECRET:
                signature = hmac.new(NOWPAYMENTS_IPN_SECRET.encode(), body, hashlib.sha512).hexdigest()
                if signature != x_nowpayments_sig:
                     return JSONResponse(content={"status": "error", "message": "Invalid signature"}, status_code=400)

            payment_data = await request.json()

            if payment_data.get('payment_status') == 'finished':
                user_id = int(payment_data.get('order_id'))
                price_amount = float(payment_data.get('price_amount'))

                # Determine plan and duration
                if price_amount == PRICES['monthly_usd']:
                    plan_duration_days = 30
                    plan_type = 'Monthly'
                else:
                    plan_duration_days = 365
                    plan_type = 'Yearly'

                await self.activate_subscription(user_id, plan_duration_days, "nowpayments", plan_type)

            return JSONResponse(content={"status": "success"}, status_code=200)

    def run_webhook_server(self):
        uvicorn.run(self.webhook_app, host="0.0.0.0", port=WEBHOOK_PORT)

    async def activate_subscription(self, user_id, duration_days, payment_method, plan_type, promo_code=None):
        """
        Activate/extend subscription. This version *stacks* promo durations by extending
        the current expiry (i.e. new_expiry = max(now, current_expiry) + duration_days).
        Also records the promo in a small per-user promo history list.

        Returns True on success, False on failure.
        """
        try:
            user_data = await self.load_user_data(user_id)
            if not user_data:
                logging.error(f"activate_subscription: user_data for {user_id} not found.")
                return False

            # --- Payment record (promo codes record amount 0.0) ---
            payment_amount = 0.0
            if payment_method != "promo_code":
                if duration_days > 360:
                    payment_amount = PRICES.get('yearly_usd', 0)
                else:
                    payment_amount = PRICES.get('monthly_usd', 0)

            payment_record = {
                "date": datetime.datetime.now().isoformat(),
                "amount": payment_amount,
                "plan": plan_type,
                "payment_method": payment_method,
                "promo_code": promo_code
            }
            user_data.setdefault('payment_history', []).append(payment_record)
            # --- end payment record ---

            # --- Compute new expiry by stacking ---
            raw_expiry = user_data.get('subscription_expiry_date')
            if isinstance(raw_expiry, str):
                try:
                    current_expiry = datetime.datetime.fromisoformat(raw_expiry)
                except Exception:
                    current_expiry = datetime.datetime.now()
            elif isinstance(raw_expiry, datetime.datetime):
                current_expiry = raw_expiry
            else:
                current_expiry = datetime.datetime.now()

            base = current_expiry if current_expiry > datetime.datetime.now() else datetime.datetime.now()
            new_expiry = base + datetime.timedelta(days=duration_days)

            user_data['subscription_status'] = 'paid'
            user_data['subscription_expiry_date'] = new_expiry.isoformat()
            user_data['plan_type'] = plan_type

            # Record the active promo (keeps last applied code for backward compatibility)
            if promo_code:
                user_data['active_promo_code'] = promo_code
                # keep a history of applied promos (stack trace, useful for admin/undo)
                user_data.setdefault('applied_promo_codes', []).append({
                    "code": promo_code,
                    "days": duration_days,
                    "applied_at": datetime.datetime.now().isoformat(),
                    "new_expiry": new_expiry.isoformat()
                })
            else:
                # don't clear 'active_promo_code' if there is one; preserve currently applied promo
                user_data.setdefault('active_promo_code', user_data.get('active_promo_code'))

            # Affiliate handling (same as before)
            referrer_id = user_data.get('referrer_id')
            if referrer_id:
                referrer_data = await self.load_user_data(referrer_id)
                if referrer_data:
                    is_first_sub = not user_data.get('has_had_first_subscription')
                    commission = 0
                    if duration_days > 360:
                        commission = COMMISSIONS['yearly_new_subscriber'] if is_first_sub else COMMISSIONS['yearly_upgrade']
                    else:
                        commission = COMMISSIONS['first_month_bonus'] if is_first_sub else COMMISSIONS['subsequent_month']
                    referrer_data['unpaid_commissions'] = referrer_data.get('unpaid_commissions', 0.0) + commission
                    await self.save_user_data(referrer_id, None, referrer_data)

            user_data['has_had_first_subscription'] = True

            # Persist the user data
            saved_ok = await self.save_user_data(user_id, None, direct_data=user_data)
            if not saved_ok:
                logging.error(f"activate_subscription: save_user_data failed for {user_id}")
                return False

            # Send confirmation message
            await self.bot.send_message(user_id, f"✅ Your subscription is now active! Your plan will renew on {new_expiry.strftime('%Y-%m-%d')}.")

            return True

        except Exception as e:
            logging.error(f"activate_subscription failed for user {user_id}: {e}", exc_info=True)
            try:
                await self.bot.send_message(user_id, "❌ Failed to activate subscription. Please contact support.")
            except Exception:
                logging.error("Also failed to send failure message to user.", exc_info=True)
            return False

    async def admin_generate_revenue_report(self, event):
        """Generates a CSV report of revenue segmented by month from payment history."""
        await event.reply("⏳ Generating revenue report from transaction history, please wait...")

        revenue_by_month = {}  # Format: {'YYYY-MM': revenue}

        # Iterate through every user data file
        for filename in os.listdir(DATA_DIR):
            if filename.startswith("user_") and filename.endswith(".dat"):
                try:
                    user_id = int(filename.split("_")[1].split(".")[0])
                    user_data = await self.load_user_data(user_id)

                    # Check if the user has a payment history
                    if user_data and 'payment_history' in user_data:
                        # Iterate through each recorded payment for the user
                        for payment in user_data['payment_history']:
                            # IMPORTANT: Skip promo codes as they are not real revenue
                            if payment.get('payment_method') == 'promo_code':
                                continue

                            # Get the payment date and amount from the record
                            payment_date = datetime.datetime.fromisoformat(payment['date'])
                            payment_amount = payment.get('amount', 0.0)

                            # Add the amount to the correct month in our report dictionary
                            month_key = payment_date.strftime('%Y-%m')
                            revenue_by_month[month_key] = revenue_by_month.get(month_key, 0.0) + payment_amount
                except Exception as e:
                    logging.error(f"Could not process data file {filename} for revenue report: {e}")
                    continue

        # --- The report generation logic remains the same ---
        if not revenue_by_month:
            await event.reply("No revenue data found in user payment histories.")
        else:
            # Create the CSV content
            report_content = "Month,Total Revenue (USD)\n"
            total_revenue = 0.0
            # Sort by month for a clean report
            for month in sorted(revenue_by_month.keys()):
                revenue = revenue_by_month[month]
                total_revenue += revenue
                report_content += f"{month},{revenue:.2f}\n"

            report_content += f"\nTotal Yearly Revenue,{total_revenue:.2f}"

            # Save the content to a temporary file
            report_filename = f"revenue_report_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
            with open(report_filename, 'w') as f:
                f.write(report_content)

            # Send the file to the admin
            await event.reply("✅ Report generated successfully!", file=report_filename)

            # Clean up the temporary file
            os.remove(report_filename)

        # This line now runs for both cases, bringing back the admin menu.
        await self.show_admin_menu(event)

    async def show_prompt_for_state(self, event, state, extra_info=""):
        """
        Replies to the user with the correct prompt AND a dynamic keyboard for a given state.
        This is the definitive, complete version.
        """
        prompt = ""
        buttons = [] # The keyboard layout

        # Default keyboard for all multi-step conversational flows
        step_keyboard = [
            [Button.text("⬅️ Back", resize=True), Button.text("❌ Cancel", resize=True)]
        ]

        # --- THIS IS THE CORRECTED AND INCLUDED BLOCK ---
        if state == "creating_job":
            # The 'start_job_creation' function is the official entry point.
            # When we need to show this prompt (either at the start or by going 'back'),
            # we call it to ensure the state machine is correctly initialized.
            await self.start_job_creation(event, is_going_back=bool(extra_info))
            return
        # --- END OF CORRECTION ---

        elif state == "awaiting_api_id":
            prompt = ("""🔧 **Account Setup**\n\nTo connect your Telegram account, I need API credentials from https://my.telegram.org.\n\nPlease send me your **API ID**.""")
            buttons = [[Button.text("❌ Cancel", resize=True)]] # No 'Back' on the very first step

        elif state == "awaiting_api_hash":
            prompt = "✅ API ID saved!\n\nNow send your **API Hash**."
            buttons = step_keyboard

        elif state == "awaiting_phone":
            prompt = "✅ API Hash saved!\n\nNow send your **phone number** (e.g., +1234567890)."
            buttons = step_keyboard

        elif state == "awaiting_code":
            prompt = ("📲 **Verification code sent!**\n\n⚡ Please check Telegram and send me the code.\n\n🔢 **IMPORTANT:** Obfuscate the code (e.g., `1 a 2 b 3`) to prevent expiration.")
            buttons = step_keyboard

        elif state == "awaiting_2fa":
            prompt = "🔐 **Two-factor authentication required**\n\nPlease send your 2FA password:"
            buttons = step_keyboard

        elif state == "job_sources":
            prompt = (f"{extra_info}\n\n📥 Send me the source chat(s).\n\n*(This is the first step, so 'Back' will cancel.)*")
            buttons = step_keyboard

        elif state == "job_destinations":
            prompt = (f"{extra_info}\n\n"
                      "📤 **Set Destination**\n\n"
                      "Send the destination chat(s) (e.g., `@username, My Channel`).\n\n"
                      "Alternatively, press the button below to use your connected **Hyperliquid Account** as the destination to automatically place trades.")

            # Add the new button to the standard step keyboard
            hyperliquid_button = [Button.text("🏦 Use Hyperliquid Account", resize=True)]
            base_wallet_button = [Button.text("🪙 Use Base Wallet Source", resize=True)]
            buttons = [hyperliquid_button, base_wallet_button] + step_keyboard

        elif state == "job_keywords":
            prompt = (f"{extra_info}\n\n🔤 Send keywords (comma-separated), or 'none' for all.")
            buttons = step_keyboard

        elif state == "job_cashtags":
            prompt = (f"{extra_info}\n\n💰 Send cashtags (e.g., $BTC), or 'none' for all.")
            buttons = step_keyboard

        elif state == "job_timer":
            prompt = (f"{extra_info}\n\n⏱️ Set a cooldown timer (e.g., '5 minutes'), or 'none'.")
            buttons = step_keyboard

        elif state == "awaiting_job_name":
            prompt = (f"{extra_info}\n\n"
                      "📝 **Name Your Job (Optional)**\n\n"
                      "Finally, please give this job a short, memorable name (e.g., 'Dannyboy solana Alerts`).\n\n"
                      "You can also type `skip` or `none` to use a default name.")
            buttons = step_keyboard

        elif state == "awaiting_redeem_code":
            prompt = "🎁 Please send me the code you wish to redeem."
            buttons = [[Button.text("❌ Cancel", resize=True)]]

        # States that show their own full-screen menus are handled here
        elif state in ["job_management", "other_settings_menu", "job_awaiting_match_logic", "deleting_job", "modifying_job"]:
            if state == "job_management": await self.show_job_management(event)
            elif state == "other_settings_menu": await self.show_other_settings_menu(event)
            elif state == "job_awaiting_match_logic": await self.ask_for_match_logic(event)
            elif state == "deleting_job": await self.show_job_deletion(event)
            elif state == "modifying_job": await self.show_job_modification_menu(event)
            return

        else:
            await event.reply("↩️ Returning to the main menu.", buttons=Button.clear())
            await self.send_main_menu(event)
            return

        if prompt:
            await event.reply(prompt, buttons=buttons)
            
    # -----------------------
    # Ensure prompts for new job states exist (safe, idempotent)
    # Paste this after your existing show_prompt_for_state(...) definition,
    # or before async def handle_job_steps(...) if you don't have a prompts map.
    # -----------------------
    try:
        # try to detect an existing prompts map used by show_prompt_for_state
        PROMPT_MAP
    except NameError:
        PROMPT_MAP = {}

    # Minimum prompts for our new states (will not overwrite existing entries)
    PROMPT_MAP.setdefault("job_source_base_wallet", "Send the wallet address you want to use as a source (0x... or alice.base).")
    PROMPT_MAP.setdefault("awaiting_job_wallet_signature", "Please paste the signature for the challenge you were given (optionally prefixed by your address). Example: `0xYourAddr 0xSIG...`")
    PROMPT_MAP.setdefault("job_onchain_amount", "Enter the native Base amount to send per trigger (example: `0.0001`). Type `skip` to skip setting an onchain amount.")

    # If show_prompt_for_state is not present in globals(), add a minimal safe wrapper so calls succeed.
    # This wrapper uses PROMPT_MAP above and will not override an existing _show_prompt_for_state_fallback.
    if 'show_prompt_for_state' not in globals():
        async def _show_prompt_for_state_fallback(event, state, extra_info=None):
            from telethon import Button
            txt = PROMPT_MAP.get(state)
            if extra_info:
                if txt:
                    txt = txt + "\n\n" + extra_info
                else:
                    txt = extra_info
            if not txt:
                txt = "Please send the required information."
            try:
                await event.reply(txt, buttons=Button.clear())
            except Exception:
                # best-effort fallback
                await event.respond(txt)


    # Add this entire new function to your TelegramBot class
    async def handle_referral_command(self, event):
        """
        Generates and displays the user's unique referral link, their current
        commission balance, and a payout button if they meet the threshold.
        """
        user_id = event.sender_id

        try:
            # --- START OF NEW LOGIC: Count Referrals ---
            referral_count = 0
            # Iterate through every user data file in the data directory
            for filename in os.listdir(DATA_DIR):
                if filename.startswith("user_") and filename.endswith(".dat"):
                    try:
                        # Extract the user ID from the filename
                        referred_user_id = int(filename.split("_")[1].split(".")[0])

                        # A user cannot refer themselves, so we can skip their own file
                        if referred_user_id == user_id:
                            continue

                        # Load the data for the other user
                        referred_user_data = await self.load_user_data(referred_user_id)

                        # Check if their referrer_id matches the current user's ID
                        if referred_user_data and referred_user_data.get('referrer_id') == user_id:
                            referral_count += 1

                    except (ValueError, IndexError):
                        # Skip corrupted filenames
                        continue
            # --- END OF NEW LOGIC ---

            bot_info = await self.bot.get_me()
            bot_username = bot_info.username
            referral_link = f"https.t.me/{bot_username}?start=ref_{user_id}"

            user_data = await self.load_user_data(user_id)
            # Safely get the commission value, defaulting to 0.0 if not present
            commissions = user_data.get('unpaid_commissions', 0.0)

            # --- MODIFIED MESSAGE STRING ---
            message = (
                f"**Your Referral Link**\n\n"
                f"Share this link with others. When they subscribe, you'll earn a commission.\n\n"
                f"`{referral_link}`\n\n"
                f"👥 **Total Referrals:** {referral_count}\n"
                f"💰 **Unpaid Commissions:** `${commissions:.2f}`"
            )

            buttons = None
            # Check if the user's commission balance is at or above the minimum required for a payout
            if commissions >= COMMISSIONS['minimum_payout_threshold']:
                buttons = [[Button.inline("Request Payout", data="request_payout")]]

            await event.reply(message, buttons=buttons, link_preview=False)

        except Exception as e:
            logging.error(f"Error generating referral info for user {user_id}: {e}")
            await event.reply("❌ Could not generate your referral information at this time. Please try again later.")

    # Add this entire new function to your TelegramBot class
    async def handle_redeem_command(self, event, code):
        """
        Validates a promo code and applies its effect if valid.
        Uses a tiny POSIX file lock around codes.json read-modify-write to avoid
        concurrent write corruption when running in Colab / multi-cell.
        """
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        codes_file = os.path.join(DATA_DIR, "codes.json")

        if not os.path.exists(codes_file):
            await event.reply("❌ Invalid code. The code database does not exist.")
            session.set_state("idle")
            await self.send_main_menu(event)
            return

        try:
            # open in r+ so we can read and rewrite safely
            with open(codes_file, 'r+') as f:
                # --- LOCK ---
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    codes = json.load(f)

                    if code in codes and not codes[code].get('used'):
                        code_type = codes[code]['type']
                        value = codes[code]['value']

                        if code_type == 'subscription':
                            success = await self.activate_subscription(
                                user_id, value, "promo_code",
                                plan_type=f"{value}-Day Pass",
                                promo_code=code
                            )

                            if success:
                                # mark used
                                codes[code]['used'] = True
                                codes[code]['used_by'] = user_id
                                codes[code]['used_date'] = datetime.datetime.now().isoformat()

                                # write back safely
                                f.seek(0)
                                json.dump(codes, f, indent=4)
                                f.truncate()

                                # --- SUCCESS MESSAGE (remove DEBUG block here if desired) ---
                                debug_user_data = await self.load_user_data(user_id)
                                expiry_iso = debug_user_data.get('subscription_expiry_date')
                                # Be defensive: expiry_iso might be None
                                if expiry_iso:
                                    try:
                                        expiry_date = datetime.datetime.fromisoformat(expiry_iso)
                                        expiry_str = expiry_date.strftime('%Y-%m-%d')
                                    except Exception:
                                        expiry_str = expiry_iso
                                else:
                                    expiry_str = "Unknown"

                                await event.reply(
                                    f"✅ **Promo Code Applied Successfully!**\n\n"
                                    f"🎁 **Code:** `{code}`\n"
                                    f"📅 **Plan:** {value}-Day Pass\n"
                                    f"🗓️ **Expires:** {expiry_str}",
                                    buttons=Button.clear()
                                )
                            else:
                                await event.reply("❌ Failed to activate subscription. Please contact support.")
                                session.set_state("idle")
                                await self.send_main_menu(event)
                                return

                        session.set_state("idle")
                        await self.send_main_menu(event)
                    else:
                        await event.reply("❌ This code is invalid or has already been used. Please try again or type 'cancel'.")
                finally:
                    # --- UNLOCK ---
                    fcntl.flock(f, fcntl.LOCK_UN)

        except (json.JSONDecodeError, FileNotFoundError):
            await event.reply("❌ An error occurred while trying to validate your code. Please contact support.")
            session.set_state("idle")
            await self.send_main_menu(event)
        except Exception as e:
            logging.error(f"Error during code redemption for user {user_id} with code '{code}': {e}")
            await event.reply("❌ A critical error occurred. Please try again later.")
            session.set_state("idle")
            await self.send_main_menu(event)

    # Add this entire new function to your TelegramBot class
    async def handle_subscription_status(self, event):
        """Displays the user's current subscription status."""
        user_id = event.sender_id
        user_data = await self.load_user_data(user_id)

        if not user_data:
            await event.reply("Could not retrieve your account details.")
            return

        status = user_data.get('subscription_status', 'free')

        if status == 'paid':
            plan = user_data.get('plan_type', 'Paid')
            expiry_iso = user_data.get('subscription_expiry_date')
            expiry_date = datetime.datetime.fromisoformat(expiry_iso).strftime('%B %d, %Y')

            message = (
                f"**Your Subscription Status**\n\n"
                f"🟢 **Status:** `PAID`\n"
                f"📄 **Plan:** `{plan}`\n"
                f"🗓️ **Next Renewal Date:** `{expiry_date}`\n\n"
                f"You have access to all premium features."
            )
            await event.reply(message)
        else:
            message = (
                "**Your Subscription Status**\n\n"
                "⚪️ **Status:** `FREE`\n\n"
                "You are currently on the free plan with limited features. "
                "Use the /subscribe command to upgrade and unlock the full power of the bot!"
            )
            buttons = [
                [Button.text("🚀 Upgrade Now (/subscribe)", resize=True)]
            ]
            await event.reply(message, buttons=buttons)

    async def handle_awaiting_job_name(self, event, message):
        """Processes the optional job name and finalizes job creation."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        job_name = message.strip()

        if job_name.lower() in ['skip', 'none', '']:
            # Generate a default name
            job_type_str = session.pending_job['type'].replace('_', ' ').capitalize()
            job_count = len(forwarder.jobs) + 1
            job_name = f"{job_type_str} Job #{job_count}"

        session.pending_job['job_name'] = job_name

        # Now we can finalize the job
        await self.finalize_job_creation(event, session, forwarder)

    async def handle_hyperliquid_menu(self, event, message):
        """Handles button clicks from the 'Manage Hyperliquid' menu."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        if message == "🔗 Connect Account":
            session.set_state("awaiting_hl_credentials")

            # We now provide the standard conversational keyboard and remove the old menu.
            prompt = (
                "🔐 **Connect Hyperliquid Account**\n\n"
                "To connect, you must create an **Agent** in your Hyperliquid API settings.\n\n"
                "Please send your Agent's **Address (API Key)** and its **Private Key** in a single message, separated by a space.\n\n"
                "**Format:** `<agent_address> <private_key>`\n\n"
                "⚠️ Your private key will be encrypted and is required to sign trades. For your security, this message will be deleted after processing."
            )
            # This is the standard keyboard for multi-step conversations
            step_keyboard = [
                [Button.text("⬅️ Back", resize=True), Button.text("❌ Cancel", resize=True)]
            ]
            await event.reply(prompt, buttons=step_keyboard)

        elif message == "⚙️ Set Trade Defaults":
            session.set_state("awaiting_hl_trade_defaults")

            # Load current defaults to show the user
            user_data = await self.load_user_data(user_id)
            profile = user_data.get('hl_trade_profile', {})

            current_defaults = (
                f"▶️ **Current Side:** `{profile.get('side', 'Not Set')}`\n"
                f"▶️ **Current Type:** `{profile.get('type', 'Not Set')}`\n"
                f"▶️ **Current Size (USD):** `{profile.get('size_usd', 'Not Set')}`\n"
                f"▶️ **Current Leverage:** `{profile.get('leverage', 'Not Set')}x`"
            )

            await event.reply(
                f"⚙️ **Set Trade Defaults**\n\n{current_defaults}\n\n"
                "Please send your desired defaults in a single message. You can set one or more at a time.\n\n"
                "**Format:** `side=<value> type=<value> size_usd=<value> leverage=<value>`\n\n"
                "**Example:** `side=buy type=market size_usd=50 leverage=10`"
            )

        elif message == "🗑️ Disconnect Account":
            user_data = await self.load_user_data(user_id)
            user_data['hl_api_key'] = None
            user_data['hl_api_secret_encrypted'] = None
            # Optionally clear the trade profile as well
            user_data['hl_trade_profile'] = {}

            await self.save_user_data(user_id, None, direct_data=user_data)

            await event.reply("✅ **Account Disconnected.** Your Hyperliquid credentials have been removed.")
            await self.show_hyperliquid_menu(event)

        elif message == "🔙 Back to Other Settings":
            await self.show_other_settings_menu(event)

        else:
            await event.reply("Please use one of the menu buttons.")

    async def handle_awaiting_hl_credentials(self, event, message):
        """Processes and saves the user's Hyperliquid API credentials."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        try:
            # Delete the user's message containing secrets for security
            await event.delete()

            parts = message.strip().split()
            if len(parts) != 2:
                await event.reply("❌ **Invalid Format.**\nPlease send the API Key and Secret separated by a single space.")
                return

            api_key, api_secret = parts[0], parts[1]

            # Encrypt the secret key
            crypto = CryptoManager(user_id)
            encrypted_secret = crypto.encrypt(api_secret).decode('utf-8')

            # Load existing data to not overwrite it
            user_data = await self.load_user_data(user_id)

            user_data['hl_api_key'] = api_key
            user_data['hl_api_secret_encrypted'] = encrypted_secret

            # Initialize trade profile if it doesn't exist
            if 'hl_trade_profile' not in user_data:
                user_data['hl_trade_profile'] = {}

            await self.save_user_data(user_id, None, direct_data=user_data)

            await event.reply("✅ **Hyperliquid Account Connected Successfully!**")

            # Return to the hyperliquid menu to show the updated status
            await self.show_hyperliquid_menu(event)

        except Exception as e:
            logging.error(f"Failed to process Hyperliquid credentials for user {user_id}: {e}", exc_info=True)
            await event.reply("❌ An unexpected error occurred while saving your credentials. Please try again.")
            session.set_state("hyperliquid_menu") # Revert state

    async def handle_awaiting_hl_trade_defaults(self, event, message):
        """Processes and saves the user's default trading parameters."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        try:
            user_data = await self.load_user_data(user_id)
            # Ensure the profile dictionary exists to avoid errors
            if 'hl_trade_profile' not in user_data or user_data['hl_trade_profile'] is None:
                user_data['hl_trade_profile'] = {}

            # Standardize input for parsing
            parts = message.strip().lower().split()

            # Temporary dictionary to hold validated updates
            updates = {}

            # --- Validation Rules ---
            # A dictionary mapping each key to a validation function.
            # This makes it easy to add more rules in the future.
            validations = {
                'side': lambda value: value in ['buy', 'sell'],
                'type': lambda value: value in ['market', 'limit'],
                'size_usd': lambda value: float(value) > 0,
                'leverage': lambda value: int(value) >= 1
            }

            if not parts:
                raise ValueError("Input cannot be empty.")

            # --- Parsing and Validation Loop ---
            for part in parts:
                if '=' not in part:
                    raise ValueError(f"Invalid format for '{part}'. All settings must be in 'key=value' format.")

                key, value = part.split('=', 1)

                # Check if the key is a valid, recognized setting
                if key not in validations:
                    raise ValueError(f"Invalid setting '{key}'. Allowed settings are: {', '.join(validations.keys())}")

                # Run the specific validation function for that key
                if not validations[key](value):
                    raise ValueError(f"Invalid value '{value}' for setting '{key}'. Please check the requirements.")

                # --- Type Conversion ---
                # Convert the validated string value to its correct data type before storing
                if key == 'size_usd':
                    updates[key] = float(value)
                elif key == 'leverage':
                    updates[key] = int(value)
                else:
                    updates[key] = value

            # --- Save the data ---
            # If all parts were validated successfully, merge the updates into the user's profile
            user_data['hl_trade_profile'].update(updates)

            # Save the entire updated user_data object
            await self.save_user_data(user_id, None, direct_data=user_data)

            # --- User Feedback ---
            await event.reply("✅ **Trade Defaults Updated Successfully!**")
            # Navigate back to the main Hyperliquid menu to show the new status
            await self.show_hyperliquid_menu(event)

        except ValueError as e:
            # Catches specific validation or formatting errors and provides clear feedback
            await event.reply(f"❌ **Validation Error:** {e}\nPlease check your input and try again.")
            # We keep the user in the same state so they can correct their input
        except Exception as e:
            # Catches any other unexpected errors during the process
            logging.error(f"Failed to process trade defaults for user {user_id}: {e}", exc_info=True)
            await event.reply("❌ An unexpected error occurred while saving your settings. Please try again.")
            # Revert state to the menu to prevent the user from being stuck
            await self.show_hyperliquid_menu(event)

    # ADD THIS NEW METHOD TO THE TelegramBot CLASS
    async def admin_create_code(self, event, code_type, value):
        """
        Generates and saves a new promo code.
        Uses a tiny POSIX file lock around codes.json read-modify-write.
        """
        codes_file = os.path.join(DATA_DIR, "codes.json")
        if not os.path.exists(codes_file):
            with open(codes_file, 'w') as f:
                json.dump({}, f)

        try:
            with open(codes_file, 'r+') as f:
                # --- LOCK ---
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    codes = json.load(f)
                    # Generate a new unique code
                    new_code = base64.urlsafe_b64encode(os.urandom(6)).decode('utf-8').replace('-', '').replace('_', '')

                    codes[new_code] = {
                        'type': code_type,
                        'value': int(value),
                        'used': False,
                        'created_date': datetime.datetime.now().isoformat()
                    }

                    f.seek(0)
                    json.dump(codes, f, indent=4)
                    f.truncate()

                    await event.reply(
                        f"✅ **Promo Code Created!**\n\n"
                        f"**Code:** `{new_code}`\n"
                        f"**Type:** {code_type}\n"
                        f"**Value:** {value} days\n\n"
                        f"You can now send this code to a user to redeem."
                    )
                finally:
                    # --- UNLOCK ---
                    fcntl.flock(f, fcntl.LOCK_UN)

        except Exception as e:
            logging.error(f"Failed to create promo code: {e}", exc_info=True)
            await event.reply("❌ An error occurred while trying to create the code.")

    async def start_bot(self):
        await self.bot.start(bot_token=BOT_TOKEN)
        print("🤖 Bot started successfully!")

        @self.bot.on(events.NewMessage)
        async def handle_message(event):
            # CRITICAL: Ignore messages from the bot itself
            if event.sender_id == (await self.bot.get_me()).id:
                return  # Don't process our own messages!
            
            # Also ignore messages from other bots
            if event.sender and getattr(event.sender, 'bot', False):
                return            

            # We modify the /start logic to be handled by its own specific handler
            if event.raw_text.startswith('/start'):
                # The new /start handler below will now manage this
                pass
            else:
                await self.handle_user_message(event)

        @self.bot.on(events.CallbackQuery)
        async def handle_callback(event):
            # Decode the data to check if it's one of the new patterns
            data = event.data.decode('utf-8')
            if any(keyword in data for keyword in ['subscribe_', 'pay_card_', 'pay_crypto_', 'page_']):
                # This will be handled by the new, more specific callback handlers
                pass
            else:
                await self.handle_callback_query(event)

        # --- PAGINATION HANDLERS ---
        @self.bot.on(events.CallbackQuery(pattern=re.compile(b"page_([a-zA-Z_]+)_(\\d+)")))
        async def handle_pagination_callback(event):
            """Handles all 'Previous' and 'Next' button clicks for paginated menus."""
            session = await self.get_user_session(event.sender_id)

            try:
                callback_type = event.pattern_match.group(1).decode('utf-8')
                page_index = int(event.pattern_match.group(2).decode('utf-8'))

                if session.pagination_type != callback_type:
                    await event.answer("This is an old menu. Please use the command again.", alert=True)
                    return

                function_map = {
                    "admin_users": self.admin_view_all_users,
                    "admin_jobs": self.admin_view_all_jobs,
                    "admin_chats": self.admin_view_all_chats,
                    "admin_delete": self.admin_delete_jobs_menu,
                    "admin_manage": self.admin_manage_users_menu,
                    "user_chats": self.list_user_chats,
                    "user_jobs": self.show_job_management,
                    "user_modify": self.show_job_modification_menu,
                    "user_delete": self.show_job_deletion,
                }

                handler_func = function_map.get(callback_type)
                if handler_func:
                    await handler_func(event, page_index)
                else:
                    await event.answer("Unknown pagination type.", alert=True)
            except Exception as e:
                logging.error(f"Error in pagination callback: {e}", exc_info=True)
                await event.answer("An error occurred.", alert=True)

        @self.bot.on(events.CallbackQuery(pattern=b"page_close"))
        async def handle_pagination_close(event):
            """Closes the paginated view."""
            session = await self.get_user_session(event.sender_id)
            session.pagination_data = []
            session.pagination_type = None
            session.pagination_page = 0
            session.last_paginated_message_id = None
            try:
                text = "View closed."
                if session.is_admin:
                    text = "View closed. Returning to Admin Menu."
                await event.edit(text, buttons=None)
                if session.is_admin:
                    await self.show_admin_menu(event)
            except Exception:
                await event.answer()

        # --- NEW SPECIFIC HANDLERS FOR SUBSCRIPTION SYSTEM ---
        # 1. New, more specific handler for the /start command to track referrals
        @self.bot.on(events.NewMessage(pattern='/start'))
        async def handle_start(event):
            # Referral tracking logic
            parts = event.raw_text.split()
            if len(parts) > 1 and parts[1].startswith('ref_'):
                try:
                    referrer_id = int(parts[1].split('_')[1])
                    user_id = event.sender_id

                    if user_id != referrer_id:
                        user_data = await self.load_user_data(user_id)
                        if user_data and not user_data.get('referrer_id'):
                            user_data['referrer_id'] = referrer_id

                            referrer_user_data = await self.load_user_data(referrer_id)
                            if referrer_user_data:
                                upline = referrer_user_data.get('upline_chain', [])
                                upline.append(referrer_id)
                                user_data['upline_chain'] = upline

                            await self.save_user_data(user_id, self.user_forwarders.get(user_id), user_data)
                            await event.reply("You've been successfully referred!")
                except (ValueError, IndexError):
                    # Handle cases where the ref link is malformed
                    pass

            # Always show the main menu after handling the start command
            await self.send_main_menu(event)

        # 2. Add handlers for the new commands
        @self.bot.on(events.NewMessage(pattern='/subscribe'))
        async def subscribe_command(event):
            await self.handle_subscribe_command(event)

        @self.bot.on(events.NewMessage(pattern='/referral'))
        async def referral_command(event):
            await self.handle_referral_command(event)

        @self.bot.on(events.NewMessage(pattern=re.compile(r'/redeem (\S+)')))
        async def redeem_command(event):
            code = event.pattern_match.group(1)
            await self.handle_redeem_command(event, code)
            
        # ---- make /help and /commands easy to use ----
        @self.bot.on(events.NewMessage(pattern='/help'))
        async def help_command(event):
            """
            Usage: /help
            Calls the existing show_help() method which provides full help & tutorial.
            """
            try:
                await self.show_help(event)
            except Exception as e:
                try:
                    await event.reply("❌ Error showing help. Please contact admin.")
                except Exception:
                    pass

        @self.bot.on(events.NewMessage(pattern='/commands'))
        async def commands_command(event):
            """
            Usage: /commands
            Short list of available slash commands (one-line each).
            """
            try:
                commands_short = (
                    "/start - Open the main menu\n"
                    "/subscribe - Start subscription / payments\n"
                    "/subscription - View your subscription\n"
                    "/referral - Referral info\n"
                    "/redeem <code> - Redeem a promo code\n"
                    "/pay <amount> - Start a payment (USD)\n"
                    "/tx <tx_hash> <reference> - Submit a payment tx for verification\n"
                    "/pay_status <reference> - Check payment status\n"
                    "/resolve <name> - Resolve a basename (e.g. alice.base)\n"
                )
                await event.reply(f"📋 **Available Commands (short):**\n\n{commands_short}")
            except Exception:
                try:
                    await event.reply("❌ Error showing commands. Please contact admin.")
                except Exception:
                    pass


        # 3. Add handlers for the new button callbacks (CallbackQuery)
        @self.bot.on(events.CallbackQuery(pattern=b'subscribe_card'))
        async def on_subscribe_card(event):
            # Show monthly/yearly options for card
            buttons = [
                [Button.inline(f"${PRICES['monthly_usd']}/Month", data=f"pay_card_{PRICES['monthly_usd']}")],
                [Button.inline(f"${PRICES['yearly_usd']}/Year", data=f"pay_card_{PRICES['yearly_usd']}")],
            ]
            await event.edit("Choose your plan:", buttons=buttons)

        @self.bot.on(events.CallbackQuery(pattern=b'subscribe_crypto'))
        async def on_subscribe_crypto(event):
            # Show monthly/yearly options for crypto
            buttons = [
                [Button.inline(f"${PRICES['monthly_usd']}/Month", data=f"pay_crypto_{PRICES['monthly_usd']}")],
                [Button.inline(f"${PRICES['yearly_usd']}/Year", data=f"pay_crypto_{PRICES['yearly_usd']}")],
            ]
            await event.edit("Choose your plan:", buttons=buttons)

        @self.bot.on(events.CallbackQuery(pattern=re.compile(b"pay_card_(\d+\.?\d*)")))
        async def on_pay_card(event):
            amount = float(event.pattern_match.group(1).decode('utf-8'))
            user_id = event.sender_id

            secret = PAYSTACK_TEST_SECRET_KEY if PAYSTACK_TEST_MODE else PAYSTACK_LIVE_SECRET_KEY
            headers = {"Authorization": f"Bearer {secret}"}
            payload = {
                "email": f"user_{user_id}@telegramforwarder.bot", # Paystack requires an email
                "amount": int(amount * 100), # Amount in kobo
                "currency": "USD",
                "metadata": {"user_id": user_id},
                "callback_url": f"{WEBHOOK_BASE_URL}/payment_success"
            }

            try:
                r = requests.post("https://api.paystack.co/transaction/initialize", headers=headers, json=payload)
                r.raise_for_status()
                response_data = r.json()
                auth_url = response_data['data']['authorization_url']
                await event.edit("Please complete your payment here:", buttons=[[Button.url("Pay Now", auth_url)]])
            except requests.exceptions.RequestException as e:
                await event.answer(f"Error creating payment link: {e}", alert=True)

        @self.bot.on(events.CallbackQuery(pattern=re.compile(b"pay_crypto_(\d+\.?\d*)")))
        async def on_pay_crypto(event):
            amount = float(event.pattern_match.group(1).decode('utf-8'))
            user_id = event.sender_id

            headers = {'x-api-key': NOWPAYMENTS_API_KEY}
            payload = {
                "price_amount": amount,
                "price_currency": "usd",
                "order_id": str(user_id),
                "ipn_callback_url": f"{WEBHOOK_BASE_URL}/webhook/nowpayments"
            }

            try:
                r = requests.post("https://api.nowpayments.io/v1/payment", headers=headers, json=payload)
                r.raise_for_status()
                invoice_data = r.json()

                payment_address = invoice_data.get('pay_address')
                pay_amount = invoice_data.get('pay_amount')
                pay_currency = invoice_data.get('pay_currency').upper()

                await event.edit(f"Please send `{pay_amount}` {pay_currency} to the following address:\n\n`{payment_address}`\n\n**Note:** This invoice will be valid for a limited time.")
            except requests.exceptions.RequestException as e:
                await event.answer(f"Error creating invoice: {e}", alert=True)

        @self.bot.on(events.CallbackQuery(pattern=b'request_payout'))
        async def on_request_payout(event):
            user_id = event.sender_id
            user_data = await self.load_user_data(user_id)
            commissions = user_data.get('unpaid_commissions', 0.0)

            if commissions >= COMMISSIONS['minimum_payout_threshold']:
                # Notify admin (implement your own notification logic)
                admin_id = "YOUR_ADMIN_USER_ID" # Replace with your Telegram User ID
                await self.bot.send_message(admin_id, f"Payout request from user {user_id} for ${commissions:.2f}.")
                await event.answer("Your payout request has been sent to the admin.", alert=True)
            else:
                await event.answer("You have not reached the minimum payout threshold.", alert=True)

        @self.bot.on(events.NewMessage(pattern='/subscription'))
        async def subscription_status_command(event):
            await self.handle_subscription_status(event)

        await self.bot.run_until_disconnected()
        
        # --- Payment command handlers ---

        @self.bot.on(events.NewMessage(pattern=re.compile(r'^/pay(?:\s+(\d+(\.\d+)?))?$', re.IGNORECASE)))
        async def on_pay_command(event):
            """
            Usage: /pay 12.50
            Starts the payment flow (Coinbase Commerce primary + direct/manual fallback).
            """
            try:
                user_id = event.sender_id
                m = event.pattern_match
                amt_str = m.group(1) if m and m.group(1) else None
                if not amt_str:
                    await event.reply("Please specify an amount in USD. Example: `/pay 12.50`")
                    return
                amount = float(amt_str)
            except Exception:
                await event.reply("Invalid amount. Usage: `/pay 12.50`")
                return

            # Acknowledge quickly
            await event.reply("Initializing payment flow...")

            # Run the synchronous helper in a threadpool to avoid blocking
            loop = asyncio.get_running_loop()
            try:
                res = await loop.run_in_executor(None, handle_pay_command, user_id, amount, None, globals().get('bot_instance'))
            except Exception as e:
                await event.reply(f"Failed to start payment flow: {e}")
                return

            # Optionally report immediate result summary
            if res and res.get("coinbase") and res["coinbase"].get("ok"):
                await event.reply("✅ Payment checkout created. Please follow the hosted URL sent to you.")
            else:
                await event.reply("⚠️ Could not create Coinbase checkout; direct/manual instructions were sent.")


        @self.bot.on(events.NewMessage(pattern=re.compile(r'^/tx\s+(0x[a-fA-F0-9]{64})\s+(\S+)', re.IGNORECASE)))
        async def on_tx_submit(event):
            """
            Usage: /tx <tx_hash> <reference>
            User submits a tx hash for verification of a direct/manual payment.
            """
            try:
                user_id = event.sender_id
                tx_hash = event.pattern_match.group(1)
                reference = event.pattern_match.group(2)
            except Exception:
                await event.reply("Invalid command format. Usage: `/tx <tx_hash> <reference>`")
                return

            await event.reply("Verifying transaction — this may take a few seconds...")

            loop = asyncio.get_running_loop()
            try:
                res = await loop.run_in_executor(None, cmd_submit_tx, user_id, tx_hash, reference, None, globals().get('bot_instance'))
            except Exception as e:
                await event.reply(f"Verification failed: {e}")
                return

            if res.get("ok"):
                await event.reply(f"✅ Payment confirmed. Tx: {res.get('tx_hash') or tx_hash}")
            else:
                await event.reply(f"❌ Verification failed: {res.get('reason') or res.get('error') or 'unknown'}")


        @self.bot.on(events.NewMessage(pattern=re.compile(r'^/pay_status\s+(\S+)', re.IGNORECASE)))
        async def on_pay_status(event):
            """
            Usage: /pay_status <reference>
            Returns the current status for a payment reference.
            """
            try:
                user_id = event.sender_id
                reference = event.pattern_match.group(1)
            except Exception:
                await event.reply("Invalid usage. Example: `/pay_status BOTPAY-12345-ABCDEF`")
                return

            await event.reply("Looking up payment status...")

            loop = asyncio.get_running_loop()
            try:
                res = await loop.run_in_executor(None, cmd_pay_status, user_id, reference, None, globals().get('bot_instance'))
            except Exception as e:
                await event.reply(f"Failed to fetch status: {e}")
                return

            # cmd_pay_status will already send the user message via bot; this is an extra guard
            if res.get("ok"):
                # nothing more to do
                return
            else:
                await event.reply(f"Error retrieving payment status: {res.get('error') or 'not found'}")


        @self.bot.on(events.NewMessage(pattern=re.compile(r'^/list_proofs$', re.IGNORECASE)))
        async def on_admin_list_proofs(event):
            """
            Admin command: /list_proofs
            Sends recent confirmed payments to admin chat (uses admin notify).
            Only callable by admins — checks session.is_admin if available.
            """
            session = None
            try:
                session = await self.get_user_session(event.sender_id)
            except Exception:
                session = None

            if not session or not getattr(session, "is_admin", False):
                await event.reply("❌ You are not authorized to run this command.")
                return

            await event.reply("Gathering recent confirmed payments...")

            loop = asyncio.get_running_loop()
            try:
                res = await loop.run_in_executor(None, admin_list_proofs, globals().get('bot_instance'), globals().get('db_conn'), 50)
            except Exception as e:
                await event.reply(f"Failed to list proofs: {e}")
                return

            if res.get("ok"):
                await event.reply("✅ Admin notification sent with recent proofs.")
            else:
                await event.reply(f"Error listing proofs: {res.get('error') or 'unknown'}")

        @self.bot.on(events.NewMessage(pattern=re.compile(r'^/resolve\s+(\S+)', re.IGNORECASE)))
        async def on_resolve_command(event):
            """
            Usage: /resolve alice.base
            Resolves a Basename to an address and replies with result.
            """
            try:
                user_id = event.sender_id
                name = event.pattern_match.group(1)
            except Exception:
                await event.reply("Usage: /resolve <name>. Example: `/resolve alice.base`")
                return

            await event.reply("Resolving...")

            loop = asyncio.get_running_loop()
            try:
                addr = await loop.run_in_executor(None, resolve_basename, name)
            except Exception as e:
                await event.reply(f"Resolve error: {e}")
                return

            if addr:
                await event.reply(f"`{name}` → `{addr}`")
            else:
                await event.reply(f"Could not resolve `{name}` (no record or zero-address).")


    # -----------------------------------------------------------------------------
    # --- PAGINATION SYSTEM ---
    # -----------------------------------------------------------------------------
    # In the TelegramBot class, REPLACE the existing show_paginated_list method with this corrected version:

    async def show_paginated_list(self, event, page_index, item_lister, item_formatter, callback_prefix, title, items_per_page=5):
        """
        A generic function to display a paginated list with navigation.
        This version contains the definitive fix for the duplication and "old menu" bugs.
        """
        session = await self.get_user_session(event.sender_id)
        user_id = event.sender_id

        is_new_list_request = not hasattr(event, 'data')
        wait_message = None

        if is_new_list_request:
            if session.last_paginated_message_id:
                try:
                    await self.bot.edit_message(
                        user_id,
                        session.last_paginated_message_id,
                        text=f"---\n*This '{session.pagination_type}' menu has expired. Please use the new one below.*\n---",
                        buttons=None
                    )
                except Exception:
                    pass # Ignore if message is gone

            wait_message = await event.reply("🔄 Fetching data, please wait...")
            session.pagination_page = 0
            session.pagination_data = await item_lister()
            session.pagination_type = callback_prefix

        total_items = len(session.pagination_data)
        if total_items == 0:
            if wait_message:
                await wait_message.delete()
            no_data_text = f"📭 **No Data**\n\nThere are no items to display in '{title}'."
            if hasattr(event, 'data'):
                await event.edit(no_data_text, buttons=None)
            else:
                await event.reply(no_data_text)
            return

        total_pages = (total_items + items_per_page - 1) // items_per_page
        session.pagination_page = page_index

        start_index = page_index * items_per_page
        end_index = start_index + items_per_page
        items_on_page = session.pagination_data[start_index:end_index]
        message_text = await item_formatter(items_on_page, start_index, title, page_index + 1, total_pages)

        buttons = []
        row = []
        if page_index > 0:
            row.append(Button.inline("⬅️ Previous", data=f"{callback_prefix}_{page_index - 1}"))
        if (page_index + 1) < total_pages:
            row.append(Button.inline("Next ➡️", data=f"{callback_prefix}_{page_index + 1}"))
        if row: # Only add the row if it has buttons
            buttons.append(row)
        buttons.append([Button.inline("❌ Close", data="page_close")])

        # --- START OF DEFINITIVE FIX for Duplication and "Old Menu" bugs ---

        new_message = None
        if wait_message:
            # This is a new list request, so we delete the "wait" message and send a new one.
            await wait_message.delete()
            new_message = await event.respond(message_text, buttons=buttons)
        else:
            # This is a callback (Next/Previous), so we edit the existing message.
            try:
                await event.edit(message_text, buttons=buttons)
                # IMPORTANT: Acknowledge the callback query
                await event.answer()
                new_message = await event.get_message()
            except errors.MessageNotModifiedError:
                await event.answer() # Still acknowledge the click
                new_message = await event.get_message() # Get message even if not modified
            except Exception as e:
                logging.error(f"Could not edit message for pagination: {e}")
                await event.answer() # Still acknowledge the callback

        # After either responding or editing, ALWAYS update the session with the correct message ID.
        if new_message:
            session.last_paginated_message_id = new_message.id

        # --- END OF DEFINITIVE FIX ---

    # THIS IS THE COMPLETE FUNCTION.
    # PASTE IT INSIDE YOUR TelegramBot CLASS.
    async def check_permission(self, user_id, action, context={}):
        """
        The central gatekeeper for all restricted features.
        Checks a user's subscription status against the rules in config.py.
        """
        # 1. Master Switch: If subscriptions are off, everyone has permission.
        if not SUBSCRIPTION_ENABLED:
            return True, ""

        # 2. Load User Data: Get the user's record from the database.
        user_data = await self.load_user_data(user_id)
        if not user_data:
            return False, "Could not load your user data. Please try again."

        # 3. Check for Expired Subscription:
        status = user_data.get('subscription_status', 'free')
        expiry = user_data.get('subscription_expiry_date')
        active_promo = user_data.get('active_promo_code')

        # DEBUG: Send debug info to user for critical actions
        # if action in ['set_and_logic', 'set_timer']:
            # await self.bot.send_message(user_id, f"**DEBUG - Permission Check for {action}:**\n"
                                              # f"• Status: {status}\n"
                                              # f"• Expiry: {expiry}\n"
                                              # f"• Active Promo: {active_promo}\n"
                                              # f"• Current Time: {datetime.datetime.now()}")

        if status == 'paid' and expiry:
            try:
                expiry_datetime = datetime.datetime.fromisoformat(expiry)
                if expiry_datetime < datetime.datetime.now():
                    # Subscription expired
                    status = 'free'
                    user_data['subscription_status'] = 'free'
                    user_data['active_promo_code'] = None
                    await self.save_user_data(user_id, None, direct_data=user_data)
                    await self.bot.send_message(user_id, "⚠️ Your subscription has expired. You have been switched to the free plan.")
            except ValueError as e:
                logging.error(f"Invalid date format for user {user_id}: {expiry}. Error: {e}")
                status = 'free'

        # 4. Get the correct set of rules for the user's current status ('free' or 'paid').
        rules = USER_LIMITS.get(status, USER_LIMITS['free'])
        forwarder = self.user_forwarders.get(user_id)

        # 5. The Action Switch: Check the specific action against the rules.
        if action == 'create_job':
            if forwarder and len(forwarder.jobs) >= rules['max_jobs']:
                return False, f"Your plan limit is {rules['max_jobs']} jobs. Please upgrade to create more."
            if context.get('job_type') == 'custom_pattern' and len(forwarder.jobs) >= rules['max_custom_pattern_jobs']:
                return False, f"Your plan limit is {rules['max_custom_pattern_jobs']} custom pattern job(s). Please upgrade to create more."

        elif action == 'add_source':
            if context.get('source_count', 0) > rules['max_sources_per_job']:
                return False, f"Your plan limit is {rules['max_sources_per_job']} sources per job. Please upgrade for more."

        elif action == 'add_destination':
            if context.get('destination_count', 0) > rules['max_destinations_per_job']:
                return False, f"Your plan limit is {rules['max_destinations_per_job']} destinations per job. Please upgrade for more."

        elif action == 'set_and_logic':
            if not rules['can_use_and_logic']:
                return False, "'AND' logic is a premium feature. Please upgrade to use it."

        elif action == 'set_timer':
            min_required = rules['min_timer_minutes']
            if min_required > 0:
                timer_str = context.get('timer_str', '').lower()
                if timer_str in ('', 'none'):
                    user_minutes = 0
                else:
                    parts = timer_str.split()
                    if len(parts) == 2 and parts[0].isdigit():
                        val, unit = int(parts[0]), parts[1]
                        if "minute" in unit: user_minutes = val
                        elif "hour" in unit: user_minutes = val * 60
                        else: user_minutes = 0  # Default for seconds or other units
                    else:
                        user_minutes = 0

                if user_minutes < min_required:
                    return False, f"Your plan requires a minimum cooldown of {min_required} minutes. Please set a longer timer or upgrade."

        elif action == 'add_keywords':
            if context.get('keyword_count', 0) > rules['max_keywords_per_job']:
                return False, f"Your plan limit is **{rules['max_keywords_per_job']} keywords** per job. Please upgrade to add more."

        elif action == 'add_patterns':
            if context.get('pattern_count', 0) > rules['max_patterns_per_job']:
                return False, f"Your plan limit is **{rules['max_patterns_per_job']} custom patterns** per job. Please upgrade to add more."

        # If none of the checks above failed, grant permission.
        return True, ""

    async def get_user_session(self, user_id: int) -> UserSession:
        """
        Gets the in-memory session for a user.
        Crucially, it also ensures a persistent data file exists for the user,
        creating a default 'free' user record on their very first interaction.
        """
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = UserSession(user_id)
            # --- THIS IS THE UPDATED LOGIC ---
            # When a new in-memory session is created, check if a persistent
            # data file exists. If not, create one with default values.
            user_data_path = os.path.join(DATA_DIR, f"user_{user_id}.dat")
            if not os.path.exists(user_data_path):
                logging.info(f"User {user_id} not found in database. Creating new default record.")
                initial_data = {
                    'subscription_status': 'free',
                    'subscription_expiry_date': None,
                    'active_promo_code': None,  # CRITICAL: Add this field!
                    'plan_type': None,  # CRITICAL: Add this field too!
                    'referrer_id': None,
                    'upline_chain': [],
                    'unpaid_commissions': 0.0,
                    'has_had_first_subscription': False,
                    'payment_history': [],  # CRITICAL: Add this field!
                    # Default fields for the forwarder
                    'api_id': None,
                    'api_hash': None,
                    'phone_number': None,
                    'jobs': [],
                    'offsets': {},
                    'saved_patterns': []
                }
                # The save_user_data function now handles both new and existing data
                # We pass `direct_data` to tell it to use this dictionary.
                await self.save_user_data(user_id, None, direct_data=initial_data)
                logging.info(f"Successfully created default data file for new user {user_id}.")
            # --- END OF UPDATED LOGIC ---

        return self.user_sessions[user_id]

    # ADDED for Custom Pattern feature
    async def handle_callback_query(self, event):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        data = event.data.decode('utf-8')

        if data.startswith("select_pattern_"):
            # This logic is now restored and adapted
            pattern_index = int(data.split("_")[2])
            forwarder = self.user_forwarders.get(user_id)

            if not forwarder:
                await event.answer("Error: Cannot find your user session.", alert=True)
                return

            if 0 <= pattern_index < len(forwarder.saved_patterns):
                selected_pattern = forwarder.saved_patterns[pattern_index]

                # Add the selected pattern to the job's list
                if 'patterns' not in session.pending_job:
                    session.pending_job['patterns'] = []
                session.pending_job['patterns'].append(selected_pattern['pattern'])

                await event.answer(f"Pattern '{selected_pattern['name']}' added!")
                await event.delete() # Remove the inline keyboard

                # Transition to the next step
                session.set_state("job_pattern_add_or_done")
                await self.handle_pattern_add_or_done(event, "") # Call the next step
            else:
                await event.answer("Invalid selection. Please try again.", alert=True)

    async def handle_user_message(self, event):
        """Handle incoming messages with anti-echo protection"""
        
        # ANTI-ECHO GUARD: Ignore our own messages
        bot_info = await self.bot.get_me()
        if event.sender_id == bot_info.id:
            logging.info(f"🛑 Ignoring bot's own message: '{event.message.text}'")
            return
        
        # ANTI-BOT GUARD: Ignore other bots
        if hasattr(event, 'sender') and event.sender and getattr(event.sender, 'bot', False):
            return        
        
        user_id = event.sender_id
        message = event.message.text.strip()
        session = await self.get_user_session(user_id)
        
        # DEBUG: Log state transitions
        logging.info(f"📍 Message from {user_id}: '{event.message.text}' | State: {session.state}")

        try:
            # --- NEW UNIFIED UNIVERSAL BUTTON HANDLER ---
            # Both "Cancel" and "Back to Main Menu" now perform the same, robust action.
            if message == "❌ Cancel" or "Back to Main Menu" in message:
                await self.return_to_main_menu(event)
                return

            if message == "⬅️ Back":
                previous_state = session.go_back()
                await self.show_prompt_for_state(event, previous_state, extra_info="↩️ **Going back.**")
                return

            if message == "🔙 Back to Admin Menu":
                await self.show_admin_menu(event)
                return

            # If it's not a universal button, delegate to the router.
            await self.route_to_state_handler(event, message, session, session.state)

        except Exception as e:
            logging.error(f"Error handling message from {user_id}: {e}", exc_info=True)
            await event.reply("❌ An unexpected error occurred. Please try again or contact support.")
            
    async def return_to_main_menu(self, event):
            """
            Thoroughly resets the user's session, sends a confirmation,
            and shows the main menu. This is the definitive exit function.
            """
            session = await self.get_user_session(event.sender_id)
            
            # This is the critical cleanup that was missing from previous attempts.
            session.state = "idle"
            session.state_history = []
            session.pending_job = {}
            session.temp_data = {}
            
            # Send a single, clear confirmation message.
            await event.reply("Action cancelled. Returning to the main menu.", buttons=Button.clear())
            
            # Now, send the main menu in a separate message.
            await self.send_main_menu(event)

    # The logic in this function is reordered to check for menu buttons first.
    async def route_to_state_handler(self, event, message, session, state):
        """
        This function contains the main routing logic based on the user's state.
        This version includes a comprehensive check for all menu buttons first to prevent state conflicts.
        """
        # --- FINAL FIX: CHECK ALL UNIVERSAL AND MENU BUTTONS FIRST ---
        # This list now includes buttons from the main menu, admin menu, and other settings.
        all_menu_buttons = [
            "📋 Manage Jobs", "📝 List Chats", "🔄 Reconnect Account", "🚪 Logout (Keep Jobs)", "⚙️ Other Settings", "🔷 Base",
            "👥 View All Users", "📋 View All Jobs", "📊 System Statistics", "💬 View All Chats",
            "🎁 Create Promo Code", "🗑️ Admin Delete Jobs", "👤 Manage Users", "🔙 Exit Admin Mode",
            "🚀 Subscribe", "🤝 Referral", "🎁 Redeem Code", "🗑️ Delete Account"
        ]

        # If the message is a recognized menu button, route it correctly regardless of the current state.
        if message in all_menu_buttons:
            if message in ["📋 Manage Jobs", "📝 List Chats", "🔄 Reconnect Account", "🚪 Logout (Keep Jobs)", "⚙️ Other Settings", "🔷 Base"]:
                await self.handle_main_menu(event, message)
                return
            elif session.is_admin and message in ["👥 View All Users", "📋 View All Jobs", "📊 System Statistics", "💬 View All Chats", "🎁 Create Promo Code", "🗑️ Admin Delete Jobs", "👤 Manage Users", "🔙 Exit Admin Mode"]:
                await self.handle_admin_functions(event, message)
                return
            elif message in ["🚀 Subscribe", "🤝 Referral", "🎁 Redeem Code", "🗑️ Delete Account"]:
                await self.handle_other_settings_menu(event, message)
                return

        # --- Handle direct, typed commands next ---
        if message.startswith('/start'):
            await self.send_main_menu(event)
            return
        elif message.lower().strip() == '/help_regex':
            await self.show_regex_tutorial(event)
            return
        elif message.lower().strip() == '/subscription':
            await self.handle_subscription_status(event)
            return
        elif message.startswith('/admin'):
            await self.handle_admin_command(event, message)
            return

        # --- If it's not a menu button or command, proceed with state-based logic ---
        if state == "idle":
            await self.handle_main_menu(event, message)

        elif session.is_admin and state.startswith("admin_"):
            # First, we handle specific states that require TEXT input from the admin.
            if state == "admin_awaiting_code_value":
                await self.admin_handle_code_value(event, message)
            elif state == "admin_delete_jobs":
                await self.admin_handle_job_deletion(event, message)
            elif state == "admin_manage_users":
                await self.admin_handle_user_management(event, message)
            # Add any other specific text-input admin states here in the future.
            else:
                # If the state is not one that expects text input (e.g., it's 'admin_menu'),
                # then the message MUST be a button click. We pass it to the
                # function that handles all the admin menu buttons.
                await self.handle_admin_functions(event, message)

        elif state == "other_settings_menu":
            await self.handle_other_settings_menu(event, message)
            
        elif state == "base_menu":
            await self.handle_base_menu(event, message)

        elif state == "awaiting_redeem_code":
            await self.handle_redeem_command(event, message)

        elif state == "demo_awaiting_api_id":
            await self.handle_demo_api_id(event, message)

        elif state == "demo_awaiting_api_hash":
            await self.handle_demo_api_hash(event, message)

        elif state == "demo_awaiting_phone":
            await self.handle_demo_phone(event, message)

        elif state == "demo_awaiting_code":
            await self.handle_demo_code(event, message)

        elif state == "awaiting_api_id":
            await self.handle_api_id(event, message)

        elif state == "awaiting_api_hash":
            await self.handle_api_hash(event, message)

        elif state == "awaiting_phone":
            await self.handle_phone(event, message)

        elif state == "awaiting_code":
            await self.handle_code(event, message)

        elif state == "awaiting_2fa":
            await self.handle_2fa(event, message)

        elif state == "job_management":
            await self.handle_job_management(event, message)

        elif state == "creating_job":
            await self.handle_job_creation(event, message)

        elif state == "modifying_job":
            await self.handle_job_modification_selection(event, message)

        elif state == "job_confirmation":
            await self.handle_job_confirmation(event, message)

        elif state == "confirming_delete":
            if message.upper() == "DELETE":
                session.temp_data['delete_confirmed'] = True
                await self.delete_account(event)
            else:
                session.set_state("idle")
                session.temp_data = {}
                await event.reply("❌ **Deletion cancelled**\n\nYour account and jobs are safe.")
                await self.send_main_menu(event)

        elif state.startswith("modify_"):
            await self.handle_job_modification_steps(event, message)

        elif state == "job_awaiting_match_logic":
            await self.handle_job_awaiting_match_logic(event, message)

        elif state == "job_custom_pattern_menu":
            await self.handle_custom_pattern_menu(event, message)

        elif state == "awaiting_new_pattern":
            await self.handle_new_pattern_input(event, message)

        elif state == "awaiting_pattern_save_prompt":
            await self.handle_pattern_save_prompt(event, message)

        elif state == "awaiting_pattern_save_name":
            await self.handle_pattern_save_name(event, message)

        # NEW STATE HANDLER ADDED
        elif state == "awaiting_job_name":
            await self.handle_awaiting_job_name(event, message)

        elif state == "job_pattern_add_or_done":
            await self.handle_pattern_add_or_done(event, message)

        elif state.startswith("job_"):
            await self.handle_job_steps(event, message)

        elif state == "deleting_job":
            await self.handle_job_deletion(event, message)

        # NEW STATES ADDED HERE
        elif state == "hyperliquid_menu":
            await self.handle_hyperliquid_menu(event, message)

        elif state == "awaiting_hl_credentials":
            await self.handle_awaiting_hl_credentials(event, message)

        elif state == "awaiting_hl_trade_defaults":
            await self.handle_awaiting_hl_trade_defaults(event, message)

    # ADDED: New method for Demo Mode
    async def setup_demo_mode(self):
        """Initializes and connects the single forwarder for Demo Mode."""
        if not self.demo_mode:
            return

        logging.info("🚀 DEMO MODE IS ACTIVE 🚀")
        logging.info("Attempting to load pre-authenticated demo session...")

        session_path = os.path.join(SESSIONS_DIR, DEMO_SESSION_NAME)
        if not os.path.exists(f"{session_path}.session"):
            logging.critical(f"FATAL: DEMO_MODE is active but session file '{session_path}.session' not found.")
            logging.critical("Please run the one-time login script to generate the demo session file first.")
            sys.exit(1)

        try:
            demo_user_id = 0 # Dummy ID for the forwarder object itself
            forwarder = TelegramForwarder(
                api_id=DEMO_API_ID,
                api_hash=DEMO_API_HASH,
                phone_number=DEMO_PHONE_NUMBER,
                user_id=demo_user_id,
                bot_instance=self,
                is_demo_forwarder=True # New flag
            )

            result = await forwarder.connect_and_authorize()

            if result == "authorized":
                self.demo_forwarder = forwarder
                await self.start_forwarder_session(0, self.demo_forwarder)
                logging.info(f"✅ Demo session for {DEMO_PHONE_NUMBER} loaded and authorized successfully.")
            else:
                logging.critical(f"FATAL: Failed to authorize demo session. Result: {result}")
                logging.critical("Please ensure the demo session file is valid and not expired.")
                sys.exit(1)

        except Exception as e:
            logging.critical(f"FATAL: An error occurred while setting up Demo Mode: {e}", exc_info=True)
            sys.exit(1)

    # ADDED: New handler for Demo Mode credentials
    async def handle_demo_api_id(self, event, message):
        """Accepts any API ID input and moves to the next step."""
        # --- FIX: Delete the user's input message ---
        await event.delete()
        session = await self.get_user_session(event.sender_id)
        session.set_state("demo_awaiting_api_hash")
        await event.reply(
            "✅ Great! The next step is the **API Hash**.\n\n"
            "In a real login, you would get this from my.telegram.org. For the demo, you can enter anything."
        )

    async def handle_demo_api_hash(self, event, message):
        """Accepts any API Hash input and moves to the next step."""
        # --- FIX: Delete the user's input message ---
        await event.delete()
        session = await self.get_user_session(event.sender_id)
        session.set_state("demo_awaiting_phone")
        await event.reply(
            "✅ Perfect. Now, you would enter your account's **phone number**.\n\n"
            "Please enter any phone number to continue the demo."
        )

    async def handle_demo_phone(self, event, message):
        """Accepts any phone number input and moves to the next step."""
        # --- FIX: Delete the user's input message ---
        await event.delete()
        session = await self.get_user_session(event.sender_id)
        session.set_state("demo_awaiting_code")
        await event.reply(
            "✅ Excellent. At this point, Telegram would send a login code to your account.\n\n"
            "Since this is a simulation, please enter any 5-digit code to finish."
        )

    async def handle_demo_code(self, event, message):
        """Accepts any code input and completes the demo login."""
        # --- FIX: Delete the user's input message ---
        await event.delete()
        session = await self.get_user_session(event.sender_id)

        # We can add a simple check for digits if you want, or remove it to accept anything.
        # This is optional, but makes the final step feel slightly more real.
        if not message.strip().isdigit() or len(message.strip()) != 5:
            await event.reply("Please enter any 5-digit code to complete the demo (e.g., 12345).")
            return

        session.set_state("idle")
        # This is the key step: associate the user with the global demo forwarder
        self.user_forwarders[event.sender_id] = self.demo_forwarder

        await event.reply(
            "✅ **Demo Login Successful!**\n\n"
            "You have now experienced the full login flow. You are connected to the shared demo account."
        )
        await self.send_main_menu(event)

    async def handle_admin_command(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # Extract password from command
        parts = message.split(' ', 1)
        if len(parts) < 2:
            await event.reply("🔐 **Admin Access**\n\nUsage: `/admin <master_secret>`")
            return

        provided_secret = parts[1]

        # Verify master secret
        if provided_secret == MASTER_SECRET:
            session.is_admin = True
            session.set_state("admin_menu")
            await self.show_admin_menu(event)
        else:
            await event.reply("❌ **Access Denied**\n\nInvalid master secret.")

    async def show_admin_menu(self, event):
        total_users = len(self.user_forwarders)
        total_jobs = sum(len(fw.jobs) for fw in self.user_forwarders.values())

        admin_text = "🔧 **Admin Control Panel**\n\n"
        admin_text += f"👥 **Total Users:** {total_users}\n"
        admin_text += f"⚙️ **Total Active Jobs:** {total_jobs}\n\n"
        admin_text += "**Admin Functions:**"

        buttons = [
            [Button.text("👥 View All Users", resize=True), Button.text("📋 View All Jobs", resize=True)],
            [Button.text("📊 System Statistics", resize=True), Button.text("💬 View All Chats", resize=True)],
            [Button.text("🎁 Create Promo Code", resize=True)],
            [Button.text("🗑️ Admin Delete Jobs", resize=True), Button.text("👤 Manage Users", resize=True)],
            [Button.text("🔙 Exit Admin Mode", resize=True)]
        ]

        await event.reply(admin_text, buttons=buttons)

    async def handle_admin_functions(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        # --- NEW LOGIC ADDED HERE ---
        if message == "🎁 Create Promo Code":
            await self.admin_start_create_code(event)
        elif message == "📄 Download Yearly Revenue Report":
            await self.admin_generate_revenue_report(event)
        # --- END OF NEW LOGIC ---
        elif message == "👥 View All Users":
            await self.admin_view_all_users(event)
        elif message == "📋 View All Jobs":
            await self.admin_view_all_jobs(event)
        elif message == "💬 View All Chats":
            await self.admin_view_all_chats(event)
        elif message == "🗑️ Admin Delete Jobs":
            await self.admin_delete_jobs_menu(event)
        elif message == "👤 Manage Users":
            await self.admin_manage_users_menu(event)
        elif message == "📊 System Statistics":
            await self.admin_show_statistics(event)
        elif message == "🔙 Exit Admin Mode":
            session.is_admin = False
            session.set_state("idle")
            await event.reply("👋 **Admin mode exited**\n\nReturning to regular user interface.")
            await self.send_main_menu(event)
        elif session.state == "admin_delete_jobs":
            await self.admin_handle_job_deletion(event, message)
        elif session.state == "admin_manage_users":
            await self.admin_handle_user_management(event, message)
        else:
            await event.reply("Please use the admin menu buttons.")

    async def admin_view_all_users(self, event, page_index=0):
        """Kicks off the paginated view of all users."""
        async def lister():
            return list(self.user_forwarders.keys())

        async def formatter(user_ids, start_index, title, current_page, total_pages):
            text = f"**{title} (Page {current_page}/{total_pages})**\n\n"

            # Load promo code database once
            codes_file = os.path.join(DATA_DIR, "codes.json")
            all_codes = {}
            if os.path.exists(codes_file):
                with open(codes_file, 'r') as f:
                    all_codes = json.load(f)

            # Create a reverse lookup: user_id -> list of codes they've used
            user_codes_map = {}
            for code, code_data in all_codes.items():
                if code_data.get('used') and 'used_by' in code_data:
                    used_by = code_data['used_by']
                    if used_by not in user_codes_map:
                        user_codes_map[used_by] = []
                    user_codes_map[used_by].append({
                        'code': code,
                        'value': code_data.get('value', 'N/A'),
                        'used_date': code_data.get('used_date', 'Unknown')
                    })

            for i, user_id in enumerate(user_ids, start=start_index + 1):
                forwarder = self.user_forwarders.get(user_id)
                user_data = await self.load_user_data(user_id)
                if not user_data: continue  # Skip if data can't be loaded

                text += f"**{i}.** 🔹 **User ID:** `{user_id}`\n"
                if forwarder:
                    text += f"   📱 **Phone:** {forwarder.phone_number}\n"

                # Check subscription status and expiry
                status = user_data.get('subscription_status', 'free')
                expiry_iso = user_data.get('subscription_expiry_date')
                active_promo = user_data.get('active_promo_code')

                # Determine if subscription is currently active
                is_subscription_active = False
                if status == 'paid' and expiry_iso:
                    try:
                        expiry_date = datetime.datetime.fromisoformat(expiry_iso)
                        is_subscription_active = expiry_date > datetime.datetime.now()
                    except:
                        is_subscription_active = False

                if is_subscription_active:
                    plan_type = user_data.get('plan_type', 'Paid').capitalize()
                    text += f"   ⭐️ **Plan:** {plan_type}\n"

                    # Show current promo code if one is active
                    if active_promo:
                        # Find the details of the active promo code
                        promo_details = all_codes.get(active_promo, {})
                        promo_value = promo_details.get('value', 'N/A')
                        text += f"   🎁 **Active Promo Code:** `{active_promo}` ({promo_value}-Day Pass)\n"
                    elif user_id in user_codes_map:
                        # Show the most recently used promo code if no active promo but codes were used
                        user_codes = user_codes_map[user_id]
                        user_codes.sort(key=lambda x: x.get('used_date', ''), reverse=True)
                        latest_code = user_codes[0]
                        text += f"   🎁 **Last Promo Code Used:** `{latest_code['code']}` ({latest_code['value']}-Day Pass)\n"
                        if len(user_codes) > 1:
                            text += f"   📊 **Total Codes Used:** {len(user_codes)}\n"

                    expiry = datetime.datetime.fromisoformat(expiry_iso)
                    text += f"   🗓️ **Expires:** {expiry.strftime('%Y-%m-%d')}\n"
                else:
                    # User is on free tier
                    if status == 'paid' and expiry_iso:
                        # Subscription existed but expired
                        text += "   ⭐️ **Plan:** Expired (Free Tier)\n"
                        try:
                            expiry = datetime.datetime.fromisoformat(expiry_iso)
                            text += f"   🗓️ **Expired:** {expiry.strftime('%Y-%m-%d')}\n"
                        except:
                            pass
                    else:
                        text += "   ⭐️ **Plan:** Free Tier\n"

                    # Show promo code usage history for free tier users
                    if user_id in user_codes_map:
                        user_codes = user_codes_map[user_id]
                        text += f"   📝 **Previously Used Codes:** {len(user_codes)}\n"
                        # Optionally show the most recent code
                        user_codes.sort(key=lambda x: x.get('used_date', ''), reverse=True)
                        latest_code = user_codes[0]
                        text += f"   📅 **Last Code:** `{latest_code['code']}` (Used: {latest_code.get('used_date', 'Unknown')[:10]})\n"

                if forwarder:
                    text += f"   ⚙️ **Jobs:** {len(forwarder.jobs)}\n"
                    text += f"   🔗 **Status:** {'✅ Connected' if forwarder.is_authorized else '❌ Disconnected'}\n\n"
                else:
                    text += "   🔗 **Status:** ❌ Disconnected\n\n"

            return text

        await self.show_paginated_list(
            event,
            page_index=page_index,
            item_lister=lister,
            item_formatter=formatter,
            callback_prefix="page_admin_users",
            title="👥 All Users"
        )

    async def admin_start_create_code(self, event):
        """Starts the interactive process for creating a promo code."""
        session = await self.get_user_session(event.sender_id)
        session.set_state("admin_awaiting_code_type")

        # For now, we only have one type of code. This can be expanded later.
        # We will automatically select "subscription" and move to the next step.
        session.temp_data['admin_code_type'] = 'subscription'
        session.set_state("admin_awaiting_code_value")
        await event.reply("✅ Code type set to: **subscription**\n\n"
                          "Now, please enter the duration for this subscription code in **days** (e.g., `30` for a 30-day pass).")

    async def admin_handle_code_value(self, event, message):
        """Handles receiving the value (duration) for the promo code."""
        session = await self.get_user_session(event.sender_id)

        if not message.isdigit() or int(message) <= 0:
            await event.reply("❌ Invalid input. Please enter a positive number for the days (e.g., `30`).")
            return

        value = int(message)
        code_type = session.temp_data.get('admin_code_type', 'subscription')

        # We can now call the original code creation logic
        await self.admin_create_code(event, code_type, value)

        # Clean up and return to the admin menu
        session.temp_data.pop('admin_code_type', None)
        session.set_state("admin_menu")
        await self.show_admin_menu(event)

    async def admin_view_all_jobs(self, event, page_index=0):
        """Kicks off the paginated view of all jobs."""
        async def lister():
            all_jobs = []
            for user_id, forwarder in self.user_forwarders.items():
                for i, job in enumerate(forwarder.jobs):
                    all_jobs.append({
                        'user_id': user_id,
                        'phone': forwarder.phone_number,
                        'job_index_in_user_list': i,
                        'job': job,
                        'forwarder': forwarder
                    })
            return all_jobs

        async def formatter(jobs_on_page, start_index, title, current_page, total_pages):
            text = f"**{title} (Page {current_page}/{total_pages})**\n\n"
            for i, job_info in enumerate(jobs_on_page, start=start_index + 1):
                job = job_info['job']
                forwarder = job_info['forwarder']
                source_names = [forwarder.chat_cache.get(sid, f"ID {sid}") for sid in job.get('source_ids', [])]
                dest_names = [forwarder.chat_cache.get(did, f"ID {did}") for did in job.get('destination_ids', [])]

                text += f"**{i}.** 👤 {job_info['phone']}\n"
                text += f"   📋 **Type:** {job.get('type', 'N/A').capitalize()}\n"
                text += f"   📥 **From:** {', '.join(source_names[:2])}{'...' if len(source_names) > 2 else ''}\n"
                text += f"   📤 **To:** {', '.join(dest_names[:2])}{'...' if len(dest_names) > 2 else ''}\n\n"
            return text

        await self.show_paginated_list(
            event,
            page_index=page_index,
            item_lister=lister,
            item_formatter=formatter,
            callback_prefix="page_admin_jobs",
            title="📋 All Active Jobs"
        )

    async def admin_view_all_chats(self, event, page_index=0):
        """Kicks off the paginated view of all accessible chats."""
        async def lister():
            all_chats = {}
            for user_id, forwarder in self.user_forwarders.items():
                for chat_id, chat_title in forwarder.chat_cache.items():
                    if chat_id not in all_chats:
                        all_chats[chat_id] = {
                            'title': chat_title,
                            'users': []
                        }
                    all_chats[chat_id]['users'].append(forwarder.phone_number)
            return [{'id': k, **v} for k, v in all_chats.items()]

        async def formatter(chats_on_page, start_index, title, current_page, total_pages):
            text = f"**{title} (Page {current_page}/{total_pages})**\n\n"
            for i, info in enumerate(chats_on_page, start=start_index + 1):
                text += f"**{i}. {info['title']}** (`{info['id']}`)\n"
                text += f"   👥 Accessible by: {', '.join(info['users'][:2])}{'...' if len(info['users']) > 2 else ''}\n\n"
            return text

        await self.show_paginated_list(
            event,
            page_index=page_index,
            item_lister=lister,
            item_formatter=formatter,
            callback_prefix="page_admin_chats",
            title="💬 All Accessible Chats",
            items_per_page=10
        )

    async def admin_delete_jobs_menu(self, event, page_index=0):
        """Starts the paginated, interactive job deletion process."""
        session = await self.get_user_session(event.sender_id)
        session.set_state("admin_delete_jobs")

        async def lister():
            all_jobs = []
            for user_id, forwarder in self.user_forwarders.items():
                for i, job in enumerate(forwarder.jobs):
                    all_jobs.append({
                        'user_id': user_id,
                        'phone': forwarder.phone_number,
                        'job_index_in_user_list': i,
                        'job': job
                    })
            return all_jobs

        async def formatter(jobs_on_page, start_index, title, current_page, total_pages):
            text = f"**{title} (Page {current_page}/{total_pages})**\n\n"
            for i, job_info in enumerate(jobs_on_page, start=start_index + 1):
                job = job_info['job']
                forwarder = self.user_forwarders[job_info['user_id']]
                source_names = [forwarder.chat_cache.get(sid, f"ID {sid}") for sid in job.get('source_ids', [])]
                text += f"**{i}.** 👤 {job_info['phone']} - {job.get('type', 'N/A').capitalize()}\n"
                text += f"   📥 From: {', '.join(source_names[:2])}\n\n"

            last_num = start_index + len(jobs_on_page)
            text += f"Reply with a number **({start_index + 1}-{last_num})** to delete."
            return text

        await self.show_paginated_list(
            event,
            page_index=page_index,
            item_lister=lister,
            item_formatter=formatter,
            callback_prefix="page_admin_delete",
            title="🗑️ Admin Job Deletion"
        )

    async def admin_handle_job_deletion(self, event, message):
        session = await self.get_user_session(event.sender_id)

        try:
            global_job_num = int(message)
            all_jobs = session.pagination_data

            if 1 <= global_job_num <= len(all_jobs):
                job_info = all_jobs[global_job_num - 1]

                user_id_to_modify = job_info['user_id']
                forwarder = self.user_forwarders.get(user_id_to_modify)
                job_index_in_user_list = job_info['job_index_in_user_list']

                if not forwarder:
                     await event.reply("❌ Error: Could not find the user for this job.")
                     return

                # This is a complex operation. If we delete job #3, the index of job #4 becomes #3.
                # We need to rebuild the user's job list carefully if multiple deletions happen
                # without refreshing the list. For now, we assume one deletion at a time.
                deleted_job = forwarder.jobs.pop(job_index_in_user_list)
                await self.save_user_data(user_id_to_modify, forwarder)

                await event.reply(f"✅ **Job #{global_job_num} Deleted Successfully**\n\n"
                                f"👤 **User:** {job_info['phone']}\n"
                                f"📋 **Job:** {deleted_job.get('type', 'Unknown').capitalize()}\n\n"
                                "The job has been removed.", buttons=Button.clear())

                session.set_state("admin_menu")
                session.pagination_data = []
                await self.show_admin_menu(event)
            else:
                await event.reply("❌ Invalid job number. Please send a number from the list.")
        except (ValueError, IndexError):
            await event.reply("❌ Please send a valid job number from the list.")

    async def admin_manage_users_menu(self, event, page_index=0):
        session = await self.get_user_session(event.sender_id)
        session.set_state("admin_manage_users")

        async def lister():
            return list(self.user_forwarders.items())

        async def formatter(users_on_page, start_index, title, current_page, total_pages):
            text = f"**{title} (Page {current_page}/{total_pages})**\n\n"
            for i, (user_id, forwarder) in enumerate(users_on_page, start=start_index + 1):
                text += f"**{i}.** {forwarder.phone_number} (`{user_id}`)\n"
                text += f"   ⚙️ Jobs: {len(forwarder.jobs)} | Status: {'✅' if forwarder.is_authorized else '❌'}\n\n"

            last_num = start_index + len(users_on_page)
            text += f"Reply with a number **({start_index + 1}-{last_num})** to manage."
            return text

        await self.show_paginated_list(
            event,
            page_index=page_index,
            item_lister=lister,
            item_formatter=formatter,
            callback_prefix="page_admin_manage",
            title="👤 User Management"
        )

    # --- FIX #5: APPLIED ---
    # The error message in the 'except' block is corrected.
    async def admin_handle_user_management(self, event, message):
        session = await self.get_user_session(event.sender_id)
        try:
            global_user_num = int(message)
            all_users = session.pagination_data

            if 1 <= global_user_num <= len(all_users):
                user_id, forwarder = all_users[global_user_num - 1]

                await event.reply(f"👤 **Managing User: {forwarder.phone_number}**\n\n"
                                f"**User ID:** `{user_id}`\n"
                                f"**Jobs:** {len(forwarder.jobs)}\n"
                                f"**Status:** {'✅ Connected' if forwarder.is_authorized else '❌ Disconnected'}\n\n"
                                f"**Available Actions:**\n"
                                f"• `stop_all` - Stop all jobs for this user\n"
                                f"• `disconnect` - Disconnect user's account\n"
                                f"• `remove` - Remove user completely\n"
                                f"• `back` - Go back to user list\n\n"
                                f"Send action:")

                session.temp_data['selected_user'] = (user_id, forwarder)
                session.set_state("admin_user_action")
            else:
                await event.reply("❌ Invalid user number. Please send a number from the list.")
        except (ValueError, IndexError):
            # --- FIX: CHANGE "job number" TO "user number" ---
            await event.reply("❌ Please send a valid user number from the list.")

    async def admin_show_statistics(self, event):
        """Shows advanced system statistics including subscription and revenue data."""
        total_users = len(self.user_forwarders)
        total_jobs = sum(len(fw.jobs) for fw in self.user_forwarders.values())
        connected_users = sum(1 for fw in self.user_forwarders.values() if fw.is_authorized)

        # --- NEW: Subscription and Revenue Calculation ---
        monthly_subs = 0
        yearly_subs = 0
        monthly_revenue = 0.0
        yearly_revenue = 0.0

        # Iterate through all user data files to gather subscription info
        for filename in os.listdir(DATA_DIR):
            if filename.startswith("user_") and filename.endswith(".dat"):
                try:
                    user_id = int(filename.split("_")[1].split(".")[0])
                    user_data = await self.load_user_data(user_id)
                    if user_data and user_data.get('subscription_status') == 'paid':
                        # Check if subscription is currently active
                        expiry = datetime.datetime.fromisoformat(user_data.get('subscription_expiry_date'))
                        if expiry > datetime.datetime.now():
                            plan_type = user_data.get('plan_type', '').lower()
                            if 'monthly' in plan_type:
                                monthly_subs += 1
                                monthly_revenue += PRICES.get('monthly_usd', 0)
                            elif 'yearly' in plan_type:
                                yearly_subs += 1
                                yearly_revenue += PRICES.get('yearly_usd', 0)
                except Exception:
                    continue # Skip if a user data file is corrupted

        stats_text = "📊 **System Statistics**\n\n"
        stats_text += f"👥 **Users:** {total_users} total, {connected_users} connected\n"
        stats_text += f"⚙️ **Jobs:** {total_jobs} active\n\n"

        stats_text += "**Active Subscriptions**\n"
        stats_text += f"📅 **Monthly:** {monthly_subs} users (`${monthly_revenue:.2f}`/month)\n"
        stats_text += f"🗓️ **Yearly:** {yearly_subs} users (`${yearly_revenue:.2f}`/year)\n\n"

        stats_text += "🔧 **Master Secret:** Active\n"
        stats_text += "💾 **Data Encryption:** Enabled"

        # Add a button to download the detailed revenue report
        buttons = [[Button.text("📄 Download Yearly Revenue Report", resize=True)], [Button.text("🔙 Back to Admin Menu", resize=True)]]
        await event.reply(stats_text, buttons=buttons)

    # REPLACED based on fixed_disconnect_logic.txt
    async def send_main_menu(self, event):
        print(f"🔄 MAIN MENU TRIGGERED | Message: '{event.message.text if hasattr(event, 'message') else 'N/A'}'")
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        session.set_state("idle")

        if user_id in self.user_forwarders and self.user_forwarders[user_id].is_authorized:
            forwarder = self.user_forwarders[user_id]
            menu_text = "🤖 **Telegram Forwarder Bot**\n\n"
            menu_text += f"📱 Connected as: {forwarder.phone_number}\n"
            menu_text += f"📊 Active jobs: {len(forwarder.jobs)}\n\n"
            menu_text += "Choose an option using the:"

            buttons = [
                [Button.text("📋 Manage Jobs", resize=True)],
                [Button.text("📝 List Chats", resize=True)],
                [Button.text("🔄 Reconnect Account", resize=True)],
                [Button.text("🚪 Logout (Keep Jobs)", resize=True)],
                [Button.text("🔷 Base", resize=True)],
                # The "Delete Account" button is replaced with "Other Settings"
                [Button.text("⚙️ Other Settings", resize=True)]
            ]
        else:
            menu_text = "🤖 **Welcome to Telegram Forwarder Bot!**\n\n"
            menu_text += "This bot helps you automatically forward messages between Telegram chats.\n\n"
            if self.demo_mode:
                menu_text += "✨ **DEMO MODE is active!** ✨\n\n"
            menu_text += "To get started, you'll need to connect your Telegram account."

            buttons = [
                [Button.text("🔗 Connect Account", resize=True)],
                [Button.text("ℹ️ Help", resize=True)]
            ]
        #
        await event.reply(menu_text, buttons=buttons)
        
    async def show_base_menu(self, event):
        """
        Display Base submenu as text-buttons (reuses same Button.text style).
        This sets session.state to 'base_menu' so handle_base_menu can pick up choices.
        """
        session = await self.get_user_session(event.sender_id)
        session.set_state("base_menu")

        menu_text = "🔷 **Base Menu**\n\nQuick actions for Base / Basenames / Wallets:"
        buttons = [
            [Button.text("🔗 Link Wallet", resize=True), Button.text("👤 My Wallet", resize=True)],
            [Button.text("🔎 Resolve Name", resize=True), Button.text("🆔 Register Name", resize=True)],
            [Button.text("💳 Payments", resize=True), Button.text("🔙 Back to Main Menu", resize=True)]
        ]
        await event.reply(menu_text, buttons=buttons)

    async def handle_base_menu(self, event, message):
        """
        Handler for Base submenu button presses.
        - Link Wallet -> creates a challenge and replies with the string to sign
        - My Wallet -> shows currently linked wallet (if any)
        - Resolve Name -> set session and prompt for a name to resolve
        - Register Name -> set session and prompt for a name to register
        - Payments -> brief instruction to run /pay or triggers payment prompt
        - Back to Main Menu -> calls send_main_menu
        """
        user_id = event.sender_id
        # Link Wallet
        if "Link Wallet" in message or "🔗 Link Wallet" in message:
            await event.reply("Creating wallet link challenge...")
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                msg = await loop.run_in_executor(None, create_link_challenge, user_id)
                help_text = (
                    "Sign the message with your wallet (MetaMask/Wallet) using personal_sign.\n\n"
                    "After signing, send:\n"
                    "`/confirm_link <address?> <signature>`\n\n"
                    "Example signature-only: `/confirm_link 0x2c64...`"
                )
                await event.reply(f"Challenge message (sign exactly):\n\n```\n{msg}\n```\n\n{help_text}")
            except Exception as e:
                await event.reply(f"Failed to create challenge: {e}")

        # My Wallet
        elif "My Wallet" in message or "👤 My Wallet" in message:
            await event.reply("Checking linked wallet...")
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                addr = await loop.run_in_executor(None, get_user_linked_address, user_id)
                if addr:
                    await event.reply(f"Your linked wallet: `{addr}`")
                else:
                    await event.reply("You do not have a linked wallet. Use `Link Wallet` to start.")
            except Exception as e:
                await event.reply(f"Error reading linked wallet: {e}")

        # Resolve Name
        elif "Resolve Name" in message or "🔎 Resolve Name" in message:
            session = await self.get_user_session(user_id)
            session.set_state("awaiting_resolve_name")
            await self.show_prompt_for_state(event, "awaiting_resolve_name")

        # Register Name
        elif "Register Name" in message or "🆔 Register Name" in message:
            session = await self.get_user_session(user_id)
            session.set_state("awaiting_register_name")
            await self.show_prompt_for_state(event, "awaiting_register_name")

        # Payments
        elif "Payments" in message or "💳 Payments" in message:
            # If you have an interactive payment prompt state, use it. Else instruct user.
            try:
                # attempt to reuse an existing prompt handler if available
                session = await self.get_user_session(user_id)
                session.set_state("awaiting_payment_amount")
                await self.show_prompt_for_state(event, "awaiting_payment_amount")
            except Exception:
                await event.reply("To start a payment, send: `/pay <amount_usd>` (e.g. `/pay 5.00`).")

        # Back to Main Menu
        elif "Back to Main Menu" in message or "🔙 Back to Main Menu" in message:
            await self.send_main_menu(event)

        else:
            await event.reply("Please use one of the Base submenu buttons.")


    # REPLACED based on fixed_disconnect_logic.txt
    async def handle_main_menu(self, event, message):
        if message == "🔗 Connect Account":
            await self.start_account_setup(event)
        elif message == "📋 Manage Jobs":
            await self.show_job_management(event)
        elif message == "📝 List Chats":
            await self.list_user_chats(event)
        elif message == "🔄 Reconnect Account":
            await self.start_account_setup(event)
        elif message == "🚪 Logout (Keep Jobs)":
            await self.logout_account(event)
        elif message == "⚙️ Other Settings":
            await self.show_other_settings_menu(event)
        elif message == "🔷 Base":
            # Open the Base submenu
            await self.show_base_menu(event)
        elif message == "❌ Disconnect Account": # Backward compatibility
            await self.logout_account(event)
        elif message == "ℹ️ Help":
            await self.show_help(event)
        else:
            await event.reply("Please use the menu buttons below.")

    async def show_other_settings_menu(self, event):
        """Displays the secondary menu for less-frequent actions."""
        session = await self.get_user_session(event.sender_id)
        session.set_state("other_settings_menu")

        menu_text = "⚙️ **Other Settings**\n\nSelect an option from the menu below."
        buttons = [
            [Button.text("🏦 Manage Hyperliquid", resize=True)],
            [Button.text("🚀 Subscribe", resize=True)],
            [Button.text("🤝 Referral", resize=True)],
            [Button.text("🎁 Redeem Code", resize=True)],
            [Button.text("🗑️ Delete Account", resize=True)],
            [Button.text("🔙 Back to Main Menu", resize=True)]
        ]
        await event.reply(menu_text, buttons=buttons)

    async def handle_other_settings_menu(self, event, message):
        """Handles button clicks from the 'Other Settings' menu."""
        if "Subscribe" in message:
            # We don't need a separate handler for the button click,
            # we can just show the subscription options directly.
            buttons = [
                [Button.inline("Pay with Card/Bank", data="subscribe_card")],
                [Button.inline("Pay with Crypto", data="subscribe_crypto")]
            ]
            await event.reply("How would you like to subscribe?", buttons=buttons)

        elif "Referral" in message:
            await self.handle_referral_command(event)

        elif "Manage Hyperliquid" in message:
            await self.show_hyperliquid_menu(event)

        elif "Redeem Code" in message:
            session = await self.get_user_session(event.sender_id)
            session.set_state("awaiting_redeem_code")
            # The prompt is now handled by the UI manager
            await self.show_prompt_for_state(event, "awaiting_redeem_code")

        elif "Delete Account" in message:
            await self.delete_account(event)

        elif "Back to Main Menu" in message:
            await self.send_main_menu(event)

        else:
            await event.reply("Please use one of the menu buttons.")

    async def show_hyperliquid_menu(self, event):
        """Displays the main menu for managing Hyperliquid settings."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        session.set_state("hyperliquid_menu")

        user_data = await self.load_user_data(user_id)

        # Check if the user has already connected their account
        is_connected = user_data and user_data.get('hl_api_key')

        if is_connected:
            status_text = f"✅ Account Connected: `{user_data['hl_api_key'][:8]}...`"
            buttons = [
                [Button.text("⚙️ Set Trade Defaults", resize=True)],
                [Button.text("🗑️ Disconnect Account", resize=True)],
                [Button.text("🔙 Back to Other Settings", resize=True)]
            ]
        else:
            status_text = "❌ Account Not Connected"
            buttons = [
                [Button.text("🔗 Connect Account", resize=True)],
                [Button.text("🔙 Back to Other Settings", resize=True)]
            ]

        menu_text = f"🏦 **Hyperliquid Management**\n\n**Status:** {status_text}\n\nPlease choose an option below."

        await event.reply(menu_text, buttons=buttons)

    async def start_account_setup(self, event):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # ADDED: Demo mode check
        # --- THIS IS THE KEY CHANGE ---
        if self.demo_mode:
            if not self.demo_forwarder or not self.demo_forwarder.is_authorized:
                await event.reply("🚧 Demo Mode is configured but the pre-authenticated session is not ready. Please contact the administrator.")
                return

            session.set_state("demo_awaiting_api_id")

            # Use the DEMO_API_ID from your loaded config
            # from your_config_module import DEMO_API_ID # Or however you access it

            await event.reply(
                "👋 **Welcome to the Demo!**\n\n"
                "We will now walk you through the real login process using a pre-configured demo account. I will provide you with the details to enter at each step.\n\n"
                "First, you need an **API ID** from my.telegram.org.\n\n"
                f"For this demo, please copy and enter the following **Demo API ID**:\n`{DEMO_API_ID}`"
            )
            return

        session.temp_data = {}
        session.set_state("awaiting_api_id")

        await self.show_prompt_for_state(event, "awaiting_api_id")

    # MODIFIED - Reintroducing API ID validation logic
    async def handle_api_id(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # --- FIX: Delete the user's message for privacy ---
        await event.delete()

        if not message.isdigit() or not (7 <= len(message) <= 8):
            await event.reply("❌ Invalid API ID. Must be 7 or 8 digits. Please try again:")
            return

        session.temp_data['api_id'] = int(message)
        session.set_state("awaiting_api_hash")
        # The prompt is now handled by the UI manager
        await self.show_prompt_for_state(event, "awaiting_api_hash")

    async def handle_api_hash(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # --- FIX: Delete the user's message for privacy ---
        await event.delete()

        if not re.match(r"^[a-f0-9]{32}$", message, re.IGNORECASE):
            await event.reply("❌ Invalid API Hash. It must be 32 hexadecimal characters. Please try again:")
            return

        session.temp_data['api_hash'] = message
        session.set_state("awaiting_phone")
        # The prompt is now handled by the UI manager
        await self.show_prompt_for_state(event, "awaiting_phone")

    # MODIFIED - Reintroducing obfuscation logic
    async def handle_phone(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # --- FIX: Delete the user's message for privacy ---
        await event.delete()

        if not re.match(r"^\+\d{10,}$", message):
            await event.reply("❌ Invalid phone number format. Please use format: +1234567890")
            return

        session.temp_data['phone_number'] = message

        # Create forwarder and attempt connection
        forwarder = TelegramForwarder(
            session.temp_data['api_id'],
            session.temp_data['api_hash'],
            session.temp_data['phone_number'],
            user_id,
            bot_instance=self # Add bot reference for notifications
        )

        await event.reply("📱 Connecting to Telegram and requesting verification code... Please wait.")

        try:
            result = await forwarder.connect_and_authorize()

            if result == "code_requested":
                session.forwarder = forwarder
                session.set_state("awaiting_code")
                # REINTRODUCED OBFUSCATION INSTRUCTION
                #await event.reply("📲 **Verification code sent!**\n\n⚡ Please check your Telegram app and send me the verification code.\n\n🔢 **IMPORTANT:** You must obfuscate the code to avoid expiration:\n• Instead of: `123456`\n• Send: `1a2b3c4d5e6f` or `1 2 3 4 5 6`\n\n⚠️ Plain numeric codes will be rejected!")
                # The prompt is now handled by the UI manager
                await self.show_prompt_for_state(event, "awaiting_code")
            elif result == "authorized":
                await self.complete_setup(event, forwarder)
            elif result.startswith("flood_wait"):
                wait_time = result.split("_")[2]
                await event.reply(f"⏱️ **Rate limited by Telegram**\n\nPlease wait {wait_time} seconds before trying again.\n\nThis is a Telegram security measure.")
                session.set_state("idle")
            else:
                await event.reply(f"❌ Connection failed: {result}\n\nPlease check your API details and try again with /start")
                session.set_state("idle")
        except Exception as e:
            logging.error(f"Error during phone setup for user {user_id}: {e}")
            await event.reply("❌ **Connection error**\n\nThere was an issue connecting to Telegram. Please check your credentials and try again with /start")
            session.set_state("idle")

    # ENTIRE METHOD REPLACED to handle obfuscation
    async def handle_code(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # --- FIX: Delete the user's message for privacy ---
        await event.delete()

        # Obfuscation protocol: Only accept obfuscated input
        # Check if input is purely numeric (not allowed)
        if message.isdigit():
            await event.reply("❌ For security, please obfuscate your code by adding letters or spaces:\n• Instead of: `123456`\n• Send: `1a2b3c4d5e6f` or `1 2 3 4 5 6`")
            return

        # Clean the obfuscated input code
        # Remove any non-digit characters (letters, spaces, special chars)
        clean_code = ''.join(char for char in message if char.isdigit())

        if not clean_code:
            await event.reply("❌ Please send the verification code with letters or spaces mixed in (e.g., '1a2b3c4d5e6f' or '1 2 3 4 5 6'):")
            return

        await event.reply("🔐 Verifying code... please wait.")

        try:
            # Use the cleaned code for authentication
            result = await session.forwarder.connect_and_authorize(phone_code=clean_code)

            if result == "authorized":
                await self.complete_setup(event, session.forwarder)
            elif result == "2fa_required":
                session.set_state("awaiting_2fa")
                # The prompt is now handled by the UI manager
                await self.show_prompt_for_state(event, "awaiting_2fa")
            elif result == "invalid_code":
                await event.reply("❌ Invalid verification code. Please try again with obfuscated format (add letters or spaces):")
            elif result == "error" or result == "code_expired":
                await event.reply("❌ **Code expired or authentication error**\n\n⚠️ This can happen if:\n• The code took too long to enter\n• The code was used elsewhere\n• Network issues occurred\n\n🔄 **Let's try again:**")
                session.set_state("awaiting_phone")
                await event.reply("Please send your phone number again to get a new verification code:")
            else:
                await event.reply(f"❌ Authentication failed: {result}\n\n🔄 Let's start fresh - send /start to try again")
                session.set_state("idle")
        except Exception as e:
            logging.error(f"Error during code verification for user {user_id}: {e}")
            await event.reply("❌ **Code verification failed**\n\n⚠️ This usually means the code expired or there was a network issue.\n\n🔄 **Let's get a fresh code:**")
            session.set_state("awaiting_phone")
            await event.reply("Please send your phone number again to request a new verification code:")

    async def handle_2fa(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # --- FIX: Delete the user's message for privacy ---
        await event.delete()

        await event.reply("🔐 Verifying 2FA password... please wait.")
        result = await session.forwarder.connect_and_authorize(password=message)

        if result == "authorized":
            await self.complete_setup(event, session.forwarder)
        elif result == "invalid_password":
            await event.reply("❌ Invalid 2FA password. Please try again:")
        else:
            await event.reply(f"❌ Authentication failed: {result}. Please start over with /start")
            session.set_state("idle")

    # REPLACED
    async def complete_setup(self, event, forwarder):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # Add bot reference to forwarder for error notifications
        forwarder.bot_instance = self

        # Check if user has existing data to restore
        existing_data = await self.load_user_data(user_id)
        if existing_data:
            # Restore existing jobs and other data
            forwarder.jobs = existing_data.get('jobs', [])
            forwarder.saved_patterns = existing_data.get('saved_patterns', [])


            job_count = len(forwarder.jobs)
            setup_message = "✅ **Account reconnected successfully!**\n\n"
            setup_message += f"📱 Phone: {forwarder.phone_number}\n"
            setup_message += f"📋 **Restored {job_count} existing jobs!**\n"
            setup_message += "🎯 All jobs are now active and running!"
        else:
            # New account setup
            setup_message = "✅ **Account connected successfully!**\n\n"
            setup_message += f"📱 Phone: {forwarder.phone_number}\n"
            setup_message += "🎯 Ready to create forwarding jobs!"

        # Save user data and start the new event-driven listener
        self.user_forwarders[user_id] = forwarder
        await self.save_user_data(user_id, forwarder)
        await self.start_forwarder_session(user_id, forwarder)

        session.set_state("idle")
        session.forwarder = None # Clear temporary forwarder
        await event.reply(setup_message, buttons=Button.clear())
        await self.send_main_menu(event)

    async def show_job_management(self, event, page_index=0):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        if user_id not in self.user_forwarders:
            await event.reply("❌ Please connect your account first.")
            return

        session.set_state("job_management")
        forwarder = self.user_forwarders[user_id]

        if not forwarder.jobs:
            menu_text = "📋 **Job Management**\n\nYou don't have any active jobs yet."
            buttons = [
                [Button.text("➕ Create Job", resize=True)],
                [Button.text("🔙 Back to Main Menu", resize=True)]
            ]
            await event.reply(menu_text, buttons=buttons)
            return

        items_per_page = 5
        total_items = len(forwarder.jobs)
        total_pages = (total_items + items_per_page - 1) // items_per_page

        start_index = page_index * items_per_page
        jobs_on_page = forwarder.jobs[start_index : start_index + items_per_page]

        menu_text = f"📋 **Job Management (Page {page_index + 1}/{total_pages})**\n\n"
        for i, job in enumerate(jobs_on_page, start=start_index):
            # --- START OF FULLY CORRECTED LOGIC ---
            job_name = job.get('job_name')
            job_type_str = job.get('type', 'N/A').replace('_', ' ').capitalize()
            match_logic = job.get('match_logic', 'OR')

            # Append match logic to the type string if applicable
            items_key = 'keywords' if job['type'] == 'keywords' else 'patterns'
            if job['type'] in ['keywords', 'custom_pattern'] and len(job.get(items_key, [])) > 1:
                job_type_str += f" ({match_logic})"

            # Display Job Name first, then the detailed type string
            if job_name:
                menu_text += f"**{i+1}. {job_name}** (`{job_type_str}`)\n"
            else: # Fallback for old, unnamed jobs
                menu_text += f"**{i+1}.** **{job_type_str}**\n"
            # --- END OF FULLY CORRECTED LOGIC ---

            source_names = [forwarder.chat_cache.get(sid, f"ID {sid}") for sid in job.get('source_ids', [])]
            dest_names = [forwarder.chat_cache.get(did, f"ID {did}") for did in job.get('destination_ids', [])]

            # Special handling to display 'Hyperliquid' correctly
            if dest_names == ['hyperliquid_account']:
                dest_display = "Hyperliquid Account"
            else:
                dest_display = f"{', '.join(dest_names[:2])}{'...' if len(dest_names) > 2 else ''}"

            menu_text += f"   📥 From: {', '.join(source_names[:2])}{'...' if len(source_names) > 2 else ''}\n"
            menu_text += f"   📤 To: {dest_display}\n\n"

        pagination_buttons = []
        if page_index > 0:
            pagination_buttons.append(Button.inline("⬅️ Previous", data=f"page_user_jobs_{page_index - 1}"))
        if (page_index + 1) < total_pages:
            pagination_buttons.append(Button.inline("Next ➡️", data=f"page_user_jobs_{page_index + 1}"))

        action_buttons = [
            [Button.text("➕ Create Job", resize=True), Button.text("✏️ Modify Job", resize=True)],
            [Button.text("🗑️ Delete Job", resize=True)],
            [Button.text("🔙 Back to Main Menu", resize=True)]
        ]

        if hasattr(event, 'data'):
            await event.edit(menu_text, buttons=[pagination_buttons] if pagination_buttons else None)
        else:
            await event.reply(menu_text, buttons=[pagination_buttons] if pagination_buttons else None)
            await event.respond("Choose an option:", buttons=action_buttons)

    async def handle_job_management(self, event, message):
        if message == "➕ Create Job":
            await self.start_job_creation(event)
        elif message == "✏️ Modify Job":
            await self.show_job_modification_menu(event)
        elif message == "🗑️ Delete Job":
            await self.show_job_deletion(event)
        elif message == "🔙 Back to Main Menu":
            await self.send_main_menu(event)
        else:
            await event.reply("Please use the menu buttons.")

    # --- FIX #2: APPLIED ---
    # A guard clause is added to handle cases with no jobs.
    async def show_job_modification_menu(self, event, page_index=0):
        user_id = event.sender_id
        forwarder = self.user_forwarders.get(user_id)

        # --- FIX: ADD THIS GUARD CLAUSE AT THE TOP ---
        if not forwarder or not forwarder.jobs:
            await event.reply("✅ You have no jobs to modify. Use 'Create Job' to make one!")
            return
        # --- END OF FIX ---

        session = await self.get_user_session(user_id)
        if not session: return # Safety check

        session.set_state("modifying_job")
        session.pagination_data = forwarder.jobs # Store for the handler

        async def lister(): return forwarder.jobs
        async def formatter(jobs_on_page, start_index, title, current_page, total_pages):
            text = f"**{title} (Page {current_page}/{total_pages})**\n\n"
            for i, job in enumerate(jobs_on_page, start=start_index + 1):
                job_name = job.get('job_name')
                job_type_str = job.get('type', 'N/A').replace('_', ' ').capitalize()
                match_logic = job.get('match_logic', 'OR')

                items_key = 'keywords' if job['type'] == 'keywords' else 'patterns'
                if job['type'] in ['keywords', 'custom_pattern'] and len(job.get(items_key, [])) > 1:
                    job_type_str += f" ({match_logic})"

                if job_name:
                    text += f"**{i}. {job_name}** (`{job_type_str}`)\n"
                else: # Fallback for old jobs
                    text += f"**{i}.** **{job_type_str}**\n"

            last_num = start_index + len(jobs_on_page)
            text += f"\nReply with a number **({start_index + 1}-{last_num})** to modify, or type 'cancel'."
            return text

        await self.show_paginated_list(
            event,
            page_index=page_index,
            item_lister=lister,
            item_formatter=formatter,
            callback_prefix="page_user_modify",
            title="✏️ Modify Job"
        )

    async def handle_job_modification_selection(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        if message.lower() == 'cancel':
            session.set_state("job_management")
            await self.show_job_management(event)
            return

        try:
            job_num = int(message)
            all_jobs = session.pagination_data
            if 1 <= job_num <= len(all_jobs):
                session.modifying_job_index = job_num - 1
                job = all_jobs[job_num - 1]

                match_logic = job.get('match_logic', 'OR')
                job_type_str = job.get('type', 'N/A').replace('_', ' ').capitalize()
                items_key = 'keywords' if job['type'] == 'keywords' else 'patterns'
                if job['type'] in ['keywords', 'custom_pattern'] and len(job.get(items_key, [])) > 1:
                     job_type_str += f" ({match_logic})"

                source_names = [forwarder.chat_cache.get(sid, f"ID {sid}") for sid in job.get('source_ids', [])]
                dest_names = [forwarder.chat_cache.get(did, f"ID {did}") for did in job.get('destination_ids', [])]

                settings_text = f"📝 **Modifying Job #{job_num}: {job_type_str}**\n\n"
                settings_text += "**Current Settings:**\n"
                settings_text += f"📥 **Sources:** {', '.join(source_names)}\n"
                settings_text += f"📤 **Destinations:** {', '.join(dest_names)}\n"

                if job.get('type') == 'keywords':
                    keywords = job.get('keywords', [])
                    settings_text += f"🔤 **Keywords:** {', '.join(keywords) if keywords else 'All messages'}\n"
                elif job.get('type') == 'cashtags':
                    cashtags = job.get('cashtags', [])
                    settings_text += f"💰 **Cashtags:** {', '.join(cashtags) if cashtags else 'All cashtags'}\n"
                elif job.get('type') == 'custom_pattern':
                    patterns = job.get('patterns', [])
                    settings_text += f"🔍 **Patterns:** `{', '.join(patterns)}`\n"

                timer = job.get('timer', '')
                settings_text += f"⏱️ **Timer:** {timer if timer else 'No cooldown'}\n\n"
                settings_text += "**What would you like to modify?**"

                buttons = [
                    [Button.text("📥 Sources", resize=True), Button.text("📤 Destinations", resize=True)],
                ]
                if job.get('type') == 'keywords': buttons.append([Button.text("🔤 Keywords", resize=True)])
                elif job.get('type') == 'cashtags': buttons.append([Button.text("💰 Cashtags", resize=True)])
                elif job.get('type') == 'custom_pattern': buttons.append([Button.text("🔍 Patterns", resize=True)])

                items_key = 'keywords' if job['type'] == 'keywords' else 'patterns'
                if job.get('type') in ['keywords', 'custom_pattern'] and len(job.get(items_key, [])) > 1:
                    buttons.append([Button.text("🔄 Match Logic", resize=True)])
                buttons.extend([[Button.text("⏱️ Timer", resize=True)], [Button.text("❌ Cancel", resize=True)]])

                session.set_state("modify_selection")
                await event.reply(settings_text, buttons=buttons)
            else:
                await event.reply(f"❌ Invalid job number. Please send a number between 1 and {len(all_jobs)}:")
        except ValueError:
            await event.reply("❌ Please send a valid job number:")

    async def handle_job_modification_steps(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]
        job_index = session.modifying_job_index
        job = forwarder.jobs[job_index]

        if session.state == "modify_selection":
            if message == "📥 Sources":
                session.set_state("modify_sources")
                current_sources = [forwarder.chat_cache.get(sid, f"ID {sid}") for sid in job.get('source_ids', [])]
                await event.reply(f"📥 **Modify Sources**\n\n**Current:** {', '.join(current_sources)}\n\nSend new source chats (comma-separated):")
            elif message == "📤 Destinations":
                session.set_state("modify_destinations")
                current_dests = [forwarder.chat_cache.get(did, f"ID {did}") for did in job.get('destination_ids', [])]
                await event.reply(f"📤 **Modify Destinations**\n\n**Current:** {', '.join(current_dests)}\n\nSend new destination chats (comma-separated):")
            elif message == "🔤 Keywords" and job.get('type') == 'keywords':
                session.set_state("modify_keywords")
                current_keywords = job.get('keywords', [])
                await event.reply(f"🔤 **Modify Keywords**\n\n**Current:** {', '.join(current_keywords) if current_keywords else 'All messages'}\n\nSend new keywords (comma-separated, or 'none' for all messages):")
            elif message == "💰 Cashtags" and job.get('type') == 'cashtags':
                session.set_state("modify_cashtags")
                current_cashtags = job.get('cashtags', [])
                await event.reply(f"💰 **Modify Cashtags**\n\n**Current:** {', '.join(current_cashtags) if current_cashtags else 'All cashtags'}\n\nSend new cashtags (comma-separated, or 'none' for all cashtags):")
            elif message == "🔍 Patterns" and job.get('type') == 'custom_pattern':
                session.set_state("modify_custom_patterns")
                current_patterns = job.get('patterns', [])
                # For modification, we will ask for the full new list to simplify the flow.
                # The one-by-one method is best for initial creation.
                await event.reply(f"🔍 **Replace Patterns**\n\n**Current:** `{', '.join(current_patterns)}`\n\nSend the new, complete list of patterns, separated by a comma. For help, send `/help regex`.")

            # ADDED: Handler for modifying match logic
            elif message == "🔄 Match Logic":
                session.set_state("modify_match_logic")
                current_logic = job.get('match_logic', 'OR')
                buttons = [
                    [Button.text("OR - Must contain any", resize=True)],
                    [Button.text("AND - Must contain all", resize=True)]
                ]
                await event.reply(f"🔄 **Modify Match Logic**\n\n**Current:** `{current_logic}`\n\nChoose new logic:", buttons=buttons)

            elif message == "⏱️ Timer":
                session.set_state("modify_timer")
                current_timer = job.get('timer', '')
                await event.reply(f"⏱️ **Modify Timer**\n\n**Current:** {current_timer if current_timer else 'No cooldown'}\n\nSend new timer (e.g., '5 minutes', '1 hour', or 'none' for no cooldown):")
            elif message == "❌ Cancel":
                session.set_state("job_management")
                session.modifying_job_index = None
                await event.reply("✅ Job modification cancelled.")
                await self.show_job_management(event)

        elif session.state == "modify_sources":
            await self._modify_job_sources(event, message, job_index)
        elif session.state == "modify_destinations":
            await self._modify_job_destinations(event, message, job_index)
        elif session.state == "modify_keywords":
            await self._modify_job_keywords(event, message, job_index)
        elif session.state == "modify_cashtags":
            await self._modify_job_cashtags(event, message, job_index)
        elif session.state == "modify_custom_patterns":
            await self._modify_job_patterns(event, message, job_index)
        # ADDED: Handler for modifying match logic
        elif session.state == "modify_match_logic":
            await self._modify_job_match_logic(event, message, job_index)
        elif session.state == "modify_timer":
            await self._modify_job_timer(event, message, job_index)

    async def _modify_job_sources(self, event, message, job_index):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        sources = [s.strip() for s in message.split(',') if s.strip()]
        try:
            source_details = []
            for source in sources:
                details = await forwarder._get_entity_details(source)
                source_details.append(details)

            # Update job
            forwarder.jobs[job_index]['source_ids'] = [d[0] for d in source_details]
            await self.save_user_data(user_id, forwarder)

            source_names = [d[1] for d in source_details]
            await event.reply(f"✅ **Sources Updated**\n\n**New sources:** {', '.join(source_names)}", buttons=Button.clear())

            session.set_state("job_management")
            session.modifying_job_index = None
            await self.show_job_management(event)

        except ValueError as e:
            await event.reply(f"❌ Error: {e}\n\nPlease try again with valid chat names, usernames, or IDs:")

    async def _modify_job_destinations(self, event, message, job_index):
        """
        Accept a comma-separated list of destinations.
        Supports:
          - Telegram chat ids / usernames (existing behaviour)
          - onchain:<0x...> or onchain:<name>.base  (special onchain destinations)
          - bare 0x... addresses or <name>.base (auto-detected)
        Stores:
          - forwarder.jobs[job_index]['destination_ids'] -> list of Telegram chat ids (integers)
          - forwarder.jobs[job_index]['onchain_destinations'] -> list of onchain dest strings
        """
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        destinations = [d.strip() for d in message.split(',') if d.strip()]
        try:
            dest_details = []
            onchain_destinations = []
            import re as _re

            for dest in destinations:
                # Normalize and detect onchain prefix or patterns
                low = dest.lower()
                if low.startswith("onchain:"):
                    raw = dest.split(":", 1)[1].strip()
                    onchain_destinations.append(raw)
                    continue
                # raw address or basename heuristics
                if low.endswith(".base") or (_re.match(r"^0x[a-f0-9]{40}$", low, flags=_re.I)):
                    onchain_destinations.append(dest)
                    continue

                # fallback: treat as telegram entity (numeric ID or username/handle)
                details = await forwarder._get_entity_details(dest)
                dest_details.append(details)

            # Update job
            forwarder.jobs[job_index]['destination_ids'] = [d[0] for d in dest_details]
            if onchain_destinations:
                forwarder.jobs[job_index]['onchain_destinations'] = onchain_destinations
            else:
                # remove field if none present
                forwarder.jobs[job_index].pop('onchain_destinations', None)

            # Persist changes
            await self.save_user_data(event.sender_id, forwarder)

            await event.reply("✅ Destinations updated.")
        except Exception as e:
            logging.exception(f"Error modifying job destinations: {e}")
            await event.reply(f"❌ Invalid selection: {e}")
            await self.show_prompt_for_state(event, "creating_job")
            return


    async def _modify_job_keywords(self, event, message, job_index):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        keywords = []
        if message.lower() != 'none':
            keywords = [k.strip() for k in message.split(',') if k.strip()]

        # Update job
        forwarder.jobs[job_index]['keywords'] = keywords
        # ADDED: Reset match logic if only one keyword now
        if len(keywords) <= 1:
            forwarder.jobs[job_index]['match_logic'] = 'OR'

        await self.save_user_data(user_id, forwarder)

        if keywords:
            await event.reply(f"✅ **Keywords Updated**\n\n**New keywords:** {', '.join(keywords)}", buttons=Button.clear())
        else:
            await event.reply("✅ **Keywords Updated**\n\n**Mode:** Forward all messages", buttons=Button.clear())

        session.set_state("job_management")
        session.modifying_job_index = None
        await self.show_job_management(event)

    async def _modify_job_cashtags(self, event, message, job_index):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        cashtags = []
        if message.lower() != 'none':
            cashtags = [c.strip() for c in message.split(',') if c.strip()]

        # Update job
        forwarder.jobs[job_index]['cashtags'] = cashtags
        await self.save_user_data(user_id, forwarder)

        if cashtags:
            await event.reply(f"✅ **Cashtags Updated**\n\n**New cashtags:** {', '.join(cashtags)}", buttons=Button.clear())
        else:
            await event.reply("✅ **Cashtags Updated**\n\n**Mode:** Forward all cashtags", buttons=Button.clear())

        session.set_state("job_management")
        session.modifying_job_index = None
        await self.show_job_management(event)

    async def _modify_job_patterns(self, event, message, job_index):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        patterns = [p.strip() for p in message.split(',') if p.strip()]
        for p in patterns:
            try:
                re.compile(p)
            except re.error as e:
                await event.reply(f"❌ **Invalid Pattern:** One of your patterns is not valid.\n`{e}`\nPlease try again:")
                return

        # Update job
        forwarder.jobs[job_index]['patterns'] = patterns
        # ADDED: Reset match logic if only one pattern now
        if len(patterns) <= 1:
            forwarder.jobs[job_index]['match_logic'] = 'OR'

        await self.save_user_data(user_id, forwarder)

        await event.reply(f"✅ **Patterns Updated**\n\n**New patterns:** `{', '.join(patterns)}`", buttons=Button.clear())

        session.set_state("job_management")
        session.modifying_job_index = None
        await self.show_job_management(event)

    # ADDED: New handler for modifying match logic
    async def _modify_job_match_logic(self, event, message, job_index):
        user_id = event.sender_id
        forwarder = self.user_forwarders[user_id]

        logic = 'OR'
        if "AND" in message:
            logic = 'AND'

        forwarder.jobs[job_index]['match_logic'] = logic
        await self.save_user_data(user_id, forwarder)

        await event.reply(f"✅ **Match Logic Updated** to `{logic}`.", buttons=Button.clear())

        session = await self.get_user_session(user_id)
        session.set_state("job_management")
        session.modifying_job_index = None
        await self.show_job_management(event)


    async def _modify_job_timer(self, event, message, job_index):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        timer = message.strip()
        if timer.lower() == 'none':
            timer = ''

        # Update job
        forwarder.jobs[job_index]['timer'] = timer
        await self.save_user_data(user_id, forwarder)

        if timer:
            await event.reply(f"✅ **Timer Updated**\n\n**New cooldown:** {timer}", buttons=Button.clear())
        else:
            await event.reply("✅ **Timer Updated**\n\n**Mode:** No cooldown", buttons=Button.clear())

        session.set_state("job_management")
        session.modifying_job_index = None
        await self.show_job_management(event)

    # This is the entry point, called when a user clicks "Create Job"
    async def start_job_creation(self, event, is_going_back=False):
        """
        Initializes the job creation flow and displays the first menu.
        """
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # Only clear the pending job if we are starting fresh.
        if not is_going_back:
            session.pending_job = {}

        session.set_state("creating_job")

        # The text block lives here, inside the function that manages this state.
        prompt = ("""➕ **Create New Job**

Choose the type of forwarding job:

1️⃣ **Keywords**
2️⃣ **Solana**
3️⃣ **Ethereum**
4️⃣ **Cashtags**
5️⃣ **Custom Pattern**

Send the number (1-5) or type the job type name.""")

        # This is a menu, not a conversational step, so it has its own keyboard
        # that should NOT include Back/Cancel.
        buttons = [[Button.text("🔙 Back to Main Menu", resize=True)]]

        await event.reply(prompt, buttons=buttons)

    # This handler processes the job type (e.g., "1" for Keywords)
    async def handle_job_creation(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        job_map = {'1': 'keywords', '2': 'solana', '3': 'ethereum', '4': 'cashtags', '5': 'custom_pattern'}
        job_type = job_map.get(message.lower().strip().split()[0]) # Get just the first word/number

        if not job_type:
            await event.reply("❌ Invalid selection. Please try again.")
            await self.show_prompt_for_state(event, "creating_job") # Re-show prompt
            return

        permission, msg = await self.check_permission(user_id, 'create_job', context={'job_type': job_type})
        if not permission:
            await event.reply(msg)
            session.go_back()
            await self.show_prompt_for_state(event, "creating_job") # Go back to job type selection
            return

        session.pending_job['type'] = job_type
        session.set_state("job_sources")
        await self.show_prompt_for_state(event, "job_sources", extra_info=f"Job type set to: **{job_type.replace('_', ' ').capitalize()}**")

    async def handle_job_steps(self, event, message):
        """
        Handles all steps of the job creation process after the type has been selected.
        This is a state machine that guides the user through providing sources,
        destinations, keywords, etc., with full back/cancel and permission enforcement.
        """
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders.get(user_id)

        # Safety check: A forwarder must exist to create a job.
        if not forwarder:
            await event.reply("❌ Your account is not properly connected. Please use /start to reconnect.")
            session.state = "idle" # Reset state
            return

        state = session.state

        try:
            # --- State 1: Awaiting Source Chats ---
            if state == "job_sources":
                sources = [s.strip() for s in message.split(',') if s.strip()]
                if not sources:
                    await event.reply("❌ Please provide at least one source chat.")
                    return

                # --- RESTORED LOGIC: Permission Check for Source Count ---
                permission, msg = await self.check_permission(user_id, 'add_source', context={'source_count': len(sources)})
                if not permission:
                    await event.reply(f"🔒 {msg}")
                    await self.show_prompt_for_state(event, "job_sources", extra_info="Please enter a new list of sources that meets your plan's limits.")
                    return
                # --- END OF RESTORED LOGIC ---

                await event.reply("🔍 Verifying source chats...")
                source_details = [await forwarder._get_entity_details(s) for s in sources]

                session.pending_job['source_ids'] = [d[0] for d in source_details]
                source_names = [d[1] for d in source_details]

                session.set_state("job_destinations")
                await self.show_prompt_for_state(event, "job_destinations", extra_info=f"Sources set: {', '.join(source_names)}")

            # --- State 2: Awaiting Destination Chats ---
            elif state == "job_destinations":
                dest_names = []

                # --- NEW LOGIC: Check if the Hyperliquid button was pressed ---
                if message == "🏦 Use Hyperliquid Account":
                    user_data = await self.load_user_data(user_id)
                    if not user_data.get('hl_api_key'):
                        await event.reply("❌ You must connect your Hyperliquid account first. Please go to 'Other Settings' -> 'Manage Hyperliquid' to connect.")
                        return # Stay in the same state

                    # Use a special identifier for the Hyperliquid destination
                    session.pending_job['destination_ids'] = ['hyperliquid_account']
                    dest_names = ["Hyperliquid Account"]
                    
                # New: onchain / Base destination button
                if message == "🪙 Use Base Onchain Destination":
                    session.pending_job['onchain_destinations'] = []
                    session.set_state("job_onchain_destinations")
                    await self.show_prompt_for_state(event, "job_onchain_destinations", extra_info="Send one or more destinations (comma-separated). Use `.base` names or 0x addresses. Set amount later.")
                    return


                # --- EXISTING LOGIC: Handle regular Telegram chat IDs ---
                else:
                    destinations = [d.strip() for d in message.split(',') if d.strip()]
                    if not destinations:
                        await event.reply("❌ Please provide at least one destination chat or use the button.")
                        return

                    permission, msg = await self.check_permission(user_id, 'add_destination', context={'destination_count': len(destinations)})
                    if not permission:
                        await event.reply(f"🔒 {msg}")
                        await self.show_prompt_for_state(event, "job_destinations", extra_info="Please enter a new list of destinations that meets your plan's limits.")
                        return

                    await event.reply("🔍 Verifying destination chats...")
                    dest_details = [await forwarder._get_entity_details(d) for d in destinations]

                    session.pending_job['destination_ids'] = [d[0] for d in dest_details]
                    dest_names = [d[1] for d in dest_details]

                # --- UNIFIED LOGIC: Proceed to the next step ---
                job_type = session.pending_job.get('type')

                # We no longer test destinations if Hyperliquid is selected, as there's nothing to test.
                # A proper connection test will happen during the trade execution itself.
                if 'hyperliquid_account' in session.pending_job.get('destination_ids', []):
                     # For Hyperliquid, we skip directly to the timer/final step for most job types
                     next_state_map = {
                         'keywords': "job_keywords",
                         'cashtags': "job_cashtags",
                         'custom_pattern': "job_custom_pattern_menu"
                     }
                     next_state = next_state_map.get(job_type, "job_timer")

                else: # This is the original logic for Telegram destinations
                    next_state = "job_timer"
                    if job_type == 'keywords': next_state = "job_keywords"
                    elif job_type == 'cashtags': next_state = "job_cashtags"

                if job_type == 'custom_pattern':
                    session.pending_job['patterns'] = []
                    session.set_state("job_custom_pattern_menu")
                    await self.handle_custom_pattern_menu(event, "init")
                    return

                session.set_state(next_state)
                await self.show_prompt_for_state(event, next_state, extra_info=f"Destinations set: {', '.join(dest_names)}")
                
                # -----------------------------------------------------------
                # NEW STATE HANDLER: Onchain destinations entry
                # -----------------------------------------------------------
                if state == "job_onchain_destinations":
                    # Parse comma-separated addresses or basenames
                    parts = [p.strip() for p in message.split(",") if p.strip()]
                    if not parts:
                        await event.reply("Please provide at least one destination.")
                        return

                    # Resolve basenames if needed
                    resolved = []
                    for p in parts:
                        if p.lower().endswith(".base"):
                            addr = resolve_basename(p)
                            if not addr:
                                await event.reply(f"❌ Could not resolve `{p}` — please check spelling or try a different `.base` name.")
                                return
                            resolved.append(addr)
                        else:
                            resolved.append(p)

                    # Store
                    session.pending_job.setdefault("onchain_destinations", [])
                    session.pending_job["onchain_destinations"] = resolved

                    # Proceed to next step: choose amount or continue normal job flow
                    session.set_state("job_timer")  # ← safest universal next step
                    await self.show_prompt_for_state(event, "job_timer")
                    return
                    
                # -----------------------------------------------------------
                # STATE: job_source_base_wallet  (now creates a challenge and asks to sign)
                # -----------------------------------------------------------
                if state == "job_source_base_wallet":
                    user_id = event.sender_id
                    text_in = message.strip()
                    if not text_in:
                        await event.reply("Please send a wallet address (0x...) or a `.base` name (e.g. alice.base).")
                        return

                    # 1) Resolve basename if provided
                    target_addr = text_in
                    try:
                        if text_in.lower().endswith(".base"):
                            resolved = resolve_basename(text_in)
                            if not resolved:
                                await event.reply(f"❌ Could not resolve `{text_in}`. Please check the name or try a direct 0x address.")
                                return
                            target_addr = resolved
                    except Exception as e:
                        await event.reply(f"Error resolving basename: {e}")
                        return

                    # 2) Normalize/check address (best-effort)
                    try:
                        from web3 import Web3
                        if isinstance(target_addr, str) and target_addr.startswith("0x") and len(target_addr) == 42:
                            try:
                                target_addr = Web3.toChecksumAddress(target_addr)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # 3) Create a challenge for the user to sign (reuse global helper if exists)
                    try:
                        # try to reuse existing create_link_challenge(user_id) helper (thread-safe)
                        try:
                            ch = await run_blocking(create_link_challenge, user_id)
                        except TypeError:
                            # some create_link_challenge implementations may not accept user_id; fallback
                            ch = await run_blocking(create_link_challenge)
                    except Exception:
                        # As a fallback, generate a short challenge locally
                        import secrets, time
                        ch = f"Verify ownership for job wallet at {int(time.time())}-{secrets.token_hex(4)}"

                    # 4) Store challenge and target address in the pending job with timestamp
                    import time
                    session.pending_job['wallet_to_verify'] = target_addr
                    session.pending_job['wallet_challenge'] = ch
                    session.pending_job['wallet_challenge_ts'] = int(time.time())

                    # 5) Move to signature-waiting state
                    session.set_state("awaiting_job_wallet_signature")

                    help_text = (
                        "Please sign the following message with the wallet you are linking (personal_sign / eth_sign).\n\n"
                        "Then paste the signature text here. You may optionally prepend the address: `0xYourAddr <signature>`\n\n"
                        f"Message to sign (exactly):\n```\n{ch}\n```\n\n"
                        "The challenge expires in 10 minutes."
                    )
                    await event.reply(help_text, buttons=Button.clear())
                    return
                    
                # -----------------------------------------------------------
                # STATE: awaiting_job_wallet_signature
                # User must paste signature (optionally prefixed by address)
                # -----------------------------------------------------------
                if state == "awaiting_job_wallet_signature":
                    user_id = event.sender_id
                    raw = message.strip()
                    if not raw:
                        await event.reply("Please paste the signature (optionally preceded by the address). Example:\n`0xabc... 0xSIG...`")
                        return

                    # Load saved challenge and target address
                    pj = session.pending_job
                    challenge = pj.get('wallet_challenge')
                    target_addr = pj.get('wallet_to_verify')
                    ts = pj.get('wallet_challenge_ts') or 0

                    import time
                    if not challenge or not target_addr:
                        await event.reply("No pending wallet verification found. Please restart by sending the wallet again.")
                        session.set_state("job_source_base_wallet")
                        return

                    # Check expiry (10 minutes)
                    if int(time.time()) - int(ts) > 600:
                        await event.reply("The challenge expired (older than 10 minutes). Please resend the wallet address to generate a new challenge.")
                        # clear stale challenge
                        pj.pop('wallet_challenge', None)
                        pj.pop('wallet_challenge_ts', None)
                        pj.pop('wallet_to_verify', None)
                        session.set_state("job_source_base_wallet")
                        return

                    # Parse input: allow "0xADDR <signature>" or just "<signature>"
                    sig = None
                    maybe_addr = None
                    parts = raw.split()
                    if len(parts) == 1:
                        sig = parts[0].strip()
                    else:
                        # if first token looks like an address, treat it as address
                        if parts[0].startswith("0x") and len(parts[0]) == 42:
                            maybe_addr = parts[0].strip()
                            sig = " ".join(parts[1:]).strip()
                        else:
                            # assume signature only
                            sig = raw

                    if not sig:
                        await event.reply("No signature found in your message. Please paste the signature text.")
                        return

                    # 1) Prefer existing helper verify_wallet_signature(user_id, address, signature)
                    verified_addr = None
                    try:
                        if 'verify_wallet_signature' in globals() and callable(globals().get('verify_wallet_signature')):
                            # helper may accept (user_id, address, signature) or (address, signature)
                            try:
                                ok = await run_blocking(verify_wallet_signature, user_id, target_addr, sig)
                            except TypeError:
                                ok = await run_blocking(verify_wallet_signature, target_addr, sig)
                            if ok:
                                verified_addr = target_addr
                        else:
                            # Fallback to eth_account recovery
                            try:
                                from eth_account.messages import encode_defunct
                                from eth_account import Account
                                msg = encode_defunct(text=challenge)
                                # signature may be hex prefixed or not
                                recovered = Account.recover_message(msg, signature=sig)
                                # normalize checksum
                                from web3 import Web3
                                recovered = Web3.toChecksumAddress(recovered)
                                # If user provided maybe_addr, check match; otherwise accept recovered
                                if maybe_addr:
                                    try:
                                        maybe_addr = Web3.toChecksumAddress(maybe_addr)
                                    except Exception:
                                        pass
                                    if recovered.lower() != maybe_addr.lower():
                                        await event.reply("Signature recovered address does not match the address you provided.")
                                        return
                                # confirm that recovered matches the target_addr we asked to verify
                                try:
                                    target_ck = Web3.toChecksumAddress(target_addr)
                                except Exception:
                                    target_ck = target_addr
                                if recovered.lower() != str(target_ck).lower():
                                    await event.reply("Signature recovered address does not match the address you requested to verify.")
                                    return
                                verified_addr = recovered
                            except Exception as e:
                                await event.reply(f"Signature verification failed: {e}")
                                return
                    except Exception as e:
                        await event.reply(f"Verification helper error: {e}")
                        return

                    if not verified_addr:
                        await event.reply("Signature did not verify. Please ensure you signed the *exact* challenge message and paste the complete signature. You may also paste `0xAddr <signature>`.")
                        return

                    # 2) Persist tracked wallet and attach to pending job
                    try:
                        res = await run_blocking(add_tracked_base_wallet, user_id, verified_addr)
                    except Exception as e:
                        await event.reply(f"Failed to persist tracked wallet: {e}")
                        return

                    if not res or not res.get("ok"):
                        # if already tracked, continue
                        if res and res.get("error") and "already tracked" in str(res.get("error")).lower():
                            pass
                        else:
                            await event.reply(f"Could not track wallet: {res.get('error') if isinstance(res, dict) else res}")
                            return

                    pj.setdefault("tracked_wallets", [])
                    if verified_addr not in pj["tracked_wallets"]:
                        pj["tracked_wallets"].append(verified_addr)
                    pj["trigger_on_tracked_wallet"] = True

                    # Clean up challenge metadata
                    pj.pop('wallet_challenge', None)
                    pj.pop('wallet_challenge_ts', None)
                    pj.pop('wallet_to_verify', None)

                    # Persist pending_job if supported
                    try:
                        await self.save_user_data(user_id, pj)
                    except Exception:
                        pass

                # Next: ask for onchain amount (user can skip)
                session.set_state("job_onchain_amount")
                await self.show_prompt_for_state(event, "job_onchain_amount")
                await event.reply(f"✅ Ownership verified and wallet `{verified_addr}` attached to this job.\n\nPlease enter the native amount to send per trigger (or type `skip` to leave it unset).", buttons=Button.clear())
                return
                
                # -----------------------------------------------------------
                # STATE: job_onchain_amount
                # Expect user to send a numeric amount (native) or "skip"
                # -----------------------------------------------------------
                if state == "job_onchain_amount":
                    user_id = event.sender_id
                    txt = message.strip().lower()

                    # allow skip
                    if txt == "" or txt == "skip":
                        # no onchain amount set — proceed to destinations
                        session.set_state("job_destinations")
                        await self.show_prompt_for_state(event, "job_destinations")
                        await event.reply("No onchain amount set. Proceeding to destination selection.", buttons=Button.clear())
                        return

                    # parse numeric amount
                    try:
                        amt = float(message.strip())
                        if amt <= 0:
                            raise ValueError("amount must be positive")
                    except Exception:
                        await event.reply("Please enter a valid positive number (example: `0.0001`) or type `skip` to skip setting an onchain amount.")
                        return

                    # optional global guard: ONCHAIN_MAX_PER_JOB
                    try:
                        import os
                        max_per_job = float(os.getenv("ONCHAIN_MAX_PER_JOB")) if os.getenv("ONCHAIN_MAX_PER_JOB") else None
                    except Exception:
                        max_per_job = None

                    if max_per_job is not None and amt > max_per_job:
                        await event.reply(f"Amount exceeds ONCHAIN_MAX_PER_JOB ({max_per_job}). Enter a smaller amount or type `skip`.")
                        return

                    # store amount and enable onchain transfer flag
                    session.pending_job['onchain_amount'] = float(amt)
                    session.pending_job['onchain_transfer'] = True

                    # advance to destination selection
                    session.set_state("job_destinations")
                    await self.show_prompt_for_state(event, "job_destinations")
                    await event.reply(f"✅ Onchain amount set to {amt} (native). Now choose where to forward the event (destinations).", buttons=Button.clear())
                    return



            # --- State 3 (Conditional): Awaiting Keywords ---
            elif state == "job_keywords":
                keywords = [k.strip() for k in message.split(',') if k.strip() and k.lower() != 'none']

                permission, msg = await self.check_permission(user_id, 'add_keywords', context={'keyword_count': len(keywords)})
                if not permission:
                    await event.reply(f"🔒 {msg}")
                    await self.show_prompt_for_state(event, "job_keywords", extra_info="⚠️ Please provide a new list of keywords that meets your plan's limits.")
                    return

                session.pending_job['keywords'] = keywords

                if len(keywords) > 1:
                    session.set_state("job_awaiting_match_logic")
                    await self.ask_for_match_logic(event)
                else:
                    session.pending_job['match_logic'] = 'OR'
                    session.set_state("job_timer")
                    await self.show_prompt_for_state(event, "job_timer", extra_info=f"Keywords set: {', '.join(keywords) if keywords else 'All Messages'}")

            # --- State 4 (Conditional): Awaiting Cashtags ---
            elif state == "job_cashtags":
                cashtags = [c.strip() for c in message.split(',') if c.strip() and c.lower() != 'none']
                session.pending_job['cashtags'] = cashtags

                session.set_state("job_timer")
                await self.show_prompt_for_state(event, "job_timer", extra_info=f"Cashtags set: {', '.join(cashtags) if cashtags else 'All Cashtags'}")

            # --- Final State: Awaiting Timer ---
            elif state == "job_timer":
                timer_str = message.strip()

                permission, msg = await self.check_permission(user_id, 'set_timer', context={'timer_str': timer_str})
                if not permission:
                    await event.reply(f"🔒 {msg}")
                    await self.show_prompt_for_state(event, "job_timer", extra_info="⚠️ Please provide a valid timer that meets your plan's limits.")
                    return

                session.pending_job['timer'] = '' if timer_str.lower() == 'none' else timer_str

                await event.reply("🧪 **Testing destinations...**")
                test_results = await forwarder.test_job_destinations(session.pending_job)
                failed_tests = [r for r in test_results if not r['success']]

                 # Instead of finalizing, we now move to the naming step.
                session.set_state("awaiting_job_name")
                await self.show_prompt_for_state(event, "awaiting_job_name", extra_info="✅ Timer set.")

                # Previous Test feature, memed out for now
                #if failed_tests:
                    #session.set_state("job_confirmation")
                    #session.temp_data['test_results'] = test_results

                    #working_chats = [r['destination'] for r in test_results if r['success']]
                    #failed_details = "\n".join([f"• **{r['destination']}**: {r['error']}" for r in failed_tests])
                    #warning_text = (f"⚠️ **Destination Test Failed**\n\n✅ **Working:** {len(working_chats)}\n❌ **Failed:** {len(failed_tests)}\n{failed_details}\n\nType `proceed` to create the job with only the working destinations, `fix` to re-enter destinations, or `cancel`.")
                    #await event.reply(warning_text)
                #else:
                    #await event.reply("✅ All destinations are working!")
                    #await self.finalize_job_creation(event, session, forwarder)

        except ValueError as e:
            await event.reply(f"❌ Error: {e}\n\nPlease check the spelling and try again.")
            await self.show_prompt_for_state(event, state)

    # ADDED: New handler for match logic step
    async def ask_for_match_logic(self, event):
        """Asks the user to choose between AND/OR logic with detailed explanations."""
        match_logic_explanation = """⚙️ **Choose Your Match Logic**

You've provided multiple items. Please choose how the bot should match them to forward a message.

*   **OR - Must contain any (Default)**
    This will forward a message if it contains **at least one** of your items.
    *Use this for related concepts or synonyms.*
    *Example:* Keywords `apple`, `orange`. A message with "I like apple" is forwarded. A message with "I like orange" is also forwarded.

*   **AND - Must contain all**
    This will only forward a message if it contains **all** of your items, anywhere in the message.
    *Use this for required combinations.*
    *Example:* Keywords `new listing`, `Binance`. A message must contain BOTH "new listing" and "Binance" to be forwarded.

Please choose one:"""

        buttons = [
            [Button.text("OR - Can contain any", resize=True)],
            [Button.text("AND - Must contain all", resize=True)]
        ]
        await event.reply(match_logic_explanation, buttons=buttons)

    async def handle_job_awaiting_match_logic(self, event, message):
        session = await self.get_user_session(event.sender_id)
        user_id = event.sender_id # Define user_id

        logic = ''
        if "AND" in message:
            logic = 'AND'
            # --- START OF ENFORCEMENT CODE ---
            # Check permission specifically for using 'AND' logic
            permission, msg = await self.check_permission(user_id, 'set_and_logic')
            if not permission:
                await event.reply(f"🔒 {msg}\n\nYour job will be created with the default 'OR' logic. To use 'AND', please /subscribe.", parse_mode='markdown')
                logic = 'OR' # Revert to default
            # --- END OF ENFORCEMENT CODE ---
        elif "OR" in message:
            logic = 'OR'

        if not logic:
            await event.reply("❌ Invalid selection. Please choose one of the buttons.")
            return

        session.pending_job['match_logic'] = logic
        session.set_state("job_timer")
        await event.reply(f"✅ Match logic set to **{logic}**.\n\n⏱️ Now, set a cooldown timer (e.g., '5 minutes', '1 hour').\n\nLeave empty or send 'none' for no cooldown:")

    # ADDED: New interactive handlers for Custom Patterns
    # --- START: Restored & Adapted Custom Pattern Methods ---

    async def handle_custom_pattern_menu(self, event, message):
        """Shows the main menu for creating a custom pattern job."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        # --- PERMISSION CHECK FOR PATTERN COUNT ---
        # Check if the user is already at their limit before allowing them to add another.
        current_pattern_count = len(session.pending_job.get('patterns', []))
        permission, msg = await self.check_permission(user_id, 'add_patterns', context={'pattern_count': current_pattern_count})
        if not permission:
            await event.reply(f"{msg}\n\nYou can now proceed to set the job timer.")
            # Transition them directly to the next step as they cannot add more patterns.
            session.set_state("job_timer")
            await event.reply("⏱️ Set a cooldown timer (e.g., '5 minutes', '1 hour').\n\nLeave empty or send 'none' for no cooldown:")
            return
        # --- END OF PERMISSION CHECK ---

        # This check is needed for when a user clicks a button
        if message not in ["init", "➕ Add Another Pattern"]:
            if message == "🤖 Help: Generate Pattern with AI":
                await self.show_regex_tutorial(event)
                # Re-show the menu so the user isn't stuck
                message = "init"
            else: # Assume it's a button click from the main menu
                pass

        if message == "init": # Initial call to show menu
            menu_text = "⚙️ **Set Custom Pattern**\n\nYou can enter a new pattern, choose a saved one, or get help."
            buttons = [
                [Button.text("➕ Enter New Pattern", resize=True)],
                [Button.text("📚 Choose Saved Pattern", resize=True)],
                [Button.text("🤖 Help: Generate Pattern with AI", resize=True)],
                [Button.text("🔙 Back to Main Menu", resize=True)],
            ]
            await event.reply(menu_text, buttons=buttons)
            return

        if message == "➕ Enter New Pattern":
            session.set_state("awaiting_new_pattern")
            await event.reply("✍️ Please send me the regex pattern to use. To get help creating one, send `/help regex`.")
        elif message == "📚 Choose Saved Pattern":
            if not forwarder.saved_patterns:
                await event.reply("You have no saved patterns yet. Please enter a new one first.")
                return

            buttons = []
            for i, p in enumerate(forwarder.saved_patterns):
                buttons.append([Button.inline(p['name'], data=f"select_pattern_{i}")])

            await event.reply("📚 **Your Saved Patterns**\n\nPlease choose a pattern from the list below:", buttons=buttons)
            # State remains job_custom_pattern_menu, waiting for a callback_query
        else:
            await event.reply("Please use the menu buttons.")

    async def handle_new_pattern_input(self, event, message):
        """Handles the text input for a new regex pattern."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # Allow user to ask for help at this stage
        if message.lower() == '/help regex':
            await self.show_regex_tutorial(event)
            await event.reply("Now, please send the regex pattern to use.")
            return

        pattern = message.strip()
        try:
            re.compile(pattern)
        except re.error as e:
            await event.reply(f"❌ **Invalid Pattern:** That pattern is not valid.\n`{e}`\nPlease try again.")
            return

        # Temporarily store the pattern while we ask if they want to save it
        session.temp_data['current_pattern'] = pattern

        session.set_state("awaiting_pattern_save_prompt")
        buttons = [
            [Button.text("Yes, save it", resize=True), Button.text("No, just use it once", resize=True)]
        ]
        await event.reply(f"✅ Pattern set to: `{pattern}`\n\n💾 Do you want to save this pattern for future use?", buttons=buttons)

    async def handle_pattern_save_prompt(self, event, message):
        """Handles the Yes/No for saving a new pattern."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        pattern_to_add = session.temp_data.get('current_pattern')

        if not pattern_to_add:
            await event.reply("An error occurred. Let's try again.")
            session.set_state("job_custom_pattern_menu")
            await self.handle_custom_pattern_menu(event, "init")
            return

        if message == "Yes, save it":
            session.set_state("awaiting_pattern_save_name")
            await event.reply("📝 Please give this pattern a short, memorable name (e.g., 'Email Finder').")
        elif message == "No, just use it once":
            # Add the pattern to the job but don't save it permanently
            if 'patterns' not in session.pending_job:
                session.pending_job['patterns'] = []
            session.pending_job['patterns'].append(pattern_to_add)
            session.temp_data.pop('current_pattern', None)

            # Transition to the multi-pattern check
            session.set_state("job_pattern_add_or_done")
            await self.handle_pattern_add_or_done(event, "") # Call the next step
        else:
            await event.reply("Please choose 'Yes' or 'No'.")

    async def handle_pattern_save_name(self, event, message):
        """Handles getting the name for a new pattern and saving it."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]
        pattern_to_add = session.temp_data.get('current_pattern')

        pattern_name = message.strip()
        new_pattern = {
            'name': pattern_name,
            'pattern': pattern_to_add
        }

        # Save to the user's permanent list
        forwarder.saved_patterns.append(new_pattern)
        await self.save_user_data(user_id, forwarder) # Save immediately

        # Also add it to the current job
        if 'patterns' not in session.pending_job:
            session.pending_job['patterns'] = []
        session.pending_job['patterns'].append(pattern_to_add)
        session.temp_data.pop('current_pattern', None)

        await event.reply(f"✅ Pattern saved as **'{pattern_name}'**.")

        # Transition to the multi-pattern check
        session.set_state("job_pattern_add_or_done")
        await self.handle_pattern_add_or_done(event, "") # Call the next step

    async def handle_pattern_add_or_done(self, event, message):
        """Asks the user if they want to add another pattern or finish."""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        if message == "➕ Add Another Pattern":
            session.set_state("job_custom_pattern_menu")
            await self.handle_custom_pattern_menu(event, "init")
            return

        # This part runs either after the first pattern is added,
        # or when the user clicks "Done Adding Patterns"
        if message == "✅ Done Adding Patterns":
            patterns = session.pending_job.get('patterns', [])

            # --- START OF ENFORCEMENT ---
            permission, msg = await self.check_permission(user_id, 'add_patterns', context={'pattern_count': len(patterns)})
            if not permission:
                await event.reply(f"🔒 {msg}\n\nPlease remove some patterns by modifying the job later, or /subscribe for more.", parse_mode='markdown')
                # We will proceed but the user is warned. Alternatively, you could block creation.
                # For now, we let them proceed to the next step.
            # --- END OF ENFORCEMENT ---

            if len(patterns) > 1:
                session.set_state("job_awaiting_match_logic")
                await self.ask_for_match_logic(event)
            else:
                session.pending_job['match_logic'] = 'OR'
                session.set_state("job_timer")
                await event.reply("✅ Patterns set!\n\n⏱️ Now, set a cooldown timer (e.g., '5 minutes', '1 hour').\n\nLeave empty or send 'none' for no cooldown:")
            return

        # This is the initial prompt after the first pattern is added
        current_patterns = session.pending_job.get('patterns', [])
        buttons = [
            [Button.text("➕ Add Another Pattern", resize=True)],
            [Button.text("✅ Done Adding Patterns", resize=True)]
        ]
        await event.reply(f"✅ Pattern added! You now have {len(current_patterns)} pattern(s) in this job.\n\nDo you want to add another?", buttons=buttons)
    # --- END: Restored & Adapted Custom Pattern Methods ---


    async def show_regex_tutorial(self, event):
        tutorial_text = """🤖 **How to Generate Patterns with AI**

You can use an AI Chat service (like Gemini, ChatGPT, Claude) to create the exact pattern you need. It's easy!

---
**1. Copy the Master Prompt Template**
---
This template is designed to get the best results from the AI.

```text
Hello! I need a regex pattern to use in a Telegram bot. The pattern should find messages that contain a specific type of text.

Here is a description of the text I want to find:
[**Describe the text here. Be as detailed as possible.**]

Here are some real examples of the text:
[**Paste 1-3 real examples here.**]

Please give me only the regex pattern itself, with no extra explanation.```

---
**2. How to Use the Template**
---
1.  **Copy the template** above.
2.  **Describe your target text** in the first bracket. The more detail, the better!
3.  **Provide real examples** in the second bracket. This is the most important step!
4.  **Send the completed prompt** to an AI chat service.
5.  The AI will give you a pattern (e.g., `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`). **Copy the entire pattern.**
6.  **Paste the pattern back here** into our chat.

---
**Pro-Tip: Test Your Pattern!**
---
Before you give a pattern to the bot, it's a great idea to test it first.

1.  Go to a website like **regex101.com**.
2.  Paste your pattern into the "Regular Expression" box.
3.  In the "Test String" box, paste some sample messages that should match, and some that shouldn't.
4.  The website will instantly highlight the matches, showing you if your pattern works exactly as you expect!

---
**3. Example Prompts (Cookbook)**
---
Here are some ready-to-use descriptions you can copy for the `[Describe the text here]` part.

▶️ **To find any Solana Address:**
`A Solana address. It's a string of 32 to 44 characters containing letters (A-Z, a-z) and numbers (1-9), but it specifically avoids the characters '0', 'O', and 'I'.`

▶️ **To find any Email Address:**
`Any standard email address, like name@example.com.`

▶️ **To find a specific Cashtag format ($ followed by 4 capital letters):**
`Cashtags starting with a '$' symbol, followed by exactly four uppercase letters. For example: $ABCD or $GENX.`
"""
        await event.reply(tutorial_text, link_preview=False)

    # --- FIX #2: APPLIED ---
    # A guard clause is added to handle cases with no jobs.
    async def show_job_deletion(self, event, page_index=0):
        user_id = event.sender_id
        forwarder = self.user_forwarders.get(user_id) # Use .get for safety

        # --- FIX: ADD THIS GUARD CLAUSE AT THE TOP ---
        if not forwarder or not forwarder.jobs:
            await event.reply("✅ You have no jobs to delete.")
            return
        # --- END OF FIX ---

        session = await self.get_user_session(user_id)
        if not session: return # Safety check

        session.set_state("deleting_job")
        session.pagination_data = forwarder.jobs

        async def lister(): return forwarder.jobs
        async def formatter(jobs_on_page, start_index, title, current_page, total_pages):
            text = f"**{title} (Page {current_page}/{total_pages})**\n\n"
            for i, job in enumerate(jobs_on_page, start=start_index + 1):
                job_name = job.get('job_name')
                job_type_str = job.get('type', 'N/A').replace('_', ' ').capitalize()
                match_logic = job.get('match_logic', 'OR')

                items_key = 'keywords' if job['type'] == 'keywords' else 'patterns'
                if job['type'] in ['keywords', 'custom_pattern'] and len(job.get(items_key, [])) > 1:
                    job_type_str += f" ({match_logic})"

                if job_name:
                    text += f"**{i}. {job_name}** (`{job_type_str}`)\n"
                else: # Fallback for old jobs
                    text += f"**{i}.** **{job_type_str}**\n"

            last_num = start_index + len(jobs_on_page)
            text += f"\nReply with a number **({start_index + 1}-{last_num})** to delete, or type 'cancel'."
            return text

        await self.show_paginated_list(
            event,
            page_index=page_index,
            item_lister=lister,
            item_formatter=formatter,
            callback_prefix="page_user_delete",
            title="🗑️ Delete Job"
        )

    async def handle_job_deletion(self, event, message):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        if message.lower() == 'cancel':
            session.set_state("idle")
            await self.show_job_management(event)
            return

        try:
            job_num = int(message)
            all_jobs = session.pagination_data
            if 1 <= job_num <= len(all_jobs):
                # The index in the actual forwarder.jobs list is job_num - 1
                deleted_job = forwarder.jobs.pop(job_num - 1)
                await self.save_user_data(user_id, forwarder)
                job_type = deleted_job.get('type', 'Unknown').replace('_', ' ').capitalize()
                await event.reply(f"✅ Job #{job_num} deleted successfully!\n\n**{job_type}** job has been removed.", buttons=Button.clear())
                session.set_state("idle")
                session.pagination_data = []
                await self.send_main_menu(event)
            else:
                await event.reply(f"❌ Invalid job number. Please send a number between 1 and {len(all_jobs)}:")
        except ValueError:
            await event.reply("❌ Please send a valid job number:")

    async def list_user_chats(self, event, page_index=0):
        user_id = event.sender_id
        if user_id not in self.user_forwarders:
            message = "❌ Please connect your account first."
            if hasattr(event, 'data'):  # It's a callback
                await event.edit(message)
            else:  # It's a regular message
                await event.reply(message)
            return

        async def lister():
            forwarder = self.user_forwarders[user_id]
            await forwarder._build_chat_cache()
            return sorted(forwarder.get_chats_list(), key=lambda x: x[1])

        async def formatter(chats_on_page, start_index, title, current_page, total_pages):
            text = f"**{title} (Page {current_page}/{total_pages})**\n\n"
            for i, (chat_id, chat_title) in enumerate(chats_on_page, start=start_index + 1):
                text += f"**{i}. {chat_title}**\n  `{chat_id}`\n"
            return text

        await self.show_paginated_list(
            event,
            page_index=page_index,
            item_lister=lister,
            item_formatter=formatter,
            callback_prefix="page_user_chats",
            title="📝 Your Accessible Chats",
            items_per_page=10
        )

    # ADDED
    async def logout_account(self, event):
        user_id = event.sender_id

        # ADDED: Demo mode logic
        if self.demo_mode:
            # In demo mode, "logging out" just removes the user from the in-memory mapping.
            # It doesn't touch the shared demo session file.
            if user_id in self.user_forwarders:
                del self.user_forwarders[user_id]
            await event.reply("🚪 **Demo Logout Successful!**\n\nYou have been logged out of the demo session.")
            await self.send_main_menu(event)
            return

        if user_id in self.user_forwarders:
            forwarder = self.user_forwarders[user_id]

            # Save current jobs before logout
            await self.save_user_data(user_id, forwarder)

            # Stop the background listener task and disconnect the client
            await self.stop_forwarder_session(user_id)

            # Remove ONLY session file (keep encrypted data)
            try:
                # Path needs to be constructed carefully, session_fix logic is applied here
                session_file_path = forwarder.session_file + ".session"
                if os.path.exists(session_file_path):
                    os.remove(session_file_path)
                    logging.info(f"Removed session file: {session_file_path}")
            except FileNotFoundError:
                pass # Already gone

            # Remove from memory
            del self.user_forwarders[user_id]

            await event.reply("🚪 **Logged out successfully!**\n\n✅ Your jobs and settings have been saved. They will be restored when you connect this account again.")
        else:
            await event.reply("❌ No account is currently connected.")

        await self.send_main_menu(event)

    # ADDED from fixed_disconnect_logic.txt
    async def delete_account(self, event):
        user_id = event.sender_id
        session = await self.get_user_session(user_id)

        # ADDED: Demo mode logic
        if self.demo_mode:
            # "Delete" is the same as "Logout" in demo mode.
            if user_id in self.user_forwarders:
                del self.user_forwarders[user_id]
            # Clear any confirmation state
            session.temp_data = {}
            session.set_state("idle")
            await event.reply("🗑️ **Demo Account Removed**\n\nYou have been logged out of the demo session. No data was stored or deleted.")
            await self.send_main_menu(event)
            return


        if user_id not in self.user_forwarders:
            await event.reply("❌ No account is currently connected.")
            await self.send_main_menu(event)
            return

        # Confirmation step
        if not session.temp_data.get('delete_confirmed'):
            forwarder = self.user_forwarders[user_id]
            job_count = len(forwarder.jobs)

            confirm_text = "⚠️ **PERMANENT ACCOUNT DELETION**\n\n"
            confirm_text += "This will permanently delete:\n"
            confirm_text += "• Your account connection from this bot\n"
            confirm_text += f"• All **{job_count}** of your active jobs\n"
            confirm_text += "• All saved data and settings\n\n"
            confirm_text += "**This action cannot be undone!**\n\n"
            confirm_text += "To confirm, please type `DELETE`"

            session.temp_data['delete_confirmed'] = False # Ensure it's set
            session.set_state("confirming_delete")
            await event.reply(confirm_text)
            return

        # Actually delete everything
        forwarder = self.user_forwarders[user_id]
        # Stop the background listener task and disconnect the client
        await self.stop_forwarder_session(user_id)

        # Remove session file (using fixed path logic)
        try:
            session_file_path = forwarder.session_file + ".session"
            if os.path.exists(session_file_path):
                os.remove(session_file_path)
        except FileNotFoundError:
            pass

        # Remove from memory
        del self.user_forwarders[user_id]

        # Remove ALL user data (encrypted data file + salt)
        await self.remove_user_data(user_id)

        session.temp_data = {}
        session.set_state("idle")
        await event.reply("🗑️ **Account deleted permanently!**\n\n✅ All your data, jobs, and settings have been removed from the bot.")
        await self.send_main_menu(event)

    async def show_help(self, event):
        help_text = """ℹ️ **🤖 Bot Help — Quick Reference**

**What this bot does:**
This bot helps you automatically forward messages between Telegram chats based on specific criteria.

**Job Types:**
• **Keywords** - Forward messages containing specific words
• **Solana/Ethereum** - Forward crypto contract addresses
• **Cashtags** - Forward messages with cashtags ($BTC, $ETH, etc.)
• **Custom Pattern** - Forward messages based on advanced text patterns (Regex).

**Getting Started:**
1. Use "🔗 Connect Account" to link your Telegram account.
2. You'll need API credentials from https://my.telegram.org.
3. Use "📋 Manage Jobs" to create and manage forwarding rules.
4. Jobs will run automatically in the background.

**Disconnecting:**
• **🚪 Logout (Keep Jobs)**: Disconnects your account but saves your jobs. Reconnect later to resume.
• **🗑️ Delete Account**: Permanently removes your account and all associated jobs and data from this bot.

**Important Notes:**
• Your account credentials are encrypted and stored securely.
• The bot tests destinations before creating jobs to ensure they work.
• You'll be notified if messages start failing to forward.
• **When logging in, you must obfuscate the verification code (e.g., add letters or spaces) for it to work correctly.**

**Commands:**
/start - Show the main menu
/help regex - Get a tutorial on generating patterns with AI
/admin <master_secret> - Admin access (for administrators)

Basic
  /start                 Open the main menu
  /help                  Show this help

Wallet linking (non-custodial)
  /link_wallet           Create a challenge message to sign with your wallet
  /confirm_link <addr?> <signature>
                         Confirm wallet linking.
                         - You can provide BOTH address and signature:
                           /confirm_link 0xAbC... 0x2c64...
                         - OR provide only the signature (bot will infer address):
                           /confirm_link 0x2c64...
  /me  or  /my_wallet    Show your currently linked wallet address

  How it works (step-by-step)
    1) Send: `/link_wallet`
       → Bot replies with a human-readable challenge message (contains nonce, expires).
    2) In your wallet (MetaMask/Coinbase Wallet) use **Sign Message / personal_sign** and paste the exact challenge string.
    3) Copy the signature and run:
       - Explicit: `/confirm_link 0xYourAddress 0xSignatureHex`
       - Or signature-only: `/confirm_link 0xSignatureHex`
    4) On success you get: `✅ Wallet linked: 0xYourAddress`

  Notes:
    • Challenge TTL = 10 minutes (default). If expired, `/link_wallet` again.
    • We never ask for private keys — only a signature.
    • Linked addresses are stored on the server in file: `data/user_profiles.json`.

Payments & subscriptions
  /pay <amount_usd>      Start a payment flow (Coinbase Commerce primary; direct/manual fallback)
  /tx <tx_hash> <ref>    Submit a tx hash for manual verification (attach your reference)
  /pay_status <reference>  Check status for a payment reference
  /subscribe             Start subscription flow
  /subscription          Show your subscription status

Basename & naming (Basenames on Base)
  /resolve <name>        Resolve a basename to an address (e.g. /resolve alice.base)
  /register <name>       Register a basename (server-wallet automatic or manual instructions)
                         - Server registration requires the server to have a funded key (env).
                         - Manual mode returns instructions + amount to pay.

Admin (restricted)
  /admin_list_links      List all linked wallet addresses (admin only)
  /list_proofs           List recent onchain proofs (admin only)
  /list_proofs (admin)   Show payments/proofs for review

How admin check works
  The bot checks admin permission by:
    • session.is_admin (if session manager exists), OR
    • environment variables ADMIN_USER_ID (single) or ADMIN_USER_IDS (comma list).
  If you need to add an admin, set one of those env vars to your Telegram user id.

Security & operational notes
  • Never paste private keys into chat. Only signatures are safe to send.
  • Server-side registration ( /register ) uses SERVER_WALLET_PRIVATE_KEY or
    SERVER_WALLET_PRIVATE_KEY_ENC (encrypted). If encrypted, a `decrypt()` helper is required.
  • Files: `data/user_profiles.json` stores linked addresses & challenges; `payments.json` stores payments.
  • If you want a purchase to go to your own address, link your wallet first or use the manual flow.

Examples
  /link_wallet
  (sign returned message in MetaMask)
  /confirm_link 0x2c64...                ← signature-only; bot infers address
  /confirm_link 0xAbC... 0x2c64...       ← explicit address + signature
  /me
  /pay 12.50
  /tx 0x8d69... REF-A12

If something fails
  • If a challenge expired: run `/link_wallet` again.
  • If signature verification fails: ensure you signed the **exact** challenge string (no extra whitespace).
  • Contact the bot admin (owner) for help if needed.
"""

        await event.reply(help_text)

# CODE FIX 1: Correct save_user_data function
    async def save_user_data(self, user_id: int, forwarder: TelegramForwarder, direct_data=None):
        """
        Save user data to an encrypted file.
        Handles both updating an existing user's forwarder data and saving a brand-new user record.
        Returns True if successful, False otherwise.
        """
        if self.demo_mode:
            return True  # In demo mode, pretend it worked

        try:
            if direct_data:
                # Use this for creating a new user or a direct data write.
                user_data = direct_data
            else:
                # This is the standard flow for updating an existing user's jobs.
                # First, load their existing data to preserve subscription fields.
                existing_data = await self.load_user_data(user_id) or {}

                # Merge the old and new data.
                user_data = {
                    # Preserve all subscription and referral data
                    'subscription_status': existing_data.get('subscription_status', 'free'),
                    'subscription_expiry_date': existing_data.get('subscription_expiry_date'),
                    'active_promo_code': existing_data.get('active_promo_code'),  # CRITICAL: Preserve this!
                    'plan_type': existing_data.get('plan_type'),  # CRITICAL: Preserve this too!
                    'referrer_id': existing_data.get('referrer_id'),
                    'upline_chain': existing_data.get('upline_chain', []),
                    'unpaid_commissions': existing_data.get('unpaid_commissions', 0.0),
                    'has_had_first_subscription': existing_data.get('has_had_first_subscription', False),
                    'payment_history': existing_data.get('payment_history', []),

                    # Update the forwarder-specific data
                    'api_id': forwarder.api_id,
                    'api_hash': forwarder.api_hash,
                    'phone_number': forwarder.phone_number,
                    'jobs': forwarder.jobs,
                    'saved_patterns': forwarder.saved_patterns,
                }

            json_string = json.dumps(user_data, indent=2)
            crypto = CryptoManager(user_id)
            encrypted_data = crypto.encrypt(json_string)

            filename = os.path.join(DATA_DIR, f"user_{user_id}.dat")
            with open(filename, "wb") as f:
                f.write(encrypted_data)
            logging.info(f"Saved data for user {user_id}")
            return True  # Success

        except Exception as e:
            logging.error(f"Failed to save user data for {user_id}: {e}", exc_info=True)
            return False  # Failed

    async def load_user_data(self, user_id: int):
        """
        Load user data from an encrypted file.
        If the file does not exist, create a default one and return it.
        """
        try:
            filename = os.path.join(DATA_DIR, f"user_{user_id}.dat")
            if not os.path.exists(filename):
                # --- THIS IS THE UPDATED LOGIC ---
                logging.info(f"No data file found for user {user_id}. Creating a new default file.")
                # Create a default structure for a brand-new user
                default_data = {
                    'api_id': None, 'api_hash': None, 'phone_number': None,
                    'jobs': [], 'saved_patterns': [],
                    'subscription_status': 'free',
                    'subscription_expiry_date': None,
                    'active_promo_code': None,  # CRITICAL: Add this field!
                    'plan_type': None,
                    'referrer_id': None,
                    'upline_chain': [],
                    'unpaid_commissions': 0.0,
                    'has_had_first_subscription': False,
                    'payment_history': [],
                    'hl_api_key': None,
                    'hl_api_secret_encrypted': None,
                    'hl_trade_profile': {}
                }
                # Encrypt and save this default file so it exists for future loads
                json_string = json.dumps(default_data, indent=2)
                crypto = CryptoManager(user_id)
                encrypted_data = crypto.encrypt(json_string)
                with open(filename, "wb") as f:
                    f.write(encrypted_data)

                return default_data
                # --- END OF UPDATED LOGIC ---

            with open(filename, "rb") as f:
                encrypted_data = f.read()

            crypto = CryptoManager(user_id)
            decrypted_json = crypto.decrypt(encrypted_data)
            return json.loads(decrypted_json)

        except Exception as e:
            logging.error(f"FATAL: Failed to load or create user data for {user_id}: {e}")
            return None # Return None only on a fatal error like decryption failure

    async def remove_user_data(self, user_id: int):
        """Remove all user data files (data and salt)."""
        try:
            filename = os.path.join(DATA_DIR, f"user_{user_id}.dat")
            if os.path.exists(filename):
                os.remove(filename)
                logging.info(f"Removed data file for user {user_id}")

            # Also remove salt file
            crypto = CryptoManager(user_id)
            if crypto.salt_path.exists():
                crypto.salt_path.unlink()
                logging.info(f"Removed salt file for user {user_id}")

        except Exception as e:
            logging.error(f"Failed to remove user data for {user_id}: {e}")

    # REPLACED
    async def restore_user_sessions(self):
        """Restore user sessions on bot startup with enhanced error tracking"""
        print("🔄 Restoring user sessions...")

        for filename in os.listdir(DATA_DIR):
            if filename.startswith("user_") and filename.endswith(".dat"):
                try:
                    user_id = int(filename.split("_")[1].split(".")[0])
                    user_data = await self.load_user_data(user_id)

                    if user_data:
                        # Create forwarder with bot reference
                        forwarder = TelegramForwarder(
                            user_data['api_id'],
                            user_data['api_hash'],
                            user_data['phone_number'],
                            user_id,
                            bot_instance=self  # Add bot reference
                        )
                        forwarder.saved_patterns = user_data.get('saved_patterns', [])


                        result = await forwarder.connect_and_authorize()

                        if result == "authorized":
                            forwarder.jobs = user_data.get('jobs', [])
                            await self.start_forwarder_session(user_id, forwarder)
                            self.user_forwarders[user_id] = forwarder
                            print(f"✅ Restored session for user {user_id} ({forwarder.phone_number})")
                        else:
                            self.user_forwarders[user_id] = forwarder # Keep it in memory so user can reconnect
                            forwarder.is_authorized = False
                            print(f"❌ Failed to restore session for user {user_id} ({forwarder.phone_number}): {result}. User will need to reconnect.")

                except Exception as e:
                    logging.error(f"Error restoring user session from {filename}: {e}")

        print(f"✅ Session restoration complete. {len(self.user_forwarders)} users found.")

    # ADD this new method to your TelegramBot class
    async def notify_user_of_disconnection(self, user_id, phone_number):
        """Notify user that their session was disconnected and polling has stopped."""
        try:
            message = (
                f"⚠️ **Account Disconnected: {phone_number}**\n\n"
                "Your Telegram account has been disconnected from the bot, and message forwarding has stopped.\n\n"
                "This can happen if the session expires or is terminated from another device.\n\n"
                "Please use the '🔄 Reconnect Account' button in the main menu to log in again and resume forwarding."
            )
            await self.bot.send_message(user_id, message)
            logging.info(f"Sent disconnection notification to user {user_id}.")
        except Exception as e:
            logging.error(f"Failed to send disconnection notification to user {user_id}: {e}")

    # --- Methods from user_notification_system.txt ---

    async def notify_user_of_forwarding_errors(self, user_id, job_results):
        """Notify user about forwarding failures"""
        if not job_results or job_results['failed'] == 0:
            return

        current_time = datetime.datetime.now()
        last_notification = self.last_error_summary.get(user_id, datetime.datetime.min)

        if (current_time - last_notification).total_seconds() < 3600:  # 1 hour cooldown
            return

        self.last_error_summary[user_id] = current_time

        failed_destinations = [r for r in job_results['results'] if not r['success']]

        error_summary = "⚠️ **Forwarding Issues Detected**\n\n"
        error_summary += f"📊 **Summary:** {job_results['successful']}/{job_results['total']} destinations were successful.\n\n"
        error_summary += "❌ **Failed Destinations:**\n"

        for failed in failed_destinations:
            error_summary += f"• **{failed['destination']}**: {failed['error']}\n"

        error_summary += "\n💡 **Possible Fixes:**\n"
        error_summary += "• Check if you still have permission to send messages in the failed chats.\n"
        error_summary += "• Verify you're still a member of the groups/channels.\n"
        error_summary += "• For channels, ensure you have admin rights.\n"
        error_summary += "• Use '📝 List Chats' to verify the bot can see the chats.\n\n"
        error_summary += "🔧 Use '📋 Manage Jobs' → '✏️ Modify Job' to update destinations."

        try:
            await self.bot.send_message(user_id, error_summary)
        except Exception as e:
            logging.error(f"Failed to send error notification to user {user_id}: {e}")

    async def handle_job_confirmation(self, event, message):
        """Handle user response to destination testing"""
        user_id = event.sender_id
        session = await self.get_user_session(user_id)
        forwarder = self.user_forwarders[user_id]

        response = message.lower().strip()

        if response == 'proceed':
            test_results = session.temp_data.get('test_results', [])
            working_destinations = [r for r in test_results if r['success']]

            working_ids = [r['dest_id'] for r in working_destinations]
            session.pending_job['destination_ids'] = working_ids

            await event.reply("✅ Proceeding with only the working destinations.")
            await self.finalize_job_creation(event, session, forwarder)

        elif response == 'fix':
            session.set_state("job_destinations")
            await event.reply("📤 Please send the corrected list of destination chat(s) where messages should be forwarded:")

        elif response == 'cancel':
            session.set_state("idle")
            session.pending_job = {}
            session.temp_data = {}
            await event.reply("❌ Job creation cancelled.")
            await self.send_main_menu(event)

        else:
            await event.reply("Please respond with 'proceed', 'fix', or 'cancel'.")

    async def finalize_job_creation(self, event, session, forwarder):
        """Finalize job creation with comprehensive summary"""
        forwarder.jobs.append(session.pending_job.copy())
        await self.save_user_data(event.sender_id, forwarder)

        job = session.pending_job
        job_type = job['type'].replace('_', ' ').capitalize()
        source_names = [forwarder.chat_cache.get(sid, f"ID {sid}") for sid in job['source_ids']]
        dest_names = [forwarder.chat_cache.get(did, f"ID {did}") for did in job['destination_ids']]

        # ADDED: Display AND/OR logic in summary
        match_logic = job.get('match_logic', 'OR')
        job_type_str = job_type
        items_key = 'keywords' if job['type'] == 'keywords' else 'patterns'
        if job['type'] in ['keywords', 'custom_pattern'] and len(job.get(items_key, [])) > 1:
            job_type_str += f" ({match_logic})"


        summary = "✅ **Job Created Successfully!**\n\n"
        summary += f"📝 Job Name: {job.get('job_name', 'Default')}\n"
        summary += f"📋 Type: {job_type_str}\n"
        summary += f"📥 **Sources:** {', '.join(source_names)}\n"
        summary += f"📤 **Destinations:** {', '.join(dest_names)}\n"

        if job['type'] == 'keywords' and job.get('keywords'):
            summary += f"🔤 **Keywords:** {', '.join(job['keywords'])}\n"
        elif job['type'] == 'cashtags' and job.get('cashtags'):
            summary += f"💰 **Cashtags:** {', '.join(job['cashtags'])}\n"
        elif job['type'] == 'custom_pattern' and job.get('patterns'):
            summary += f"🔍 **Patterns:** `{', '.join(job['patterns'])}`\n"


        if job.get('timer'):
            summary += f"⏱️ **Cooldown:** {job['timer']}\n"

        summary += "\n🚀 The job is now active and monitoring messages!"
        summary += "\n\n💡 **Tip:** You'll be notified if any destinations become inaccessible."

        session.set_state("idle")
        session.pending_job = {}
        session.temp_data = {}
        await event.reply(summary, buttons=Button.clear())
        await self.send_main_menu(event)
        
def register_bot_payment_handlers(bot_instance):
    """
    Idempotent runtime registration of handlers for a TelegramBot instance.
    Call this from async def main() AFTER `bot = TelegramBot()` is created.
    Note: bot_instance is the TelegramBot object; the Telethon client is bot_instance.bot.
    """
    # idempotent guard per bot instance
    if hasattr(register_bot_payment_handlers, "_registered") and register_bot_payment_handlers._registered.get(id(bot_instance)):
        return
    if not hasattr(register_bot_payment_handlers, "_registered"):
        register_bot_payment_handlers._registered = {}

    # helper to run blocking functions in threadpool
    def run_blocking(fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    client = getattr(bot_instance, "bot", None)
    if client is None:
        raise RuntimeError("register_bot_payment_handlers: passed object has no `.bot` Telethon client attribute")

    # ---- /resolve (optional arg) ----
    # ---- /resolve (optional arg) ----
    # ---- /resolve (optional arg) ----
    @client.on(events.NewMessage(pattern=re.compile(r'^/resolve(?:\s+(\S+))?$', re.IGNORECASE)))
    async def _on_resolve(event):
        try:
            name = event.pattern_match.group(1) if event.pattern_match else None
        except Exception:
            name = None

        if not name:
            await event.reply("Usage: `/resolve <name>` — e.g. `/resolve alice.base`")
            return

        await event.reply("Resolving...")

        # Direct call (module-level resolve_basename is now guaranteed to exist)
        try:
            addr = await run_blocking(resolve_basename, name)
        except Exception as e:
            await event.reply(f"Resolve error: {e}")
            return

        if addr:
            await event.reply(f"`{name}` → `{addr}`")
        else:
            await event.reply(f"Could not resolve `{name}` (no addr record).")


    # ---- /pay (basic adapter) ----
    # ---- /pay (basic adapter) ----
    # ---- /pay (basic adapter) ----
    @client.on(events.NewMessage(pattern=re.compile(r'^/pay(?:\s+(\d+(\.\d+)?))?$', re.IGNORECASE)))
    async def _on_pay(event):
        try:
            user_id = event.sender_id
            amt_str = event.pattern_match.group(1) if event.pattern_match else None
            if not amt_str:
                await event.reply("Please specify an amount in USD. Example: `/pay 12.50`")
                return
            amount = float(amt_str)
        except Exception:
            await event.reply("Invalid amount. Usage: `/pay 12.50`")
            return

        await event.reply("Initializing payment flow...")

        # Direct call to module-level handle_pay_command
        try:
            res = await run_blocking(handle_pay_command, user_id, amount, None, bot_instance)
        except Exception as e:
            await event.reply(f"Failed to start payment flow: {e}")
            return

        if res and isinstance(res, dict) and res.get("coinbase") and res["coinbase"].get("ok"):
            await event.reply("✅ Payment checkout created. Please follow the hosted URL sent to you.")
        else:
            await event.reply("⚠️ Could not create Coinbase checkout; direct/manual instructions were sent.")

    # ---- /tx (submit tx) ----
    @client.on(events.NewMessage(pattern=re.compile(r'^/tx\s+(0x[a-fA-F0-9]{64})\s+(\S+)', re.IGNORECASE)))
    async def _on_tx(event):
        try:
            user_id = event.sender_id
            tx_hash = event.pattern_match.group(1)
            reference = event.pattern_match.group(2)
        except Exception:
            await event.reply("Invalid command format. Usage: `/tx <tx_hash> <reference>`")
            return

        await event.reply("Verifying transaction — this may take a few seconds...")
        try:
            res = await run_blocking(cmd_submit_tx, user_id, tx_hash, reference, globals().get('db_conn'), bot_instance)
        except Exception as e:
            await event.reply(f"Verification failed: {e}")
            return

        if res.get("ok"):
            await event.reply(f"✅ Payment confirmed. Tx: {res.get('tx_hash') or tx_hash}")
        else:
            await event.reply(f"❌ Verification failed: {res.get('reason') or res.get('error') or 'unknown'}")
            
    # ---- /register <name> (register basename using server-wallet by default) ----
    # ---- /register <name> (register basename using server-wallet by default) ----
    @client.on(events.NewMessage(pattern=re.compile(r'^/register\s+(\S+)', re.IGNORECASE)))
    async def _on_register(event):
        try:
            label = event.pattern_match.group(1)
        except Exception:
            await event.reply("Usage: `/register <name>` (e.g. `/register alice`)")
            return

        await event.reply("Preparing registration — checking price and options...")

        use_server_wallet = True

        try:
            res = await run_blocking(handle_register_request, event.sender_id, label, use_server_wallet, bot_instance)
        except Exception as e:
            await event.reply(f"Registration failed (internal): {e}")
            return

        if not res.get("ok"):
            if res.get("instructions"):
                await event.reply(f"Could not register automatically: {res.get('error')}\n\nManual instructions:\n{res.get('instructions')}")
            else:
                await event.reply(f"Registration failed: {res.get('error') or 'unknown error'}")
            return

        if res.get("mode") == "server":
            tx = res.get("tx") or {}
            if tx.get("ok"):
                await event.reply(f"✅ Registration transaction submitted.\nTx: {tx.get('tx_hash')}\nExplorer: {tx.get('explorer_url')}")
            else:
                await event.reply(f"Server registration failed: {tx.get('error')}")
        else:
            await event.reply(f"Manual registration info:\n\n{res.get('instructions')}")

    # ---- /link_wallet (create a challenge to sign) ----
    @client.on(events.NewMessage(pattern=re.compile(r'^/link_wallet\s*$', re.IGNORECASE)))
    async def _on_link_wallet(event):
        user_id = event.sender_id
        await event.reply("Creating wallet link challenge...")

        try:
            # run in threadpool to avoid blocking event loop (file IO, secrets)
            msg = await run_blocking(create_link_challenge, user_id)
        except Exception as e:
            await event.reply(f"Failed to create challenge: {e}")
            return

        help_text = (
            "1) Use your wallet (MetaMask, Coinbase Wallet, etc.) to sign the following message using personal_sign / Sign Message.\n\n"
            "2) After signing, copy the signature (hex) and send back to the bot using:\n"
            "`/confirm_link <your_address> <signature>`\n\n"
            "Example:\n"
            "`/confirm_link 0xAbC... 0x1234abcd...`\n\n"
            "Important: DO NOT share your private key. Only send the signature string."
        )

        await event.reply(f"Challenge message (sign this exactly):\n\n```\n{msg}\n```\n\n{help_text}")

    # ---- /confirm_link <address> <signature> (verify & bind) ----
    @client.on(events.NewMessage(pattern=re.compile(r'^/confirm_link\s+(\S+)\s+(\S+)', re.IGNORECASE)))
    async def _on_confirm_link(event):
        user_id = event.sender_id
        try:
            addr = event.pattern_match.group(1)
            sig = event.pattern_match.group(2)
        except Exception:
            await event.reply("Usage: `/confirm_link <address> <signature>`")
            return

        await event.reply("Verifying signature and linking wallet...")

        try:
            res = await run_blocking(verify_wallet_signature, user_id, addr, sig)
        except Exception as e:
            await event.reply(f"Verification error: {e}")
            return

        if res.get("ok"):
            await event.reply(f"✅ Wallet linked: `{res.get('address')}`")
        else:
            await event.reply(f"❌ Link failed: {res.get('reason')}")
            
    # ---- /my_wallet or /me: show current linked address ----
    @client.on(events.NewMessage(pattern=re.compile(r'^/(?:me|my_wallet)\s*$', re.IGNORECASE)))
    async def _on_my_wallet(event):
        user_id = event.sender_id
        await event.reply("Checking linked wallet...")

        try:
            addr = await run_blocking(get_user_linked_address, user_id)
        except Exception as e:
            await event.reply(f"Error reading linked wallet: {e}")
            return

        if addr:
            await event.reply(f"Your linked wallet: `{addr}`")
        else:
            await event.reply("You do not have a linked wallet. Use `/link_wallet` to create a challenge and `/confirm_link` to link.")

    # ---- /confirm_link <address?> <signature> (address optional) ----
    # Accept both: "/confirm_link 0xAddr 0xSignature" and "/confirm_link 0xSignatureOnly" (signature only)
    @client.on(events.NewMessage(pattern=re.compile(r'^/confirm_link(?:\s+(\S+))?(?:\s+(\S+))?$', re.IGNORECASE)))
    async def _on_confirm_link(event):
        user_id = event.sender_id
        # pattern groups: group(1) may be address or signature; group(2) may be signature if group(1) was address
        g1 = event.pattern_match.group(1) if event.pattern_match else None
        g2 = event.pattern_match.group(2) if event.pattern_match else None

        # Heuristic: if two groups -> first is address, second is signature
        if g1 and g2:
            addr = g1
            sig = g2
        else:
            # single token provided — ambiguous: decide if it's a signature (starts with 0x and long)
            token = g1 or ""
            if token.lower().startswith("0x") and len(token) > 120:
                # treat as signature only (user omitted address)
                addr = None
                sig = token
            else:
                # treat as address only (but missing signature) => error
                await event.reply("Usage: `/confirm_link <address?> <signature>`\nEither provide both address and signature, or provide just the signature (bot will infer the address).")
                return

        await event.reply("Verifying signature and linking wallet...")

        try:
            res = await run_blocking(verify_wallet_signature, user_id, addr, sig)
        except Exception as e:
            await event.reply(f"Verification error: {e}")
            return

        if res.get("ok"):
            await event.reply(f"✅ Wallet linked: `{res.get('address')}`")
        else:
            await event.reply(f"❌ Link failed: {res.get('reason')}")

    # ---- /admin_list_links (admin-only) ----
    @client.on(events.NewMessage(pattern=re.compile(r'^/admin_list_links\s*$', re.IGNORECASE)))
    async def _on_admin_list_links(event):
        user_id = event.sender_id

        # admin check: attempt multiple approaches safely
        is_admin = False
        try:
            # 1) if there's a session manager with is_admin flag, prefer that
            sess_fn = globals().get("get_user_session")
            if callable(sess_fn):
                sess = sess_fn(user_id)
                if sess and getattr(sess, "is_admin", False):
                    is_admin = True
        except Exception:
            is_admin = is_admin

        # 2) fallback: check env ADMIN_USER_IDS (comma separated) or ADMIN_USER_ID single
        if not is_admin:
            try:
                env_single = os.getenv("ADMIN_USER_ID")
                env_list = os.getenv("ADMIN_USER_IDS")
                if env_single and str(user_id) == str(env_single):
                    is_admin = True
                elif env_list:
                    ids = [s.strip() for s in env_list.split(",") if s.strip()]
                    if str(user_id) in ids:
                        is_admin = True
            except Exception:
                pass

        if not is_admin:
            await event.reply("Unauthorized: admin-only command.")
            return

        await event.reply("Retrieving linked addresses...")

        try:
            links = await run_blocking(admin_list_links)
        except Exception as e:
            await event.reply(f"Failed to list links: {e}")
            return

        if not links:
            await event.reply("No linked addresses found.")
            return

        # build a compact message (limit size)
        lines = []
        for uid, addr in list(links.items())[:200]:
            lines.append(f"user {uid} -> `{addr}`")
        msg = "\n".join(lines)
        # If many entries, only send first N and note truncated
        if len(links) > 200:
            msg += f"\n\n(Showing first 200 of {len(links)} entries.)"

        # send in preformatted block
        await event.reply(f"Linked addresses:\n\n```\n{msg}\n```")
        
    # /track_wallet <0x...>
    @client.on(events.NewMessage(pattern=re.compile(r'^/track_wallet\s+(\S+)', re.IGNORECASE)))
    async def _on_track_wallet(event):
        user_id = event.sender_id
        addr = event.pattern_match.group(1)
        await event.reply("Adding tracked wallet...")
        res = await run_blocking(add_tracked_base_wallet, user_id, addr)
        if res.get("ok"):
            await event.reply(f"✅ Now tracking {res.get('address')}")
        else:
            await event.reply(f"Failed to track: {res.get('error')}")

    # /untrack_wallet <0x...>
    @client.on(events.NewMessage(pattern=re.compile(r'^/untrack_wallet\s+(\S+)', re.IGNORECASE)))
    async def _on_untrack_wallet(event):
        user_id = event.sender_id
        addr = event.pattern_match.group(1)
        res = await run_blocking(remove_tracked_base_wallet, user_id, addr)
        if res.get("ok"):
            await event.reply(f"✅ Untracked {res.get('address')}")
        else:
            await event.reply(f"Failed to untrack: {res.get('error')}")

    # /list_tracked_wallets
    @client.on(events.NewMessage(pattern=re.compile(r'^/list_tracked_wallets\s*$', re.IGNORECASE)))
    async def _on_list_tracked_wallets(event):
        user_id = event.sender_id
        res = await run_blocking(list_tracked_base_wallets, user_id)
        if res.get("ok"):
            tracked = res.get("tracked", {})
            if not tracked:
                await event.reply("You have no tracked wallets.")
                return
            lines = [f"{addr} -> last_seen_block:{meta.get('last_seen_block')} last_seen_time:{meta.get('last_seen_time')}" for addr,meta in tracked.items()]
            await event.reply("Tracked wallets:\n" + "\n".join(lines))
        else:
            await event.reply(f"Error reading tracked wallets: {res.get('error')}")
            
    # ----------------------
    # /base  -> show Base submenu (inline keyboard)
    # ----------------------
    @client.on(events.NewMessage(pattern=re.compile(r'^/base\s*$', re.IGNORECASE)))
    async def _on_base_menu(event):
        """
        Shows the Base submenu with the most important base-related user commands.
        Buttons emit CallbackQuery events handled by _on_base_menu_callback below.
        """
        kb = [
            [Button.inline("🔗 Link Wallet", b"base:link"), Button.inline("👤 My Wallet", b"base:my")],
            [Button.inline("🔎 Resolve Name", b"base:resolve"), Button.inline("🆔 Register Name", b"base:register")],
            [Button.inline("💳 Payments", b"base:pay"), Button.inline("⚙️ Settings", b"base:settings")],
        ]
        try:
            await event.reply("Base menu — quick actions:", buttons=kb)
        except Exception:
            # fallback for clients where reply with buttons may differ
            await event.respond("Base menu — please use commands: /link_wallet, /me, /resolve, /register, /pay")

    @client.on(events.CallbackQuery())
    async def _on_base_menu_callback(ev):
        """
        Handle callback data from the Base submenu.
        Data bytes: b"base:link", b"base:my", b"base:resolve", b"base:register", b"base:pay", b"base:settings"
        """
        try:
            data = ev.data.decode("utf-8") if isinstance(ev.data, (bytes, bytearray)) else str(ev.data)
        except Exception:
            data = str(ev.data)

        user_id = ev.sender_id or (ev.query.user_id if hasattr(ev, "query") else None)

        # Acknowledge the callback quickly
        try:
            await ev.answer()  # silent answer to remove "loading"
        except Exception:
            pass

        # route actions
        if data == "base:link":
            # call the link flow (create challenge) and reply with message
            try:
                msg = await run_blocking(create_link_challenge, user_id)
                help_text = (
                    "Sign the message with your wallet (MetaMask/Wallet). Then use:\n"
                    "`/confirm_link <address?> <signature>`\n\n"
                    "Example signature-only: `/confirm_link 0x2c64...`"
                )
                await ev.respond(f"Challenge message (sign exactly):\n\n```\n{msg}\n```\n\n{help_text}")
            except Exception as e:
                await ev.respond(f"Failed to create challenge: {e}")

        elif data == "base:my":
            try:
                addr = await run_blocking(get_user_linked_address, user_id)
                if addr:
                    await ev.respond(f"Your linked wallet: `{addr}`")
                else:
                    await ev.respond("You do not have a linked wallet. Use `Link Wallet` to start.")
            except Exception as e:
                await ev.respond(f"Error reading linked wallet: {e}")

        elif data == "base:resolve":
            # prompt user to use /resolve (quick help)
            await ev.respond("To resolve a basename, send: `/resolve <name>`. Example: `/resolve alice.base`")

        elif data == "base:register":
            await ev.respond("To register a basename, send: `/register <name>` (e.g. `/register alice`).")

        elif data == "base:pay":
            await ev.respond("To start a payment, send `/pay <amount_usd>` (e.g. `/pay 5.00`).")

        elif data == "base:settings":
            # route to the settings submenu (see Area C)
            kb = [
                [Button.inline("➕ Track Wallet", b"settings:track"), Button.inline("➖ Untrack Wallet", b"settings:untrack")],
                [Button.inline("📋 List Tracked", b"settings:list_tracked"), Button.inline("🔗 Linked Addr (admin)", b"settings:admin_links")],
                [Button.inline("⬅️ Back", b"settings:back")]
            ]
            await ev.respond("Base settings:", buttons=kb)

        else:
            await ev.respond("Unknown action.")

    # ----------------------
    # Job helpers (Phase 1): /add_dest, /add_trader, /enable_onchain, /disable_onchain, /set_onchain_amount
    # These are minimal helpers that call your forwarder instance methods for the current user's forwarder.
    # Usage:
    #   /add_dest <job_index> <dest1[,dest2,...]>
    #   /add_trader <job_index> <https://webhook.url>
    #   /enable_onchain <job_index> <amount_native>
    #   /disable_onchain <job_index>
    #   /set_onchain_amount <job_index> <amount_native>
    # ----------------------

    @client.on(events.NewMessage(pattern=re.compile(r'^/add_dest\s+(\d+)\s+(.+)', re.IGNORECASE)))
    async def _on_add_dest(event):
        try:
            job_index = int(event.pattern_match.group(1))
            message = event.pattern_match.group(2).strip()
            user_id = event.sender_id
            # get user forwarder
            fwd = getattr(bot_instance, "user_forwarders", {}).get(user_id)
            if not fwd:
                await event.reply("No forwarder instance found for you. Create or restore a forwarder session first.")
                return
            # delegate to existing method (it handles parsing multi destinations)
            await fwd._modify_job_destinations(event, message, job_index)
        except Exception as e:
            await event.reply(f"Error adding destination: {e}")

    @client.on(events.NewMessage(pattern=re.compile(r'^/add_trader\s+(\d+)\s+(\S+)', re.IGNORECASE)))
    async def _on_add_trader(event):
        try:
            job_index = int(event.pattern_match.group(1))
            url = event.pattern_match.group(2).strip()
            user_id = event.sender_id
            fwd = getattr(bot_instance, "user_forwarders", {}).get(user_id)
            if not fwd:
                await event.reply("No forwarder instance found for you.")
                return
            # call the same destination parser - it understands trader: prefix and raw http(s)
            await fwd._modify_job_destinations(event, f"trader:{url}", job_index)
        except Exception as e:
            await event.reply(f"Error adding trader endpoint: {e}")

    @client.on(events.NewMessage(pattern=re.compile(r'^/enable_onchain\s+(\d+)\s+([\d\.]+)', re.IGNORECASE)))
    async def _on_enable_onchain(event):
        try:
            job_index = int(event.pattern_match.group(1))
            amount = float(event.pattern_match.group(2))
            user_id = event.sender_id
            fwd = getattr(bot_instance, "user_forwarders", {}).get(user_id)
            if not fwd:
                await event.reply("No forwarder instance found for you.")
                return
            # set flags on the job and persist
            job = fwd.jobs[job_index]
            job['onchain_transfer'] = True
            job['onchain_amount'] = float(amount)
            # persist via bot_instance if available
            try:
                await bot_instance.save_user_data(user_id, fwd)
            except Exception:
                pass
            await event.reply(f"✅ Onchain transfer enabled for job {job_index} amount={amount}")
        except Exception as e:
            await event.reply(f"Error enabling onchain transfer: {e}")

    @client.on(events.NewMessage(pattern=re.compile(r'^/disable_onchain\s+(\d+)', re.IGNORECASE)))
    async def _on_disable_onchain(event):
        try:
            job_index = int(event.pattern_match.group(1))
            user_id = event.sender_id
            fwd = getattr(bot_instance, "user_forwarders", {}).get(user_id)
            if not fwd:
                await event.reply("No forwarder instance found for you.")
                return
            job = fwd.jobs[job_index]
            job.pop('onchain_transfer', None)
            job.pop('onchain_amount', None)
            try:
                await bot_instance.save_user_data(user_id, fwd)
            except Exception:
                pass
            await event.reply(f"✅ Onchain transfer disabled for job {job_index}")
        except Exception as e:
            await event.reply(f"Error disabling onchain transfer: {e}")

    @client.on(events.NewMessage(pattern=re.compile(r'^/set_onchain_amount\s+(\d+)\s+([\d\.]+)', re.IGNORECASE)))
    async def _on_set_onchain_amount(event):
        try:
            job_index = int(event.pattern_match.group(1))
            amount = float(event.pattern_match.group(2))
            user_id = event.sender_id
            fwd = getattr(bot_instance, "user_forwarders", {}).get(user_id)
            if not fwd:
                await event.reply("No forwarder instance found for you.")
                return
            job = fwd.jobs[job_index]
            job['onchain_amount'] = float(amount)
            try:
                await bot_instance.save_user_data(user_id, fwd)
            except Exception:
                pass
            await event.reply(f"✅ Set onchain amount for job {job_index} to {amount}")
        except Exception as e:
            await event.reply(f"Error setting onchain amount: {e}")

    @client.on(events.NewMessage(pattern=re.compile(r'^/base_settings\s*$', re.IGNORECASE)))
    async def _on_base_settings(event):
        kb = [
            [Button.inline("➕ Track Wallet", b"settings:track"), Button.inline("➖ Untrack Wallet", b"settings:untrack")],
            [Button.inline("📋 List Tracked", b"settings:list_tracked"), Button.inline("🔗 Linked Addr (admin)", b"settings:admin_links")],
            [Button.inline("⬅️ Back to Base", b"settings:back")],
        ]
        try:
            await event.reply("Base settings — quick actions:", buttons=kb)
        except Exception:
            await event.respond("Use commands: /track_wallet, /untrack_wallet, /list_tracked_wallets, /admin_list_links")
            
    @client.on(events.CallbackQuery())
    async def _on_main_menu_callback(ev):
        try:
            data = ev.data.decode("utf-8") if isinstance(ev.data, (bytes, bytearray)) else str(ev.data)
        except Exception:
            data = str(ev.data)
        if data == "open_base_menu":
            # trigger the same logic as /base
            await _on_base_menu(ev)  # if ev is compatible; else send "/base"

    # mark registered for this bot instance
    register_bot_payment_handlers._registered[id(bot_instance)] = True
    try:
        import logging
        if "resolve_basename" not in globals():
            logging.warning("register_bot_payment_handlers: WARNING — resolve_basename not in globals() at registration time.")
        else:
            logging.info("register_bot_payment_handlers: resolve_basename present at registration time.")
    except Exception:
        pass

    # debug log
    try:
        logging.info(f"register_bot_payment_handlers: handlers registered for bot_instance id {id(bot_instance)}")
    except Exception:
        pass


async def main():
    """Main function to start the bot."""
    print("🤖 **Enhanced Telegram Forwarder Bot**")
    print("=" * 50)

    bot = TelegramBot()
    register_bot_payment_handlers(bot)
    
    # --- PAYMENT DB INIT (INSERT BELOW `bot = TelegramBot()` in async def main()) ---
    # Ensure psycopg2 is installed on the server (pip install psycopg2-binary)
    try:
        pass
    except Exception as e:
        raise RuntimeError("psycopg2 is required for payments DB. Install with: pip install psycopg2-binary") from e

    # File-backed payments store startup (no Postgres required)
    globals()['bot_instance'] = bot
    # Ensure file-backed payments store exists (init_payments_db ignores the db param)
    try:
        init_payments_db(None)
        globals()['db_conn'] = None  # compatibility placeholder for legacy callsites
        print("✅ payments file-backed store initialized (payments.json)")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize file-backed payments store: {e}")
    # --- END PAYMENT DB INIT ---
    init_user_profiles_db()
    asyncio.create_task(wallet_poll_loop())
    # --- START: start reconcile thread now that helpers are defined ---
    try:
        # create the reconcile thread (defined at module-level by _create_reconcile_thread)
        reconcile_thread = None
        if '_create_reconcile_thread' in globals():
            reconcile_thread = _create_reconcile_thread(poll_interval_seconds=300)
            # start it now that init_payments_db() and _load_payments_store() exist
            reconcile_thread.start()
            print("🔁 Background reconcile thread started.")
        else:
            print("⚠️ _create_reconcile_thread not found; reconcile thread not started.")
    except Exception as e:
        print(f"⚠️ Failed to start reconcile thread: {e}")

    # --- START OF NEW WEBHOOK LOGIC ---
    if SUBSCRIPTION_ENABLED:
        # Run the FastAPI server in a separate thread.
        # daemon=True ensures the thread will close when the main script exits.
        webhook_thread = threading.Thread(
            target=bot.run_webhook_server,
            daemon=True
        )
        webhook_thread.start()
        print(f"🚀 Webhook server listening on http://0.0.0.0:{WEBHOOK_PORT}...")
    # --- END OF NEW WEBHOOK LOGIC ---

    try:
        # MODIFIED: Choose startup routine based on DEMO_MODE
        if DEMO_MODE:
            await bot.setup_demo_mode()
        else:
            await bot.restore_user_sessions()

        print("🚀 Bot is ready and listening for messages...")
        print("📱 Users can start chatting with the bot!")
        print("🔧 Admins can use /admin <master_secret> for admin access")
        print("=" * 50)

        await bot.start_bot()

    except Exception as e:
        logging.critical(f"A critical error occurred during bot startup or operation: {e}", exc_info=True)
        print(f"❌ BOT CRITICAL ERROR: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")
    except Exception as e:
        print(f"❌ Bot crashed unexpectedly: {e}")
        logging.critical(f"Bot crashed: {e}", exc_info=True)
