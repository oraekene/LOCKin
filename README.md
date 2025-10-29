# README — Lockin / TelegramForwarder

> **Status:** single-file prototype ( `main_bot(lastworkingasat4pm_beforeinlinemenuedits).py` ). This README explains installation, configuration, running, and how to use the bot’s features.

---

## Table of contents

1. Features (short)
2. Requirements & dependencies
3. Quick start (install + run)
4. Environment variables (`creds.env`) — full list + example
5. Running (local / Colab / production notes)
6. Using the bot — commands & flows (step-by-step)
7. Payments & subscription flow
8. Wallet linking (non-custodial) — step-by-step
9. Basename registration (Base)
10. Troubleshooting & common issues
11. Security & privacy notes
12. File layout & internals (quick reference)
13. Contact / contributing

---

## 1) Features (short)

* Connect your Telegram *user account* (Telethon) so the bot can read private groups you belong to and forward selected messages.
* Create forwarding jobs (keywords, cashtags, contract addresses, custom regex patterns).
* Demo / paper trading mode, then flip to live execution.
* Track & rank signal authors by realized PnL.
* Wallet linking via message-signature (no private keys shared).
* Payments/subscriptions using Coinbase Commerce / Paystack / NowPayments (file-backed store available if no Postgres).
* Register/resolve Base basenames and helper onchain utilities.
  These capabilities and the high-level UX are documented inside the script. 

---

## 2) Requirements & dependencies

Minimum tested Python version: **3.10+**

Key Python packages imported by the script (install via `pip` or `requirements.txt`):

* `telethon` (Telegram client)
* `uvicorn`, `fastapi`, `starlette` (webhook server & API)
* `web3`, `eth-account` (onchain / Base interactions)
* `cryptography` (Fernet + KDF)
* `python-dotenv` (load `.env` / `creds.env`)
* `requests`
* `nest_asyncio`
* `hyperliquid` (exchange SDK usage — optional depending on features used)
* (Optional) `psycopg2-binary` if you plan to use Postgres for payments DB.

You can confirm these imports inside the script (top-level imports). 

Example `pip` install (after creating a venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install telethon uvicorn fastapi starlette web3 eth-account cryptography python-dotenv requests nest_asyncio hyperliquid
# Optional:
pip install psycopg2-binary
```

---

## 3) Quick start — install & run (local)

1. Clone repo:

```bash
git clone <your-repo-url> lockin-bot
cd lockin-bot
```

2. Create and activate Python venv & install dependencies (see above).

3. Create `creds.env` in repo root (see section **Environment variables** below for full example).

4. Ensure `data/`, `sessions/`, `crypto/` directories are writable. The script creates them automatically, but make sure the user running the bot has write permission. The bot uses `BASE_PATH` detection that supports Google Colab and local VPS. 

5. Run the bot:

```bash
python "main_bot(lastworkingasat4pm_beforeinlinemenuedits).py"
```

On startup you should see printed lines like:

* `payments file-backed store initialized (payments.json)`
* `Background reconcile thread started.`
* `Webhook server listening on http://0.0.0.0:<WEBHOOK_PORT>...`
* `Bot is ready and listening for messages...`
  These lines are emitted from the `main()` startup logic. 

---

## 4) Environment variables (`creds.env`) — full list & example

**Required (minimum)** — the bot will refuse to start if these are missing:

* `MASTER_SECRET` — master secret for per-user encryption derivation (critical). 
* `BOT_TOKEN` — bot token (if you use bot features). 
* `BOT_API_ID` — Telegram API ID (integer) for Telethon. 
* `BOT_API_HASH` — Telegram API hash. 

**Common / optional / useful variables**

* `DEMO_MODE` — toggles demo startup path; note: code uses a particular check so set documentation value explicitly (the script checks `os.getenv('DEMO_MODE') == 'False'` — follow the bot’s instructions). 
* `DEMO_API_ID`, `DEMO_API_HASH`, `DEMO_PHONE_NUMBER`, `DEMO_SESSION_NAME` — used when `DEMO_MODE` is enabled. 
* `PAYSTACK_TEST_SECRET_KEY`, `PAYSTACK_LIVE_SECRET_KEY` — Paystack credentials (script has defaults but you should replace). 
* `NOWPAYMENTS_API_KEY`, `NOWPAYMENTS_IPN_SECRET` — crypto payment provider keys. 
* `PAYSTACK_TEST_MODE` — `True`/`False` toggle for Paystack test vs live. 
* `WEBHOOK_BASE_URL` — your public URL for webhooks (ngrok or domain). 
* `WEBHOOK_PORT` — port for webhook server (defaults to 8000). 
* `SERVER_WALLET_PRIVATE_KEY` or `SERVER_WALLET_PRIVATE_KEY_ENC` — required if you want server-side onchain registration / server wallet flows. If using encrypted key you must supply a `decrypt()` helper (the script attempts to call a decrypt helper). 
* `BASE_RPC_URL`, `BASE_SEPOLIA_RPC`, `BASE_RPC` — Base network RPC (used by web3 helpers). If not set, `connect_base_web3()` will raise. 
* `ETHEREUM_RPC_URL`, `POLYGON_RPC_URL`, `FALLBACK_RPC_URL` — other chain RPCs used by `get_web3_for_chain()`. 
* `USER_PROFILES_PATH` — override where user profiles file is stored (default `data/user_profiles.json`). 
* `LINK_CHALLENGE_TTL` — challenge expiry seconds (default 600). 
* `APP_NAME` — optional string for challenge messages. 
* `TRACK_POLL_INTERVAL` — wallet tracking poll interval in seconds. 
* `ADMIN_USER_ID`, `ADMIN_USER_IDS` — admin user IDs for admin-only commands (optional). 

**Example `creds.env` (DO NOT COMMIT to git):**

```
MASTER_SECRET=changeme_master_secret
BOT_TOKEN=123456:ABC-DEF...
BOT_API_ID=1234567
BOT_API_HASH=abcd1234abcd1234abcd1234abcd1234
DEMO_MODE=False
# Demo values (if DEMO_MODE enabled)
DEMO_API_ID=1111111
DEMO_API_HASH=demohash
DEMO_PHONE_NUMBER=+15551234567
# Payments
PAYSTACK_TEST_SECRET_KEY=sk_test_xxx
PAYSTACK_LIVE_SECRET_KEY=sk_live_xxx
NOWPAYMENTS_API_KEY=amt_xxx
NOWPAYMENTS_IPN_SECRET=ipn_secret
PAYSTACK_TEST_MODE=True
# Webhooks / RPC
WEBHOOK_BASE_URL=https://yourdomain.example
WEBHOOK_PORT=8000
BASE_RPC_URL=https://base.rpc.provider/...
ETHEREUM_RPC_URL=https://eth.rpc.provider/...
POLYGON_RPC_URL=https://polygon.rpc.provider/...
# Server wallet (ONLY if you want server to do txs)
# SERVER_WALLET_PRIVATE_KEY=0x...
# or
# SERVER_WALLET_PRIVATE_KEY_ENC=ENC:...
```

**Security note:** Add `creds.env` to `.gitignore`. Never commit credentials. See Security & privacy section. (The script will refuse to start if required env vars are missing.) 

---

## 5) Running modes & environment notes

* **Local / VPS**: run `python main_bot...py` as above. The script auto-creates `data/`, `crypto/`, and `sessions/` directories under detected `BASE_PATH`. 
* **Google Colab**: the script attempts to detect Colab and set `BASE_PATH` to `/content/drive/MyDrive/telegram_forwarder` — if you use Colab, ensure drive is mounted or override `TELEGRAM_BOT_BASE_PATH`. 
* **Production**: consider running via a process manager (`systemd`, `supervisord`) or containerizing. The script starts a FastAPI webhook server in a background thread when subscriptions are enabled. Ensure the server has a reachable `WEBHOOK_BASE_URL`. 

---

## 6) Using the bot — commands & flows (step-by-step)

The bot exposes both menu button flows and slash commands. Use `/start` to open the main menu. Key commands:

**Core commands**

* `/start` — open the main menu. 
* `/help` — shows full help and usage. 
* `/commands` — short command list. 

**Wallet (non-custodial)**

1. `/link_wallet` — bot returns a human-readable challenge message containing a nonce and expiry. Sign it in MetaMask (personal_sign). 
2. `/confirm_link <addr?> <signature>` — confirm and bind your wallet. You may provide only the signature and the bot will infer the address. On success the address is saved to `data/user_profiles.json`. 
3. `/me` or `/my_wallet` — show your linked wallet. 

**Payments & subscriptions**

* `/pay <amount_usd>` — starts a payment flow (Coinbase Commerce primary; direct/manual fallback). The bot will create a checkout and also provide manual transfer fallback instructions. 
* `/tx <tx_hash> <ref>` — submit a tx for manual verification. Useful for manual USDC transfers. 
* `/pay_status <reference>` — check a payment’s status. 
* `/subscribe` and menu buttons — subscription purchase flows (card/crypto choices). Callback handlers and Paystack calls are implemented in the file. 

**Basename (Base)**

* `/resolve <name>` — resolve `alice.base`.
* `/register <name>` — attempt server-side registration (requires server wallet) or returns manual instructions & price if server cannot sign/send. The manual instructions include contract and priced `wei` amount. 

**Examples:**

```
/link_wallet
# sign returned string in MetaMask
/confirm_link 0x2c64... 0xSIGNATUREHEX
/pay 12.50
/tx 0x8d69... REF-A12
/register alice
```

All these examples are included in the script help. 

---

## 7) Payments & subscriptions — specifics

* The script supports Coinbase Commerce (hosted checkouts) as the recommended flow and falls back to a manual verification path where users submit tx hash and reference. It also implements Paystack flows (card) and NowPayments for crypto. The payments are stored in a file-backed JSON store `data/payments.json` by default if Postgres is not configured. 

* **File-backed payments DB**: the bot runs `init_payments_db()` at start and will create `data/payments.json` if missing. You can optionally integrate Postgres (install `psycopg2-binary`) and provide a DB connection if you want a real DB. 

---

## 8) Wallet linking (detailed)

This is non-custodial — the bot never asks for private keys.

1. `/link_wallet` → bot replies with a challenge message (includes nonce and expiry). Stored in `data/user_profiles.json`. 
2. In your wallet, use *Sign Message* (personal_sign) and copy the signature.
3. Send `/confirm_link <signature>` or `/confirm_link <address> <signature>`. The bot uses `eth_account` to recover the address and compares. If matches, the address is saved in the profiles file and challenge cleared. 

**Troubleshooting tips:** if challenge expired (defaults 10 minutes), run `/link_wallet` again. If verification fails, ensure you signed the exact challenge string (no extra whitespace). 

---

## 9) Basename (Base) registration (high-level)

* `/register <name>` will attempt server-side registration if `SERVER_WALLET_PRIVATE_KEY` (or encrypted variant) is present. If server-side fails or is not configured the function returns instructions and `price_wei` so the user can pay manually via BaseScan contract Write interface. The script includes minimal RegistrarController ABI and logic to compute price and build the payable request. 

---

## 10) Troubleshooting & common issues

* **Missing environment variables / bot aborts on startup**: the script validates required env vars (`MASTER_SECRET`, `BOT_TOKEN`, `BOT_API_ID`, `BOT_API_HASH`) and will raise a `ValueError` if any are missing. Ensure they are in `creds.env` or system environment. 

* **RPC connection error**: `connect_base_web3()` raises if `BASE_RPC_URL` is not set or RPC fails. Provide a valid provider URL (Alchemy, Infura, or other). 

* **Payments failing / webhooks**: ensure `WEBHOOK_BASE_URL` is reachable by external payment providers and `WEBHOOK_PORT` is open & forwarded if needed. The script launches a FastAPI webhook server in a background thread when subscriptions are enabled. 

* **Telegram login codes rejected**: NOTE — the bot includes a known workaround: when the login verification code is generated programmatically, Telegram sometimes rejects pasted codes; the author found obfuscating the code (adding spaces/characters) lets it be accepted. If you see login failure, try obfuscating the code when entering it during the Telethon login flow. This behavior is described in the repo docs/heavily-commented code. 

* **If signature verification fails**: ensure you signed the *exact* challenge string and did not alter whitespace or add characters. 

---

## 11) Security & privacy

* **Never paste private keys into chat.** The bot only accepts signed messages (personal_sign) to link wallets. The code explicitly warns about never asking for private keys. 
* User data (per-user state and secrets) is encrypted using a master secret + per-user salt (PBKDF2 + Fernet) before writing files to disk. The encryption derivation is implemented in `CryptoManager`. 
* **Do not commit `creds.env` to Git**. Use `.gitignore` and for hosted deployments use the platform’s environment variable storage or GitHub Actions Secrets / Render secrets.
* Server-side keys (if used for onchain actions) should be limited in scope (small funds, separate account) and stored encrypted (or in platform secret manager). The script can read `SERVER_WALLET_PRIVATE_KEY_ENC` but requires a decrypt helper. 

---

## 12) File layout & internals (quick)

* `main_bot(lastworking...).py` — monolithic implementation (all core logic). Top of file shows required imports and configuration. 
* `data/` — stores `user_profiles.json` and `payments.json` by default. The script includes init helpers. 
* `sessions/` — Telethon session files. Created automatically. 
* `crypto/` — salt keys and encrypted per-user blobs (created by `CryptoManager`). 

---

## 13) Contact & contributing

* This is a developer-built prototype. If you want to contribute, open issues / PRs with focused changes (eg. split monolith into modules, add unit tests, add Dockerfile, or add CI). The code contains extensive inline documentation and help text. 

---

## ASCII flow (high-level)

```
[Telegram user account] <---(Telethon user client)---> [Bot monolith]
      |                                              |
      |-- create job (keywords/regex)                 |
      |-- /link_wallet (sign challenge)               |
      |-- /pay /subscribe                             |
      v                                              v
[Forwarder engine] ----> [destinations: TG chats, webhooks, Hyperliquid, wallet txs]
```

The webhook server runs in a background thread and handles payment provider callbacks (Coinbase/Paystack). 

---
