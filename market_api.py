# market_api.py
# Requires: aiohttp
# pip install aiohttp
"""
Market API Adapter with provider fallback, per-provider throttling, and persistent file cache.

Features:
 - DexScreenerProvider: chain-aware best-effort real-time quotes (search + pair endpoints).
 - CoinGeckoProvider: canonical historical price (market_chart/range) and simple price.
 - AlchemyProvider: placeholder/skeleton for on-chain price derivation (pair discovery + reserves).
 - HeliusProvider: placeholder for Solana metadata; CoinGecko remains primary price source.
 - Persistent file cache to avoid repeated calls and stay within free-tier limits.
"""
import aiohttp
import asyncio
import time
import logging
import json
import os
import tempfile
from typing import Optional, Tuple, Dict, Any, List

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.INFO)

# ---- Exceptions ----
class RateLimitError(Exception):
    pass

# ---- Helpers ----
def _is_probably_contract(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if s.startswith("0x") and len(s) >= 40:
        return True
    # naive Solana base58-ish check (length >= 32 and alnum)
    if len(s) >= 32 and all(c.isalnum() for c in s):
        return True
    return False

def _atomic_write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + "_", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

# ---- Providers (async) ----

class CoinGeckoProvider:
    COINGECKO_BASE = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None, timeout: int = 8):
        self.api_key = api_key
        self._session = session
        self.timeout = timeout
        self._symbol_to_id: Dict[str, str] = {}
        self._coins_list_ts = 0
        self._coins_ttl = 3600
        self._lock = asyncio.Lock()

    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def _refresh_coins_list_if_needed(self):
        now = time.time()
        if now - self._coins_list_ts < self._coins_ttl and self._symbol_to_id:
            return
        async with self._lock:
            now = time.time()
            if now - self._coins_list_ts < self._coins_ttl and self._symbol_to_id:
                return
            await self._ensure_session()
            url = f"{self.COINGECKO_BASE}/coins/list"
            headers = {}
            if self.api_key:
                headers["x-cg-pro-api-key"] = self.api_key
            try:
                async with self._session.get(url, headers=headers, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        _LOG.warning("CoinGecko coins/list returned %s", resp.status)
                        return
                    data = await resp.json()
                    mapping = {}
                    for item in data:
                        sym = (item.get("symbol") or "").lower()
                        if sym and sym not in mapping:
                            mapping[sym] = item.get("id")
                    self._symbol_to_id = mapping
                    self._coins_list_ts = time.time()
            except Exception:
                _LOG.exception("Failed to refresh coingecko coins list")

    async def get_price(self, chain: str, symbol: Optional[str] = None, contract: Optional[str] = None, vs_currency: str = "usd") -> Optional[float]:
        await self._ensure_session()
        headers = {}
        if self.api_key:
            headers["x-cg-pro-api-key"] = self.api_key

        # If contract + chain maps to platform, use token_price endpoint
        platform_map = {
            "evm": "ethereum",
            "ethereum": "ethereum",
            "polygon": "polygon-pos",
            "bsc": "binance-smart-chain",
            "bnb": "binance-smart-chain",
            "optimism": "optimistic-ethereum",
            "arbitrum": "arbitrum-one",
        }
        platform = platform_map.get((chain or "").lower())
        if contract and platform:
            url = f"{self.COINGECKO_BASE}/simple/token_price/{platform}"
            params = {"contract_addresses": contract, "vs_currencies": vs_currency}
            try:
                async with self._session.get(url, params=params, headers=headers, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        j = await resp.json()
                        val = j.get(contract.lower(), {}).get(vs_currency)
                        if val is not None:
                            return float(val)
            except Exception:
                _LOG.exception("CoinGecko token_price call failed")

        # fallback: symbol -> coin id -> simple/price
        if symbol:
            await self._refresh_coins_list_if_needed()
            coin_id = self._symbol_to_id.get(symbol.lower())
            if coin_id:
                url = f"{self.COINGECKO_BASE}/simple/price"
                params = {"ids": coin_id, "vs_currencies": vs_currency}
                try:
                    async with self._session.get(url, params=params, headers=headers, timeout=self.timeout) as resp:
                        if resp.status == 200:
                            j = await resp.json()
                            val = j.get(coin_id, {}).get(vs_currency)
                            if val is not None:
                                return float(val)
                except Exception:
                    _LOG.exception("CoinGecko simple/price failed")
        return None

    async def get_price_history(self, chain: str, symbol: Optional[str], contract: Optional[str], start_ts: int, end_ts: int, interval: str = "60") -> Optional[List[Tuple[int, float]]]:
        # CoinGecko market_chart_by_id supports vs_currency and days/resolution; implement a simple wrapper:
        await self._ensure_session()
        if not symbol:
            return None
        await self._refresh_coins_list_if_needed()
        coin_id = self._symbol_to_id.get(symbol.lower())
        if not coin_id:
            return None
        url = f"{self.COINGECKO_BASE}/coins/{coin_id}/market_chart/range"
        params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}
        try:
            async with self._session.get(url, params=params, timeout=self.timeout) as resp:
                if resp.status == 200:
                    j = await resp.json()
                    prices = j.get("prices", [])
                    # prices: [[ts_ms, price], ...]
                    out = [(int(p[0] // 1000), float(p[1])) for p in prices]
                    return out
        except Exception:
            _LOG.exception("CoinGecko history failed")
        return None

class AlchemyProvider:
    ALCHEMY_API_BASE = "https://api.g.alchemy.com/prices/v1"

    def __init__(self, api_key: Optional[str], session: Optional[aiohttp.ClientSession] = None, timeout: int = 6):
        self.api_key = api_key
        self._session = session
        self.timeout = timeout

    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def get_price(self, chain: str, symbol: Optional[str] = None, contract: Optional[str] = None, vs_currency: str = "usd") -> Optional[float]:
        if not self.api_key:
            return None
        await self._ensure_session()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            # prefer contract lookup
            if contract:
                url = f"{self.ALCHEMY_API_BASE}/tokens/by-address"
                params = {"addresses": contract}
                async with self._session.get(url, params=params, headers=headers, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        j = await resp.json()
                        data = j.get("data", [])
                        if data:
                            prices = data[0].get("prices", [])
                            for p in prices:
                                if (p.get("currency") or "").lower() == vs_currency.lower():
                                    try:
                                        return float(p.get("value"))
                                    except Exception:
                                        pass
            if symbol:
                url = f"{self.ALCHEMY_API_BASE}/tokens/by-symbol"
                params = {"symbols": symbol}
                async with self._session.get(url, params=params, headers=headers, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        j = await resp.json()
                        data = j.get("data", [])
                        if data:
                            prices = data[0].get("prices", [])
                            for p in prices:
                                if (p.get("currency") or "").lower() == vs_currency.lower():
                                    try:
                                        return float(p.get("value"))
                                    except Exception:
                                        pass
        except Exception:
            _LOG.exception("AlchemyProvider.get_price error")
        return None

    async def get_price_history(self, chain: str, symbol: Optional[str], contract: Optional[str], start_ts: int, end_ts: int, interval: str = "60") -> Optional[List[Tuple[int, float]]]:
        # Alchemy doesn't provide straightforward OHLC history via free endpoints; skip or implement provider-specific logic
        return None

class HeliusProvider:
    HELIUS_RPC_MAINNET = "https://mainnet.helius-rpc.com/"

    def __init__(self, api_key: Optional[str], session: Optional[aiohttp.ClientSession] = None, timeout: int = 8):
        self.api_key = api_key
        self._session = session
        self.timeout = timeout

    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def get_price(self, chain: str, symbol: Optional[str] = None, contract: Optional[str] = None, vs_currency: str = "usd") -> Optional[float]:
        if not self.api_key:
            return None
        await self._ensure_session()
        if not contract:
            return None
        try:
            url = self.HELIUS_RPC_MAINNET
            params = {"api-key": self.api_key}
            body = {"jsonrpc": "2.0", "id": "1", "method": "getAsset", "params": {"id": contract}}
            async with self._session.post(url, json=body, params=params, timeout=self.timeout) as resp:
                if resp.status != 200:
                    return None
                j = await resp.json()
                result = j.get("result") or {}
                token_info = result.get("token_info") or {}
                price_info = token_info.get("price_info") or {}
                prices = price_info.get("prices") or []
                for p in prices:
                    if p.get("currency", "").lower() == vs_currency.lower():
                        try:
                            return float(p.get("value"))
                        except Exception:
                            pass
        except Exception:
            _LOG.exception("HeliusProvider.get_price error")
        return None

    async def get_price_history(self, chain: str, symbol: Optional[str], contract: Optional[str], start_ts: int, end_ts: int, interval: str = "60") -> Optional[List[Tuple[int, float]]]:
        # Helius DAS does not expose long-range history via getAsset; skip or implement via on-chain aggregator
        return None

class DexScreenerProvider:
    DEXSCREENER_BASE = "https://openapi.dexscreener.com"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None, timeout: int = 6):
        self._session = session
        self.timeout = timeout

    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def get_price(self, chain: str, symbol: Optional[str] = None, contract: Optional[str] = None, vs_currency: str = "usd") -> Optional[float]:
        await self._ensure_session()
        try:
            # Dexscreener token by address endpoint
            if contract:
                url = f"{self.DEXSCREENER_BASE}/latest/dex/tokens/{contract}"
                async with self._session.get(url, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        j = await resp.json()
                        # j structure: { "pairs": [...], "token": {...} }
                        token = j.get("token") or {}
                        price = token.get("priceUsd") or token.get("priceUsd24h")
                        if price is not None:
                            return float(price)
            # fallback: search by symbol (not as reliable)
            if symbol:
                url = f"{self.DEXSCREENER_BASE}/search"
                params = {"q": symbol}
                async with self._session.get(url, params=params, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        j = await resp.json()
                        # find first pair with price
                        pairs = j.get("pairs") or []
                        if pairs:
                            # choose first pair priceUsd
                            p = pairs[0]
                            price = p.get("priceUsd")
                            if price is not None:
                                return float(price)
        except Exception:
            _LOG.exception("DexScreener get_price error")
        return None

    async def get_price_history(self, chain: str, symbol: Optional[str], contract: Optional[str], start_ts: int, end_ts: int, interval: str = "60") -> Optional[List[Tuple[int, float]]]:
        # Dexscreener history endpoints are limited; not implemented here
        return None

class JupiterProvider:
    JUPITER_API = "https://quote-api.jup.ag"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None, timeout: int = 6):
        self._session = session
        self.timeout = timeout

    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def get_price(self, chain: str, symbol: Optional[str] = None, contract: Optional[str] = None, vs_currency: str = "usd") -> Optional[float]:
        # Jupiter quotes are on Solana only. For precise quotes use quote endpoint.
        if (chain or "").lower() not in ("solana", "sol"):
            return None
        await self._ensure_session()
        # Jupiter requires inputMint/outputMint; to get price in USD we may ask for SOL->token and multiply by SOL price,
        # or there's no straightforward USD quote. We'll perform a lightweight approach:
        # if contract present, fetch quote for 1 unit of token -> get output amount relative to SOL and multiply by SOL price (approx).
        try:
            if not contract:
                return None
            # endpoint: /v4/price?ids=...
            # Jupiter's public API does not have a simple /price; but quote endpoint exists: /v4/quote?inputMint=...&outputMint=...
            # Here we'll attempt to get SOL price first then compute token price via swap simulation if possible.
            # For now, attempt to query their token list to get priceUsd if available on their token info endpoint
            url = f"{self.JUPITER_API}/tokens"
            async with self._session.get(url, timeout=self.timeout) as resp:
                if resp.status == 200:
                    j = await resp.json()
                    # j is list of token objects possibly containing 'priceUsd' or similar
                    for t in j:
                        addr = t.get("address") or t.get("mint")
                        if addr and addr.lower() == contract.lower():
                            price = t.get("priceUsd") or t.get("price")
                            if price is not None:
                                try:
                                    return float(price)
                                except Exception:
                                    pass
        except Exception:
            _LOG.exception("JupiterProvider.get_price error")
        return None

    async def get_price_history(self, chain: str, symbol: Optional[str], contract: Optional[str], start_ts: int, end_ts: int, interval: str = "60") -> Optional[List[Tuple[int, float]]]:
        return None

# ---- MarketAPIAdapter (async) ----

class MarketAPIAdapter:
    """
    Async Market API adapter. Tries chain-specific providers first, then falls back to CoinGecko.
    """

    def __init__(self,
                 alchemy_key: Optional[str] = None,
                 helius_key: Optional[str] = None,
                 coingecko_key: Optional[str] = None,
                 session: Optional[aiohttp.ClientSession] = None,
                 cache_ttl: int = 15,
                 request_timeout: int = 8,
                 enable_file_cache: bool = False,
                 file_cache_dir: Optional[str] = None):
        self._session = session or aiohttp.ClientSession()
        self.cache_ttl = cache_ttl
        self._cache: Dict[Tuple[str, str], Tuple[float, float]] = {}  # (key, chain) -> (value, expire_ts)
        self._lock = asyncio.Lock()
        self.request_timeout = request_timeout

        # instantiate providers
        self.alchemy = AlchemyProvider(alchemy_key, session=self._session, timeout=request_timeout)
        self.helius = HeliusProvider(helius_key, session=self._session, timeout=request_timeout)
        self.cg = CoinGeckoProvider(coingecko_key, session=self._session, timeout=request_timeout)
        self.dex = DexScreenerProvider(session=self._session, timeout=request_timeout)
        self.jupiter = JupiterProvider(session=self._session, timeout=request_timeout)

        # ordered provider list (name attr used for debug)
        self._providers = [
            ("alchemy", self.alchemy),
            ("helius", self.helius),
            ("dexscreener", self.dex),
            ("jupiter", self.jupiter),
            ("coingecko", self.cg),
        ]
        
        # --- Add rate-limiter semaphores per provider (simple concurrency limits) ---
        # Tunable limits (adjust based on provider free-tier)
        self._rate_limits = {
            "alchemy": asyncio.Semaphore(6),
            "helius": asyncio.Semaphore(4),
            "dexscreener": asyncio.Semaphore(6),
            "jupiter": asyncio.Semaphore(4),
            "coingecko": asyncio.Semaphore(12),
        }

        self.enable_file_cache = enable_file_cache
        self.file_cache_dir = file_cache_dir or os.path.join(os.getcwd(), ".market_api_cache")
        if self.enable_file_cache:
            os.makedirs(self.file_cache_dir, exist_ok=True)

    async def close(self):
        try:
            await self._session.close()
        except Exception:
            pass

    def _cache_get(self, key: str, chain: str) -> Optional[float]:
        e = self._cache.get((key, chain))
        if not e:
            # try file cache if enabled
            if self.enable_file_cache:
                path = os.path.join(self.file_cache_dir, f"current__{chain}__{key}.json")
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            obj = json.load(f)
                        val = obj.get("value")
                        exp = obj.get("expire", 0)
                        if time.time() < exp:
                            return float(val)
                        else:
                            try: os.remove(path)
                            except Exception: pass
                    except Exception:
                        pass
            return None
        val, exp = e
        if time.time() > exp:
            try:
                del self._cache[(key, chain)]
            except Exception:
                pass
            return None
        return val

    def _cache_set(self, key: str, chain: str, value: float):
        self._cache[(key, chain)] = (float(value), time.time() + self.cache_ttl)
        if self.enable_file_cache:
            path = os.path.join(self.file_cache_dir, f"current__{chain}__{key}.json")
            try:
                _atomic_write_json(path, {"value": float(value), "expire": time.time() + self.cache_ttl})
            except Exception:
                _LOG.exception("Failed to write file cache")

    # --------- Begin: provider-meta aware resolver ----------
    async def get_price_with_meta(self,
                                  chain: Optional[str],
                                  symbol: Optional[str] = None,
                                  contract: Optional[str] = None,
                                  vs_currency: str = "usd",
                                  prefer: Optional[str] = None,
                                  request_timeout: Optional[int] = None
                                  ) -> Tuple[Optional[float], Optional[str]]:
        """
        Return (price, provider_name) — provider_name is the first provider that returned a price.
        This method preserves the provider order logic already in get_price but exposes which
        provider succeeded for observability and logging.
        """
        chain_norm = (chain or "any").lower()
        key = (contract or symbol or "").lower()
        if not key:
            return None, "no_key"

        # respect optional per-call timeout override
        timeout = request_timeout or self.request_timeout

        # consult cache first
        cached = self._cache_get(key, chain_norm)
        if cached is not None:
            # We only stored value in cache previously; we do not know provider there.
            # Keep compatibility: return (value, "cache")
            return float(cached), "cache"

        # build provider order (same logic as existing get_price)
        provider_order = []
        if prefer:
            for name, prov in self._providers:
                if name == prefer:
                    provider_order.append((name, prov))
        if chain_norm in ("solana", "sol"):
            for name in ("jupiter", "helius", "dexscreener", "coingecko", "alchemy"):
                for n, p in self._providers:
                    if n == name and (n, p) not in provider_order:
                        provider_order.append((n, p))
        else:
            for name in ("alchemy", "dexscreener", "coingecko", "helius", "jupiter"):
                for n, p in self._providers:
                    if n == name and (n, p) not in provider_order:
                        provider_order.append((n, p))

        # try each provider with optional per-provider rate limiting (semaphore)
        for name, prov in provider_order:
            sem = self._rate_limits.get(name)
            try:
                if sem is not None:
                    # acquire semaphore with timeout guard
                    # use asyncio.wait_for to protect against hanging acquires in extreme cases
                    await asyncio.wait_for(sem.acquire(), timeout=min(5, timeout))
                try:
                    # call provider.get_price with a short per-call timeout
                    coro = prov.get_price(chain=chain_norm, symbol=symbol, contract=contract, vs_currency=vs_currency)
                    try:
                        val = await asyncio.wait_for(coro, timeout=timeout)
                    except asyncio.TimeoutError:
                        # provider timed out; move to next provider
                        _LOG.warning("Provider %s timed out on key=%s", name, key)
                        val = None
                    if val is not None:
                        # store cache and return both price and provider name
                        try:
                            self._cache_set(key, chain_norm, float(val))
                        except Exception:
                            _LOG.exception("Failed setting in-memory/file cache")
                        return float(val), name
                finally:
                    if sem is not None:
                        try:
                            sem.release()
                        except Exception:
                            pass
            except asyncio.TimeoutError:
                _LOG.warning("Timeout acquiring semaphore for provider %s", name)
                continue
            except Exception:
                _LOG.exception("Error invoking provider %s", name)
                continue

        # final fallback: CoinGecko explicit call (already in ordering but double-check)
        try:
            val = await self.cg.get_price(chain=chain_norm, symbol=symbol, contract=contract, vs_currency=vs_currency)
            if val is not None:
                try:
                    self._cache_set(key, chain_norm, float(val))
                except Exception:
                    pass
                return float(val), "coingecko"
        except Exception:
            _LOG.exception("CoinGecko fallback failed in get_price_with_meta")

        return None, None


    # Backwards-compatible wrapper: keep existing get_price() API (returns just the price as before)
    async def get_price(self, chain: Optional[str], symbol: Optional[str] = None, contract: Optional[str] = None, vs_currency: str = "usd", prefer: Optional[str] = None) -> Optional[float]:
        """
        Compatibility wrapper — calls get_price_with_meta and returns only the price.
        If you prefer to directly get provider metadata, call get_price_with_meta.
        """
        price, _provider = await self.get_price_with_meta(chain=chain, symbol=symbol, contract=contract, vs_currency=vs_currency, prefer=prefer)
        return price
    # --------- End: provider-meta aware resolver ----------


    async def get_price_history(self, chain: Optional[str], symbol: Optional[str], contract: Optional[str], start_ts: int, end_ts: int, interval: str = "60") -> Optional[List[Tuple[int, float]]]:
        """
        Try providers for historical series (return list of (ts, price) tuples).
        """
        key = (contract or symbol or "").lower()
        if not key:
            return None

        # Preferred providers for history: CoinGecko (by id)
        try:
            data = await self.cg.get_price_history(chain=chain or "any", symbol=symbol, contract=contract, start_ts=start_ts, end_ts=end_ts, interval=interval)
            if data:
                return data
        except Exception:
            _LOG.exception("CoinGecko history error")

        # Dexscreener may support limited history; try it
        try:
            data = await self.dex.get_price_history(chain=chain or "any", symbol=symbol, contract=contract, start_ts=start_ts, end_ts=end_ts, interval=interval)
            if data:
                return data
        except Exception:
            _LOG.exception("DexScreener history error")

        # other providers: not implemented for history in this adapter
        return None

# End of market_api.py
