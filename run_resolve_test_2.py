#!/usr/bin/env python3
"""
run_resolve_test_2.py

Standalone resolver tester — does NOT import your monolith. Uses BASE_RPC_URL in env.

Usage:
  set -a; . ./creds.env; set +a
  source .venv2/bin/activate
  python3 run_resolve_test_2.py alice alice.base example.base
"""

import os, sys
from decimal import Decimal

# small helper to create a Web3 instance
def connect_base_web3_local():
    try:
        from web3 import Web3, HTTPProvider
    except Exception as e:
        raise RuntimeError("web3.py required (pip install web3)") from e
    rpc = os.getenv("BASE_RPC_URL")
    if not rpc:
        raise RuntimeError("BASE_RPC_URL not set in environment")
    w3 = Web3(HTTPProvider(rpc, request_kwargs={"timeout": 30}))
    # modern method name
    try:
        ok = w3.is_connected()
    except Exception:
        try:
            ok = w3.isConnected()
        except Exception:
            ok = False
    if not ok:
        raise RuntimeError(f"Web3 cannot connect to RPC at {rpc}")
    return w3

def resolve_basename_local(name: str) -> str | None:
    """
    Local copy of resolver that does not depend on monolith import.
    Returns a checksum address string or None.
    """
    if not name or not isinstance(name, str):
        return None

    from web3 import Web3

    # normalize
    n = name.strip().rstrip(".")
    if n.endswith(".base.eth"):
        n = n[: -len(".base.eth")]
    if n.endswith(".base"):
        label = n
    else:
        label = n
    full_name = label if label.endswith(".base") else f"{label}.base"

    # ENS namehash implementation
    def namehash(name_str: str) -> bytes:
        node = b"\x00" * 32
        if not name_str:
            return node
        labels = name_str.split(".")
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

    # default resolver address (canonical Sepolia L2Resolver) — override with env var if needed
    default_resolver = os.getenv("BASE_SEPOLIA_L2RESOLVER", os.getenv("BASE_L2RESOLVER", "0x6533C94869D28fAA8dF77cc63f9e2b2D6Cf77eBA"))

    w3 = connect_base_web3_local()

    node_bytes = namehash(full_name)

    try:
        resolver_addr = Web3.to_checksum_address(default_resolver)
    except Exception:
        resolver_addr = default_resolver

    try:
        resolver = w3.eth.contract(address=resolver_addr, abi=resolver_abi)
        # call with bytes32
        resolved = resolver.functions.addr(node_bytes).call()
        # resolved returned as hex string or address; handle both
        if not resolved:
            return None
        # if returned as bytes-like, convert
        if isinstance(resolved, (bytes, bytearray)):
            resolved_hex = "0x" + resolved.hex()
            if int(resolved_hex, 16) == 0:
                return None
            return Web3.to_checksum_address(resolved_hex)
        # if string
        if isinstance(resolved, str):
            if resolved == "0x0000000000000000000000000000000000000000":
                return None
            return Web3.to_checksum_address(resolved)
        return None
    except Exception as e:
        # return None but include debug info in raised RuntimeError for interactive runs
        raise RuntimeError(f"resolver call failed: {e}")

def main():
    # args as list of names to test
    args = sys.argv[1:] or ["alice", "alice.base", "example.base"]
    print("Using BASE_RPC_URL =", os.getenv("BASE_RPC_URL"))
    try:
        w3 = connect_base_web3_local()
        print("Connected to RPC, latest block:", w3.eth.block_number)
    except Exception as e:
        print("Error connecting to RPC:", e)
        sys.exit(2)

    for n in args:
        try:
            res = resolve_basename_local(n)
        except Exception as e:
            print(f"{n} -> ERROR: {e}")
            continue
        print(f"{n} -> {res}")

if __name__ == "__main__":
    main()
