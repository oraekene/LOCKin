#!/usr/bin/env python3
import os, runpy, sys, json

# load monolith to access resolve_basename
MONO = os.path.join(os.getcwd(), "main_bot.py")  # adjust filename if different
ns = runpy.run_path(MONO, run_name="__main__")

if "resolve_basename" not in ns:
    print("resolve_basename not found in monolith. Ensure function added and MONO path is correct.")
    sys.exit(2)

resolve_basename = ns["resolve_basename"]

tests = ["alice", "alice.base", "example.base", "notrealname"]
for t in tests:
    try:
        addr = resolve_basename(t)
    except Exception as e:
        addr = f"ERROR: {e}"
    print(f"{t} -> {addr}")
