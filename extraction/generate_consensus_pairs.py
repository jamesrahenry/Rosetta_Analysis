#!/usr/bin/env python3
"""Wrapper — consensus pair generation lives in rosetta_tools now.

Usage:
    python src/generate_consensus_pairs.py --all --n-pairs 100
    python src/generate_consensus_pairs.py --concept credibility --n-pairs 2
"""
import sys
from pathlib import Path

from rosetta_tools.consensus_generator import main

if __name__ == "__main__":
    # Default output to caz_scaling/data/ when called from here
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", str(Path(__file__).parent.parent / "data")])
    main()
