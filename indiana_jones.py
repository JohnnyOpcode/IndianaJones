import os
import sys

# Ensure UTF-8 output encoding with line buffering and charmap resilience
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

# Setup Windows CUDA DLL search paths before importing llama_cpp
import core.dll_setup

from core.config import ExplorerConfig, parse_args
from core.explorer import LatentExplorer
from engines.cohomology import CohomologyEngine
from engines.novelty import NoveltyEngine, RepulsionLogitsProcessor
from engines.steering import VectorSteering
from engines.graph import ExpeditionGraph


def main():
    config = parse_args()
    explorer = LatentExplorer(config)
    explorer.start_expedition()


if __name__ == "__main__":
    main()