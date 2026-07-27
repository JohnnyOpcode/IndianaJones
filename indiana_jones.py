import os
import sys

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Workaround for llama-cpp-python CUDA DLL loading issue on Windows
cuda_paths = [
    os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"),
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
]

for cp in cuda_paths:
    if not cp:
        continue
    bin_x64 = os.path.join(cp, "bin", "x64")
    bin_path = os.path.join(cp, "bin")
    for p in [bin_x64, bin_path]:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

# Add virtualenv or site-packages llama_cpp/lib directory to DLL search path if present
venv_lib = os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages", "llama_cpp", "lib")
if os.path.exists(venv_lib):
    try:
        os.add_dll_directory(venv_lib)
    except Exception:
        pass

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