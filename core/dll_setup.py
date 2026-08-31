import os
import sys

# Configure UTF-8 stdout/stderr with resilient error handling and line buffering on Windows
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

def setup_cuda_dlls():
    """Configures Windows CUDA and llama_cpp DLL directories for ctypes loading."""
    cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    cuda_paths = []
    
    if "CUDA_PATH" in os.environ:
        cuda_paths.append(os.environ["CUDA_PATH"])
        
    if os.path.exists(cuda_root):
        for entry in os.listdir(cuda_root):
            full_path = os.path.join(cuda_root, entry)
            if os.path.isdir(full_path):
                cuda_paths.append(full_path)
                
    # Fallback paths
    cuda_paths.extend([
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    ])

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
                if p not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

    # Add llama_cpp/lib directory if in virtualenv or site-packages
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_libs = [
        os.path.join(base_dir, ".venv", "Lib", "site-packages", "llama_cpp", "lib"),
        os.path.join(sys.prefix, "Lib", "site-packages", "llama_cpp", "lib"),
    ]
    for venv_lib in venv_libs:
        if os.path.exists(venv_lib):
            try:
                os.add_dll_directory(venv_lib)
            except Exception:
                pass
    # Prevent OpenMP and Tokenizer initialization deadlocks on Windows with PyTorch + llama_cpp
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

setup_cuda_dlls()
