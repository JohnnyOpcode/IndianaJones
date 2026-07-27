import os
import sys

def setup_cuda_dlls():
    """Configures Windows CUDA and llama_cpp DLL directories for ctypes loading."""
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

    # Add llama_cpp/lib directory if in virtualenv or site-packages
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_lib = os.path.join(base_dir, ".venv", "Lib", "site-packages", "llama_cpp", "lib")
    if os.path.exists(venv_lib):
        try:
            os.add_dll_directory(venv_lib)
        except Exception:
            pass

setup_cuda_dlls()
