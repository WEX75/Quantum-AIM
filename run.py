"""Cross-platform task runner (works on Windows without `make`).

    python run.py install     # install dependencies
    python run.py test        # all tests (~15s)
    python run.py test-fast   # skip slow tests, stop at first failure
    python run.py qai-fast    # 15s Quantum AI smoke run
    python run.py qai         # full Quantum AI benchmark (~2.5 min)
    python run.py summary     # one-screen results summary
    python run.py check       # verify the environment is ready
"""
from __future__ import annotations

import subprocess
import sys

PY = sys.executable

TASKS = {
    "install": [PY, "-m", "pip", "install", "-r", "requirements.txt"],
    "data": [PY, "scripts/get_data.py"],
    "fraud-fast": [PY, "-m", "qai.bench", "--dataset", "creditcard", "--fast"],
    "hybrid-fast": [PY, "-m", "qai.experiments.exp_hybrid_torch", "--dataset", "creditcard", "--fast"],
    "test": [PY, "-m", "pytest"],
    "test-fast": [PY, "-m", "pytest", "-m", "not slow", "-x"],
    "qai": [PY, "-m", "qai.bench", "--dataset", "moons", "--qubits", "4", "--layers", "3",
            "--epochs", "40"],
    "qai-fast": [PY, "-m", "qai.bench", "--fast"],
    "summary": [PY, "-m", "qai.summary"],
}


def check() -> int:
    ok = sys.version_info >= (3, 10)
    print(f"python {sys.version.split()[0]}: {'ok' if ok else 'NEED >= 3.10'}")
    for mod in ["pennylane", "torch", "numpy", "scipy", "pandas", "sklearn", "matplotlib", "pytest"]:
        try:
            m = __import__(mod)
            print(f"{mod} {getattr(m, '__version__', '')}: ok")
        except ImportError:
            ok = False
            print(f"{mod}: MISSING -> python run.py install")
    from pathlib import Path
    data = Path("data/creditcard.csv").exists()
    print(f"data/creditcard.csv: {'ok' if data else 'missing -> make data'}")
    print("READY" if ok and data else "NOT READY")
    return 0 if ok and data else 1


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else ""
    if task == "check":
        return check()
    if task not in TASKS:
        print(__doc__)
        return 1
    return subprocess.call(TASKS[task] + sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
