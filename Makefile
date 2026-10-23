PY ?= python3

.PHONY: install test test-fast qai qai-fast qhack summary clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow" -x

# Full Quantum AI benchmark (classical baselines, clean vs noise-aware VQC, ZNE)
qai:
	$(PY) -m qai.bench --dataset moons --qubits 4 --layers 3 --epochs 40

# 30-second smoke run, use this while iterating
qai-fast:
	$(PY) -m qai.bench --fast

# Quantum Hacking: BB84 intercept-resend + Trojan-horse sweeps
qhack:
	$(PY) -m qhack.bench

# One-screen summary of the latest results (cheap to read for Claude Code)
summary:
	$(PY) -m qai.summary

clean:
	rm -rf results/*.json results/*.png .pytest_cache __pycache__ */__pycache__
