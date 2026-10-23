PY ?= python3

.PHONY: install data fraud fraud-fast hybrid-fast check test test-fast qai qai-fast qhack summary clean sync merge

install:
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

# Download the fraud dataset once (OpenML 42175 -> data/creditcard.csv) and verify it
data:
	$(PY) scripts/get_data.py

# Quantum AI on the real fraud data (needs `make data`)
fraud-fast:
	$(PY) -m qai.bench --dataset creditcard --fast

fraud:
	$(PY) -m qai.bench --dataset creditcard --qubits 4 --layers 3 --epochs 30

hybrid-fast:
	$(PY) -m qai.experiments.exp_hybrid_torch --dataset creditcard --fast

check:
	$(PY) run.py check

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

# --- git team workflow (see docs/GIT.md) ---
# Bring the latest main into your branch. Run every time you start something new.
sync:
	git fetch origin && git rebase origin/main

# Tests, rebase on main, push, open + squash-merge a PR, back on a fresh branch from main.
merge:
	$(PY) -m pytest -m "not slow" -x -q
	git fetch origin && git rebase origin/main
	git push -u origin HEAD --force-with-lease
	gh pr create --fill --base main || true
	gh pr merge --squash --delete-branch
	git checkout main && git pull --ff-only
