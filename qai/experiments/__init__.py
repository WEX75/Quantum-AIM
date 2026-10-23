"""One file per experiment (exp_<topic>.py) so teammates never edit the same file.

Each file exposes main(argv=None) and is run with: python3 -m qai.experiments.exp_<topic>
Shared modules (model.py, noise.py, data.py, train.py) change only through small, reviewed PRs.
"""
