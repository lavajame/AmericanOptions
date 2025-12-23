import os
import sys


# Ensure repo root is on sys.path so tests can import the local `american_options` package
# regardless of pytest's import mode / rootdir heuristics.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
