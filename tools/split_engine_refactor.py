from __future__ import annotations

from pathlib import Path
import re

root = Path(r"C:\workspace\AmericanOptions\american_options")
engine = root / "engine.py"
text = engine.read_text(encoding="utf-8")

def slice_between(start_pat: str, end_pat: str) -> str:
    s = re.search(start_pat, text, flags=re.M)
    if not s:
        raise RuntimeError(f"start pattern not found: {start_pat}")
    e = re.search(end_pat, text, flags=re.M)
    if not e:
        raise RuntimeError(f"end pattern not found: {end_pat}")
    if e.start() <= s.start():
        raise RuntimeError("end before start")
    return text[s.start():e.start()]

block_divs = slice_between(r"^def cash_divs_to_proportional_divs\(", r"^def softmax\(")
block_num = slice_between(r"^def softmax\(", r"^# -+ #\n# 2\. Abstract base class")
block_base = slice_between(r"^class CharacteristicFunction:\n", r"^# -+ #\n# 2\.x Put-Call Duality")
block_dual = slice_between(r"^class DualModel\(CharacteristicFunction\):\n", r"^class COSPricer:\n")
block_pricer = slice_between(r"^class COSPricer:\n", r"^# -+ #\n# 3\. Concrete model implementations")

model_starts = [
    ("gbm", r"^class GBMCHF\(CharacteristicFunction\):"),
    ("merton", r"^class MertonCHF\(CharacteristicFunction\):"),
    ("kou", r"^class KouCHF\(CharacteristicFunction\):"),
    ("vg", r"^class VGCHF\(CharacteristicFunction\):"),
    ("cgmy", r"^class CGMYCHF\(CharacteristicFunction\):"),
    ("composite", r"^class CompositeLevyCHF\(CharacteristicFunction\):"),
]

positions: list[tuple[str, int]] = []
for name, pat in model_starts:
    m = re.search(pat, text, flags=re.M)
    if not m:
        raise RuntimeError(f"model start not found: {name}")
    positions.append((name, m.start()))
positions.sort(key=lambda t: t[1])

vol_marker = "# --------------------------------------------------------------------------- #\n# 4. Volatility inversion helper"
vol_ix = text.find(vol_marker)
if vol_ix == -1:
    raise RuntimeError("vol inversion marker not found")

blocks_models: dict[str, str] = {}
for i, (name, start) in enumerate(positions):
    end = positions[i + 1][1] if i + 1 < len(positions) else vol_ix
    blocks_models[name] = text[start:end]

block_implvol = slice_between(r"^def invert_vol_for_american_price\(", r"\Z")

(root / "models").mkdir(exist_ok=True)
(root / "cos").mkdir(exist_ok=True)

(root / "dividends.py").write_text(
    '"""Dividend utilities.\n\nProject-wide convention: discrete dividends are provided as cash amounts (mean, std)\nin spot currency. Internally we convert to proportional multiplicative factors.\n"""\n\nfrom __future__ import annotations\n\nfrom typing import Dict, Tuple\n\nimport numpy as np\n\n'
    + block_divs
    + "\n",
    encoding="utf-8",
)

(root / "numerics.py").write_text(
    '"""Numerical helpers (smooth max, stability utilities)."""\n\nfrom __future__ import annotations\n\nimport numpy as np\n\n'
    + block_num
    + "\n",
    encoding="utf-8",
)

(root / "base_cf.py").write_text(
    '"""Characteristic-function base model API.\n\nContains:\n- CharacteristicFunction base class\n- dividend-aware increment handling\n- parameter-sensitivity hooks\n- convenience wrappers for COS pricing\n"""\n\nfrom __future__ import annotations\n\nfrom typing import Any, Dict, Tuple\n\nimport numpy as np\n\nfrom .dividends import cash_divs_to_proportional_divs, _dividend_adjustment, _dividend_adjustment_window\n\n'
    + block_base
    + "\n",
    encoding="utf-8",
)

(root / "models" / "duality.py").write_text(
    '"""Put-call duality wrapper model."""\n\nfrom __future__ import annotations\n\nfrom typing import Tuple\n\nimport numpy as np\n\nfrom ..base_cf import CharacteristicFunction\n\n'
    + block_dual
    + "\n",
    encoding="utf-8",
)

(root / "cos" / "pricer.py").write_text(
    '"""COS pricing engine (European + American rollback)."""\n\nfrom __future__ import annotations\n\nfrom collections import OrderedDict\nfrom dataclasses import dataclass\nfrom typing import Any, Dict, Hashable, Iterable, Optional, Tuple\n\nimport numpy as np\n\nfrom ..base_cf import CharacteristicFunction\nfrom ..dividends import _dividend_adjustment, _dividend_adjustment_window\nfrom ..events import DiscreteEventJump\nfrom .. import numerics\nfrom ..models.duality import DualModel\n\n'
    + block_pricer
    + "\n",
    encoding="utf-8",
)

for mod, header in [
    ("gbm", '"""GBM (Black-Scholes) characteristic function model."""'),
    ("merton", '"""Merton jump-diffusion characteristic function model."""'),
    ("kou", '"""Kou double-exponential jump-diffusion characteristic function model."""'),
    ("vg", '"""Variance-Gamma characteristic function model."""'),
    ("cgmy", '"""CGMY characteristic function model."""'),
    ("composite", '"""Composite exponential-Levy characteristic function model."""'),
]:
    extra = "" if mod not in ("cgmy",) else "from scipy.special import gamma as sp_gamma\n\n"
    (root / "models" / f"{mod}.py").write_text(
        header
        + "\n\nfrom __future__ import annotations\n\nfrom typing import Any, Dict, Tuple\n\nimport numpy as np\n"
        + ("\n" + extra if extra else "\n")
        + "from ..base_cf import CharacteristicFunction\nfrom ..dividends import _dividend_adjustment\n\n"
        + blocks_models[mod]
        + "\n",
        encoding="utf-8",
    )

(root / "implied_vol.py").write_text(
    '"""Volatility inversion and proxy helpers."""\n\nfrom __future__ import annotations\n\nfrom typing import Dict, Tuple\n\nimport numpy as np\nimport scipy.optimize as opt\n\nfrom .base_cf import CharacteristicFunction\nfrom .models.gbm import GBMCHF\n\n'
    + block_implvol
    + "\n",
    encoding="utf-8",
)

(root / "models" / "__init__.py").write_text(
    '"""Model exports."""\n\nfrom .gbm import GBMCHF\nfrom .merton import MertonCHF\nfrom .kou import KouCHF\nfrom .vg import VGCHF\nfrom .cgmy import CGMYCHF\nfrom .composite import CompositeLevyCHF\nfrom .duality import DualModel\n\n__all__ = [\n    "GBMCHF",\n    "MertonCHF",\n    "KouCHF",\n    "VGCHF",\n    "CGMYCHF",\n    "CompositeLevyCHF",\n    "DualModel",\n]\n'
,
    encoding="utf-8",
)

print("Split complete")
