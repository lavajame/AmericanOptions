"""Vectorised option-pricing engine (public shim).

This project used to keep models, COS pricer, dividend helpers, and implied-vol tools
in a single large module. It has been split into smaller modules under:
- american_options.dividends
- american_options.numerics
- american_options.base_cf
- american_options.cos.pricer
- american_options.models.*
- american_options.implied_vol

This file remains as a compatibility layer so existing imports keep working:
    from american_options.engine import COSPricer, GBMCHF, ...
"""

from __future__ import annotations

# Re-export core APIs
from .base_cf import CharacteristicFunction
from .cos.pricer import COSPricer
from .dividends import (
    cash_divs_to_proportional_divs,
    forward_price,
)
from .implied_vol import (
    equivalent_gbm,
    invert_vol_for_american_price,
)
from .models import (
    CGMYCHF,
    CompositeLevyCHF,
    GBMCHF,
    KouCHF,
    MertonCHF,
    VGCHF,
)
from .models.duality import DualModel
from .numerics import (
    SOFTMAX_FN,
    softmax,
    softmax_pair,
    softmax_sqrt_ab,
)

__all__ = [
    # Base
    "CharacteristicFunction",
    # Pricer
    "COSPricer",
    # Models
    "GBMCHF",
    "MertonCHF",
    "KouCHF",
    "VGCHF",
    "CGMYCHF",
    "CompositeLevyCHF",
    "DualModel",
    # Dividends
    "cash_divs_to_proportional_divs",
    "forward_price",
    # Numerics
    "softmax",
    "softmax_sqrt_ab",
    "softmax_pair",
    "SOFTMAX_FN",
    # Helpers
    "invert_vol_for_american_price",
    "equivalent_gbm",
]
