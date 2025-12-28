"""Model specifications for calibration framework."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class ParamSpec:
    """Specification for a single parameter."""
    name: str
    transform: Literal["linear", "log"]  # z-space transform
    z0: float  # Initial guess in z-space
    z_lb: float  # Lower bound in z-space
    z_ub: float  # Upper bound in z-space
    
    def z_to_phys(self, z: float) -> float:
        """Convert z-space to physical space."""
        if self.transform == "log":
            return float(np.exp(z))
        return float(z)
    
    def grad_factor(self, phys_val: float) -> float:
        """Jacobian factor dPhys/dZ."""
        if self.transform == "log":
            return phys_val  # d(exp(z))/dz = exp(z)
        return 1.0


@dataclass
class LinkedParamSpec:
    """Specification for a parameter linked to another parameter."""
    name: str
    link_to: str  # Reference to parameter name in another component, e.g., "Merton.sigma"
    
    def z_to_phys(self, z: float, phys_link: dict[str, float]) -> float:
        """Get physical value from linked parameter."""
        if self.link_to not in phys_link:
            raise KeyError(f"Linked parameter '{self.link_to}' not found in physical parameters")
        return phys_link[self.link_to]
    
    def grad_factor(self, phys_val: float) -> float:
        """Jacobian factor for linked parameter (pass-through)."""
        return 1.0  # Gradient flows through from linked source


@dataclass
class ModelSpec:
    """Specification for a single Lévy model component."""
    model_type: str  # "merton", "vg", "kou", "cgmy"
    params: list[ParamSpec | LinkedParamSpec]
    linked_params: dict[str, str] | None = None  # Maps param names to components they're linked from
    
    @staticmethod
    def _format_model_type(model_type: str) -> str:
        """Format model type to match CompositeLevyCHF convention."""
        mt = model_type.lower()
        if mt == "vg":
            return "VG"
        elif mt == "cgmy":
            return "CGMY"
        elif mt == "gbm":
            return "GBM"
        elif mt == "merton":
            return "Merton"
        elif mt == "kou":
            return "Kou"
        elif mt == "nig":
            return "NIG"
        return model_type
    
    @property
    def param_names(self) -> list[str]:
        """Full parameter names with model prefix."""
        prefix = self._format_model_type(self.model_type)
        return [f"{prefix}.{p.name}" for p in self.params if not isinstance(p, LinkedParamSpec)]
    
    @property
    def all_param_names(self) -> list[str]:
        """All parameter names including linked ones."""
        prefix = self._format_model_type(self.model_type)
        return [f"{prefix}.{p.name}" for p in self.params]
    
    @property
    def dim(self) -> int:
        """Number of independent parameters (excluding linked ones)."""
        return sum(1 for p in self.params if not isinstance(p, LinkedParamSpec))
    
    def z_to_phys(self, z: np.ndarray, full_phys: dict[str, float] | None = None) -> dict[str, float]:
        """Convert z-vector to physical parameter dict.
        
        Args:
            z: z-space parameters for independent parameters
            full_phys: Full physical parameter dict for resolving links (optional)
        """
        if len(z) != self.dim:
            raise ValueError(f"Expected {self.dim} parameters, got {len(z)}")
        result = {}
        prefix = self._format_model_type(self.model_type)
        
        z_idx = 0
        for p in self.params:
            if isinstance(p, LinkedParamSpec):
                # Resolve linked parameter if full_phys provided, else skip
                if full_phys is not None:
                    result[f"{prefix}.{p.name}"] = p.z_to_phys(0.0, full_phys)
            else:
                result[f"{prefix}.{p.name}"] = p.z_to_phys(z[z_idx])
                z_idx += 1
        return result
    
    def get_bounds(self) -> tuple[list[float], list[float]]:
        """Get z-space bounds (only for independent parameters)."""
        z_lb = [p.z_lb for p in self.params if not isinstance(p, LinkedParamSpec)]
        z_ub = [p.z_ub for p in self.params if not isinstance(p, LinkedParamSpec)]
        return z_lb, z_ub
    
    def get_z0(self) -> list[float]:
        """Get initial guess in z-space (only for independent parameters)."""
        return [p.z0 for p in self.params if not isinstance(p, LinkedParamSpec)]
    
    def grad_factors(self, phys: dict[str, float]) -> np.ndarray:
        """Get Jacobian factors for independent parameters."""
        factors = np.zeros(self.dim, dtype=float)
        prefix = self._format_model_type(self.model_type)
        
        z_idx = 0
        for p in self.params:
            if not isinstance(p, LinkedParamSpec):
                key = f"{prefix}.{p.name}"
                factors[z_idx] = p.grad_factor(phys[key])
                z_idx += 1
        return factors
    
    def to_component_dict(self, phys: dict[str, float]) -> dict:
        """Convert physical params to component dict for CompositeLevyCHF."""
        params = {}
        prefix = self._format_model_type(self.model_type)
        for p in self.params:
            key = f"{prefix}.{p.name}"
            params[p.name] = phys[key]
        return {"type": self.model_type, "params": params}


# Define all supported models
MERTON_SPEC = ModelSpec(
    model_type="merton",
    params=[
        ParamSpec("sigma", "log", np.log(0.10), np.log(0.01), np.log(1.0)),
        ParamSpec("lam", "log", np.log(0.5), np.log(1e-4), np.log(10.0)),
        ParamSpec("muJ", "linear", -0.3, -1.0, 1.0),
        ParamSpec("sigmaJ", "log", np.log(0.1), np.log(0.01), np.log(2.0)),
    ]
)

VG_SPEC = ModelSpec(
    model_type="vg",
    params=[
        ParamSpec("theta", "linear", -0.2, -2.0, 6.0),
        ParamSpec("sigma", "log", np.log(0.12), np.log(0.01), np.log(1.0)),
        ParamSpec("nu", "log", np.log(0.05), np.log(1e-3), np.log(5.0)),
    ]
)

# VG with linked sigma (for use in combinations with Merton/Kou)
VG_LINKED_SPEC = ModelSpec(
    model_type="vg",
    params=[
        ParamSpec("theta", "linear", -0.2, -2.0, 6.0),
        LinkedParamSpec("sigma", "Merton.sigma"),  # Will be linked to Merton
        ParamSpec("nu", "log", np.log(0.05), np.log(1e-3), np.log(5.0)),
    ]
)

VG_LINKED_KOU_SPEC = ModelSpec(
    model_type="vg",
    params=[
        ParamSpec("theta", "linear", -0.2, -2.0, 6.0),
        LinkedParamSpec("sigma", "Kou.sigma"),  # Will be linked to Kou
        ParamSpec("nu", "log", np.log(0.05), np.log(1e-3), np.log(5.0)),
    ]
)

KOU_SPEC = ModelSpec(
    model_type="kou",
    params=[
        ParamSpec("sigma", "log", np.log(0.16), np.log(0.03), np.log(1.0)),
        ParamSpec("lam", "log", np.log(0.9), np.log(1e-4), np.log(10.0)),
        ParamSpec("p", "linear", 0.35, 0.0, 1.0),
        ParamSpec("eta1", "log", np.log(12.0), np.log(1.0), np.log(50.0)),
        ParamSpec("eta2", "log", np.log(6.0), np.log(1.0), np.log(50.0)),
    ]
)

CGMY_SPEC = ModelSpec(
    model_type="cgmy",
    params=[
        ParamSpec("C", "log", np.log(0.02), np.log(0.001), np.log(5.0)),
        ParamSpec("G", "log", np.log(5.0), np.log(2.0), np.log(50.0)),
        ParamSpec("M", "log", np.log(5.0), np.log(2.0), np.log(50.0)),
        ParamSpec("Y", "linear", 1.2, 0.5, 1.9),
    ]
)

NIG_SPEC = ModelSpec(
    model_type="nig",
    params=[
        ParamSpec("alpha", "log", np.log(15.0), np.log(5.0), np.log(50.0)),
        ParamSpec("beta", "linear", -4.0, -10.0, 10.0),
        ParamSpec("delta", "log", np.log(0.5), np.log(0.1), np.log(5.0)),
        ParamSpec("mu", "linear", 0.01, -0.5, 0.5),
    ]
)

# NIG with linked sigma (for use in combinations like Kou+NIG)
# Note: NIG doesn't have sigma, but this would be for future extensions
NIG_LINKED_KOU_SPEC = ModelSpec(
    model_type="nig",
    params=[
        ParamSpec("alpha", "log", np.log(15.0), np.log(5.0), np.log(50.0)),
        ParamSpec("beta", "linear", -4.0, -10.0, 10.0),
        ParamSpec("delta", "log", np.log(0.5), np.log(0.1), np.log(5.0)),
        ParamSpec("mu", "linear", 0.01, -0.5, 0.5),
    ]
)

# Registry of all models
MODEL_REGISTRY = {
    "merton": MERTON_SPEC,
    "vg": VG_SPEC,
    "kou": KOU_SPEC,
    "cgmy": CGMY_SPEC,
    "nig": NIG_SPEC,
}


@dataclass
class CompositeModelSpec:
    """Specification for a composite model with q parameter."""
    components: list[ModelSpec]
    
    def __init__(self, component_names: list[str]):
        """Build from list of model names.
        
        For combinations with two sigmas (merton_vg, kou_vg), automatically uses
        linked specs to share the sigma parameter across components.
        """
        # Handle linked sigma combinations
        if set(component_names) == {"merton", "vg"}:
            self.components = [MODEL_REGISTRY["merton"], VG_LINKED_SPEC]
        elif set(component_names) == {"kou", "vg"}:
            self.components = [MODEL_REGISTRY["kou"], VG_LINKED_KOU_SPEC]
        elif set(component_names) == {"kou", "nig"}:
            self.components = [MODEL_REGISTRY["kou"], NIG_LINKED_KOU_SPEC]
        else:
            self.components = [MODEL_REGISTRY[name] for name in component_names]
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        comp_names = [c.model_type for c in self.components]
        return "_".join(comp_names) + "_q"
    
    @property
    def dim(self) -> int:
        """Total dimension including q."""
        return 1 + sum(c.dim for c in self.components)
    
    @property
    def param_names(self) -> list[str]:
        """All parameter names including q."""
        names = ["q"]
        for comp in self.components:
            names.extend(comp.param_names)
        return names
    
    def z_to_phys(self, z: np.ndarray) -> dict[str, float]:
        """Convert full z-vector to physical parameters."""
        if len(z) != self.dim:
            raise ValueError(f"Expected {self.dim} parameters, got {len(z)}")
        
        phys = {"q": float(z[0])}  # q is always linear
        idx = 1
        
        # First pass: add all independent parameters
        for comp in self.components:
            comp_z = z[idx:idx + comp.dim]
            phys.update(comp.z_to_phys(comp_z, full_phys=None))
            idx += comp.dim
        
        # Second pass: resolve linked parameters using full phys dict
        idx = 1
        for comp in self.components:
            comp_z = z[idx:idx + comp.dim]
            comp_phys_linked = comp.z_to_phys(comp_z, full_phys=phys)
            phys.update(comp_phys_linked)
            idx += comp.dim
        
        return phys
    
    def get_bounds(self) -> tuple[list[float], list[float]]:
        """Get full z-space bounds."""
        z_lb = [-0.10]  # q lower bound
        z_ub = [0.10]   # q upper bound
        for comp in self.components:
            lb, ub = comp.get_bounds()
            z_lb.extend(lb)
            z_ub.extend(ub)
        return z_lb, z_ub
    
    def get_z0(self) -> np.ndarray:
        """Get initial guess."""
        z0 = [0.0]  # q initial guess
        for comp in self.components:
            z0.extend(comp.get_z0())
        return np.array(z0, dtype=float)
    
    def grad_factors(self, phys: dict[str, float]) -> np.ndarray:
        """Get Jacobian factors (dPhys/dZ) for all parameters."""
        factors = np.zeros(self.dim, dtype=float)
        factors[0] = 1.0  # q is linear
        idx = 1
        for comp in self.components:
            factors[idx:idx + comp.dim] = comp.grad_factors(phys)
            idx += comp.dim
        return factors
    
    def to_component_list(self, phys: dict[str, float]) -> list[dict]:
        """Convert to component list for CompositeLevyCHF."""
        return [comp.to_component_dict(phys) for comp in self.components]


# Preset generation models with default parameters
GENERATION_MODELS = {
    "merton": {"components": [{"type": "merton", "params": {"sigma": 0.15, "lam": 0.5, "muJ": -0.2, "sigmaJ": 0.1}}]},
    "kou": {"components": [{"type": "kou", "params": {"sigma": 0.16, "lam": 0.9, "p": 0.35, "eta1": 12.0, "eta2": 6.0}}]},
    "vg": {"components": [{"type": "vg", "params": {"theta": -0.2, "sigma": 0.12, "nu": 0.05}}]},
    "cgmy": {"components": [{"type": "cgmy", "params": {"C": 0.5, "G": 10.0, "M": 10.0, "Y": 0.5}}]},
    "nig": {"components": [{"type": "nig", "params": {"alpha": 15.0, "beta": -4.0, "delta": 0.55, "mu": 0.02}}]},
    "kou_vg": {"components": [
        {"type": "kou", "params": {"sigma": 0.16, "lam": 0.9, "p": 0.35, "eta1": 12.0, "eta2": 6.0}},
        {"type": "vg", "params": {"theta": -0.2, "sigma": 0.12, "nu": 0.05}}
    ]},
    "merton_vg": {"components": [
        {"type": "merton", "params": {"sigma": 0.15, "lam": 0.5, "muJ": -0.2, "sigmaJ": 0.1}},
        {"type": "vg", "params": {"theta": -0.2, "sigma": 0.12, "nu": 0.05}}
    ]},
    "cgmy_vg": {"components": [
        {"type": "cgmy", "params": {"C": 0.5, "G": 10.0, "M": 10.0, "Y": 0.5}},
        {"type": "vg", "params": {"theta": -0.2, "sigma": 0.12, "nu": 0.05}}
    ]},
    "kou_nig": {"components": [
        {"type": "kou", "params": {"sigma": 0.16, "lam": 0.9, "p": 0.35, "eta1": 12.0, "eta2": 6.0}},
        {"type": "nig", "params": {"alpha": 15.0, "beta": -4.0, "delta": 0.55, "mu": 0.02}}
    ]},
}
