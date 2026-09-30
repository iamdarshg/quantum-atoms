"""GPU-capable time-dependent Dirac equation demonstrator."""
from .dirac import DiracGrid, DiracPropagator, parse_formula, demo_geometry
from .many_electron import ErrorBounds, MolecularDiracHartree, MolecularDiracHartreeFock, max_separation_sphere

__all__ = ["DiracGrid", "DiracPropagator", "parse_formula", "demo_geometry",
           "ErrorBounds", "MolecularDiracHartree", "MolecularDiracHartreeFock", "max_separation_sphere"]
