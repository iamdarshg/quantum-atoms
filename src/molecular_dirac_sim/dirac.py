"""3-D time-dependent Dirac equation in a prescribed electrostatic field.

Units are atomic units (hbar = electron mass = |electron charge| = 1;
 c ~= 137.036). A Strang split applies the local scalar potential and the
 exact free-particle Dirac propagator in Fourier space. The model follows one
 electron in a classical field from fixed/prescribed nuclei; it does not model
 molecular ground states, electron-electron correlation, or spontaneous
 chemical reactions.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import re
from typing import Sequence

import numpy as np

try:
    import torch
except ImportError:  # optional; NumPy/SciPy fallback remains runnable
    torch = None

C_AU = 137.035999084
_ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
Z = {symbol: i + 1 for i, symbol in enumerate(_ELEMENTS)}


def parse_formula(formula: str) -> Counter:
    """Parse a simple formula (e.g. H2O, CH4, C6H6); parentheses are supported."""
    tokens = re.findall(r"[A-Z][a-z]?|\d+|[()]", formula)
    if "".join(tokens) != formula or not tokens:
        raise ValueError(f"Invalid formula: {formula!r}")
    stack = [Counter()]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "(":
            stack.append(Counter()); i += 1
        elif t == ")":
            if len(stack) == 1: raise ValueError("Unmatched )")
            group = stack.pop(); i += 1
            n = int(tokens[i]) if i < len(tokens) and tokens[i].isdigit() else 1
            if n != 1 or (i < len(tokens) and tokens[i].isdigit()): i += 1
            stack[-1].update({k: v*n for k, v in group.items()})
        elif t.isdigit():
            raise ValueError("Count must follow an element or parenthesized group")
        else:
            if t not in Z: raise ValueError(f"Unsupported element {t}; extend the element table")
            i += 1
            n = int(tokens[i]) if i < len(tokens) and tokens[i].isdigit() else 1
            if i < len(tokens) and tokens[i].isdigit(): i += 1
            stack[-1][t] += n
    if len(stack) != 1: raise ValueError("Unmatched (")
    return stack[0]


def demo_geometry(formula: str) -> tuple[list[str], np.ndarray]:
    """Return illustrative final coordinates in bohr for H2O/CH4 only.

    Other formulas need user-supplied XYZ coordinates. The coordinates are
    idealized pedagogical geometries, not optimized quantum-chemistry results.
    """
    f = parse_formula(formula)
    if formula == "H2O":
        r = 1.81; a = math.radians(104.5/2)
        return ["O", "H", "H"], np.array([[0,0,0], [r*math.sin(a),r*math.cos(a),0],[-r*math.sin(a),r*math.cos(a),0]], float)
    if formula == "CH4":
        r = 2.06
        return ["C", "H", "H", "H", "H"], np.array([[0,0,0], [r,r,r], [r,-r,-r], [-r,r,-r], [-r,-r,r]], float)/math.sqrt(3)
    raise ValueError(f"No built-in illustrative geometry for {formula}; provide final XYZ coordinates (atoms {dict(f)})")


@dataclass
class DiracGrid:
    shape: tuple[int, int, int] = (16, 16, 16)
    spacing_bohr: float = 0.5
    device: str = "cuda"


class DiracPropagator:
    """Split-step propagator for i*d_t psi=[c alpha.(p+A)+beta*c²+V]psi.

    ``A`` is a spatially uniform vector potential in atomic units. Scalar
    potential is the electron potential energy, sampled on the grid.
    Torch uses CUDA when requested/available; otherwise NumPy is used.
    """
    def __init__(self, grid: DiracGrid = DiracGrid(), device: str | None = None):
        self.grid = grid
        want = device or grid.device
        if want == "cuda" and (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("CUDA requested but PyTorch with a CUDA-enabled runtime is not available")
        self.use_torch = torch is not None and want != "cpu" and (want == "cuda" or (want == "auto" and torch.cuda.is_available()))
        self.device = torch.device("cuda" if want == "auto" and self.use_torch else (want if self.use_torch else "cpu")) if self.use_torch else "cpu"
        n = grid.shape; dx = grid.spacing_bohr
        axes = [2*np.pi*np.fft.fftfreq(m, d=dx) for m in n]
        self.p = np.stack(np.meshgrid(*axes, indexing="ij"), axis=0)
        self.dx = dx
        # Dirac representation: alpha_i=[[0,sigma_i],[sigma_i,0]], beta=diag(I,-I).
        self.sigma = [np.array([[0,1],[1,0]],complex), np.array([[0,-1j],[1j,0]],complex), np.diag([1,-1]).astype(complex)]

    def _kinetic(self, psi, dt: float, vector_potential: Sequence[float]):
        p = self.p - np.asarray(vector_potential, float)[:, None, None, None]
        p2 = np.sum(p*p, axis=0); energy = np.sqrt(C_AU**4 + C_AU**2*p2)
        if self.use_torch:
            P = torch.as_tensor(p, dtype=torch.float64, device=self.device)
            E = torch.as_tensor(energy, dtype=torch.float64, device=self.device)
            kpsi = torch.fft.fftn(psi, dim=(-3,-2,-1))
            upper, lower = kpsi[..., :2, :, :, :], kpsi[..., 2:, :, :, :]
            # sigma.p coupling, with exact closed-form exponential of H_free.
            sp = sum(torch.as_tensor(s, dtype=torch.complex128, device=self.device)[...,None,None,None]*P[i] for i,s in enumerate(self.sigma))
            coupled = torch.cat((C_AU*torch.einsum('abxyz,...bxyz->...axyz', sp, lower), C_AU*torch.einsum('abxyz,...bxyz->...axyz', sp, upper)), dim=-4)
            beta_psi = torch.cat((upper, -lower), dim=-4)
            co = torch.cos(E*dt); si = torch.sin(E*dt)/E
            out = co*kpsi - 1j*si*(coupled + C_AU**2*beta_psi)
            return torch.fft.ifftn(out, dim=(-3,-2,-1))
        kpsi = np.fft.fftn(psi, axes=(-3,-2,-1)); upper, lower = kpsi[..., :2, :, :, :], kpsi[..., 2:, :, :, :]
        sp = sum(s[...,None,None,None]*p[i] for i,s in enumerate(self.sigma))
        coupled = np.concatenate((C_AU*np.einsum('abxyz,...bxyz->...axyz',sp,lower), C_AU*np.einsum('abxyz,...bxyz->...axyz',sp,upper)), axis=-4)
        beta_psi = np.concatenate((upper,-lower), axis=-4)
        out = np.cos(energy*dt)*kpsi - 1j*(np.sin(energy*dt)/energy)*(coupled+C_AU**2*beta_psi)
        return np.fft.ifftn(out, axes=(-3,-2,-1))

    def step(self, psi, scalar_potential, dt: float, vector_potential=(0.,0.,0.)):
        """Advance one norm-preserving symmetric split step."""
        if self.use_torch:
            if not isinstance(psi, torch.Tensor): psi = torch.as_tensor(psi, dtype=torch.complex128, device=self.device)
            v = torch.as_tensor(scalar_potential, dtype=torch.float64, device=self.device)
            if psi.ndim == 5 and v.ndim == 4: v = v.unsqueeze(1)
            psi = psi * torch.exp(-0.5j*dt*v)
            psi = self._kinetic(psi, dt, vector_potential)
            return psi * torch.exp(-0.5j*dt*v)
        v=np.asarray(scalar_potential)
        if np.ndim(psi)==5 and v.ndim==4: v=v[:,None,...]
        psi = np.asarray(psi, dtype=np.complex128) * np.exp(-0.5j*dt*v)
        psi = self._kinetic(psi, dt, vector_potential)
        return psi * np.exp(-0.5j*dt*v)

    def norm(self, psi) -> float:
        a = psi.detach().cpu().numpy() if self.use_torch and isinstance(psi, torch.Tensor) else np.asarray(psi)
        return float(np.sum(np.abs(a)**2)*self.dx**3)

    def nuclear_potential(self, elements: Sequence[str], positions_bohr: np.ndarray, softening: float = 0.25):
        coords = (np.indices(self.grid.shape).transpose(1,2,3,0) - (np.array(self.grid.shape)-1)/2)*self.dx
        v = np.zeros(self.grid.shape, float)
        for el, pos in zip(elements, positions_bohr):
            r = np.sqrt(np.sum((coords-pos)**2,axis=-1)+softening**2)
            v -= Z[el]/r
        return v

    def initial_packet(self, center_bohr=(0.,0.,0.), width_bohr=1.0):
        coords = (np.indices(self.grid.shape).transpose(1,2,3,0) - (np.array(self.grid.shape)-1)/2)*self.dx
        envelope = np.exp(-np.sum((coords-np.asarray(center_bohr))**2,axis=-1)/(2*width_bohr**2)).astype(complex)
        psi = np.zeros((4,*self.grid.shape), complex); psi[0] = envelope
        psi /= math.sqrt(np.sum(np.abs(psi)**2)*self.dx**3)
        if self.use_torch: return torch.as_tensor(psi, dtype=torch.complex128, device=self.device)
        return psi
