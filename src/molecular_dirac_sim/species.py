"""Charge- and isotope-aware atomic species bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass
import re

ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
Z = {s: i + 1 for i, s in enumerate(ELEMENTS)}
MASS = {"H": 1.00782503223, "D": 2.01410177812, "T": 3.01604928199,
        "C": 12.0, "N": 14.00307400443, "O": 15.99491461957}


@dataclass(frozen=True)
class Atom:
    symbol: str
    nuclear_charge: int
    charge: int
    electrons: int
    mass_amu: float
    term: str | None = None

    @classmethod
    def parse(cls, label: str) -> "Atom":
        """Parse H/H0/H+/O/O+ and optional term labels such as ``O(^3P)``."""
        raw = label.strip()
        term = None
        if "(" in raw and raw.endswith(")"):
            raw, term = raw.split("(", 1)
            term = term[:-1].strip()
        m = re.fullmatch(r"([A-Z][a-z]?|D|T)(?:(\d*)([+-])|0)?", raw)
        if not m:
            raise ValueError(f"Invalid atomic species {label!r}; use labels such as H, H0, H+, or O(^3P)")
        sym, mag, sign = m.groups()
        if sym == "D" or sym == "T":
            z, base = 1, "H"
        else:
            if sym not in Z:
                raise ValueError(f"Unknown element {sym!r}")
            z, base = Z[sym], sym
        q = (int(mag or "1") * (1 if sign == "+" else -1)) if sign else 0
        ne = z - q
        if ne < 0:
            raise ValueError(f"{label}: charge +{z+1} would imply a negative electron count")
        mass = MASS.get(sym, float(z) * 2.0)
        return cls(sym, z, q, ne, mass, term)


@dataclass(frozen=True)
class ReactionSystem:
    atoms: tuple[Atom, ...]

    @classmethod
    def parse(cls, labels) -> "ReactionSystem":
        atoms = tuple(Atom.parse(s) for s in labels)
        if not atoms:
            raise ValueError("A system must contain at least one nucleus")
        return cls(atoms)

    @property
    def charge(self) -> int:
        return sum(a.charge for a in self.atoms)

    @property
    def electron_count(self) -> int:
        return sum(a.electrons for a in self.atoms)

    @property
    def nuclear_mass_amu(self) -> float:
        return sum(a.mass_amu for a in self.atoms)

    def check_charge_conservation(self, products: "ReactionSystem", free_electrons_in: int = 0,
                                  free_electrons_out: int = 0) -> None:
        qin = self.charge - free_electrons_in
        qout = products.charge - free_electrons_out
        if qin != qout:
            raise ValueError(f"Charge is not conserved: reactants carry {qin:+d}e, products {qout:+d}e")

    def as_pyscf_atoms(self, coords_angstrom):
        import numpy as np
        coords = np.asarray(coords_angstrom, dtype=float)
        if coords.shape != (len(self.atoms), 3):
            raise ValueError(f"Expected coordinates with shape ({len(self.atoms)}, 3)")
        # PySCF does not recognize isotopic symbols D/T as elements; keep Z=1
        # in the electronic Hamiltonian and retain isotope masses separately.
        symbols = ["H" if a.symbol in {"D", "T"} else a.symbol for a in self.atoms]
        return [(s, tuple(map(float, r))) for s, r in zip(symbols, coords)]

