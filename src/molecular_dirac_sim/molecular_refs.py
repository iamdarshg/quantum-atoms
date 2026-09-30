"""Correlated water/methane reference structures and GIAO NMR predictions."""
from __future__ import annotations

import numpy as np
import os

from .species import ReactionSystem
from .abinitio import run_electronic, nmr_shieldings


def reference_geometry(name: str):
    name = name.lower()
    if name == "water":
        r = 0.9572
        a = np.deg2rad(104.52 / 2)
        xyz = np.array([[0., 0., 0.], [r*np.sin(a), r*np.cos(a), 0.],
                        [-r*np.sin(a), r*np.cos(a), 0.]])
        return ReactionSystem.parse(["O", "H", "H"]), xyz
    if name == "methane":
        h = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)
        h *= 1.087 / np.sqrt(3.)
        return ReactionSystem.parse(["C", "H", "H", "H", "H"]), np.vstack(([0.,0.,0.], h))
    raise ValueError("reference name must be water or methane")


def tms_geometry():
    """Ideal tetrahedral Si(CH3)4 reference, coordinates in Angstrom."""
    si_c = 1.87
    c_h = 1.09
    tetra = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float) / np.sqrt(3.)
    atoms = [("Si", np.zeros(3))]
    for u in tetra:
        cpos = si_c * u
        atoms.append(("C", cpos))
        # CH3: each C-H direction forms the tetrahedral angle to the Si-C bond.
        e1 = np.cross(u, [0.,0.,1.])
        if np.linalg.norm(e1) < 1e-8:
            e1 = np.cross(u, [0.,1.,0.])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        for phi in (0., 2*np.pi/3, 4*np.pi/3):
            v = (1./3.)*u + np.sqrt(8./9.)*(np.cos(phi)*e1 + np.sin(phi)*e2)
            atoms.append(("H", cpos + c_h*v))
    return atoms


def run_nmr_reference(name: str, *, basis="cc-pVTZ", xc="PBE0", threads=4):
    """Return gas-phase GIAO shielding and same-level TMS-relative shifts.

    Solvent and finite-temperature shifts are intentionally not folded in.
    """
    from pyscf import gto, dft, lib
    from pyscf.prop import nmr
    from pyscf.dft import numint
    sys, xyz = reference_geometry(name)
    if not os.path.exists(f"/proc/{os.getpid()}/statm"):
        lib.current_memory = lambda: [0.0]
    lib.num_threads(int(threads))
    mol = gto.M(atom=sys.as_pyscf_atoms(xyz), basis=basis, charge=0, spin=0,
                unit="Angstrom", verbose=0, max_memory=5000)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-10
    mf.grids.level = 4
    mf.kernel()
    _pad_nmr_grid(mf, numint.BLKSIZE)
    tensor = np.asarray(nmr.RKS(mf).kernel())
    iso = np.trace(tensor, axis1=1, axis2=2)/3.

    tms = gto.M(atom=tms_geometry(), basis=basis, charge=0, spin=0,
                unit="Angstrom", verbose=0, max_memory=5000)
    ref = dft.RKS(tms); ref.xc = xc; ref.conv_tol = 1e-9; ref.grids.level = 3
    ref.kernel()
    _pad_nmr_grid(ref, numint.BLKSIZE)
    ref_tensor = np.asarray(nmr.RKS(ref).kernel())
    ref_iso = np.trace(ref_tensor, axis1=1, axis2=2)/3.
    # Symmetry average over chemically equivalent target protons.
    h_indices = [i for i, a in enumerate(sys.atoms) if a.symbol in {"H","D","T"}]
    sigma = float(np.mean(iso[h_indices]))
    tms_h = [i for i, (sym, _) in enumerate(tms_geometry()) if sym == "H"]
    sigma_tms = float(np.mean(ref_iso[tms_h]))
    return {"molecule": name, "method": f"{xc}/{basis} GIAO SCF response",
            "phase": "isolated gas-phase equilibrium geometry; no solvent/thermal average",
            "proton_absolute_shielding_ppm": sigma,
            "tms_proton_reference_shielding_ppm": sigma_tms,
            "proton_chemical_shift_vs_tms_ppm": sigma_tms - sigma,
            "all_atom_isotropic_shieldings_ppm": iso.tolist(),
            "reference_isotropic_shieldings_ppm": ref_iso.tolist()}


def _pad_nmr_grid(mf, block_size: int):
    """Work around pyscf-properties GIAO grid chunk assertion for non-multiple sizes."""
    grids = mf.grids
    grids.build(with_non0tab=True)
    n = len(grids.weights)
    pad = (-n) % int(block_size)
    if pad:
        # Zero-weight dummy points make ngrids a valid GIAO block multiple.
        grids.coords = np.vstack((grids.coords, np.zeros((pad, 3))))
        grids.weights = np.concatenate((grids.weights, np.zeros(pad)))
        grids.non0tab = None
