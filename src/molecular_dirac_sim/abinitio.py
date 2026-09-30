"""PySCF electronic-structure backend for molecular and reaction-path calculations.

This module supplies actual all-electron Gaussian-basis wavefunctions (RHF/ROHF,
CCSD(T), state-averaged CASSCF and NEVPT2 where available). It is not a QED or
reactive-scattering solver; callers must retain that distinction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Sequence
import os

import numpy as np

from .species import ReactionSystem


@dataclass
class ElectronicResult:
    method: str
    basis: str
    charge: int
    spin: int
    n_electrons: int
    n_basis: int
    e_scf_hartree: float
    e_corr_hartree: float
    e_total_hartree: float
    wall_seconds: float
    converged: bool
    diagnostics: dict

    def to_dict(self):
        return asdict(self)


def run_electronic(system: ReactionSystem, coords_angstrom: Sequence[Sequence[float]],
                   *, basis="cc-pVTZ", spin=0, method="ccsd(t)",
                   active_electrons: int | None=None, active_orbitals: int | None=None,
                   frozen_core: int=0, max_memory_mb=4096, threads=4,
                   roots=1, density_fit=False) -> tuple[ElectronicResult, object, object]:
    """Run an all-electron SCF/post-SCF calculation at fixed nuclear geometry.

    ``spin`` is 2S (= N_alpha - N_beta), matching PySCF. CCSD(T) uses the
    spin-appropriate restricted/unrestricted implementation. CASSCF optionally
    state-averages a small set of roots; CAS-SCF results must be checked for
    active-space/root continuity along any geometry scan.
    """
    from pyscf import gto, scf, cc, mcscf, mrpt, lib
    # Some container namespaces remap PID without exposing /proc/<pid>/statm.
    # PySCF only uses this helper for memory heuristics; use a safe fallback.
    if not os.path.exists(f"/proc/{os.getpid()}/statm"):
        lib.current_memory = lambda: [0.0]
    if spin < 0 or spin > system.electron_count:
        raise ValueError("spin must be a nonnegative 2S not exceeding the electron count")
    if (system.electron_count - spin) % 2:
        raise ValueError(f"electron count {system.electron_count} and 2S={spin} have inconsistent parity")
    atoms = system.as_pyscf_atoms(coords_angstrom)
    mol = gto.M(atom=atoms, basis=basis, charge=system.charge, spin=int(spin),
                unit="Angstrom", verbose=0, max_memory=int(max_memory_mb))
    if mol.nelectron != system.electron_count:
        raise RuntimeError(f"PySCF electron count {mol.nelectron} disagrees with species bookkeeping {system.electron_count}")
    lib.num_threads(int(threads))
    method = method.lower().replace(" ", "")
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol))
    if density_fit:
        mf = mf.density_fit()
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    t0 = time.perf_counter()
    e_scf = float(mf.kernel())
    wavefunction = mf
    if not mf.converged:
        # CDIIS/second-order fallback; retain failed status if it still does not converge.
        mf = mf.newton()
        e_scf = float(mf.kernel())
    if method in {"hf", "rhf", "rohf"}:
        total, corr = e_scf, 0.0
        diagnostics = {"scf_converged": bool(mf.converged)}
    elif method in {"ccsd", "ccsd(t)"}:
        obj = cc.CCSD(mf, frozen=frozen_core)
        obj.conv_tol = 1e-9
        obj.max_cycle = 100
        obj.kernel()
        if not obj.converged:
            raise RuntimeError("CCSD did not converge; inspect multireference character or increase limits")
        corr = float(obj.e_corr)
        et = float(obj.ccsd_t()) if method == "ccsd(t)" else 0.0
        total = e_scf + corr + et
        diagnostics = {"ccsd_converged": bool(obj.converged), "triples_hartree": et,
                       "t1_diagnostic": float(np.linalg.norm(obj.t1)),
                       "frozen_core_orbitals": int(frozen_core)}
        wavefunction = obj
    elif method in {"casscf", "casscf-nevpt2"}:
        if active_electrons is None or active_orbitals is None:
            raise ValueError("CASSCF requires active_electrons and active_orbitals")
        mc = mcscf.CASSCF(mf, int(active_orbitals), int(active_electrons))
        mc.max_cycle_macro = 100
        if roots > 1:
            mc = mc.state_average_([1.0 / roots] * int(roots))
        mc.kernel()
        if not mc.converged:
            raise RuntimeError("CASSCF orbital optimization did not converge")
        roots_e = np.atleast_1d(getattr(mc, "e_states", [mc.e_tot])).astype(float)
        # Energies remain root-resolved. e_total denotes the lowest root only;
        # an average over distinct electronic states is not a physical PES.
        e_cas = float(np.min(roots_e))
        corr = e_cas - e_scf
        total = e_cas
        diagnostics = {"casscf_converged": bool(mc.converged), "state_energies_hartree": roots_e.tolist(),
                       "active_electrons": int(active_electrons), "active_orbitals": int(active_orbitals),
                       "state_average_roots": int(roots)}
        wavefunction = mc
        if method == "casscf-nevpt2":
            if roots != 1:
                raise ValueError("NEVPT2 currently requires a single CASSCF state")
            e2 = float(mrpt.NEVPT(mc).kernel())
            total += e2
            corr += e2
            diagnostics["nevpt2_hartree"] = e2
    else:
        raise ValueError("method must be hf, ccsd, ccsd(t), casscf, or casscf-nevpt2")
    result = ElectronicResult(method, basis, system.charge, int(spin), system.electron_count,
                              mol.nao_nr(), e_scf, float(corr), float(total),
                              time.perf_counter() - t0, bool(mf.converged), diagnostics)
    return result, mol, wavefunction


def casscf_ao_density(mol, mc, root: int=0) -> np.ndarray:
    """Spin-summed AO 1-RDM for one state-averaged CASSCF root, including core."""
    if hasattr(mc.fcisolver, "states_make_rdm1"):
        dm_active = mc.fcisolver.states_make_rdm1(mc.ci, mc.ncas, mc.nelecas)[root]
    else:
        dm_active = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
    c = np.asarray(mc.mo_coeff)
    ncore = int(mc.ncore)
    ccore = c[:, :ncore]
    cactive = c[:, ncore:ncore+int(mc.ncas)]
    return 2.0 * (ccore @ ccore.T) + cactive @ dm_active @ cactive.T


def nmr_shieldings(mol, mf, nuclei: Sequence[int] | None=None) -> dict:
    """Compute GIAO isotropic shielding tensors using the PySCF properties extension.

    Values are absolute shieldings in ppm. Chemical shifts require same-method,
    same-basis reference shieldings (typically TMS); solvent/temperature effects
    are not implied by this gas-phase calculation.
    """
    from pyscf.prop import nmr
    cls = nmr.RHF if getattr(mf, "spin", 0) == 0 else nmr.UHF
    obj = cls(mf)
    tensors = np.asarray(obj.kernel())
    iso = np.trace(tensors, axis1=1, axis2=2) / 3.0
    idx = list(range(mol.natm)) if nuclei is None else list(nuclei)
    return {"unit": "ppm", "gauge": "GIAO", "method": "SCF linear response",
            "isotropic_ppm": {str(i): float(iso[i]) for i in idx},
            "tensor_ppm": {str(i): tensors[i].tolist() for i in idx}}


def regular_tetrahedron(radius: float) -> np.ndarray:
    """Four H coordinates about a central atom, in Angstrom."""
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    return radius * v / np.sqrt(3.0)


def symmetric_triatomic_path(radii: Sequence[float], bond_angle_deg=104.5) -> list[np.ndarray]:
    """O-centered symmetric O+H+H line cut, preserving H permutation symmetry."""
    a = np.deg2rad(bond_angle_deg / 2.0)
    unit = np.array([[np.sin(a), np.cos(a), 0.0], [-np.sin(a), np.cos(a), 0.0]])
    return [np.vstack(([0.0, 0.0, 0.0], float(r) * unit)) for r in radii]
