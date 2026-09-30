import importlib.util

import numpy as np
import pytest

from molecular_dirac_sim.species import Atom, ReactionSystem


def test_neutral_hydrogen_is_not_proton_and_triplet_state_is_metadata():
    h = Atom.parse("H")
    hp = Atom.parse("H+")
    o = Atom.parse("O(^3P)")
    assert (h.charge, h.electrons) == (0, 1)
    assert (hp.charge, hp.electrons) == (1, 0)
    assert (o.charge, o.electrons, o.term) == (0, 8, "^3P")


def test_charge_conservation_rejects_two_protons_making_neutral_water():
    reactants = ReactionSystem.parse(["O", "H+", "H+"])
    water = ReactionSystem.parse(["O", "H", "H"])
    assert reactants.charge == 2 and reactants.electron_count == 8
    with pytest.raises(ValueError, match="Charge is not conserved"):
        reactants.check_charge_conservation(water)
    reactants.check_charge_conservation(water, free_electrons_in=2)


@pytest.mark.skipif(importlib.util.find_spec("pyscf") is None, reason="optional PySCF backend")
def test_correlated_water_and_methane_energies_are_finite():
    from molecular_dirac_sim.abinitio import run_electronic
    from molecular_dirac_sim.molecular_refs import reference_geometry
    for name in ("water", "methane"):
        system, xyz = reference_geometry(name)
        result, _, _ = run_electronic(system, xyz, basis="cc-pVDZ", method="ccsd(t)", spin=0, threads=2)
        assert result.converged
        assert np.isfinite(result.e_total_hartree)
        assert result.e_total_hartree < result.e_scf_hartree
        assert result.diagnostics["triples_hartree"] <= 0


@pytest.mark.skipif(importlib.util.find_spec("pyscf") is None, reason="optional PySCF backend")
def test_state_averaged_reactive_casscf_contains_multiple_surfaces():
    from molecular_dirac_sim.abinitio import run_electronic
    sys = ReactionSystem.parse(["O(^3P)", "H", "H"])
    geom = [[0.,0.,0.],[0.7,0.7,0.],[ -0.7,0.7,0.]]
    result, mol, mc = run_electronic(sys, geom, basis="cc-pVDZ", spin=0, method="casscf",
                                     active_electrons=8, active_orbitals=6, roots=3, threads=2)
    roots = result.diagnostics["state_energies_hartree"]
    assert len(roots) == 3
    assert np.all(np.isfinite(roots))
    assert roots == sorted(roots)
    from molecular_dirac_sim.abinitio import casscf_ao_density
    density = casscf_ao_density(mol, mc, root=0)
    # Integral of the spin-summed density must count all ten electrons,
    # including the doubly occupied inactive core orbitals.
    assert np.einsum("ij,ji->", density, mol.intor("int1e_ovlp")) == pytest.approx(10.0, abs=1e-8)
