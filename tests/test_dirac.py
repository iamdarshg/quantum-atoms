import numpy as np
import pytest

from molecular_dirac_sim import DiracGrid, DiracPropagator, parse_formula
from molecular_dirac_sim.many_electron import ErrorBounds, MolecularDiracHartree, max_separation_sphere


def test_formula_parser_supports_common_and_nested_formulas():
    assert parse_formula("H2O") == {"H": 2, "O": 1}
    assert parse_formula("CH4") == {"C": 1, "H": 4}
    assert parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}


def test_free_dirac_split_step_preserves_norm():
    solver = DiracPropagator(DiracGrid((8, 8, 8), 0.5, "cpu"))
    psi = solver.initial_packet()
    initial_norm = solver.norm(psi)
    for _ in range(5):
        psi = solver.step(psi, np.zeros((8, 8, 8)), 1e-6)
    assert solver.norm(psi) == pytest.approx(initial_norm, rel=1e-12)


def test_cuda_request_fails_clearly_without_cuda():
    import molecular_dirac_sim.dirac as module
    if module.torch is None or not module.torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA requested"):
            DiracPropagator(DiracGrid((8, 8, 8), 0.5, "cuda"))


def test_many_electron_initialization_and_step():
    grid=DiracGrid((10,10,10),0.5,"cpu")
    errors=ErrorBounds(1e-8,1e-8,1e-7,1e-6,0.5)
    sim=MolecularDiracHartree(["O","H","H"],grid,errors,temperature_k=300,
                              temperature_end_k=900,temperature_ramp_steps=4)
    assert sim.nelectrons == 10
    assert sim.psi.shape == (10,4,10,10,10)
    assert sim.temperature_at(2) == pytest.approx(600)
    result=sim.step()
    assert result["norm"] == pytest.approx(10,abs=1e-10)
    assert result["overlap_error"] < errors.orbital_overlap


def test_atoms_are_equidistant_from_origin_and_separated():
    positions=max_separation_sphere(6,3.0)
    assert np.linalg.norm(positions,axis=1)==pytest.approx(np.full(6,3.0))
    distances=np.linalg.norm(positions[:,None,:]-positions[None,:,:],axis=-1)
    assert distances[np.triu_indices(6,1)].min()>2.0


def test_custom_proton_numbers_and_independent_electron_count():
    sim=MolecularDiracHartree(["custom-a","custom-b"],grid=DiracGrid((8,8,8),.5,"cpu"),
                              proton_numbers=[2,1],electron_count=2)
    assert sim.nuclear_charges.tolist()==[2,1]
    assert sim.nelectrons==2
    assert sim.psi.shape[0]==2
