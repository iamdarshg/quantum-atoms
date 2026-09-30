# quantum-atoms

Computational experiments for small-molecule electronic structure and reduced-dimensional reactive dynamics. This repository now includes an all-electron PySCF backend for HF, CCSD(T), and state-averaged CASSCF, fixed-geometry GIAO NMR calculations, symmetric HHO/CH4 association-path scans, and a one-coordinate quantum nuclear wavepacket whose time-dependent Born–Oppenheimer electron density can be exported as MP4.

These components are **not** a full QED, full-dimensional reactive-scattering, or predictive radiative-association simulator. In particular, the wavepacket animation is the electronic one-particle density from a state-specific CASSCF 1-RDM, interpolated along a single symmetric association coordinate and averaged over the propagated nuclear packet. It does not calculate a full electron–nuclear wavefunction, spontaneous photon emission, product stabilization, reaction rates, or branching fractions. The earlier `formation_demo` remains a separate TD-DHF prototype and is not an accurate chemistry model.

## Install

```bash
python -m pip install -e '.[science,test]'
```

## Run electronic structure and NMR calculations

```bash
quantum-atoms species 'O(^3P)' H H
quantum-atoms reference water --basis cc-pVTZ --method ccsd\(t\) --nmr --nmr-basis cc-pVDZ
quantum-atoms reference methane --basis cc-pVTZ --method ccsd\(t\) --nmr --nmr-basis cc-pVDZ
```

Energies are all-electron fixed-geometry CCSD(T) results in the requested finite Gaussian basis. GIAO shieldings use PBE0/cc-pVDZ and a same-method TMS reference. The reported chemical shift convention is `delta = sigma(TMS) - sigma(sample)`. The supplied equilibrium geometries are not optimized; solvent, temperature, rovibrational averaging, and experimental calibration corrections are absent.

## Compute and animate a reactive path

```bash
quantum-atoms scan water --points 17 --r-min 0.78 --r-max 6 --grid-size 40 --basis cc-pVDZ --out results
quantum-atoms wavepacket results/water_cc-pVDZ_casscf_path.npz --collision-k 1000 --initial-r 4.8 --frames 45 --out results/water_electron_cloud.mp4
```

For methane use `scan methane` and the resulting `methane_cc-pVDZ_casscf_path.npz`. The water path uses a CAS(8e,6o) average over three singlet roots; methane uses CAS(8e,8o) over four singlet roots. These are valence active spaces in cc-pVDZ; the remaining electrons are represented in inactive orbitals. The 1D nuclear coordinate preserves H permutation symmetry and fixes the molecule's shape to its symmetric path. The nuclear Hamiltonian uses a finite-difference kinetic operator, interpolated CASSCF potential, Gaussian incoming packet, and complex absorbing boundaries. The output JSON reports transient probability and surviving packet norm. `stable_product_probability` is intentionally null because no radiative stabilization is implemented.

The electron-density movie is not a classical particle animation. Its frames are the probability-weighted, computed CASSCF electron density, in a fixed spatial slice and scale; a video file cannot show the full 3D field at once. Current example runs and reference outputs are in `results/`.

## Physics boundaries

No GPU acceleration is implemented in this PySCF route. It currently includes neither multiple spin manifolds/spin-orbit or nonadiabatic couplings, multidimensional nuclear scattering, threshold-resolved three-body rates, radiative dipole transition surfaces, photon modes, resonances, detailed balance, a hot-water rovibrational cascade, nor explicit relativistic/QED corrections. A CASSCF path and a transient packet reaching the well do not establish stable chemical formation. `CCSD(T)` is a high-quality single-reference correlation method, not a QED calculation; its suitability is geometry-dependent (the reported T1 diagnostic should be checked).
