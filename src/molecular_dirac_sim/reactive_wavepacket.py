"""Reduced-dimensional BO quantum wavepacket, CASSCF PES scan, and density movie.

The implemented coordinate is a permutation-symmetric collective association
path. It is a real nuclear wavepacket on an ab-initio electronic surface, but
it is not a full-dimensional three-/five-body scattering calculation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

from .abinitio import run_electronic, casscf_ao_density, symmetric_triatomic_path, regular_tetrahedron
from .species import ReactionSystem, MASS

AMU_TO_ME = 1822.888486209
KB_HARTREE_K = 3.166811563e-6
ANGSTROM_TO_BOHR = 1.889726125


def path_geometries(kind: str, radii_angstrom):
    radii = np.asarray(radii_angstrom, float)
    if kind == "water":
        return symmetric_triatomic_path(radii, 104.5), ReactionSystem.parse(["O(^3P)", "H", "H"]), (8, 6, 3)
    if kind == "methane":
        geoms = []
        tetra = regular_tetrahedron(1.0)
        for r in radii:
            geoms.append(np.vstack(([0.0, 0.0, 0.0], r * tetra)))
        return geoms, ReactionSystem.parse(["C(^3P)", "H", "H", "H", "H"]), (8, 8, 4)
    raise ValueError("kind must be water or methane")


def effective_mass_au(kind: str) -> float:
    masses = np.array([MASS["O"], MASS["H"], MASS["H"]]) if kind == "water" else np.array([MASS["C"], *([MASS["H"]] * 4)])
    # With all coordinates scaled by R, remove center-of-mass motion analytically.
    if kind == "water":
        a = np.deg2rad(104.5 / 2)
        deriv = np.array([[0., 0., 0.], [np.sin(a), np.cos(a), 0.], [-np.sin(a), np.cos(a), 0.]])
    else:
        deriv = np.vstack(([0., 0., 0.], regular_tetrahedron(1.0)))
    dcom = (masses[:, None] * deriv).sum(axis=0) / masses.sum()
    mu_amu = np.sum(masses * np.sum(deriv**2, axis=1)) - masses.sum() * np.dot(dcom, dcom)
    return float(mu_amu * AMU_TO_ME)


def run_surface_scan(kind: str, output: Path, *, basis="cc-pVDZ", npoints=25,
                     r_min=0.78, r_max=6.0, threads=4, grid_size=52):
    """Compute a state-averaged active-space electronic curve and AO densities."""
    radii = np.linspace(r_min, r_max, int(npoints))
    geoms, system, (ne, no, roots) = path_geometries(kind, radii)
    energies, roots_all, times, densities = [], [], [], []
    # One shared real-space grid; density is stored as e / bohr^3.
    extent_ang = max(7.0, r_max + 1.0)
    axis_ang = np.linspace(-extent_ang, extent_ang, grid_size)
    xyz_ang = np.stack(np.meshgrid(axis_ang, axis_ang, axis_ang, indexing="ij"), axis=-1)
    xyz_bohr = xyz_ang.reshape(-1, 3) * ANGSTROM_TO_BOHR
    from pyscf.dft import numint
    t0 = time.perf_counter()
    for i, (r, geom) in enumerate(zip(radii, geoms)):
        result, mol, mc = run_electronic(system, geom, basis=basis, spin=0, method="casscf",
                                         active_electrons=ne, active_orbitals=no, roots=roots,
                                         threads=threads, max_memory_mb=4096)
        energies.append(result.e_total_hartree)
        roots_all.append(result.diagnostics["state_energies_hartree"])
        times.append(result.wall_seconds)
        dm = casscf_ao_density(mol, mc, root=0)
        ao = numint.eval_ao(mol, xyz_bohr, deriv=0)
        rho = np.einsum("pi,ij,pj->p", ao, dm, ao, optimize=True).reshape((grid_size,) * 3)
        densities.append(rho.astype(np.float32))
        del ao, dm, mc, mol
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, kind=kind, basis=basis, radii_angstrom=radii,
                        energies_hartree=np.asarray(energies), root_energies_hartree=np.asarray(roots_all),
                        electronic_wall_seconds=np.asarray(times), electronic_density=np.asarray(densities),
                        density_axis_angstrom=axis_ang, active_electrons=ne, active_orbitals=no,
                        state_average_roots=roots, scan_runtime_seconds=time.perf_counter() - t0)
    return output


def propagate_packet(scan_file: Path, output: Path, *, collision_energy_k=1000.0,
                     initial_r_angstrom=4.8, packet_width_angstrom=0.35,
                     frames=90, grid_points=700, cap_strength=0.001):
    """Propagate an inward nuclear wavepacket and make a probability-density movie.

    The wavepacket is a single collective coordinate. The movie's electron field
    is the nuclear-probability-weighted CASSCF 1-RDM field on the precomputed path.
    This does not contain spontaneous photon emission or transverse nuclear motion.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    data = np.load(scan_file, allow_pickle=False)
    kind = str(data["kind"])
    r_ang = data["radii_angstrom"].astype(float)
    v = data["energies_hartree"].astype(float)
    rho_path = data["electronic_density"].astype(np.float32)
    rho_axis = data["density_axis_angstrom"].astype(float)
    # Set zero at the large-R asymptote; this is an electronic threshold on this path.
    v = v - float(np.mean(v[-3:]))
    r = r_ang * ANGSTROM_TO_BOHR
    n = int(grid_points)
    x = np.linspace(r[0], r[-1], n)
    dx = x[1] - x[0]
    vint = PchipInterpolator(r, v, extrapolate=False)(x)
    vint = np.nan_to_num(vint, nan=float(v[-1]))
    mu = effective_mass_au(kind)
    kinetic_diag = 1.0 / (mu * dx * dx)
    off = -0.5 / (mu * dx * dx)
    h = diags([np.full(n - 1, off), kinetic_diag + vint, np.full(n - 1, off)], [-1, 0, 1], format="csc", dtype=np.complex128)
    # Smooth complex absorbing layers suppress end reflections; the absorbed norm is reported.
    cap_n = max(12, n // 18)
    cap = np.zeros(n)
    q = np.linspace(0, 1, cap_n)
    cap[:cap_n] = cap_strength * q**4
    cap[-cap_n:] = cap_strength * q[::-1]**4
    h.setdiag(h.diagonal() - 1j * cap)
    e0 = collision_energy_k * KB_HARTREE_K
    p0 = -np.sqrt(2.0 * mu * e0)
    sigma = packet_width_angstrom * ANGSTROM_TO_BOHR
    r0 = initial_r_angstrom * ANGSTROM_TO_BOHR
    psi = np.exp(-0.5 * ((x - r0) / sigma) ** 2 + 1j * p0 * (x - r0))
    psi = psi.astype(np.complex128)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    # Atomic time span estimated from travel to the product well plus return margin.
    velocity = abs(p0) / mu
    t_end = min(120000.0, max(6000.0, 1.6 * (r0 - 1.8 * ANGSTROM_TO_BOHR) / max(velocity, 1e-8)))
    psi_t = expm_multiply(-1j * h, psi, start=0.0, stop=t_end, num=int(frames), endpoint=True)
    prob_t = np.abs(psi_t) ** 2
    capture = (x <= 1.35 * ANGSTROM_TO_BOHR)
    pwell = np.sum(prob_t[:, capture], axis=1) * dx
    survival = np.sum(np.abs(psi_t) ** 2, axis=1) * dx

    # Interpolate precomputed CASSCF electron densities along R, then average over |chi(R,t)|^2.
    from scipy.interpolate import interp1d
    dens_interp = interp1d(r * 0 + r_ang, rho_path, axis=0, kind="linear", bounds_error=False,
                           fill_value=(rho_path[0], rho_path[-1]), assume_sorted=True)
    rho = dens_interp(x / ANGSTROM_TO_BOHR).astype(np.float32)
    center = rho.shape[1] // 2
    density_frames = np.einsum("tr,rxyz->txyz", prob_t.astype(np.float32) * dx, rho, optimize=True)
    # Show a molecular symmetry plane. The default xy plane is useful for water
    # but misses every C-H axis in tetrahedral methane, so rotate that slice to
    # contain one tetrahedral C-H bond. These remain sections through the 3D
    # electron density, not projected/blurred cartoons.
    from scipy.ndimage import map_coordinates
    uv, vv = np.meshgrid(rho_axis, rho_axis, indexing="ij")
    if kind == "methane":
        e1 = np.array([1., 1., 1.]) / np.sqrt(3.)
        e2 = np.array([1., -1., 0.]) / np.sqrt(2.)
        normal = np.cross(e1, e2)
        xyz_slice = uv[..., None] * e1 + vv[..., None] * e2
    else:
        xyz_slice = np.stack((uv, vv, np.zeros_like(uv)), axis=-1)
    grid_index = np.moveaxis((xyz_slice - rho_axis[0]) / (rho_axis[1] - rho_axis[0]), -1, 0)
    slice_frames = np.stack([map_coordinates(frame, grid_index, order=1, mode="constant", cval=0.0)
                             for frame in density_frames]) * ANGSTROM_TO_BOHR**3
    vmin = max(float(np.max(slice_frames)) * 1e-5, 1e-12)
    extent = [rho_axis[0], rho_axis[-1], rho_axis[0], rho_axis[-1]]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(slice_frames[0].T, origin="lower", extent=extent, cmap="magma", vmin=vmin,
                      vmax=max(float(np.max(slice_frames)) * 0.35, vmin * 10), interpolation="bilinear")
    cb = fig.colorbar(image, ax=ax); cb.set_label("electron probability density (e Å⁻³)")
    ax.set(xlabel="x (Å)", ylabel="y (Å)", title=f"{kind.title()} association: CASSCF electron field / BO nuclear packet")
    nucleus_colors = ["#31d6ff"] + ["#ffffff"] * (2 if kind == "water" else 4)
    dots = ax.scatter(np.zeros(len(nucleus_colors)), np.zeros(len(nucleus_colors)),
                      s=[75] + [32] * (2 if kind == "water" else 4),
                      c=nucleus_colors, edgecolors="#111111", linewidths=0.8, zorder=5)
    subtitle = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="white", va="top",
                       bbox={"facecolor": "black", "alpha": .65, "pad": 4})
    def update(i):
        image.set_data(slice_frames[i].T)
        mean_r = float(np.sum(prob_t[i] * x) * dx / max(survival[i], 1e-15) / ANGSTROM_TO_BOHR)
        if kind == "water":
            aa = np.deg2rad(104.5 / 2)
            atom_xyz = np.vstack(([0., 0., 0.],
                                  [mean_r*np.sin(aa), mean_r*np.cos(aa), 0.],
                                  [-mean_r*np.sin(aa), mean_r*np.cos(aa), 0.]))
            atom_uv = atom_xyz[:, :2]
        else:
            atom_xyz = np.vstack(([0., 0., 0.], mean_r * regular_tetrahedron(1.0)))
            atom_uv = atom_xyz @ np.stack((e1, e2), axis=1)
        dots.set_offsets(atom_uv)
        subtitle.set_text(f"t = {i*t_end/(frames-1):.0f} a.u.  |  Ecoll = {collision_energy_k:g} K-equivalent\n"
                          f"P(R < 1.35 Å) = {pwell[i]:.3f}  |  packet norm = {survival[i]:.5f}")
        return image, subtitle, dots
    animation = FuncAnimation(fig, update, frames=int(frames), interval=60, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4":
        animation.save(output, writer=FFMpegWriter(fps=15, bitrate=1800))
    else:
        animation.save(output.with_suffix(".gif"), writer=PillowWriter(fps=12))
        output = output.with_suffix(".gif")
    plt.close(fig)
    report = {"kind": kind, "mode": "1D permutation-symmetric BO wavepacket; no radiation",
              "collision_energy_temperature_equivalent_K": collision_energy_k,
              "effective_mass_electron_masses": mu, "time_end_au": t_end,
              "max_transient_probability_R_below_1p35A": float(pwell.max()),
              "final_probability_R_below_1p35A": float(pwell[-1]),
              "final_surviving_packet_norm": float(survival[-1]),
              "stable_product_probability": None,
              "reason_stable_product_not_computed": "no photon continuum/radiative transition operator in this reduced model",
              "movie": str(output)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report
