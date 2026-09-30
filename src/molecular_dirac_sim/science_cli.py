"""First-principles molecular and reaction-coordinate command-line workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .species import ReactionSystem
from .abinitio import run_electronic
from .molecular_refs import reference_geometry, run_nmr_reference
from .reactive_wavepacket import run_surface_scan, propagate_packet


def main():
    p = argparse.ArgumentParser(prog="quantum-atoms", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("species", help="check charge/electron bookkeeping")
    sp.add_argument("labels", nargs="+", help="e.g. O(^3P) H H or O H+ H+")
    ref = sub.add_parser("reference", help="CCSD(T) energies and optional GIAO NMR references")
    ref.add_argument("molecule", choices=["water", "methane"])
    ref.add_argument("--basis", default="cc-pVTZ")
    ref.add_argument("--method", default="ccsd(t)", choices=["hf", "ccsd", "ccsd(t)"])
    ref.add_argument("--nmr", action="store_true")
    ref.add_argument("--nmr-basis", default="cc-pVDZ", help="GIAO NMR basis; defaults to cc-pVDZ for practical TMS cost")
    ref.add_argument("--threads", type=int, default=4)
    scan = sub.add_parser("scan", help="state-averaged CASSCF symmetric reactive-coordinate scan")
    scan.add_argument("system", choices=["water", "methane"])
    scan.add_argument("--basis", default="cc-pVDZ")
    scan.add_argument("--points", type=int, default=25)
    scan.add_argument("--r-min", type=float, default=.78)
    scan.add_argument("--r-max", type=float, default=6.0)
    scan.add_argument("--grid-size", type=int, default=52)
    scan.add_argument("--threads", type=int, default=4)
    scan.add_argument("--out", type=Path, default=Path("runs"))
    vid = sub.add_parser("wavepacket", help="propagate one reduced-dimensional quantum association packet")
    vid.add_argument("scan_file", type=Path)
    vid.add_argument("--collision-k", type=float, default=1000.)
    vid.add_argument("--initial-r", type=float, default=4.8)
    vid.add_argument("--frames", type=int, default=90)
    vid.add_argument("--out", type=Path, default=Path("runs/electron_cloud.mp4"))
    a = p.parse_args()
    if a.cmd == "species":
        s = ReactionSystem.parse(a.labels)
        print(json.dumps({"species": [x.__dict__ for x in s.atoms], "total_charge": s.charge,
                          "total_electrons": s.electron_count, "nuclear_mass_amu": s.nuclear_mass_amu}, indent=2))
    elif a.cmd == "reference":
        sys, xyz = reference_geometry(a.molecule)
        t0 = time.perf_counter()
        result, _, _ = run_electronic(sys, xyz, basis=a.basis, method=a.method, spin=0, threads=a.threads)
        output = {"electronic_structure": result.to_dict(), "geometry_angstrom": xyz.tolist(),
                  "runtime_total_seconds": time.perf_counter()-t0}
        if a.nmr:
            output["nmr"] = run_nmr_reference(a.molecule, basis=a.nmr_basis, threads=a.threads)
        print(json.dumps(output, indent=2))
    elif a.cmd == "scan":
        out = a.out / f"{a.system}_{a.basis}_casscf_path.npz"
        print(run_surface_scan(a.system, out, basis=a.basis, npoints=a.points,
                               r_min=a.r_min, r_max=a.r_max, threads=a.threads,
                               grid_size=a.grid_size))
    elif a.cmd == "wavepacket":
        result = propagate_packet(a.scan_file, a.out, collision_energy_k=a.collision_k,
                                  initial_r_angstrom=a.initial_r, frames=a.frames)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
