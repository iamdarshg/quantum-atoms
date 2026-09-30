"""Run all-electron, time-dependent Dirac-Hartree-Fock demos and save PNG frames.

This prototype propagates every electron in the neutral molecule using a
self-consistent direct and nonlocal Fock exchange fields with Ehrenfest nuclei.
Electron correlation beyond TD-DHF is outside the present model.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dirac import parse_formula, Z
from .dirac import DiracGrid
from .many_electron import ErrorBounds, MolecularDiracHartreeFock, max_separation_sphere


def xyz_elements(path: Path, formula: str):
    lines=[s.strip() for s in path.read_text().splitlines() if s.strip()]
    n=int(lines[0]); rows=[line.split() for line in lines[2:2+n]]
    elements=[r[0] for r in rows]
    if Counter(elements)!=parse_formula(formula):
        raise ValueError("XYZ atom list does not match requested molecular formula")
    return elements


def simulate(formula: str, outdir: Path, steps=8, device="cuda", xyz: Path | None=None,
             temperature_k=300.0, ramp_to_k: float | None=None, ramp_steps=0,
             grid_size=16, spacing_bohr=0.5, dt_au=1e-7,
             norm_bound=1e-10, overlap_bound=1e-10, poisson_bound=1e-10,
             radius_bohr=3.0):
    comp=parse_formula(formula)
    elements=xyz_elements(xyz,formula) if xyz else [el for el,count in comp.items() for _ in range(count)]
    grid=DiracGrid((grid_size,)*3,spacing_bohr,device)
    errors=ErrorBounds(dt_au,norm_bound,overlap_bound,poisson_bound,spacing_bohr)
    sim=MolecularDiracHartreeFock(elements,grid,errors,temperature_k,ramp_to_k,ramp_steps,radius_bohr=radius_bohr)
    outdir.mkdir(parents=True,exist_ok=True)
    selected=sorted(set([0,steps//2,steps]))
    frames=[]
    for step in range(steps+1):
        if step in selected:
            frames.append((step,sim.positions.copy(),sim.norm(),sim.temperature_at(),sim.last_overlap_error if hasattr(sim,"last_overlap_error") else 0.0))
        if step<steps: sim.step()
    palette={"H":"#f6f4ef","C":"#4c566a","N":"#5e81ac","O":"#bf616a","F":"#a3be8c","Cl":"#8fbcbb"}
    for step,pos,norm,temp,overlap in frames:
        fig=plt.figure(figsize=(10,5),layout="constrained")
        grid_spec=fig.add_gridspec(1,2,width_ratios=[1.05,1])
        ax=fig.add_subplot(grid_spec[0,0],projection="3d")
        for el,p in zip(elements,pos):
            ax.scatter(*p,s=130+20*Z[el],color=palette.get(el,"#d08770"),edgecolor="#222")
            ax.text(*p,el,fontsize=9)
        ax.scatter([0],[0],[0],marker="+",color="#444",s=50)
        lim=radius_bohr+1
        ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim),xlabel="x (bohr)",ylabel="y (bohr)",zlabel="z (bohr)")
        ax.set_title(f"All nuclei: {formula} | step {step}/{steps}")
        info=fig.add_subplot(grid_spec[0,1]); info.axis("off")
        info.text(.03,.92,"Many-electron time-dependent Dirac",fontsize=14,weight="bold",transform=info.transAxes)
        info.text(.03,.78,f"Electrons propagated: {sim.nelectrons}\nOccupied spinors: {sim.nelectrons}\nTemperature: {temp:.1f} K\nBackend: {sim.device}\nGrid: {grid_size}³ at {spacing_bohr:g} bohr\nSpinor norm: {norm:.8f}\nOverlap error: {overlap:.2e}\nΔt: {dt_au:g} atomic units",va="top",linespacing=1.6,transform=info.transAxes)
        info.text(.03,.28,"Time-dependent Dirac-Hartree-Fock + classical nuclei.\nExchange included; correlation and QED radiative\ncorrections omitted. Not a chemistry prediction.",color="#a33",va="top",transform=info.transAxes)
        fig.savefig(outdir/f"{formula.lower()}_step_{step:04d}.png",dpi=150); plt.close(fig)
    return frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("formula",help="Molecular formula, e.g. H2O, CH4, C2H6O")
    p.add_argument("--xyz",type=Path,help="Optional XYZ atom ordering; coordinates are not imposed")
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--device",choices=["cuda","cpu","auto"],default="cuda")
    p.add_argument("--temperature-k",type=float,default=300.0)
    p.add_argument("--ramp-to-k",type=float,help="Automatically ramp from --temperature-k to this value")
    p.add_argument("--ramp-steps",type=int,default=0)
    p.add_argument("--grid-size",type=int,default=16)
    p.add_argument("--spacing-bohr",type=float,default=0.5)
    p.add_argument("--dt-au",type=float,default=1e-7)
    p.add_argument("--norm-bound",type=float,default=1e-10)
    p.add_argument("--overlap-bound",type=float,default=1e-10)
    p.add_argument("--poisson-bound",type=float,default=1e-10)
    p.add_argument("--radius-bohr",type=float,default=3.0)
    p.add_argument("--out",type=Path,default=Path("molecular_frames"))
    a=p.parse_args()
    frames=simulate(a.formula,a.out,a.steps,a.device,a.xyz,a.temperature_k,a.ramp_to_k,a.ramp_steps,a.grid_size,a.spacing_bohr,a.dt_au,a.norm_bound,a.overlap_bound,a.poisson_bound,a.radius_bohr)
    print(f"Saved {len(frames)} frames to {a.out.resolve()}")

if __name__=="__main__": main()
