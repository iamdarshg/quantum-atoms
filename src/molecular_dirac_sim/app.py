"""Streamlit desktop-style GUI for the molecular Dirac-Hartree-Fock simulator."""
from __future__ import annotations
from collections import Counter
import math

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .dirac import DiracGrid, parse_formula, Z
from .many_electron import ErrorBounds, MolecularDiracHartree

st.set_page_config(page_title="Molecular Dirac Lab", page_icon="⚛️", layout="wide")
st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#0b1020 0%,#111a2e 60%,#0b1020 100%);color:#ecf2ff}
[data-testid="stSidebar"] {background:#111827;border-right:1px solid #29364f}
.hero {padding:1.4rem 1.7rem;border:1px solid #31415f;border-radius:18px;background:linear-gradient(120deg,#16233b,#182c45);margin-bottom:1rem}
.hero h1 {margin:0;color:#f3f7ff;font-size:2rem}.hero p {color:#9fb2d1;margin:.45rem 0 0}
.metric {padding:1rem;border:1px solid #293a57;border-radius:14px;background:#111a2c}
div.stButton>button {background:#6e8cff;color:white;border:0;border-radius:10px;font-weight:700;height:3rem}
</style>
<div class="hero"><h1>⚛️ Molecular Dirac Lab</h1><p>Time-dependent multi-electron spinors · self-consistent Coulomb field · CUDA-first</p></div>
""",unsafe_allow_html=True)

st.markdown("""This experimental app propagates every requested electron in a **time-dependent Dirac-Hartree-Fock mean field**, including nonlocal Fock exchange and classical moving nuclei. Correlation beyond Hartree-Fock, quantized photons, radiative QED corrections, and exact many-body dynamics are not implemented. This is a research prototype, not a fourth-order QED solver.""")

with st.sidebar:
    st.header("System")
    entry_mode=st.radio("Define atoms",["Molecular formula","Per-atom proton counts"],help="Enter any formula or specify each nucleus directly.")
    if entry_mode=="Molecular formula":
        formula=st.text_input("Formula",value="H2O",help="Examples: H2O, CH4, C2H6O, Ca(OH)2")
        try:
            composition=parse_formula(formula)
            atom_symbols=[el for el,n in composition.items() for _ in range(n)]
            protons=[Z[el] for el in atom_symbols]
        except Exception as exc:
            st.error(str(exc)); atom_symbols=[]; protons=[]
    else:
        proton_text=st.text_area("One proton count per atom",value="8, 1, 1",height=100,help="Comma, space, or newline separated. Example H₂O: 8, 1, 1")
        try:
            protons=[int(x) for x in proton_text.replace(","," ").split()]
            if not protons or any(z<1 for z in protons): raise ValueError("Use positive integers, one per atom.")
            atom_symbols=[next((el for el,z in Z.items() if z==p),f"Z={p}") for p in protons]
            formula=" + ".join(atom_symbols)
        except Exception as exc:
            st.error(f"Invalid proton list: {exc}"); protons=[]; atom_symbols=[]; formula="custom"
    total_protons=sum(protons)
    electrons=st.number_input("Electron count",min_value=1,value=max(1,total_protons),step=1,help="Set independently for ions or custom systems.")
    radius=st.number_input("Initial equal-radius shell (bohr)",min_value=0.5,max_value=100.0,value=3.0,step=0.5)

    st.header("Thermal conditions")
    t0=st.number_input("Reaction temperature (K)",min_value=0.0,value=300.0,step=50.0)
    ramp=st.toggle("Ramp temperature automatically",value=False)
    if ramp:
        t1=st.number_input("Ramp to (K)",min_value=0.0,value=1200.0,step=50.0)
        ramp_steps=st.number_input("Ramp duration (steps)",min_value=1,value=500,step=100)
    else:
        t1=t0; ramp_steps=0

    st.header("Compute & error limits")
    backend=st.selectbox("Compute device",["cuda","cpu","auto"],index=0)
    grid_size=st.select_slider("Grid points per dimension",options=[8,10,12,16,20,24,32,40],value=16)
    spacing=st.number_input("Grid spacing (bohr)",min_value=0.1,max_value=2.0,value=0.5,step=0.05)
    steps=st.number_input("Propagation steps",min_value=1,max_value=100000,value=100,step=10)
    dt=st.number_input("Time step (atomic units)",min_value=1e-12,max_value=1e-2,value=1e-7,format="%.1e")
    norm_tol=st.number_input("Norm drift bound",min_value=1e-14,max_value=1e-2,value=1e-10,format="%.1e")
    overlap_tol=st.number_input("Orbital overlap bound",min_value=1e-14,max_value=1e-2,value=1e-10,format="%.1e")
    poisson_tol=st.number_input("Poisson residual bound",min_value=1e-14,max_value=1e-2,value=1e-10,format="%.1e")
    exchange_iterations=st.slider("Fock exchange solve iterations",min_value=1,max_value=8,value=3,help="More fixed-point iterations improve the implicit exchange substep at higher cost.")

st.info(f"System: **{len(protons)} nuclei · {total_protons} protons · {int(electrons)} electrons**. Approximate spinor storage: **{len(protons) and int(electrons)*4*grid_size**3*16/1024**2:.1f} MiB** (not including FFT work buffers). Larger systems can require substantial RAM/VRAM.")

run=st.button("▶ Run time-dependent simulation",use_container_width=True,disabled=not protons)
if run:
    try:
        grid=DiracGrid((grid_size,)*3,spacing,backend)
        bounds=ErrorBounds(dt,norm_tol,overlap_tol,poisson_tol,spacing)
        sim=MolecularDiracHartree(atom_symbols,grid,bounds,t0,t1,int(ramp_steps),radius_bohr=radius,
                                  proton_numbers=protons,electron_count=int(electrons),
                                  exchange_iterations=exchange_iterations)
        progress=st.progress(0,text="Building occupied spinors and initial fields…")
        snapshots=[]
        for i in range(int(steps)+1):
            if i in {0,int(steps)//2,int(steps)}:
                snapshots.append((i,sim.positions.copy(),sim._density().copy(),sim.norm(),sim.temperature_at()))
            if i<int(steps):
                report=sim.step()
                if i%max(1,int(steps)//100)==0:
                    progress.progress(min(1,(i+1)/int(steps)),text=f"Step {i+1}/{int(steps)} · T={report['temperature_k']:.1f} K · norm drift={report['norm_drift']:.2e}")
        progress.progress(1.0,text="Run complete")
        st.session_state["run_result"]=(formula,atom_symbols,sim,snapshots,grid_size,spacing)
    except Exception as exc:
        st.exception(exc)

if "run_result" in st.session_state:
    formula,atom_symbols,sim,snapshots,n,dx=st.session_state["run_result"]
    final_step,positions,density,norm,temp=snapshots[-1]
    fig=make_subplots(rows=1,cols=2,specs=[[{"type":"scene"},{"type":"heatmap"}]],
                      subplot_titles=("Nuclei · spherical start → Ehrenfest motion","Electron density · central slice"))
    palette={"H":"#f4f4f2","C":"#53627a","N":"#5e81ac","O":"#d45b66","F":"#7aa874","Cl":"#61a6a1"}
    for idx,(el,pos) in enumerate(zip(atom_symbols,positions)):
        label=el if el in Z and Z[el]==int(sim.nuclear_charges[idx]) else f"Z={int(sim.nuclear_charges[idx])}"
        fig.add_trace(go.Scatter3d(x=[pos[0]],y=[pos[1]],z=[pos[2]],mode="markers+text",text=[label],textposition="top center",marker={"size":8+min(8,math.log10(sim.nuclear_charges[idx]+1)*3),"color":palette.get(label,"#d08770"),"line":{"color":"#e4eaf5","width":1}},name=label,showlegend=False),row=1,col=1)
    mid=n//2
    fig.add_trace(go.Heatmap(z=density[:,:,mid].T,colorscale="Viridis",colorbar={"title":"e⁻ / bohr³"}),row=1,col=2)
    fig.update_layout(height=520,template="plotly_dark",paper_bgcolor="#0b1020",plot_bgcolor="#0b1020",margin={"l":5,"r":5,"t":55,"b":5})
    fig.update_scenes(aspectmode="cube",xaxis_title="x (bohr)",yaxis_title="y (bohr)",zaxis_title="z (bohr)")
    fig.update_xaxes(title="x grid",row=1,col=2); fig.update_yaxes(title="y grid",row=1,col=2)
    st.plotly_chart(fig,use_container_width=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Electrons / spinors",f"{sim.nelectrons}")
    m2.metric("Current temperature",f"{temp:.1f} K")
    m3.metric("Total spinor norm",f"{norm:.8f}")
    m4.metric("Orbital overlap error",f"{getattr(sim,'last_overlap_error',0):.2e}")
    st.caption(f"Poisson residual {getattr(sim,'last_poisson_residual',0):.2e} · backend {sim.device} · t={sim.time_au:.3e} atomic units · spherical initial radius {np.mean(np.linalg.norm(snapshots[0][1],axis=1)):.2f} bohr")

with st.expander("Model assumptions and interpreting results"):
    st.markdown("""
    - One four-component Dirac spinor is propagated for each electron. Atom count, proton number per atom, and electron count are separate inputs.
    - The electron interaction includes a direct Coulomb Hartree field and four-component nonlocal Fock exchange, applied with an iterative Cayley substep. This is time-dependent Dirac-Hartree-Fock (TD-DHF), not a correlated method.
    - MP2/MP3/MP4 correlation, radiative self-energy diagrams, vacuum-polarization diagrams, and a photon field are not implemented. There is no no-pair projection or renormalized Dirac-sea treatment. No fourth-order QED claim is made; these require a specified renormalized QED formulation and substantially more machinery than the current real-space prototype.
    - Nuclei are classical Ehrenfest particles with approximate masses. Temperature rescales nuclear kinetic energy; it is not an electronic thermal occupation model.
    - The initial atom positions have equal distance from the origin and use a maximin spherical layout. Finite grid and periodic Poisson solve introduce finite-size errors.
    - The displayed tolerances monitor norm, orthogonality, and Poisson residual. They are not guarantees of physical accuracy. Repeat with finer grids and smaller time steps to assess convergence.
    """)
