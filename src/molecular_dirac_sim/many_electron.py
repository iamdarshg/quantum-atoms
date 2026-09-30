"""Finite-grid time-dependent Dirac-Hartree-Fock molecular prototype.

Propagates one occupied four-spinor per electron (neutral molecule: sum Z),
with direct Coulomb and nonlocal Fock exchange fields. Correlation beyond
time-dependent Hartree-Fock is not included.
Nuclei are classical Ehrenfest particles and may be thermostatted/ramped.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .dirac import C_AU, DiracGrid, DiracPropagator, Z

KB_HARTREE_PER_K = 3.166811563e-6
BOHR_PER_ANGSTROM = 1.889726125


@dataclass
class ErrorBounds:
    """User-set numerical controls; these report discretization limits, not certified bounds."""
    dt_au: float = 1e-7
    norm_drift: float = 1e-10
    orbital_overlap: float = 1e-10
    poisson_residual: float = 1e-10
    grid_spacing_bohr: float = 0.5

    def validate(self):
        for name in ("dt_au", "norm_drift", "orbital_overlap", "poisson_residual", "grid_spacing_bohr"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def max_separation_sphere(count: int, radius_bohr: float = 7.0) -> np.ndarray:
    """Equal-radius coordinates with approximately maximal pairwise spacing.

    A deterministic Fibonacci-sphere initialization is improved by tangent
    repulsion. Every atom remains exactly ``radius_bohr`` from the origin.
    """
    if count < 1 or radius_bohr <= 0:
        raise ValueError("count and radius must be positive")
    if count == 1:
        return np.zeros((1, 3))
    i = np.arange(count, dtype=float)
    z = 1.0 - 2.0*(i+0.5)/count
    phi = np.pi*(3.0-np.sqrt(5.0))*i
    p = np.stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z), axis=1)
    for _ in range(300):
        delta = p[:,None,:]-p[None,:,:]
        d2 = np.sum(delta*delta,axis=2)+np.eye(count)
        force = np.sum(delta/(d2[:,:,None]**1.5),axis=1)
        force -= np.sum(force*p,axis=1)[:,None]*p
        p += 0.012*force
        p /= np.linalg.norm(p,axis=1)[:,None]
    return p*radius_bohr


def _orthonormalize(states: np.ndarray, dv: float) -> np.ndarray:
    result=[]
    for state in states:
        v=state.copy()
        for prior in result:
            v -= np.sum(np.conj(prior)*v)*dv*prior
        norm=np.sqrt(np.sum(np.abs(v)**2)*dv)
        if norm < 1e-12:
            raise RuntimeError("Initial orbital basis became linearly dependent; increase grid size")
        result.append(v/norm)
    return np.asarray(result)


class MolecularDiracHartree:
    """Many-electron 3-D Dirac spinor dynamics with direct Coulomb mean field.

    ``elements`` lists atom labels; ``proton_numbers`` and ``electron_count``
    may be supplied independently for ions or custom nuclear charges.
    """
    def __init__(self, elements, grid: DiracGrid | None=None, errors: ErrorBounds | None=None,
                 temperature_k: float=300.0, temperature_end_k: float | None=None,
                 temperature_ramp_steps: int=0, seed: int=7, radius_bohr: float=3.0,
                 proton_numbers=None, electron_count: int | None=None,
                 exchange_iterations: int=3):
        raw_elements=list(elements)
        if not raw_elements: raise ValueError("Provide at least one atom")
        if proton_numbers is None:
            proton_numbers=[Z[el] if isinstance(el,str) and el in Z else int(el) for el in raw_elements]
        if len(proton_numbers)!=len(raw_elements) or any(int(p)<1 for p in proton_numbers):
            raise ValueError("Each atom needs one positive integer proton count")
        self.nuclear_charges=np.asarray(proton_numbers,dtype=float)
        if any(not np.isfinite(p) or p!=int(p) for p in self.nuclear_charges):
            raise ValueError("Proton counts must be positive integers")
        self.elements=[el if isinstance(el,str) and el in Z and Z[el]==int(p) else f"Z={int(p)}" for el,p in zip(raw_elements,self.nuclear_charges)]
        self.nelectrons=int(self.nuclear_charges.sum()) if electron_count is None else int(electron_count)
        if self.nelectrons<1: raise ValueError("Electron count must be at least one")
        self.exchange_iterations=int(exchange_iterations)
        if self.exchange_iterations<1: raise ValueError("exchange_iterations must be at least one")
        self.grid=grid or DiracGrid(device="cuda")
        basis_capacity=4*int(np.prod(self.grid.shape))
        if self.nelectrons>basis_capacity:
            raise ValueError(f"{self.nelectrons} electrons exceed the {basis_capacity}-spinor grid basis; increase grid dimensions")
        self.errors=errors or ErrorBounds(grid_spacing_bohr=self.grid.spacing_bohr)
        self.errors.validate()
        if not np.isclose(self.errors.grid_spacing_bohr,self.grid.spacing_bohr):
            raise ValueError("ErrorBounds.grid_spacing_bohr must match DiracGrid.spacing_bohr")
        self.propagator=DiracPropagator(self.grid)
        self.device=self.propagator.device
        self.positions=max_separation_sphere(len(elements),radius_bohr)
        self.masses=np.maximum(2*self.nuclear_charges,1.0)  # rough isotope masses in electron masses
        self.temperature_start=float(temperature_k)
        self.temperature_end=float(temperature_end_k if temperature_end_k is not None else temperature_k)
        self.ramp_steps=int(temperature_ramp_steps)
        if min(self.temperature_start,self.temperature_end)<0 or self.ramp_steps<0:
            raise ValueError("Temperatures and ramp steps must be nonnegative")
        rng=np.random.default_rng(seed)
        self.velocities=rng.normal(size=(len(elements),3))*np.sqrt(KB_HARTREE_PER_K*self.temperature_start/self.masses[:,None])
        self.velocities-=np.average(self.velocities,axis=0,weights=self.masses)
        self.time_au=0.0; self.step_index=0
        self.coords=self._grid_coordinates()
        self.psi=self._initial_orbitals()
        if self.device != "cpu":
            import torch
            self.psi=torch.as_tensor(self.psi,dtype=torch.complex128,device=self.device)
        self.initial_norm=self.norm()
        self._kx=2*np.pi*np.fft.fftfreq(self.grid.shape[0],d=self.grid.spacing_bohr)[:,None,None]
        self._ky=2*np.pi*np.fft.fftfreq(self.grid.shape[1],d=self.grid.spacing_bohr)[None,:,None]
        self._kz=2*np.pi*np.fft.fftfreq(self.grid.shape[2],d=self.grid.spacing_bohr)[None,None,:]
        self._k2=self._kx**2+self._ky**2+self._kz**2

    def _grid_coordinates(self):
        shape=np.array(self.grid.shape); dx=self.grid.spacing_bohr
        return (np.indices(self.grid.shape).transpose(1,2,3,0)-(shape-1)/2)*dx

    def _initial_orbitals(self):
        """Localized Gaussian spin orbitals, filled two spin states per mode."""
        dx=self.grid.spacing_bohr; coords=self.coords
        # One localized spatial mode per atom, then shell-like radial modes.
        states=[]; atom_electrons=np.zeros(len(self.elements),int)
        quotas=self.nelectrons*self.nuclear_charges/self.nuclear_charges.sum()
        counts=np.floor(quotas).astype(int)
        remainder=self.nelectrons-int(counts.sum())
        if remainder: counts[np.argsort(-(quotas-counts))[:remainder]]+=1
        hosts=[idx for idx,count in enumerate(counts) for _ in range(int(count))]
        for eidx in hosts:
                atom_electrons[eidx]+=1
                shell=(atom_electrons[eidx]-1)//2
                spin=(atom_electrons[eidx]-1)%2
                center=self.positions[eidx]
                width=0.65+0.22*shell
                r=coords-center
                env=np.exp(-np.sum(r*r,axis=-1)/(2*width*width))
                # Distinct orbital modes; hydrogenic-inspired radial nodes and
                # angular factors form a seed basis, not an SCF eigenbasis.
                mode=(atom_electrons[eidx]-1)//2
                if shell:
                    env *= (1.0-np.sqrt(np.sum(r*r,axis=-1))/(width*(1+shell)))
                direction=np.eye(3)[mode%3]
                scalar=env*(1.0+0.08*np.dot(r,direction))
                state=np.zeros((4,*self.grid.shape),complex)
                state[spin]=scalar
                states.append(state)
        return _orthonormalize(np.asarray(states),dx**3)

    def temperature_at(self, step_index: int | None=None):
        k=self.step_index if step_index is None else step_index
        if self.ramp_steps <= 0: return self.temperature_end
        f=min(max(k/self.ramp_steps,0.0),1.0)
        return self.temperature_start+(self.temperature_end-self.temperature_start)*f

    def _density(self):
        return np.sum(np.abs(self._to_numpy(self.psi))**2,axis=(0,1))

    def electrostatic_potential(self):
        """Nuclear attraction plus direct Hartree field.

        The nonlocal exchange operator is applied separately; retaining each
        occupied orbital's direct self-field lets its Fock self-term cancel it.
        """
        soft=max(0.2,self.grid.spacing_bohr/2)
        v_nuc=np.zeros(self.grid.shape,float)
        for q,r0 in zip(self.nuclear_charges,self.positions):
            r2=np.sum((self.coords-r0)**2,axis=-1)+soft**2
            v_nuc-=q/np.sqrt(r2)
        density=self._density()
        rho_k=np.fft.fftn(density)
        kernel=np.zeros_like(self._k2); mask=self._k2>0
        kernel[mask]=4*np.pi/self._k2[mask]
        vh=np.fft.ifftn(kernel*rho_k).real
        residual=np.zeros_like(rho_k)
        residual[mask]=self._k2[mask]*np.fft.fftn(vh)[mask]-4*np.pi*rho_k[mask]
        self.last_poisson_residual=float(np.linalg.norm(residual[mask])/max(np.linalg.norm((4*np.pi*rho_k)[mask]),1e-30))
        if self.last_poisson_residual>self.errors.poisson_residual:
            raise RuntimeError(f"Poisson residual {self.last_poisson_residual:.3g} exceeds bound {self.errors.poisson_residual:.3g}")
        return np.broadcast_to(v_nuc+vh,(self.nelectrons,*self.grid.shape)).copy()

    def _exchange_action(self, orbitals):
        r"""Apply K psi_i(r)=-sum_j psi_j(r) int(psi_j†psi_i/|r-r'|)dr'.

        This four-component nonlocal Fock operator is evaluated by spectral
        Poisson solves. Cost is O(N_e^2 FFTs); the finite periodic-box Coulomb
        kernel and grid convergence remain material numerical limitations.
        """
        kernel=np.zeros_like(self._k2); mask=self._k2>0
        kernel[mask]=4*np.pi/self._k2[mask]
        dv=self.grid.spacing_bohr**3
        if self.device != "cpu":
            import torch
            a=orbitals if isinstance(orbitals,torch.Tensor) else torch.as_tensor(orbitals,dtype=torch.complex128,device=self.device)
            k=torch.as_tensor(kernel,dtype=torch.complex128,device=self.device)
            out=torch.zeros_like(a)
            for i in range(a.shape[0]):
                for j in range(a.shape[0]):
                    overlap=torch.sum(torch.conj(a[j])*a[i],dim=0)
                    phi=torch.fft.ifftn(k*torch.fft.fftn(overlap))*dv
                    out[i]-=a[j]*phi.unsqueeze(0)
            return out
        a=self._to_numpy(orbitals)
        out=np.zeros_like(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                overlap=np.sum(np.conj(a[j])*a[i],axis=0)
                phi=np.fft.ifftn(kernel*np.fft.fftn(overlap))*dv
                out[i]-=a[j]*phi[None,...]
        return out

    def _exchange_cayley(self, orbitals, dt):
        """Apply a fixed-point Cayley step for the frozen Hermitian Fock field."""
        if self.device != "cpu":
            import torch
            old=orbitals if isinstance(orbitals,torch.Tensor) else torch.as_tensor(orbitals,dtype=torch.complex128,device=self.device)
        else:
            old=self._to_numpy(orbitals)
        rhs=old-0.5j*dt*self._exchange_action(old)
        guess=rhs
        for _ in range(self.exchange_iterations):
            guess=rhs-0.5j*dt*self._exchange_action(guess)
        return guess

    def _gram_matrix(self):
        flat=self._to_numpy(self.psi).reshape(self.nelectrons,-1)
        return flat.conj()@flat.T*self.grid.spacing_bohr**3

    def step(self):
        """Advance all occupied electron spinors and classical nuclei by one dt."""
        dt=self.errors.dt_au
        vfields=self.electrostatic_potential()
        # Batch axis is orbitals; each orbital sees its own corrected scalar field.
        if self.device != "cpu":
            import torch
            fields=torch.as_tensor(vfields,dtype=torch.float64,device=self.device)
        else: fields=vfields
        self.psi=self.propagator.step(self.psi,fields,dt)
        self.psi=self._exchange_cayley(self.psi,dt)
        if self.device == "cpu":
            self.psi=_orthonormalize(self.psi,self.grid.spacing_bohr**3)
        else:
            # QR keeps the occupied-orbital subspace orthonormal on device.
            import torch
            dv=self.grid.spacing_bohr**3
            flat=self.psi.reshape(self.nelectrons,-1).T*math.sqrt(dv)
            q,_=torch.linalg.qr(flat,mode="reduced")
            self.psi=(q.T/math.sqrt(dv)).reshape_as(self.psi)
        self._update_nuclei(dt)
        self.time_au+=dt; self.step_index+=1
        norm_drift=abs(self.norm()-self.initial_norm)/self.initial_norm
        if norm_drift>self.errors.norm_drift:
            raise RuntimeError(f"Spinor norm drift {norm_drift:.3g} exceeds bound {self.errors.norm_drift:.3g}")
        gram=self._gram_matrix()
        self.last_overlap_error=float(np.max(np.abs(gram-np.eye(self.nelectrons))))
        if self.last_overlap_error>self.errors.orbital_overlap:
            raise RuntimeError(f"Orbital overlap error {self.last_overlap_error:.3g} exceeds bound {self.errors.orbital_overlap:.3g}")
        return {"time_au":self.time_au,"temperature_k":self.temperature_at(),"norm":self.norm(),"norm_drift":norm_drift,"overlap_error":self.last_overlap_error,"poisson_residual":getattr(self,"last_poisson_residual",0.0)}

    def _to_numpy(self, value):
        if hasattr(value,"detach"): return value.detach().cpu().numpy()
        return np.asarray(value)

    def _update_nuclei(self,dt):
        """Ehrenfest Coulomb forces and optional linear kinetic-temperature ramp."""
        rho=self._density(); forces=np.zeros_like(self.positions); soft=max(.2,self.grid.spacing_bohr/2)
        for a,(za,ra) in enumerate(zip(self.nuclear_charges,self.positions)):
            dr=self.coords-ra; r2=np.sum(dr*dr,axis=-1)+soft**2
            forces[a]+=za*np.sum(rho[...,None]*dr/(r2[...,None]**1.5),axis=(0,1,2))*self.grid.spacing_bohr**3
            for b,(zb,rb) in enumerate(zip(self.nuclear_charges,self.positions)):
                if a==b: continue
                d=ra-rb; r=np.linalg.norm(d)+1e-9
                forces[a]+=za*zb*d/r**3
        acc=forces/self.masses[:,None]
        self.positions+=self.velocities*dt+0.5*acc*dt*dt
        self.velocities+=acc*dt
        target=self.temperature_at(self.step_index+1)
        self.velocities-=np.average(self.velocities,axis=0,weights=self.masses)
        ke=.5*np.sum(self.masses[:,None]*self.velocities**2)
        dof=max(1,3*len(self.elements)-3)
        if target>0 and ke>1e-30:
            self.velocities*=math.sqrt((.5*dof*KB_HARTREE_PER_K*target)/ke)

    def norm(self):
        return float(np.sum(np.abs(self._to_numpy(self.psi))**2)*self.grid.spacing_bohr**3)
