"""
propagator.py
-------------
Split-Operator Fourier Transform (SOFT) wavefunction propagator.

Implements the symmetric Trotter-Suzuki decomposition of the time-evolution
operator U(dt) = exp(-i H dt) ≈ exp(-i V dt/2) · exp(-i T dt) · exp(-i V dt/2),
where T is diagonal in G-space (kinetic energy) and V is diagonal in real space
(local potential). This is a symplectic integrator: time-reversible and
energy-conserving to machine precision for the split scheme.

Units
-----
All quantities are in atomic units unless stated otherwise:
  - Length : bohr (a₀ ≈ 0.529 Å)
  - Energy : Hartree (Ha ≈ 27.21 eV)
  - Time   : ℏ/Ha ≈ 24.19 attoseconds (1 a.u. of time)
  - ℏ = 1, mₑ = 1

Physics
-------
For a single real eigenstate at Γ: j(r,t) = 0 at all times.
For a complex eigenstate (k ≠ 0) or superposition of bands: j ≠ 0.
Use WavefunctionLoader.load_multi_band to build superpositions.

Performance
-----------
~10-20 ms per step on a 100³ grid (numpy FFT). Achieve 50-100 Hz update rate.

Main class
----------
WavefunctionPropagator
    - step()                     : advance by one dt
    - step_n(n)                  : advance by n steps
    - get_density()              : |ψ(r,t)|²
    - get_probability_current()  : j(r,t) = Im[ψ* ∇ψ]  (atomic units)
    - get_autocorrelation()      : C(t) = ⟨ψ(0)|ψ(t)⟩
    - get_energy()               : ⟨H⟩ = ⟨T⟩ + ⟨V⟩ (Ha)
    - get_norm()                 : ‖ψ‖²  (should stay ≈ 1)
    - reset()                    : restore ψ to initial state
"""

import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len

from ..lattice import rec_lat
from .loaders import WavefunctionData


class WavefunctionPropagator:
    """
    Real-time wavefunction propagator using the split-operator SOFT method.

    Parameters
    ----------
    data : WavefunctionData
        Loaded (and optionally supercell-expanded) wavefunction.
    dt : float
        Time step in atomic units (default 0.01 a.u. ≈ 0.24 as).
        A smaller dt gives more accurate dynamics; a larger dt is faster.
    V : ndarray, float, shape (nx, ny, nz), optional
        Local potential energy grid in Hartree. If None, V = 0 (free particle).
        Can be set to a constant eigenvalue to enforce correct phase evolution
        for a single DFT eigenstate: V = E_n * ones (approximation).
    workers : int, optional
        Number of CPU threads for scipy FFT (default -1 = all cores).

    Attributes
    ----------
    psi : ndarray, complex, shape (nx, ny, nz)
        Current wavefunction (evolves in place).
    time : float
        Elapsed time in atomic units.
    step_count : int
        Number of steps taken.
    """

    def __init__(self, data: WavefunctionData, dt: float = 0.01,
                 V: np.ndarray = None, workers: int = -1):
        if data.psi is None:
            raise ValueError("WavefunctionData.psi is None — load data first.")

        self.dt = float(dt)
        self.workers = workers
        self.lat = data.lat
        self.grid = tuple(data.grid)     # always (nx, ny, nz) — spatial only
        self.volume = data.volume

        # Working copy and initial state
        self.psi = data.psi.astype(complex, copy=True)
        self._psi0 = self.psi.copy()

        # Potential — always (nx, ny, nz) regardless of spinor mode
        if V is None:
            self.V = np.zeros(self.grid, dtype=float)
        else:
            if V.shape != self.grid:
                raise ValueError(
                    f"V.shape {V.shape} does not match grid {self.grid}."
                )
            self.V = np.asarray(V, dtype=float)

        self.time = 0.0
        self.step_count = 0

        # Precompute propagators
        self._setup_kinetic_propagator()
        self._setup_potential_propagators()

    # ------------------------------------------------------------------
    # Spinor detection helpers
    # ------------------------------------------------------------------

    @property
    def is_spinor(self) -> bool:
        """True when psi has shape (2, nx, ny, nz)."""
        return self.psi.ndim == 4

    # ------------------------------------------------------------------
    # Setup (called once at construction)
    # ------------------------------------------------------------------

    def _setup_kinetic_propagator(self):
        """
        Build exp(-i T dt) in G-space.

        T(G) = ℏ²|G|²/2m = |G_cart|²/2  [atomic units, ℏ=m=1]

        The physical G-vectors corresponding to FFT index (gx, gy, gz) are:
            G_cart = 2π * (gx/nx * b1 + gy/ny * b2 + gz/nz * b3)
        where b_i = rec_lat(lat)[i] (without 2π), so the 2π must be inserted.
        """
        nx, ny, nz = self.grid
        rlat = rec_lat(self.lat)   # (3,3) without 2π factor

        # FFT integer frequency indices (zero-centered via fftfreq convention)
        gx = np.fft.fftfreq(nx) * nx   # [0, 1, ..., nx//2-1, -nx//2, ..., -1]
        gy = np.fft.fftfreq(ny) * ny
        gz = np.fft.fftfreq(nz) * nz
        gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')  # (nx, ny, nz)

        # Physical G-vectors: G = 2π * (g1*b1 + g2*b2 + g3*b3)
        # where g are INTEGER FFT indices and b_i = rec_lat rows (a_i·b_j = δ_ij)
        g_int = np.stack([gx, gy, gz], axis=-1)   # (..., 3)  integer indices
        g_cart = 2.0 * np.pi * (g_int @ rlat)     # (..., 3) in 1/bohr

        k2 = np.einsum('...i,...i->...', g_cart, g_cart)   # |G|²

        T = k2 / 2.0   # kinetic energy in Hartree (a.u.)
        self._kinetic_exp = np.exp(-1j * T * self.dt)

    def _setup_potential_propagators(self):
        """Build exp(-i V dt/2) and exp(-i V dt) in real space."""
        self._half_V_exp = np.exp(-1j * self.V * (self.dt / 2.0))
        self._full_V_exp = np.exp(-1j * self.V * self.dt)

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    def _soft_step_scalar(self, psi):
        """One SOFT step for a scalar (nx, ny, nz) wavefunction. Returns updated psi."""
        psi *= self._half_V_exp
        psi_k = fftn(psi, workers=self.workers)
        psi_k *= self._kinetic_exp
        psi = ifftn(psi_k, workers=self.workers)
        psi *= self._half_V_exp
        return psi

    def _soft_step_n_scalar(self, psi, n):
        """n merged SOFT steps for a scalar wavefunction. Returns updated psi."""
        psi *= self._half_V_exp
        for _ in range(n - 1):
            psi_k = fftn(psi, workers=self.workers)
            psi_k *= self._kinetic_exp
            psi = ifftn(psi_k, workers=self.workers)
            psi *= self._full_V_exp
        psi_k = fftn(psi, workers=self.workers)
        psi_k *= self._kinetic_exp
        psi = ifftn(psi_k, workers=self.workers)
        psi *= self._half_V_exp
        return psi

    def step(self):
        """
        Advance the wavefunction by one time step dt using the SOFT algorithm:
            ψ → exp(-i V dt/2) · IFFT[exp(-i T dt) · FFT[exp(-i V dt/2) · ψ]]

        For spinor wavefunctions (Option A), each component is propagated
        independently with the same scalar kinetic + local-potential SOFT step.
        Spin-orbit coupling between components is NOT included (Option A).
        """
        if self.is_spinor:
            self.psi[0] = self._soft_step_scalar(self.psi[0])
            self.psi[1] = self._soft_step_scalar(self.psi[1])
        else:
            self.psi = self._soft_step_scalar(self.psi)

        self.time += self.dt
        self.step_count += 1

    def step_n(self, n: int):
        """
        Advance by n steps (uses merged potential half-steps for efficiency).

        For spinor wavefunctions both components are advanced independently.
        """
        if n <= 0:
            return

        if self.is_spinor:
            self.psi[0] = self._soft_step_n_scalar(self.psi[0], n)
            self.psi[1] = self._soft_step_n_scalar(self.psi[1], n)
        else:
            self.psi = self._soft_step_n_scalar(self.psi, n)

        self.time += self.dt * n
        self.step_count += n

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def get_density(self) -> np.ndarray:
        """
        Probability density ρ(r,t) = |ψ(r,t)|².

        For spinor wavefunctions: ρ = |ψ↑|² + |ψ↓|²  (total density).

        Returns
        -------
        rho : ndarray, float, shape (nx, ny, nz)
        """
        if self.is_spinor:
            return np.abs(self.psi[0]) ** 2 + np.abs(self.psi[1]) ** 2
        return np.abs(self.psi) ** 2

    def get_spin_density(self) -> np.ndarray:
        """
        Spin density S_z(r,t) = |ψ↑|² − |ψ↓|².

        Only available for spinor wavefunctions (load_spinor=True).
        Positive values = spin-up dominates; negative = spin-down dominates.

        Returns
        -------
        Sz : ndarray, float, shape (nx, ny, nz)

        Raises
        ------
        ValueError if psi is scalar.
        """
        if not self.is_spinor:
            raise ValueError(
                "Spin density requires a spinor wavefunction. "
                "Load with load_spinor=True."
            )
        return np.abs(self.psi[0]) ** 2 - np.abs(self.psi[1]) ** 2

    def get_spin_vector(self) -> np.ndarray:
        """
        Full spin density vector S(r,t) = (S_x, S_y, S_z).

        Computed via Pauli matrices: S_i = ψ† σ_i ψ
            S_x = 2 Re[ψ↑* ψ↓]
            S_y = 2 Im[ψ↑* ψ↓]
            S_z = |ψ↑|² − |ψ↓|²

        Returns
        -------
        S : ndarray, float, shape (3, nx, ny, nz)
        """
        if not self.is_spinor:
            raise ValueError("Spin vector requires a spinor wavefunction.")
        off = np.conj(self.psi[0]) * self.psi[1]
        S_x = 2.0 * np.real(off)
        S_y = 2.0 * np.imag(off)
        S_z = np.abs(self.psi[0]) ** 2 - np.abs(self.psi[1]) ** 2
        return np.stack([S_x, S_y, S_z], axis=0)

    def get_probability_current(self) -> np.ndarray:
        """
        Probability current j(r,t) = Im[ψ*(r,t) ∇ψ(r,t)]  in atomic units.

        For spinor wavefunctions: j = j↑ + j↓  (total current, Option A).

        Uses FFT-based gradient: ∂ψ/∂x_i = IFFT[ i G_i · FFT[ψ] ]

        Returns
        -------
        j : ndarray, float, shape (3, nx, ny, nz)
        """
        nx, ny, nz = self.grid
        rlat = rec_lat(self.lat)

        gx = np.fft.fftfreq(nx) * nx
        gy = np.fft.fftfreq(ny) * ny
        gz = np.fft.fftfreq(nz) * nz
        gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')
        g_int = np.stack([gx, gy, gz], axis=-1)
        g_cart = 2.0 * np.pi * (g_int @ rlat)   # (nx, ny, nz, 3)

        j = np.zeros((3, nx, ny, nz), dtype=float)

        def _current_scalar(psi_s):
            psi_k = fftn(psi_s, workers=self.workers)
            psi_conj = np.conj(psi_s)
            j_s = np.empty((3, nx, ny, nz), dtype=float)
            for i in range(3):
                dpsi = ifftn(1j * g_cart[..., i] * psi_k, workers=self.workers)
                j_s[i] = np.imag(psi_conj * dpsi)
            return j_s

        if self.is_spinor:
            j += _current_scalar(self.psi[0])
            j += _current_scalar(self.psi[1])
        else:
            j = _current_scalar(self.psi)

        return j

    def get_autocorrelation(self) -> complex:
        """
        Autocorrelation function C(t) = ⟨ψ(0)|ψ(t)⟩.

        A real eigenstate gives |C(t)| = 1 (pure phase oscillation).
        A superposition gives oscillating |C(t)| revealing the energy spectrum.

        Returns
        -------
        C : complex scalar
        """
        return complex(np.sum(np.conj(self._psi0) * self.psi))

    def get_energy(self) -> float:
        """
        Energy expectation value ⟨H⟩ = ⟨T⟩ + ⟨V⟩ in Hartree.

        For spinor: ⟨T⟩ = ⟨T⟩↑ + ⟨T⟩↓, ⟨V⟩ = ⟨V⟩↑ + ⟨V⟩↓.

        Returns
        -------
        E : float (Hartree)
        """
        nx, ny, nz = self.grid
        rlat = rec_lat(self.lat)

        gx = np.fft.fftfreq(nx) * nx
        gy = np.fft.fftfreq(ny) * ny
        gz = np.fft.fftfreq(nz) * nz
        gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')
        g_int = np.stack([gx, gy, gz], axis=-1)
        g_cart = 2.0 * np.pi * (g_int @ rlat)
        k2 = np.einsum('...i,...i->...', g_cart, g_cart)
        T_op = k2 / 2.0
        N = nx * ny * nz

        if self.is_spinor:
            T_exp = 0.0
            V_exp = 0.0
            for s in range(2):
                psi_k = fftn(self.psi[s], workers=self.workers)
                T_exp += float(np.sum(np.abs(psi_k) ** 2 * T_op) / N)
                V_exp += float(np.sum(np.abs(self.psi[s]) ** 2 * self.V))
        else:
            psi_k = fftn(self.psi, workers=self.workers)
            T_exp = float(np.sum(np.abs(psi_k) ** 2 * T_op) / N)
            V_exp = float(np.sum(np.abs(self.psi) ** 2 * self.V))

        return T_exp + V_exp

    def get_norm(self) -> float:
        """
        Squared norm ‖ψ‖² = Σ |ψ(r)|² (should remain ≈ 1).

        For spinor: ||ψ||² = ||ψ↑||² + ||ψ↓||².

        Returns
        -------
        norm : float
        """
        return float(np.sum(np.abs(self.psi) ** 2))

    def get_current_magnitude(self) -> np.ndarray:
        """
        |j(r,t)| = sqrt(jx² + jy² + jz²).

        Returns
        -------
        jmag : ndarray, float, shape (nx, ny, nz)
        """
        j = self.get_probability_current()
        return np.sqrt(np.sum(j ** 2, axis=0))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Restore wavefunction to the initial state ψ(t=0)."""
        self.psi = self._psi0.copy()
        self.time = 0.0
        self.step_count = 0

    def set_potential(self, V: np.ndarray):
        """
        Update the potential and rebuild propagators.

        Parameters
        ----------
        V : ndarray, float, shape (nx, ny, nz), in Hartree
        """
        if V.shape != self.grid:
            raise ValueError(f"V.shape {V.shape} != grid {self.grid}.")
        self.V = np.asarray(V, dtype=float)
        self._setup_potential_propagators()

    def set_dt(self, dt: float):
        """Change the time step and rebuild propagators."""
        self.dt = float(dt)
        self._setup_kinetic_propagator()
        self._setup_potential_propagators()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> str:
        norm = self.get_norm()
        E = self.get_energy()
        C = self.get_autocorrelation()
        return (
            f"t={self.time:.4f} a.u. | step={self.step_count} | "
            f"norm={norm:.8f} | E={E:.6f} Ha | |C(t)|={abs(C):.6f}"
        )

    def __repr__(self):
        return (
            f"WavefunctionPropagator("
            f"grid={self.grid}, dt={self.dt} a.u., "
            f"t={self.time:.3f} a.u., step={self.step_count})"
        )
