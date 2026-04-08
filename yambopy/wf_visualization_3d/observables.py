"""
observables.py
--------------
Physical observables computed from the wavefunction at each time step.

All functions operate on raw numpy arrays (no class dependencies) so they can be
called directly from a propagator loop, a notebook, or the visualizer.

All quantities are in atomic units unless otherwise noted.

Functions
---------
compute_density(psi)
    ρ(r) = |ψ(r)|²

compute_probability_current(psi, lat)
    j(r) = Im[ψ*(r) ∇ψ(r)]   (FFT-based gradient, periodic BCs)

compute_current_magnitude(psi, lat)
    |j(r)| = sqrt(jx² + jy² + jz²)

compute_autocorrelation(psi0, psi_t)
    C(t) = ⟨ψ(0)|ψ(t)⟩   (complex overlap)

compute_energy(psi, lat, V=None)
    ⟨H⟩ = ⟨T⟩ + ⟨V⟩  in Hartree

compute_norm(psi)
    ‖ψ‖² = Σ |ψ|²

compute_current_flux(j, lat, axis, position)
    ∮ j · n̂ dA  integrated across a cross-sectional plane

compute_vorticity(j, lat)
    ω(r) = ∇ × j(r)  (curl of the probability current)

ObservableLogger
    Lightweight class to record time-series of all observables during a run.
"""

import numpy as np
from scipy.fft import fftn, ifftn
from ..lattice import rec_lat, vol_lat


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _build_g_cart(grid, lat):
    """
    Build Cartesian G-vector array for an (nx, ny, nz) grid.

    Parameters
    ----------
    grid : tuple (nx, ny, nz)
    lat  : ndarray (3, 3)  — real-space lattice vectors in bohr (rows)

    Returns
    -------
    g_cart : ndarray, shape (nx, ny, nz, 3)  in 1/bohr
    """
    nx, ny, nz = grid
    rlat = rec_lat(lat)   # (3,3) without 2π

    gx = np.fft.fftfreq(nx) * nx   # integer indices [0, 1, ..., N//2-1, -N//2, ..., -1]
    gy = np.fft.fftfreq(ny) * ny
    gz = np.fft.fftfreq(nz) * nz
    gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')

    # Physical G-vectors: G = 2π * (g1*b1 + g2*b2 + g3*b3)
    # where g are INTEGER FFT indices and b_i = rec_lat rows (a_i·b_j = δ_ij)
    g_int = np.stack([gx, gy, gz], axis=-1)
    return 2.0 * np.pi * (g_int @ rlat)


# ---------------------------------------------------------------------------
# Core observables
# ---------------------------------------------------------------------------

def compute_density(psi: np.ndarray) -> np.ndarray:
    """
    Probability density ρ(r) = |ψ(r)|².

    Parameters
    ----------
    psi : ndarray, complex, shape (nx, ny, nz)

    Returns
    -------
    rho : ndarray, float, shape (nx, ny, nz)
    """
    return np.abs(psi) ** 2


def compute_probability_current(psi: np.ndarray, lat: np.ndarray,
                                 workers: int = -1) -> np.ndarray:
    """
    Probability current j(r) = Im[ψ*(r) ∇ψ(r)] in atomic units.

    Uses the FFT-based gradient (exact for periodic BCs):
        ∂ψ/∂r_i = IFFT[i G_i(cart) · FFT[ψ]]

    Parameters
    ----------
    psi  : ndarray, complex, shape (nx, ny, nz)
    lat  : ndarray, float, shape (3, 3)  — lattice vectors in bohr
    workers : int, optional
        scipy FFT thread count (-1 = all).

    Returns
    -------
    j : ndarray, float, shape (3, nx, ny, nz)
        j[0]=jx, j[1]=jy, j[2]=jz  in atomic units (1/bohr⁴ for normalised ψ)
    """
    nx, ny, nz = psi.shape
    g_cart = _build_g_cart((nx, ny, nz), lat)    # (nx, ny, nz, 3)

    psi_k = fftn(psi, workers=workers)
    psi_conj = np.conj(psi)

    j = np.empty((3, nx, ny, nz), dtype=float)
    for i in range(3):
        dpsi = ifftn(1j * g_cart[..., i] * psi_k, workers=workers)
        # Im[ψ*(r) · dpsi(r)]
        j[i] = np.imag(psi_conj * dpsi)

    return j


def compute_current_magnitude(psi: np.ndarray, lat: np.ndarray,
                               workers: int = -1) -> np.ndarray:
    """
    |j(r)| = sqrt(jx² + jy² + jz²).

    Returns
    -------
    jmag : ndarray, float, shape (nx, ny, nz)
    """
    j = compute_probability_current(psi, lat, workers=workers)
    return np.sqrt(np.einsum('i...,i...->...', j, j))


def compute_norm(psi: np.ndarray) -> float:
    """
    Squared norm ‖ψ‖² = Σ |ψ(r)|².

    For a properly normalised wavefunction on a discrete grid this equals 1.

    Returns
    -------
    norm : float
    """
    return float(np.sum(np.abs(psi) ** 2))


def compute_autocorrelation(psi0: np.ndarray, psi_t: np.ndarray) -> complex:
    """
    Autocorrelation C(t) = ⟨ψ(0)|ψ(t)⟩ = Σ_r ψ*(0,r) ψ(t,r).

    |C(t)| = 1  for a single eigenstate (pure phase oscillation).
    |C(t)| < 1  for a superposition (quantum beating reveals E_n - E_m gaps).

    Parameters
    ----------
    psi0  : ndarray, complex, shape (nx, ny, nz)  — initial wavefunction
    psi_t : ndarray, complex, shape (nx, ny, nz)  — wavefunction at time t

    Returns
    -------
    C : complex scalar
    """
    return complex(np.sum(np.conj(psi0) * psi_t))


def compute_energy(psi: np.ndarray, lat: np.ndarray,
                   V: np.ndarray = None, workers: int = -1) -> float:
    """
    Energy expectation value ⟨H⟩ = ⟨T⟩ + ⟨V⟩ in Hartree.

    ⟨T⟩ = Σ_G |ψ_G|² |G|²/2  (Parseval, with normalisation)
    ⟨V⟩ = Σ_r |ψ(r)|² V(r)

    Parameters
    ----------
    psi : ndarray, complex, shape (nx, ny, nz)
    lat : ndarray, float, shape (3, 3)  lattice vectors in bohr
    V   : ndarray, float, shape (nx, ny, nz), optional
        Local potential in Hartree. Default: zero (free particle).
    workers : int, optional
        scipy FFT thread count.

    Returns
    -------
    E : float  (Hartree)
    """
    nx, ny, nz = psi.shape
    g_cart = _build_g_cart((nx, ny, nz), lat)
    k2 = np.einsum('...i,...i->...', g_cart, g_cart)   # |G|² in 1/bohr²
    T_op = k2 / 2.0

    psi_k = fftn(psi, workers=workers)
    N = nx * ny * nz
    T_exp = float(np.sum(np.abs(psi_k) ** 2 * T_op) / N)

    if V is not None:
        V_exp = float(np.sum(np.abs(psi) ** 2 * V))
    else:
        V_exp = 0.0

    return T_exp + V_exp


def compute_vorticity(j: np.ndarray, lat: np.ndarray,
                      workers: int = -1) -> np.ndarray:
    """
    Vorticity ω(r) = ∇ × j(r) using FFT-based curl.

    Highlights rotational features in the probability current (e.g. orbital
    angular momentum vortex patterns around atomic cores).

    Parameters
    ----------
    j    : ndarray, float, shape (3, nx, ny, nz)
        Probability current from compute_probability_current.
    lat  : ndarray, float, shape (3, 3)
    workers : int

    Returns
    -------
    omega : ndarray, float, shape (3, nx, ny, nz)
        ω = (∂jz/∂y - ∂jy/∂z,  ∂jx/∂z - ∂jz/∂x,  ∂jy/∂x - ∂jx/∂y)
    """
    nx, ny, nz = j.shape[1:]
    g_cart = _build_g_cart((nx, ny, nz), lat)   # (nx, ny, nz, 3)

    # FFT each component
    jk = np.array([fftn(j[i], workers=workers) for i in range(3)])  # (3, nx, ny, nz)

    # Gradient of each component: ∂j_a/∂r_b = IFFT[i G_b · jk_a]
    dj = np.empty((3, 3, nx, ny, nz), dtype=complex)
    for a in range(3):
        for b in range(3):
            dj[a, b] = ifftn(1j * g_cart[..., b] * jk[a], workers=workers)

    omega = np.empty((3, nx, ny, nz), dtype=float)
    omega[0] = np.real(dj[2, 1] - dj[1, 2])   # ∂jz/∂y - ∂jy/∂z
    omega[1] = np.real(dj[0, 2] - dj[2, 0])   # ∂jx/∂z - ∂jz/∂x
    omega[2] = np.real(dj[1, 0] - dj[0, 1])   # ∂jy/∂x - ∂jx/∂y

    return omega


def compute_current_flux(j: np.ndarray, lat: np.ndarray,
                          axis: int = 2, position: int = None) -> float:
    """
    Integrated probability current flux ∮ j · n̂ dA across a cross-section.

    Integrates j_axis over a plane perpendicular to *axis* at grid index *position*.
    Useful for measuring net electron flow through a cross-section of the supercell.

    Parameters
    ----------
    j        : ndarray, float, shape (3, nx, ny, nz)
    lat      : ndarray, float, shape (3, 3)
    axis     : int  (0=x, 1=y, 2=z)
    position : int, optional  (default: mid-plane)

    Returns
    -------
    flux : float  (atomic units)
    """
    nx, ny, nz = j.shape[1:]
    if position is None:
        position = [nx, ny, nz][axis] // 2

    if axis == 0:
        plane = j[0, position, :, :]
    elif axis == 1:
        plane = j[1, :, position, :]
    else:
        plane = j[2, :, :, position]

    # Area element for the cross-section in bohr² (parallelogram of the other two vectors)
    a = [0, 1, 2]
    a.remove(axis)
    v1 = lat[a[0]]
    v2 = lat[a[1]]
    cell_area = np.linalg.norm(np.cross(v1, v2))
    grid_area = plane.size
    dA = cell_area / grid_area

    return float(np.sum(plane) * dA)


# ---------------------------------------------------------------------------
# Observable logger
# ---------------------------------------------------------------------------

class ObservableLogger:
    """
    Record time-series of physical observables during a propagation run.

    Usage
    -----
    logger = ObservableLogger(psi0=propagator.psi, lat=data.lat)
    for _ in range(n_steps):
        propagator.step()
        logger.record(propagator)

    Arrays are available as logger.times, logger.norms, logger.energies,
    logger.autocorrelations, logger.current_flux_z.

    Parameters
    ----------
    psi0 : ndarray, complex
        Initial wavefunction for autocorrelation reference.
    lat  : ndarray, float, shape (3, 3)
        Lattice vectors (bohr).
    track_flux_axis : int or None
        Axis (0/1/2) along which to track current flux. None = skip.
    """

    def __init__(self, psi0: np.ndarray, lat: np.ndarray,
                 track_flux_axis: int = None):
        self._psi0 = psi0.copy()
        self._lat = lat
        self._flux_axis = track_flux_axis

        self.times: list = []
        self.norms: list = []
        self.energies: list = []
        self.autocorrelations: list = []
        self.current_flux: list = []

    def record(self, propagator) -> None:
        """
        Sample all observables from a WavefunctionPropagator and append to logs.

        Parameters
        ----------
        propagator : WavefunctionPropagator
        """
        psi = propagator.psi
        lat = self._lat

        self.times.append(propagator.time)
        self.norms.append(compute_norm(psi))
        self.energies.append(compute_energy(psi, lat, V=propagator.V))
        self.autocorrelations.append(compute_autocorrelation(self._psi0, psi))

        if self._flux_axis is not None:
            j = compute_probability_current(psi, lat)
            flux = compute_current_flux(j, lat, axis=self._flux_axis)
            self.current_flux.append(flux)

    def as_arrays(self) -> dict:
        """Return all logged data as numpy arrays in a dictionary."""
        d = {
            'time': np.array(self.times),
            'norm': np.array(self.norms),
            'energy': np.array(self.energies),
            'autocorrelation': np.array(self.autocorrelations),
        }
        if self.current_flux:
            d['current_flux'] = np.array(self.current_flux)
        return d

    def print_summary(self) -> None:
        """Print a brief statistical summary of the logged data."""
        d = self.as_arrays()
        t = d['time']
        norms = d['norm']
        energies = d['energy']
        C = np.abs(d['autocorrelation'])

        print(f"  Steps recorded : {len(t)}")
        print(f"  Time range     : {t[0]:.3f} → {t[-1]:.3f} a.u.")
        print(f"  Norm drift     : {norms.max() - norms.min():.2e}")
        print(f"  Energy drift   : {energies.max() - energies.min():.2e} Ha")
        print(f"  |C(t)| range   : {C.min():.4f} → {C.max():.4f}")

    def __len__(self):
        return len(self.times)
