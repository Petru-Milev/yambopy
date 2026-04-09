"""
loaders.py
----------
Data loading from Yambo/QE calculations into the wavefunction visualization pipeline.

Reads wavefunctions from the Yambo SAVE directory (ns.wf + ns.db1 netCDF files),
converts G-space plane-wave expansions to real-space grids via FFT, and bundles
structural information (lattice vectors, atomic positions, eigenvalues) into a
WavefunctionData container ready for propagation and visualization.

Main class
----------
WavefunctionLoader
    - load(ik, ib, grid)         : single band → WavefunctionData
    - load_multi_band(ik, bands) : superposition of bands (gives non-zero current)
    - get_eigenvalues(ik)        : band energies at k-point ik (eV)
    - get_info()                 : print database summary

WavefunctionData
    Container holding psi, lattice, atoms, and metadata after loading.
"""

import numpy as np

from ..dbs.wfdb import YamboWFDB
from ..dbs.electronsdb import YamboElectronsDB
from ..dbs.latticedb import YamboLatticeDB
from ..units import bohr2ang, ha2ev, chemical_symbols
from ..lattice import car_red, red_car, rec_lat, vol_lat


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class WavefunctionData:
    """
    Container for all data needed by propagator and visualizer.

    Attributes
    ----------
    psi : ndarray, complex
        Wavefunction in real space on the unit-cell grid (atomic units, normalized).
        Shape is (nx, ny, nz) for scalar (non-magnetic or collinear spin-polarized)
        or (2, nx, ny, nz) for a 2-component spinor (non-collinear / SOC).
    spinor_mode : str
        'scalar'  — single spin channel, shape (nx, ny, nz)
        'spinor'  — both spinor components, shape (2, nx, ny, nz)
    grid : ndarray, int, shape (3,)
        Grid dimensions [nx, ny, nz].
    lat : ndarray, float, shape (3, 3)
        Lattice vectors in bohr (rows = vectors).
    lat_ang : ndarray, float, shape (3, 3)
        Lattice vectors in Angstroms.
    rlat : ndarray, float, shape (3, 3)
        Reciprocal lattice vectors WITHOUT 2π (use 2π*rlat for physical G).
    atom_pos_car : ndarray, float, shape (natoms, 3)
        Atomic positions in Cartesian bohr.
    atom_pos_red : ndarray, float, shape (natoms, 3)
        Atomic positions in reduced (fractional) coordinates.
    atom_num : ndarray, int, shape (natoms,)
        Atomic numbers (Z).
    atom_sym : list of str, length natoms
        Element symbols.
    eigenvalues_ev : ndarray or None, shape (nspin, nbands) in eV
        Band eigenvalues at ik (loaded lazily via get_eigenvalues).
    kpoint : ndarray, shape (3,)
        K-point in crystal (reduced) coordinates.
    ik : int
        K-point index used.
    ib : int or list of int
        Band index (or list if superposition).
    spin : int
        Spin component used (0 or 1; for spinor mode both are loaded together).
    volume : float
        Unit cell volume in bohr³.
    norm : float
        Norm of the wavefunction (should be ≈ 1).
    """

    def __init__(self):
        self.psi = None
        self.spinor_mode = 'scalar'   # 'scalar' or 'spinor'
        self.grid = None
        self.lat = None
        self.lat_ang = None
        self.rlat = None
        self.atom_pos_car = None
        self.atom_pos_red = None
        self.atom_num = None
        self.atom_sym = None
        self.eigenvalues_ev = None
        self.kpoint = None
        self.ik = None
        self.ib = None
        self.spin = None
        self.volume = None
        self.norm = None

    @property
    def natoms(self):
        return len(self.atom_num) if self.atom_num is not None else 0

    @property
    def shape(self):
        return tuple(self.grid) if self.grid is not None else None

    @property
    def is_spinor(self):
        """True if psi has shape (2, nx, ny, nz) (2-component spinor)."""
        return self.spinor_mode == 'spinor'

    def __repr__(self):
        if self.psi is None:
            return "WavefunctionData(empty)"
        return (
            f"WavefunctionData("
            f"grid={list(self.grid)}, ik={self.ik}, ib={self.ib}, "
            f"spinor_mode={self.spinor_mode!r}, "
            f"natoms={self.natoms}, norm={self.norm:.6f})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class WavefunctionLoader:
    """
    Load wavefunctions and structural data from a Yambo SAVE folder.

    Wraps YamboWFDB (wavefunction G→r conversion) and YamboElectronsDB
    (eigenvalues, lattice, atoms). Lazy-loads databases on first access.

    Parameters
    ----------
    path : str
        Directory that contains the SAVE subfolder.
    save : str
        Name of the SAVE subfolder (default: 'SAVE').
    bands_range : list [first, last], optional
        Restrict which bands are read from disk to save memory.
        Both indices are inclusive and 1-based (Yambo convention).

    Example
    -------
    >>> loader = WavefunctionLoader('/path/to/run', save='SAVE')
    >>> data   = loader.load(ik=0, ib=3, grid=[60, 60, 60])
    >>> loader.get_info()
    """

    def __init__(self, path='.', save='SAVE', bands_range=None):
        self.path = path
        self.save = save
        self.bands_range = bands_range if bands_range is not None else []

        self._wfdb = None
        self._edb = None

    # ------------------------------------------------------------------
    # Private helpers — lazy database loading
    # ------------------------------------------------------------------

    def _ensure_wfdb(self):
        if self._wfdb is not None:
            return
        import os
        full_path = os.path.join(self.path, self.save)
        print(f"[WavefunctionLoader] Reading wavefunction DB from {full_path!r} ...")
        self._wfdb = YamboWFDB(
            path=self.path,
            save=self.save,
            bands_range=self.bands_range,
        )
        wf = self._wfdb
        print(
            f"  nkpoints={wf.nkpoints}, nbands={wf.nbands}, "
            f"nspin={wf.nspin}, nspinor={wf.nspinor}, "
            f"fft_box={list(wf.fft_box)}"
        )

    def _ensure_edb(self):
        if self._edb is not None:
            return
        import os
        db1 = os.path.join(self.path, self.save)
        print(f"[WavefunctionLoader] Reading electrons DB from {db1!r} ...")
        self._edb = YamboElectronsDB.from_db_file(folder=db1, filename='ns.db1')
        print(f"  nelectrons={self._edb.nelectrons}, nbands={self._edb.nbands}")

    # ------------------------------------------------------------------
    # Internal: build WavefunctionData from the latdb embedded in wfdb
    # ------------------------------------------------------------------

    def _build_structural_data(self, data: WavefunctionData):
        """Fill lattice and atomic fields of *data* from wfdb.ydb."""
        latdb = self._wfdb.ydb

        lat = latdb.lat                          # (3,3) bohr, rows = vectors
        atom_pos_car = latdb.car_atomic_positions # (natoms,3) bohr
        atom_pos_red = latdb.red_atomic_positions # (natoms,3) fractional
        atom_num = np.asarray(latdb.atomic_numbers, dtype=int)

        # Element symbols from atomic numbers
        atom_sym = []
        for Z in atom_num:
            try:
                atom_sym.append(chemical_symbols[Z])
            except (IndexError, KeyError):
                atom_sym.append(str(Z))

        data.lat = lat
        data.lat_ang = lat * bohr2ang
        data.rlat = rec_lat(lat)          # without 2π
        data.atom_pos_car = atom_pos_car
        data.atom_pos_red = atom_pos_red
        data.atom_num = atom_num
        data.atom_sym = atom_sym
        data.volume = vol_lat(lat)

    def _extract_psi(self, wfc_rs, spin=0, spinor=0):
        """
        Extract a single (nx, ny, nz) complex array from the full
        (nspin, nspinor, nx, ny, nz) output of wfcG2r.
        """
        if wfc_rs.ndim == 5:
            # (nspin, nspinor, nx, ny, nz)
            psi = wfc_rs[spin, spinor]
        elif wfc_rs.ndim == 3:
            # already (nx, ny, nz)
            psi = wfc_rs
        else:
            raise ValueError(
                f"Unexpected wfcG2r output shape {wfc_rs.shape}. "
                "Expected (nspin, nspinor, nx, ny, nz) or (nx, ny, nz)."
            )
        return psi.copy()

    def _extract_spinor_psi(self, wfc_rs, spin=0):
        """
        Extract both spinor components as a (2, nx, ny, nz) complex array.

        Used for non-collinear / SOC systems where nspinor == 2.
        Both ψ↑ and ψ↓ are stacked on axis 0.

        Parameters
        ----------
        wfc_rs : ndarray
            Output of wfcG2r, shape (nspin, nspinor, nx, ny, nz) or (nx, ny, nz).
        spin : int
            Spin row to use (only relevant when nspin > 1, rare for non-collinear).

        Returns
        -------
        psi : ndarray, complex, shape (2, nx, ny, nz)
        """
        if wfc_rs.ndim == 5:
            # (nspin, nspinor, nx, ny, nz)
            nspinor = wfc_rs.shape[1]
            psi_up = wfc_rs[spin, 0].copy()
            if nspinor >= 2:
                psi_dn = wfc_rs[spin, 1].copy()
            else:
                # Only one spinor component available (collinear calc loaded in spinor mode)
                psi_dn = np.zeros_like(psi_up)
        elif wfc_rs.ndim == 3:
            # scalar wfcG2r output — treat as spin-up only
            psi_up = wfc_rs.copy()
            psi_dn = np.zeros_like(psi_up)
        else:
            raise ValueError(
                f"Unexpected wfcG2r output shape {wfc_rs.shape}."
            )
        return np.stack([psi_up, psi_dn], axis=0)  # (2, nx, ny, nz)

    @staticmethod
    def _normalize(psi):
        """
        Normalize psi in place. Works for both scalar (nx, ny, nz)
        and spinor (2, nx, ny, nz) arrays. Returns (psi, norm).
        """
        norm = float(np.sqrt(np.sum(np.abs(psi) ** 2)))
        if norm > 0.0:
            psi = psi / norm
        return psi, norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, ik=0, ib=0, grid=None, spin=0, spinor=0, load_spinor=False):
        """
        Load a single wavefunction band at a given k-point and convert to real space.

        Parameters
        ----------
        ik : int
            K-point index in the irreducible BZ (0-based).
        ib : int
            Band index (0-based, relative to bands_range[0] if set).
        grid : list [nx, ny, nz], optional
            Real-space FFT grid. Must be >= Yambo's fft_box on each axis.
            Defaults to the minimal fft_box from the database.
        spin : int
            Spin component to extract (0 or 1).
            For collinear spin-polarized systems: spin=0 loads spin-up,
            spin=1 loads spin-down (each as a scalar wavefunction).
        spinor : int
            Spinor component to extract when load_spinor=False (0 or 1).
            Ignored when load_spinor=True.
        load_spinor : bool
            If True, load both spinor components (ψ↑ and ψ↓) simultaneously
            into a single (2, nx, ny, nz) array. Required for correct treatment
            of non-collinear / SOC wavefunctions (Option A propagation).
            If False (default), load a single scalar component as before.

        Returns
        -------
        WavefunctionData
            data.spinor_mode == 'spinor' when load_spinor=True,
            data.spinor_mode == 'scalar' otherwise.

        Notes
        -----
        For a real eigenstate at Γ (k=0), the probability current j = Im[ψ*∇ψ]
        is identically zero. Use load_multi_band or choose k≠0 for non-trivial
        current dynamics.

        For spin-polarized (collinear) systems, each spin channel is a valid
        scalar wavefunction. Load spin=0 and spin=1 separately to compare them.
        """
        self._ensure_wfdb()

        grid = list(grid) if grid is not None else list(self._wfdb.fft_box)

        mode_str = "spinor" if load_spinor else f"spinor={spinor}"
        print(f"[WavefunctionLoader] FFT ik={ik}, ib={ib}, spin={spin}, "
              f"{mode_str}, grid={grid} ...")
        wfc_rs = self._wfdb.wfcG2r(ik=ik, ib=ib, grid=grid)

        if load_spinor:
            psi = self._extract_spinor_psi(wfc_rs, spin=spin)
            spinor_mode = 'spinor'
        else:
            psi = self._extract_psi(wfc_rs, spin=spin, spinor=spinor)
            spinor_mode = 'scalar'

        psi, norm = self._normalize(psi)

        data = WavefunctionData()
        data.psi = psi
        data.spinor_mode = spinor_mode
        data.grid = np.array(grid, dtype=int)
        data.ik = ik
        data.ib = ib
        data.spin = spin
        data.norm = norm
        data.kpoint = self._wfdb.get_iBZ_kpt(ik)

        self._build_structural_data(data)
        return data

    def load_multi_band(self, ik=0, bands=None, grid=None, spin=0, spinor=0,
                        weights=None, load_spinor=False):
        """
        Build a coherent superposition of multiple bands.

        A superposition ψ = Σ_n c_n ψ_n(r) has a non-zero probability current
        even at Γ, making it useful for visualizing orbital flow dynamics.

        Parameters
        ----------
        ik : int
            K-point index in the irreducible BZ.
        bands : list of int
            Band indices to include in the superposition.
        grid : list [nx, ny, nz], optional
            Real-space FFT grid.
        spin : int
            Spin channel (0 or 1).
        spinor : int
            Spinor component when load_spinor=False.
        weights : array-like of complex, optional
            Mixing coefficients c_n. Internally normalized to unit total weight.
            Default: equal-weight mixture (all 1/sqrt(N)).
        load_spinor : bool
            If True, load both spinor components for each band and build the
            superposition in spinor space → psi shape (2, nx, ny, nz).

        Returns
        -------
        WavefunctionData
            data.ib is set to the list of band indices used.
        """
        self._ensure_wfdb()

        if bands is None:
            bands = [0, 1]
        if len(bands) < 2:
            raise ValueError("load_multi_band requires at least 2 bands.")

        grid = list(grid) if grid is not None else list(self._wfdb.fft_box)

        if weights is None:
            weights = np.ones(len(bands), dtype=complex) / np.sqrt(len(bands))
        else:
            weights = np.asarray(weights, dtype=complex)
            weights = weights / np.linalg.norm(weights)

        # First band to get shape
        wfc_rs = self._wfdb.wfcG2r(ik=ik, ib=bands[0], grid=grid)
        if load_spinor:
            psi0 = self._extract_spinor_psi(wfc_rs, spin=spin)
        else:
            psi0 = self._extract_psi(wfc_rs, spin=spin, spinor=spinor)
        psi0, _ = self._normalize(psi0)
        psi_super = weights[0] * psi0

        for i, ib in enumerate(bands[1:], 1):
            wfc_rs = self._wfdb.wfcG2r(ik=ik, ib=ib, grid=grid)
            if load_spinor:
                psi_i = self._extract_spinor_psi(wfc_rs, spin=spin)
            else:
                psi_i = self._extract_psi(wfc_rs, spin=spin, spinor=spinor)
            psi_i, _ = self._normalize(psi_i)
            psi_super = psi_super + weights[i] * psi_i

        psi_super, norm = self._normalize(psi_super)

        data = WavefunctionData()
        data.psi = psi_super
        data.spinor_mode = 'spinor' if load_spinor else 'scalar'
        data.grid = np.array(grid, dtype=int)
        data.ik = ik
        data.ib = list(bands)
        data.spin = spin
        data.norm = norm
        data.kpoint = self._wfdb.get_iBZ_kpt(ik)

        self._build_structural_data(data)
        return data

    def get_eigenvalues(self, ik=0):
        """
        Return band eigenvalues at k-point ik.

        Returns
        -------
        eigenvalues : ndarray, shape (nspin, nbands) in eV
        """
        self._ensure_edb()
        return self._edb.eigenvalues_ibz[:, ik, :]

    def get_info(self):
        """Print a summary of the loaded databases."""
        self._ensure_wfdb()
        wf = self._wfdb
        latdb = wf.ydb
        print("=" * 60)
        print("  Wavefunction Database (ns.wf)")
        print("=" * 60)
        print(f"  nkpoints (IBZ) : {wf.nkpoints}")
        print(f"  nbands         : {wf.nbands}")
        print(f"  nspin          : {wf.nspin}")
        print(f"  nspinor        : {wf.nspinor}")
        print(f"  FFT box        : {list(wf.fft_box)}")
        print()
        print("  Lattice vectors (bohr):")
        for i, v in enumerate(latdb.lat):
            print(f"    a{i+1} = [{v[0]:10.5f}  {v[1]:10.5f}  {v[2]:10.5f}]")
        print()
        print(f"  Cell volume    : {vol_lat(latdb.lat):.4f} bohr³")
        print()
        print("  Atoms:")
        for Z, pos in zip(latdb.atomic_numbers, latdb.red_atomic_positions):
            try:
                sym = chemical_symbols[Z]
            except Exception:
                sym = str(Z)
            print(f"    {sym:3s}  [{pos[0]:.4f}  {pos[1]:.4f}  {pos[2]:.4f}]")
        print("=" * 60)

    @property
    def nkpoints(self):
        self._ensure_wfdb()
        return self._wfdb.nkpoints

    @property
    def nbands(self):
        self._ensure_wfdb()
        return self._wfdb.nbands

    @property
    def fft_box(self):
        self._ensure_wfdb()
        return self._wfdb.fft_box
