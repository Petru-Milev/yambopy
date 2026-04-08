"""
supercell.py
------------
Expand a unit-cell wavefunction to a periodic supercell for visualization.

Takes a real-space wavefunction on an (nx × ny × nz) unit-cell grid and tiles
it to a (rx·nx × ry·ny × rz·nz) supercell grid using periodic boundary
conditions (simple np.tile). Lattice vectors are scaled by the replication
factors and all atomic positions are replicated accordingly.

Main class
----------
SupercellExpander
    - expand(data, replicas)  : WavefunctionData → expanded WavefunctionData
    - expand_psi(psi, reps)   : convenience for raw arrays
"""

import numpy as np
from .loaders import WavefunctionData
from ..lattice import rec_lat, vol_lat, red_car
from ..units import bohr2ang


class SupercellExpander:
    """
    Tile a unit-cell wavefunction into a periodic supercell.

    The wavefunction is replicated by straightforward np.tile — valid because
    the Bloch form ψ_k(r) is periodic on the unit cell (modulo the Bloch phase,
    which is absorbed into the plane-wave coefficients for Gamma-point or when
    forming real combinations).

    Parameters
    ----------
    replicas : tuple of int, length 3
        (rx, ry, rz) — number of unit-cell copies along each lattice direction.
        E.g. (3, 3, 1) gives the 3×3×1 supercell from the project spec.

    Example
    -------
    >>> expander = SupercellExpander(replicas=(3, 3, 1))
    >>> sc_data  = expander.expand(uc_data)
    """

    def __init__(self, replicas=(3, 3, 1)):
        self.replicas = tuple(int(r) for r in replicas)
        if len(self.replicas) != 3:
            raise ValueError("replicas must be a 3-tuple (rx, ry, rz).")
        if any(r < 1 for r in self.replicas):
            raise ValueError("All replica counts must be >= 1.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, data: WavefunctionData) -> WavefunctionData:
        """
        Expand all fields of a WavefunctionData to the supercell.

        Parameters
        ----------
        data : WavefunctionData
            Loaded unit-cell wavefunction (output of WavefunctionLoader.load).

        Returns
        -------
        WavefunctionData
            New container with psi, lattice, and atoms replicated to the supercell.
            The supercell norm is re-normalized to 1.
        """
        rx, ry, rz = self.replicas

        # --- wavefunction ---
        # wfcG2r computes u_k(r), the cell-periodic part of the Bloch wavefunction.
        # The full Bloch state is ψ_k(r) = e^{ik·r} u_k(r).
        # For k=Γ (k=0), ψ = u, so simple tiling is correct.
        # For k≠0, we must apply the Bloch phase before tiling for physical correctness.
        kpt = data.kpoint  # crystal (reduced) coordinates
        k_norm = np.linalg.norm(kpt)
        if k_norm > 1e-8:
            print(f"[SupercellExpander] k = {kpt} (non-Gamma): applying Bloch phase e^{{ik·r}}")
            psi_with_phase = self._apply_bloch_phase(data.psi, kpt, data.lat)
            psi_sc = self._tile_psi(psi_with_phase)
        else:
            psi_sc = self._tile_psi(data.psi)

        # re-normalize on supercell grid
        norm = float(np.sqrt(np.sum(np.abs(psi_sc) ** 2)))
        if norm > 0:
            psi_sc = psi_sc / norm

        # --- lattice vectors: scale each row by its replication factor ---
        lat_sc = data.lat.copy()
        lat_sc[0] *= rx
        lat_sc[1] *= ry
        lat_sc[2] *= rz

        # --- atomic positions: replicate across all unit cells ---
        atom_pos_red_sc, atom_pos_car_sc, atom_num_sc, atom_sym_sc = \
            self._replicate_atoms(data)

        # --- assemble output ---
        sc = WavefunctionData()
        sc.psi = psi_sc
        sc.grid = np.array([
            data.grid[0] * rx,
            data.grid[1] * ry,
            data.grid[2] * rz,
        ], dtype=int)
        sc.lat = lat_sc
        sc.lat_ang = lat_sc * bohr2ang
        sc.rlat = rec_lat(lat_sc)
        sc.atom_pos_car = atom_pos_car_sc
        sc.atom_pos_red = atom_pos_red_sc
        sc.atom_num = atom_num_sc
        sc.atom_sym = atom_sym_sc
        sc.eigenvalues_ev = data.eigenvalues_ev
        sc.kpoint = data.kpoint
        sc.ik = data.ik
        sc.ib = data.ib
        sc.spin = data.spin
        sc.volume = vol_lat(lat_sc)
        sc.norm = norm

        return sc

    def expand_psi(self, psi: np.ndarray) -> np.ndarray:
        """
        Tile a raw complex 3D wavefunction array without structural data.

        Parameters
        ----------
        psi : ndarray, shape (nx, ny, nz)

        Returns
        -------
        psi_sc : ndarray, shape (rx·nx, ry·ny, rz·nz)
        """
        return self._tile_psi(psi)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tile_psi(self, psi: np.ndarray) -> np.ndarray:
        """np.tile psi along all three spatial axes."""
        rx, ry, rz = self.replicas
        return np.tile(psi, (rx, ry, rz))

    @staticmethod
    def _apply_bloch_phase(psi, kpt_red, lat):
        """
        Multiply u_k(r) by exp(ik·r) to obtain the full Bloch wavefunction ψ_k(r).

        Parameters
        ----------
        psi : ndarray, shape (nx, ny, nz)
            Cell-periodic part u_k(r) from wfcG2r.
        kpt_red : ndarray, shape (3,)
            K-point in reduced (crystal) coordinates.
        lat : ndarray, shape (3, 3)
            Lattice vectors in bohr (rows).

        Returns
        -------
        psi_k : ndarray, shape (nx, ny, nz)
            ψ_k(r) = exp(ik·r) u_k(r)
        """
        nx, ny, nz = psi.shape
        rlat = rec_lat(lat)  # b_i with a_i · b_j = δ_ij

        # k in Cartesian: k_cart = 2π * (k_red @ rlat)
        k_cart = 2.0 * np.pi * (kpt_red @ rlat)   # (3,)  in 1/bohr

        # Real-space position grid: r(n) = (n1/N1)*a1 + (n2/N2)*a2 + (n3/N3)*a3
        fx = np.arange(nx, dtype=float) / nx
        fy = np.arange(ny, dtype=float) / ny
        fz = np.arange(nz, dtype=float) / nz
        fx, fy, fz = np.meshgrid(fx, fy, fz, indexing='ij')
        frac = np.stack([fx, fy, fz], axis=-1)       # (nx, ny, nz, 3)
        r_cart = frac @ lat                           # (nx, ny, nz, 3) bohr

        # Phase factor exp(i k · r)
        phase = np.exp(1j * np.einsum('i,...i->...', k_cart, r_cart))

        return psi * phase

    def _replicate_atoms(self, data: WavefunctionData):
        """
        Replicate atomic positions to fill all unit cells in the supercell.

        Reduced coordinates in the supercell: shift each atom by (ix/rx, iy/ry, iz/rz)
        and scale its original reduced position by (1/rx, 1/ry, 1/rz).
        """
        rx, ry, rz = self.replicas
        red_uc = data.atom_pos_red       # (natoms, 3) in unit-cell fractional
        lat_uc = data.lat                # unit-cell lattice (bohr)
        lat_sc = data.lat.copy()
        lat_sc[0] *= rx
        lat_sc[1] *= ry
        lat_sc[2] *= rz

        positions_red_sc = []
        positions_car_sc = []

        for ix in range(rx):
            for iy in range(ry):
                for iz in range(rz):
                    shift_red = np.array([ix / rx, iy / ry, iz / rz])
                    for j, pos_uc in enumerate(red_uc):
                        # position in supercell reduced coords
                        pos_sc_red = pos_uc / np.array([rx, ry, rz]) + shift_red
                        # Cartesian in bohr: pos_sc_red @ lat_sc
                        pos_sc_car = pos_sc_red @ lat_sc
                        positions_red_sc.append(pos_sc_red)
                        positions_car_sc.append(pos_sc_car)

        ntotal = rx * ry * rz
        atom_num_sc = np.tile(data.atom_num, ntotal)
        atom_sym_sc = data.atom_sym * ntotal

        return (
            np.array(positions_red_sc),
            np.array(positions_car_sc),
            atom_num_sc,
            atom_sym_sc,
        )

    def __repr__(self):
        return f"SupercellExpander(replicas={self.replicas})"
