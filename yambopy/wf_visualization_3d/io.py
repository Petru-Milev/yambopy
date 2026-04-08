"""
io.py
-----
Save and load wavefunction trajectories, observables, and individual frames.

Supports
--------
HDF5 trajectory
    Full time-series: ψ(r,t), ρ(r,t), j(r,t), and scalar observables.
    Readable by h5py, HDFView, or Python scripts for post-processing.

Gaussian cube files
    Export a single density or wavefunction frame for use with VESTA,
    XCrySDen, or VMD.

XSF files
    Export density as XCrySDen Structure Format for XCrySDen visualization.

Checkpoint save/restore
    Dump/restore the full propagator state (ψ, t, step_count) to resume a run.

Main class
----------
WavefunctionIO
    - save_frame_cube(filename, prop, data, mode)  : single cube frame
    - save_frame_xsf(filename, prop, data)         : single XSF frame
    - open_trajectory(filename)                    : open HDF5 file for writing
    - append_frame(h5file, prop, data, ...)        : append frame to trajectory
    - close_trajectory(h5file)                     : flush and close HDF5
    - save_checkpoint(filename, prop)              : save propagator state
    - load_checkpoint(filename, prop)              : restore propagator state
    - load_trajectory_frame(filename, frame_idx)   : read a single frame back
"""

import os
import numpy as np

from .observables import (
    compute_density,
    compute_probability_current,
    compute_norm,
    compute_energy,
    compute_autocorrelation,
)


class WavefunctionIO:
    """
    Utility class for saving and loading wavefunction data.

    All methods are static / class methods — no instance state needed.
    Use as a namespace: WavefunctionIO.save_frame_cube(...), etc.
    """

    # ------------------------------------------------------------------
    # Gaussian cube export
    # ------------------------------------------------------------------

    @staticmethod
    def save_frame_cube(filename: str, prop, data,
                        mode: str = 'density'):
        """
        Export a single frame to a Gaussian .cube file.

        Parameters
        ----------
        filename : str
            Output file path (e.g. 'frame_0001.cube').
        prop : WavefunctionPropagator
            Current propagator state.
        data : WavefunctionData
            Structural data (lattice, atoms) for the supercell.
        mode : str
            'density'   — write |ψ(r)|²
            'real'      — write Re[ψ(r)]
            'imag'      — write Im[ψ(r)]
            'phase'     — write arg(ψ(r)) / π

        Notes
        -----
        Uses yambopy.io.cubetools.write_cube internally.
        """
        try:
            from ..io.cubetools import write_cube
        except ImportError:
            raise ImportError("yambopy.io.cubetools not found. "
                              "Make sure yambopy is installed.")

        psi = prop.psi
        lat = data.lat          # bohr
        atom_pos = data.atom_pos_car   # (natoms, 3) bohr
        atom_num = data.atom_num

        if mode == 'density':
            field = np.abs(psi) ** 2
            header = f'Density |psi|^2  t={prop.time:.4f} a.u.'
        elif mode == 'real':
            field = np.real(psi)
            header = f'Re[psi]  t={prop.time:.4f} a.u.'
        elif mode == 'imag':
            field = np.imag(psi)
            header = f'Im[psi]  t={prop.time:.4f} a.u.'
        elif mode == 'phase':
            field = np.angle(psi) / np.pi
            header = f'phase(psi)/pi  t={prop.time:.4f} a.u.'
        else:
            raise ValueError(f"Unknown mode {mode!r}. "
                             "Choose 'density', 'real', 'imag', or 'phase'.")

        write_cube(
            filename=filename,
            data=field.astype(np.float64),
            lat_vec=lat,
            atom_pos=atom_pos,
            atomic_num=atom_num,
            header=header,
        )
        print(f"[WavefunctionIO] Wrote cube: {filename}")

    # ------------------------------------------------------------------
    # XSF export
    # ------------------------------------------------------------------

    @staticmethod
    def save_frame_xsf(filename: str, prop, data):
        """
        Export density |ψ(r)|² at current time as an XSF file.

        Parameters
        ----------
        filename : str
        prop : WavefunctionPropagator
        data : WavefunctionData
        """
        try:
            from ..io.xsffile import YamboXsf
        except ImportError:
            raise ImportError("yambopy.io.xsffile not found.")

        rho = np.abs(prop.psi) ** 2
        nx, ny, nz = rho.shape
        bohr2ang = 0.52917720859

        xsf = YamboXsf()
        xsf.set_dim(3)   # CRYSTAL
        xsf.set_cell_parameters(data.lat * bohr2ang)

        from ..units import chemical_symbols
        for Z, pos in zip(data.atom_num, data.atom_pos_car):
            try:
                sym = chemical_symbols[int(Z)]
            except Exception:
                sym = str(Z)
            xsf.add_atom(sym, pos * bohr2ang)

        # Add volumetric data block (matches YamboXsf.add_grid_data signature)
        origin = np.zeros(3)
        lat_ang = data.lat * bohr2ang
        xsf.add_grid_data(
            grid_name=f'density_t{prop.time:.4f}',
            block_dim=3,                           # 3D block
            sub_grid_name='density',
            sub_grid_size=1,                       # single sub-grid
            sub_grid_dim=np.array([nx, ny, nz]),
            sub_grid_origin=origin,
            sub_grid_vectors=lat_ang,
            data_array=rho.astype(np.float64),
        )

        xsf.write_xsf(filename)
        print(f"[WavefunctionIO] Wrote XSF: {filename}")

    # ------------------------------------------------------------------
    # HDF5 trajectory
    # ------------------------------------------------------------------

    @staticmethod
    def open_trajectory(filename: str, mode: str = 'w',
                        store_psi: bool = True,
                        store_density: bool = True,
                        store_current: bool = False,
                        compression: str = 'gzip',
                        compression_opts: int = 4):
        """
        Open an HDF5 file to stream trajectory frames.

        Parameters
        ----------
        filename : str
            Path to output HDF5 file.
        mode : str
            'w' = create new (default), 'a' = append to existing.
        store_psi : bool
            Save full complex wavefunction ψ(r,t) (large).
        store_density : bool
            Save density |ψ(r,t)|² (half size of psi).
        store_current : bool
            Save probability current j(r,t) (3× density; can be large).
        compression : str
            HDF5 compression filter ('gzip', 'lzf', or None).
        compression_opts : int
            Compression level (1-9 for gzip).

        Returns
        -------
        h5file : h5py.File  — keep open and pass to append_frame/close_trajectory.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 output. "
                              "Install with: pip install h5py")

        h5 = h5py.File(filename, mode)
        # Store metadata as attributes
        h5.attrs['store_psi'] = store_psi
        h5.attrs['store_density'] = store_density
        h5.attrs['store_current'] = store_current
        h5.attrs['compression'] = str(compression)
        h5.attrs['compression_opts'] = compression_opts
        h5.attrs['frame_count'] = 0

        # Groups
        h5.require_group('observables')
        if store_psi:
            h5.require_group('psi')
        if store_density:
            h5.require_group('density')
        if store_current:
            h5.require_group('current')

        print(f"[WavefunctionIO] Opened trajectory: {filename!r}")
        return h5

    @staticmethod
    def write_structure(h5file, data):
        """
        Write structural information (lattice, atoms) to an open HDF5 file.
        Call once before appending frames.

        Parameters
        ----------
        h5file : h5py.File
        data : WavefunctionData
        """
        grp = h5file.require_group('structure')
        grp.create_dataset('lattice_bohr', data=data.lat)
        grp.create_dataset('lattice_ang', data=data.lat_ang)
        grp.create_dataset('atom_pos_car_bohr', data=data.atom_pos_car)
        grp.create_dataset('atom_pos_red', data=data.atom_pos_red)
        grp.create_dataset('atom_num', data=data.atom_num)
        grp.create_dataset('grid', data=data.grid)
        if data.atom_sym is not None:
            sym_bytes = [s.encode('utf-8') for s in data.atom_sym]
            grp.create_dataset('atom_sym', data=sym_bytes)
        h5file.flush()

    @staticmethod
    def append_frame(h5file, prop, psi0=None):
        """
        Append one time step to an open HDF5 trajectory.

        Parameters
        ----------
        h5file : h5py.File  — opened with open_trajectory
        prop   : WavefunctionPropagator
        psi0   : ndarray, optional — initial wavefunction for autocorrelation.
                 If None, autocorrelation is not recorded.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required.")

        n = int(h5file.attrs.get('frame_count', 0))
        comp = str(h5file.attrs.get('compression', 'gzip'))
        opts = int(h5file.attrs.get('compression_opts', 4))
        kw = dict(compression=comp, compression_opts=opts) if comp != 'None' else {}

        psi = prop.psi

        # Scalar observables
        obs = h5file['observables']
        def _append_scalar(name, value):
            if name in obs:
                ds = obs[name]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = value
            else:
                obs.create_dataset(name, data=[value], maxshape=(None,), **kw)

        _append_scalar('time', prop.time)
        _append_scalar('step_count', prop.step_count)
        _append_scalar('norm', compute_norm(psi))

        if psi0 is not None:
            C = compute_autocorrelation(psi0, psi)
            _append_scalar('autocorrelation_real', C.real)
            _append_scalar('autocorrelation_imag', C.imag)
            _append_scalar('autocorrelation_abs', abs(C))

        # Volumetric data
        def _append_volume(group, arr):
            if 'data' in group:
                ds = group['data']
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = arr
            else:
                shape = (1,) + arr.shape
                maxshape = (None,) + arr.shape
                group.create_dataset('data', data=arr[np.newaxis],
                                     maxshape=maxshape, chunks=shape, **kw)

        if h5file.attrs.get('store_psi', False):
            # Store complex as two float datasets (real/imag)
            grp = h5file['psi']
            _append_volume(grp.require_group('real'), np.real(psi).astype(np.float32))
            _append_volume(grp.require_group('imag'), np.imag(psi).astype(np.float32))

        if h5file.attrs.get('store_density', True):
            rho = compute_density(psi).astype(np.float32)
            _append_volume(h5file['density'], rho)

        if h5file.attrs.get('store_current', False):
            j = compute_probability_current(psi, prop.lat).astype(np.float32)
            _append_volume(h5file['current'], j)

        h5file.attrs['frame_count'] = n + 1
        h5file.flush()

    @staticmethod
    def close_trajectory(h5file):
        """Flush and close an open HDF5 trajectory file."""
        n = int(h5file.attrs.get('frame_count', 0))
        fname = h5file.filename
        h5file.flush()
        h5file.close()
        print(f"[WavefunctionIO] Closed trajectory: {fname!r}  ({n} frames)")

    # ------------------------------------------------------------------
    # Load frames back
    # ------------------------------------------------------------------

    @staticmethod
    def load_trajectory_frame(filename: str, frame_idx: int = -1):
        """
        Read a single frame from a saved HDF5 trajectory.

        Parameters
        ----------
        filename  : str
        frame_idx : int  (negative indexing supported)

        Returns
        -------
        dict with keys: 'time', 'norm', 'density', and optionally 'psi', 'current'
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required.")

        result = {}
        with h5py.File(filename, 'r') as h5:
            obs = h5['observables']
            result['time'] = float(obs['time'][frame_idx])
            result['norm'] = float(obs['norm'][frame_idx])
            if 'step_count' in obs:
                result['step_count'] = int(obs['step_count'][frame_idx])
            if 'autocorrelation_abs' in obs:
                result['autocorrelation_abs'] = float(obs['autocorrelation_abs'][frame_idx])

            if 'density' in h5:
                result['density'] = np.array(h5['density']['data'][frame_idx])

            if 'psi' in h5:
                re = np.array(h5['psi']['real']['data'][frame_idx])
                im = np.array(h5['psi']['imag']['data'][frame_idx])
                result['psi'] = re.astype(complex) + 1j * im.astype(complex)

            if 'current' in h5:
                result['current'] = np.array(h5['current']['data'][frame_idx])

        return result

    @staticmethod
    def load_trajectory_observables(filename: str) -> dict:
        """
        Load all scalar observables from a trajectory (time, norm, autocorrelation, etc.)

        Returns
        -------
        dict of ndarray
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required.")

        with h5py.File(filename, 'r') as h5:
            obs = h5['observables']
            return {key: np.array(obs[key]) for key in obs.keys()}

    # ------------------------------------------------------------------
    # Checkpoint (propagator state save/restore)
    # ------------------------------------------------------------------

    @staticmethod
    def save_checkpoint(filename: str, prop):
        """
        Save the current propagator state to an NPZ file for later resumption.

        Saves: psi, psi0 (initial), time, step_count, dt, V (potential), lat.

        Parameters
        ----------
        filename : str  (e.g. 'checkpoint_step1000.npz')
        prop     : WavefunctionPropagator
        """
        np.savez_compressed(
            filename,
            psi=prop.psi,
            psi0=prop._psi0,
            time=np.array([prop.time]),
            step_count=np.array([prop.step_count]),
            dt=np.array([prop.dt]),
            V=prop.V,
            lat=prop.lat,
        )
        print(f"[WavefunctionIO] Checkpoint saved: {filename!r}  "
              f"(t={prop.time:.4f}, step={prop.step_count})")

    @staticmethod
    def load_checkpoint(filename: str, prop):
        """
        Restore propagator state from a checkpoint NPZ file.

        Parameters
        ----------
        filename : str
        prop     : WavefunctionPropagator
            Existing propagator whose state will be overwritten.
            Grid and lattice must match the checkpoint.
        """
        data = np.load(filename)

        if data['psi'].shape != prop.psi.shape:
            raise ValueError(
                f"Checkpoint shape {data['psi'].shape} does not match "
                f"propagator grid {prop.psi.shape}."
            )

        prop.psi = data['psi'].copy()
        prop._psi0 = data['psi0'].copy()
        prop.time = float(data['time'][0])
        prop.step_count = int(data['step_count'][0])

        # Rebuild propagators if dt or V changed
        new_dt = float(data['dt'][0])
        if abs(new_dt - prop.dt) > 1e-12:
            prop.dt = new_dt
            prop._setup_kinetic_propagator()

        prop.V = data['V'].copy()
        prop._setup_potential_propagators()

        print(f"[WavefunctionIO] Checkpoint loaded: {filename!r}  "
              f"(t={prop.time:.4f}, step={prop.step_count})")

    # ------------------------------------------------------------------
    # Batch export helpers
    # ------------------------------------------------------------------

    @staticmethod
    def export_animation_frames(prop, data, output_dir: str = 'frames',
                                n_frames: int = 100,
                                steps_per_frame: int = 1,
                                mode: str = 'density',
                                fmt: str = 'cube'):
        """
        Propagate and export n_frames cube/XSF files for offline rendering.

        Parameters
        ----------
        prop          : WavefunctionPropagator
        data          : WavefunctionData
        output_dir    : str  — directory to write files into
        n_frames      : int
        steps_per_frame : int  — propagation steps between exported frames
        mode          : str   — passed to save_frame_cube ('density', 'real', etc.)
        fmt           : str   — 'cube' or 'xsf'
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            from tqdm import tqdm
            frames = tqdm(range(n_frames), desc='Exporting frames')
        except ImportError:
            frames = range(n_frames)

        for i in frames:
            prop.step_n(steps_per_frame)
            if fmt == 'cube':
                fname = os.path.join(output_dir, f'frame_{i:05d}.cube')
                WavefunctionIO.save_frame_cube(fname, prop, data, mode=mode)
            elif fmt == 'xsf':
                fname = os.path.join(output_dir, f'frame_{i:05d}.xsf')
                WavefunctionIO.save_frame_xsf(fname, prop, data)
            else:
                raise ValueError(f"Unknown format {fmt!r}. Use 'cube' or 'xsf'.")

        print(f"[WavefunctionIO] Exported {n_frames} frames to {output_dir!r}")
