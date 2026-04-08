"""
wf_visualization_3d
===================
Real-time wavefunction propagation and interactive 3D visualization
for the Yambo / Quantum ESPRESSO ecosystem.

Pipeline
--------
1. Load wavefunction from a Yambo SAVE folder  →  WavefunctionLoader
2. (Optional) Expand unit cell to supercell    →  SupercellExpander
3. Propagate in real time (SOFT algorithm)     →  WavefunctionPropagator
4. Compute observables                         →  observables module
5. Visualize interactively with Vispy          →  WavefunctionVisualizer
6. Save trajectory / frames to disk           →  WavefunctionIO

Minimal example
---------------
>>> from wf_visualization_3d import (
...     WavefunctionLoader, SupercellExpander,
...     WavefunctionPropagator, WavefunctionVisualizer
... )
>>>
>>> loader  = WavefunctionLoader('/path/to/yambo/run', save='SAVE')
>>> data    = loader.load_multi_band(ik=0, bands=[3, 4], grid=[60, 60, 60])
>>> sc_data = SupercellExpander(replicas=(3, 3, 1)).expand(data)
>>> prop    = WavefunctionPropagator(sc_data, dt=0.02)
>>> viz     = WavefunctionVisualizer(prop, sc_data, steps_per_frame=2)
>>> viz.run()

Files
-----
loaders.py      WavefunctionLoader, WavefunctionData
supercell.py    SupercellExpander
propagator.py   WavefunctionPropagator  (SOFT split-operator algorithm)
observables.py  compute_density, compute_probability_current, ObservableLogger, …
visualizer.py   WavefunctionVisualizer  (Vispy interactive 3D window)
io.py           WavefunctionIO          (HDF5 trajectory, cube, XSF, checkpoints)
"""

from .loaders import WavefunctionLoader, WavefunctionData
from .supercell import SupercellExpander
from .propagator import WavefunctionPropagator
from .observables import (
    compute_density,
    compute_probability_current,
    compute_current_magnitude,
    compute_norm,
    compute_autocorrelation,
    compute_energy,
    compute_vorticity,
    compute_current_flux,
    ObservableLogger,
)
from .visualizer import WavefunctionVisualizer
from .io import WavefunctionIO

__all__ = [
    # Data I/O
    'WavefunctionLoader',
    'WavefunctionData',
    # Supercell
    'SupercellExpander',
    # Propagation
    'WavefunctionPropagator',
    # Observables
    'compute_density',
    'compute_probability_current',
    'compute_current_magnitude',
    'compute_norm',
    'compute_autocorrelation',
    'compute_energy',
    'compute_vorticity',
    'compute_current_flux',
    'ObservableLogger',
    # Visualization
    'WavefunctionVisualizer',
    # File I/O
    'WavefunctionIO',
]

__version__ = '0.1.0'
