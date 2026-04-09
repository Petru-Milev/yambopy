"""
visualizer.py
-------------
Interactive 3D Vispy visualization window for real-time wavefunction dynamics.

Scene components
----------------
1. **Density volume**    — semi-transparent color-mapped volume of |ψ(r,t)|²
2. **Current arrows**    — downsampled quiver of j(r,t) probability current vectors
3. **Atom spheres**      — Markers at atomic positions, colored by element
4. **Cell box**          — Wire-frame outline of the supercell parallelepiped
5. **Info HUD**          — 2D text overlay: time, step, norm, energy, fps

Keyboard controls (printed on startup)
---------------------------------------
SPACE          play / pause animation
+  /  -        double / halve propagation speed (steps per frame)
D              toggle density volume
J              toggle probability current arrows
A              toggle atom markers
B              toggle cell box
I  /  K        raise / lower isosurface threshold by 10%
O  /  L        decrease / increase volume opacity by 10%
C              cycle colormap (grays → hot → viridis → cool → grays)
R              reset camera to default view
Q / Escape     quit

Mouse controls (Vispy turntable camera)
----------------------------------------
Left-drag      rotate
Scroll         zoom
Middle-drag    pan

Requirements
------------
pip install vispy PyOpenGL

Usage
-----
>>> from wf_visualization_3d import WavefunctionLoader, SupercellExpander
>>> from wf_visualization_3d import WavefunctionPropagator, WavefunctionVisualizer
>>>
>>> loader  = WavefunctionLoader('/path/to/run')
>>> data    = loader.load_multi_band(ik=0, bands=[3, 4], grid=[60, 60, 60])
>>> sc_data = SupercellExpander(replicas=(3, 3, 1)).expand(data)
>>> prop    = WavefunctionPropagator(sc_data, dt=0.02)
>>> viz     = WavefunctionVisualizer(prop, sc_data)
>>> viz.run()
"""

import numpy as np

from .observables import (
    compute_density,
    compute_probability_current,
    compute_energy,
    compute_norm,
)


# ---------------------------------------------------------------------------
# Element colour palette (CPK convention, subset)
# ---------------------------------------------------------------------------

_ELEMENT_COLORS = {
    1:  (1.00, 1.00, 1.00, 1.0),   # H   white
    2:  (0.85, 1.00, 1.00, 1.0),   # He  light cyan
    3:  (0.80, 0.50, 1.00, 1.0),   # Li  violet
    5:  (1.00, 0.71, 0.71, 1.0),   # B   salmon
    6:  (0.56, 0.56, 0.56, 1.0),   # C   gray
    7:  (0.19, 0.31, 0.97, 1.0),   # N   blue
    8:  (1.00, 0.05, 0.05, 1.0),   # O   red
    9:  (0.56, 0.83, 0.31, 1.0),   # F   green
    14: (0.94, 0.78, 0.63, 1.0),   # Si  tan
    15: (1.00, 0.50, 0.00, 1.0),   # P   orange
    16: (1.00, 1.00, 0.19, 1.0),   # S   yellow
    17: (0.12, 0.94, 0.12, 1.0),   # Cl  bright green
    22: (0.75, 0.76, 0.78, 1.0),   # Ti  silver-gray
    24: (0.54, 0.60, 0.78, 1.0),   # Cr  steel blue
    25: (0.61, 0.48, 0.78, 1.0),   # Mn  purple
    26: (0.88, 0.40, 0.20, 1.0),   # Fe  rust
    27: (0.94, 0.56, 0.63, 1.0),   # Co  pink
    28: (0.31, 0.82, 0.31, 1.0),   # Ni  mint
    29: (0.78, 0.50, 0.20, 1.0),   # Cu  copper
    30: (0.49, 0.50, 0.69, 1.0),   # Zn  slate
    34: (1.00, 0.63, 0.00, 1.0),   # Se  orange-gold
    42: (0.33, 0.71, 0.71, 1.0),   # Mo  teal
    46: (0.00, 0.85, 0.85, 1.0),   # Pd  cyan
    47: (0.75, 0.75, 0.75, 1.0),   # Ag  silver
    74: (0.13, 0.58, 0.84, 1.0),   # W   blue
    78: (0.80, 0.82, 0.85, 1.0),   # Pt  platinum
    79: (1.00, 0.82, 0.14, 1.0),   # Au  gold
    52: (0.83, 0.48, 0.00, 1.0),   # Te  bronze
    83: (0.62, 0.31, 0.71, 1.0),   # Bi  violet
}
_DEFAULT_ELEMENT_COLOR = (0.70, 0.70, 0.70, 1.0)


def _elem_color(Z: int):
    return _ELEMENT_COLORS.get(int(Z), _DEFAULT_ELEMENT_COLOR)


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

class WavefunctionVisualizer:
    """
    Real-time 3D visualization of wavefunction dynamics.

    Parameters
    ----------
    propagator : WavefunctionPropagator
        Running propagator (mutated each frame).
    data : WavefunctionData
        Structural data (lattice, atoms) for the supercell being visualized.
    steps_per_frame : int
        Number of propagator steps between rendered frames. Increase for faster
        effective time evolution; decrease for smoother animation. (default: 1)
    arrow_downsample : int
        Show one current arrow every N grid points along each axis. (default: 6)
    density_threshold : float
        Initial isosurface / clim threshold as a fraction of max density. (default: 0.05)
    timer_interval : float
        Milliseconds between animation frames (default 50 ms → 20 Hz display).
    bgcolor : str
        Background colour of the canvas (default '#3a3a3a' dark gray).
    """

    def __init__(
        self,
        propagator,
        data,
        steps_per_frame: int = 1,
        arrow_downsample: int = 6,
        density_threshold: float = 0.05,
        timer_interval: float = 50.0,
        bgcolor='#3a3a3a',
    ):
        self.prop = propagator
        self.data = data
        self.steps_per_frame = steps_per_frame
        self.arrow_ds = arrow_downsample
        self.density_threshold = density_threshold
        self.timer_interval = timer_interval
        self.bgcolor = bgcolor

        # State flags
        self._playing = True
        self._show_density = True
        self._show_current = True
        self._show_atoms = True
        self._show_box = True
        self._colormaps = ['grays', 'hot', 'viridis', 'cool', 'blues']
        self._cmap_idx = 1  # start with 'hot'
        self._volume_opacity = 1.0   # 0.0 = fully transparent, 1.0 = full
        self._frame_count = 0
        self._fps_timer = 0.0

        # Deferred Vispy objects (built in _build_scene)
        self._canvas = None
        self._view = None
        self._vol_visual = None
        self._arrow_visual = None
        self._atom_visual = None
        self._box_visual = None
        self._hud_text = None
        self._legend_lines = []       # list of Text visuals, one per legend row
        self._cbar_visual = None      # colormap bar Image visual
        self._timer = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        """
        Build the Vispy scene and start the interactive event loop.

        This call blocks until the window is closed (or Q/Escape is pressed).
        """
        try:
            from vispy import scene, app
        except ImportError:
            raise ImportError(
                "Vispy is required for visualization. Install with:\n"
                "    pip install vispy PyOpenGL"
            )

        self._app = app
        self._scene = scene

        self._build_scene()
        self._print_controls()

        self._timer = app.Timer(
            interval=self.timer_interval / 1000.0,
            connect=self._on_timer,
            start=True,
        )

        self._canvas.show()
        self._app.run()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self):
        scene = self._scene

        self._canvas = scene.SceneCanvas(
            title='Wavefunction Dynamics — 3D Visualization',
            keys='interactive',
            bgcolor=self.bgcolor,
            size=(1200, 900),
            show=False,
        )
        self._canvas.events.key_press.connect(self._on_key)
        self._canvas.events.close.connect(lambda e: self._app.quit())
        self._canvas.events.resize.connect(self._on_resize)

        self._view = self._canvas.central_widget.add_view()
        self._view.camera = 'turntable'
        self._view.camera.fov = 45

        # Centre camera on the supercell midpoint
        lat = self.data.lat   # (3,3) bohr
        mid = (lat[0] + lat[1] + lat[2]) / 2.0
        self._view.camera.center = mid
        self._view.camera.distance = np.linalg.norm(lat[0] + lat[1] + lat[2]) * 1.5

        self._build_box()
        self._build_atoms()
        self._build_density_volume()
        self._build_current_arrows()
        self._build_hud()
        self._build_legend()

    def _build_box(self):
        """Wire-frame parallelepiped showing the supercell boundary."""
        lat = self.data.lat
        corners = self._cell_corners(lat)   # (8, 3)
        edges = self._cell_edges(corners)   # list of (2, 3) arrays
        pos = np.vstack(edges)              # (24, 3)

        self._box_visual = self._scene.visuals.Line(
            pos=pos.astype(np.float32),
            color=(0.6, 0.8, 1.0, 0.8),
            width=1.5,
            connect='segments',
            method='gl',
            parent=self._view.scene,
        )
        self._box_visual.order = 2

    def _build_atoms(self):
        """Sphere markers at atomic positions."""
        pos = self.data.atom_pos_car.astype(np.float32)
        Z_list = self.data.atom_num
        colors = np.array([_elem_color(Z) for Z in Z_list], dtype=np.float32)

        self._atom_visual = self._scene.visuals.Markers(parent=self._view.scene)
        self._atom_visual.set_data(
            pos=pos,
            face_color=colors,
            edge_color=(0.2, 0.2, 0.2, 1.0),
            edge_width=0.5,
            size=12,
            symbol='disc',
        )
        self._atom_visual.order = 2

    def _build_density_volume(self):
        """
        Semi-transparent volume rendering of the probability density |ψ|².

        Volume data is mapped from (nx, ny, nz) on the grid to world coordinates
        via an affine transform derived from the lattice vectors.

        We use a custom RGBA colormap where the alpha channel ramps with the
        data value.  This makes low-density voxels truly transparent rather
        than accumulating into an opaque dark block (the usual 'translucent'
        issue with standard colormaps that have alpha=1 everywhere).
        """
        rho = compute_density(self.prop.psi).astype(np.float32)

        # Normalise to [0, 1] for the colormap
        rho_max = rho.max()
        if rho_max > 0:
            rho_norm = rho / rho_max
        else:
            rho_norm = rho

        self._rho_max = float(rho_max)

        self._vol_visual = self._scene.visuals.Volume(
            rho_norm,
            cmap=self._make_transparent_cmap(),
            clim=(self.density_threshold, 1.0),
            method='translucent',
            parent=self._view.scene,
        )

        # Disable depth writing so the volume's bounding-box geometry does
        # not occlude arrows / atoms that are inside the cell.  The volume
        # still reads the depth buffer (depth_test=True) so it composites
        # correctly behind opaque objects, but it never blocks them.
        self._vol_visual.set_gl_state(
            'translucent',
            depth_test=True,
            cull_face=False,
        )

        # Render order: volume draws first (order=0), arrows draw on top (order=1)
        self._vol_visual.order = 0

        # Apply affine transform so the volume fills the lattice parallelepiped
        self._vol_visual.transform = self._grid_to_world_transform()

    def _build_current_arrows(self):
        """
        Downsampled quiver plot of the probability current j(r,t).

        Uses Vispy's ArrowVisual which renders proper 3D arrow heads
        (stealth / triangle style) natively without any manual geometry.
        """
        from vispy.scene.visuals import Arrow as ArrowVisual

        segments, seg_colors, arrow_data, arrow_colors, alpha = \
            self._compute_arrow_geometry()

        if segments is None:
            self._arrow_visual = None
            return

        # Compute initial arrowhead size proportional to magnitude distribution
        mean_alpha = float(np.mean(alpha)) if len(alpha) > 0 else 0.5
        initial_arrow_size = 4.0 + mean_alpha * 8.0

        self._arrow_visual = ArrowVisual(
            pos=segments,
            color=seg_colors,
            connect='segments',
            method='gl',
            width=2.0,
            arrows=arrow_data,
            arrow_type='stealth',
            arrow_size=initial_arrow_size,
            arrow_color=arrow_colors,
            parent=self._view.scene,
        )

        # Always visible, even when inside the volume
        self._arrow_visual.set_gl_state('translucent', depth_test=False)
        self._arrow_visual.order = 1

    def _build_hud(self):
        """2D text overlay — top-left, pixel coordinates via .pos on canvas.scene."""
        self._hud_text = self._scene.visuals.Text(
            'initialising...',
            color='white',
            font_size=10,
            anchor_x='left',
            anchor_y='top',
            parent=self._canvas.scene,
        )
        self._hud_text.pos = (10, 20)

    def _build_legend(self):
        """
        Build a column of individual Text visuals for the legend (top-right).

        Each line is its own Text visual parented to canvas.scene and placed
        with .pos in pixel coordinates.  This avoids all known Vispy issues
        with multi-line text, anchor_x='right', and STTransform.
        """
        # Gather the legend content (static + dynamic parts)
        lines, colors = self._legend_content()

        w, _h = self._canvas.size
        x0 = max(w - 370, 10)
        y0 = 15
        line_h = 16   # pixels between lines

        self._legend_lines = []
        self._legend_x0 = x0
        self._legend_y0 = y0
        self._legend_line_h = line_h

        for i, (text, color) in enumerate(zip(lines, colors)):
            tv = self._scene.visuals.Text(
                text,
                color=color,
                font_size=9,
                anchor_x='left',
                anchor_y='top',
                parent=self._canvas.scene,
            )
            tv.pos = (x0, y0 + i * line_h)
            self._legend_lines.append(tv)

        # Colormap bar image just below the "colormap: ..." row
        self._build_colorbar()

    # ------------------------------------------------------------------
    # Frame update (animation loop)
    # ------------------------------------------------------------------

    def _on_timer(self, event):
        import time
        from scipy.fft import fftn

        t0 = time.perf_counter()

        if self._playing:
            if self.steps_per_frame > 1:
                self.prop.step_n(self.steps_per_frame)
            else:
                self.prop.step()

        # Compute FFT(psi) once and share it with all consumers that need it.
        # This avoids redundant FFTs: the arrow geometry needs psi_k for the
        # gradient, and any future observable that needs G-space data can
        # reuse it.
        # Compute FFT(psi) once and share with arrow geometry.
        # For spinor wavefunctions, compute a tuple (fft_up, fft_dn).
        psi_k = None
        if self._show_current and self._arrow_visual is not None:
            if self.prop.is_spinor:
                psi_k = (fftn(self.prop.psi[0], workers=-1),
                         fftn(self.prop.psi[1], workers=-1))
            else:
                psi_k = fftn(self.prop.psi, workers=-1)

        self._update_density()
        self._update_arrows(psi_k=psi_k)
        self._update_hud()

        self._frame_count += 1
        dt = time.perf_counter() - t0
        self._fps_timer = dt

        self._canvas.update()

    # ------------------------------------------------------------------
    # Per-frame visual updates
    # ------------------------------------------------------------------

    def _update_density(self):
        if not self._show_density or self._vol_visual is None:
            return

        rho = compute_density(self.prop.psi).astype(np.float32)
        rho_max = rho.max()
        if rho_max > 0:
            rho_norm = rho / rho_max
        else:
            rho_norm = rho

        self._vol_visual.set_data(rho_norm)
        self._vol_visual.clim = (self.density_threshold, 1.0)

    def _update_arrows(self, psi_k=None):
        if not self._show_current or self._arrow_visual is None:
            return

        segments, seg_colors, arrow_data, arrow_colors, alpha = \
            self._compute_arrow_geometry(psi_k=psi_k)
        if segments is None:
            return

        self._arrow_visual.set_data(
            pos=segments,
            color=seg_colors,
            arrows=arrow_data,
        )
        self._arrow_visual.arrow_color = arrow_colors

        # Scale arrowhead size proportional to magnitude distribution
        # Min magnitude → 4 pixels, max magnitude → 12 pixels
        mean_alpha = float(np.mean(alpha)) if len(alpha) > 0 else 0.5
        scaled_arrow_size = 4.0 + mean_alpha * 8.0
        self._arrow_visual.arrow_size = scaled_arrow_size

    def _update_hud(self):
        if self._hud_text is None:
            return

        t_au = self.prop.time
        t_as = t_au * 24.189          # 1 a.u. ≈ 24.189 attoseconds
        norm = compute_norm(self.prop.psi)
        fps = 1.0 / max(self._fps_timer, 1e-6)
        status = 'PLAY' if self._playing else 'PAUSE'
        spf = self.steps_per_frame
        dt_au = self.prop.dt
        dt_as = dt_au * 24.189

        opa = self._volume_opacity

        # Spinor-specific info line
        if self.prop.is_spinor:
            Sz_total = float(np.sum(self.prop.get_spin_density()))
            spin_line = f" Sz_tot = {Sz_total:+.6f}  (spin-up − spin-down)\n"
            wf_type = "spinor"
        else:
            spin_line = ""
            wf_type = "scalar"

        info = (
            f" t  = {t_au:.4f} a.u.  ({t_as:.3f} as)  [{wf_type}]\n"
            f" dt = {dt_au:.4f} a.u.  ({dt_as:.3f} as)\n"
            f" step = {self.prop.step_count}  |  spf = {spf}  |  {status}\n"
            f" ||psi||^2 = {norm:.8f}  |  opacity = {opa:.0%}\n"
            + spin_line +
            f" FPS ~ {fps:.1f}\n"
            f" [SPACE] play/pause  [+/-] speed  [D/J/A/B] toggle\n"
            f" [I/K] threshold  [O/L] opacity  [C] cmap  [R] reset  [Q] quit"
        )
        self._hud_text.text = info

        # Update dynamic legend rows (threshold/cmap may have changed)
        self._update_legend_dynamic()

    # ------------------------------------------------------------------
    # Legend helpers
    # ------------------------------------------------------------------

    def _legend_content(self):
        """
        Return (list_of_strings, list_of_rgba_colors) for every legend row.

        Some rows are 'dynamic' (threshold, cmap, opacity) and will be
        updated each frame via _update_legend_dynamic.
        """
        white  = (1.0, 1.0, 1.0, 1.0)
        gray   = (0.75, 0.75, 0.75, 1.0)
        cyan   = (0.6, 0.8, 1.0, 1.0)
        yellow = (1.0, 1.0, 0.2, 1.0)
        red    = (1.0, 0.4, 0.3, 1.0)

        cmap = self._colormaps[self._cmap_idx]
        thr  = self.density_threshold
        opa  = self._volume_opacity
        dt_au = self.prop.dt
        dt_as = dt_au * 24.189
        nx, ny, nz = self.prop.grid

        lines  = []
        colors = []
        idx = 0

        def add(text, color=white):
            nonlocal idx
            lines.append(text)
            colors.append(color)
            idx += 1

        add("--- LEGEND ---", gray)                                  # 0
        add("")                                                      # 1
        add("Density |psi|^2:", white)                               # 2
        add(f"  colormap  : {cmap}  [C]", gray)                      # 3 — dynamic
        self._dyn_cmap = 3
        add("")                                                      # 4  (colorbar goes here)
        self._dyn_cbar_row = 4
        add(f"  threshold : {thr:.3f}  [I/K]", gray)                 # 5 — dynamic
        self._dyn_thr = 5
        add(f"  opacity   : {opa:.0%}  [O-/L+]", gray)              # 6 — dynamic
        self._dyn_opa = 6
        add(f"  grid      : {nx} x {ny} x {nz}", gray)              # 7
        add("")                                                      # 8
        add("Cell box:  wire frame", cyan)                           # 9
        add("")                                                      # 10
        add("Current arrows:", white)                                # 11
        add("  low |j|", yellow)                                     # 12
        add("  high |j|", red)                                       # 13
        add("")                                                      # 14
        add("Atoms:", white)                                         # 15

        # One line per unique element, coloured by its CPK color
        seen = {}
        for Z, sym in zip(self.data.atom_num, self.data.atom_sym):
            if Z not in seen:
                seen[Z] = sym
        for Z, sym in sorted(seen.items()):
            col = _elem_color(Z)
            add(f"  {sym}  (Z={Z})", col)

        add("")
        add(f"dt = {dt_au:.4f} a.u. ({dt_as:.2f} as)", gray)
        add("---", gray)

        return lines, colors

    def _update_legend_dynamic(self):
        """Update only the legend rows that can change at runtime."""
        if not self._legend_lines:
            return
        cmap = self._colormaps[self._cmap_idx]
        thr  = self.density_threshold
        opa  = self._volume_opacity
        try:
            self._legend_lines[self._dyn_cmap].text = f"  colormap  : {cmap}  [C]"
            self._legend_lines[self._dyn_thr].text  = f"  threshold : {thr:.3f}  [I/K]"
            self._legend_lines[self._dyn_opa].text  = f"  opacity   : {opa:.0%}  [O-/L+]"
        except (IndexError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Colormap with transparency
    # ------------------------------------------------------------------

    def _make_transparent_cmap(self, n=256):
        """
        Build an Nx4 RGBA colormap from the current base colormap, with the
        alpha channel ramping from 0 (for data value 0) to _volume_opacity
        (for data value 1).

        This makes low-density voxels truly transparent instead of accumulating
        into a dark opaque block.  Pressing O/L adjusts _volume_opacity,
        which directly scales the peak alpha.
        """
        from vispy.color import get_colormap, Colormap as VispyColormap

        base = get_colormap(self._colormaps[self._cmap_idx])
        t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        rgba = np.array(base.map(t), dtype=np.float32)     # (n, 4)

        # Alpha ramp: t^1.5 gives a nice non-linear fade (nearly invisible
        # at low density, opaque only for high density), scaled by user opacity
        alpha_ramp = np.power(t.ravel(), 1.5) * self._volume_opacity
        rgba[:, 3] = np.clip(alpha_ramp, 0.0, 1.0)

        return VispyColormap(rgba)

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------

    def _cmap_to_rgba_image(self, width=200, height=12):
        """
        Sample the current colormap into a (height, width, 4) uint8 RGBA image.
        """
        from vispy.color import get_colormap
        cm = get_colormap(self._colormaps[self._cmap_idx])
        t = np.linspace(0.0, 1.0, width).reshape(-1, 1)
        rgba_float = cm.map(t)                        # (width, 4) float64 in [0,1]
        rgba_uint8 = (np.clip(rgba_float, 0, 1) * 255).astype(np.uint8)
        # Tile vertically to give the bar some height
        bar = np.tile(rgba_uint8[np.newaxis, :, :], (height, 1, 1))  # (H, W, 4)
        return bar

    def _build_colorbar(self):
        """
        Create a small Image visual showing the current colormap, positioned
        just below the 'colormap: ...' legend row.
        """
        from vispy.visuals.transforms import STTransform

        bar_img = self._cmap_to_rgba_image()
        x0 = self._legend_x0
        y_row = self._legend_y0 + self._dyn_cbar_row * self._legend_line_h

        self._cbar_visual = self._scene.visuals.Image(
            bar_img,
            interpolation='linear',
            parent=self._canvas.scene,
        )
        self._cbar_visual.transform = STTransform(translate=(x0, y_row))

    def _rebuild_colorbar(self):
        """Regenerate the colorbar image after a cmap change."""
        if self._cbar_visual is None:
            return
        bar_img = self._cmap_to_rgba_image()
        self._cbar_visual.set_data(bar_img)

    # ------------------------------------------------------------------
    # Keyboard handler
    # ------------------------------------------------------------------

    def _on_key(self, event):
        key = event.key.name if hasattr(event.key, 'name') else str(event.key)

        if key == 'Space':
            self._playing = not self._playing

        elif key in ('+', '=', 'Equal'):
            self.steps_per_frame = min(self.steps_per_frame * 2, 128)

        elif key == '-':
            self.steps_per_frame = max(self.steps_per_frame // 2, 1)

        elif key.lower() == 'd':
            self._show_density = not self._show_density
            if self._vol_visual is not None:
                self._vol_visual.visible = self._show_density

        elif key.lower() == 'j':
            self._show_current = not self._show_current
            if self._arrow_visual is not None:
                self._arrow_visual.visible = self._show_current

        elif key.lower() == 'a':
            self._show_atoms = not self._show_atoms
            if self._atom_visual is not None:
                self._atom_visual.visible = self._show_atoms

        elif key.lower() == 'b':
            self._show_box = not self._show_box
            if self._box_visual is not None:
                self._box_visual.visible = self._show_box

        elif key.lower() == 'i':
            self.density_threshold = min(self.density_threshold * 1.25, 0.95)

        elif key.lower() == 'k':
            self.density_threshold = max(self.density_threshold * 0.8, 0.001)

        elif key.lower() == 'c':
            self._cmap_idx = (self._cmap_idx + 1) % len(self._colormaps)
            if self._vol_visual is not None:
                self._vol_visual.cmap = self._make_transparent_cmap()
            self._rebuild_colorbar()

        elif key.lower() == 'o':
            # Decrease opacity (more transparent) — lets arrows show through
            self._volume_opacity = max(self._volume_opacity - 0.1, 0.0)
            if self._vol_visual is not None:
                self._vol_visual.cmap = self._make_transparent_cmap()

        elif key.lower() == 'l':
            # Increase opacity (more opaque)
            self._volume_opacity = min(self._volume_opacity + 0.1, 1.0)
            if self._vol_visual is not None:
                self._vol_visual.cmap = self._make_transparent_cmap()

        elif key.lower() == 'r':
            lat = self.data.lat
            mid = (lat[0] + lat[1] + lat[2]) / 2.0
            self._view.camera.center = mid
            self._view.camera.distance = (
                np.linalg.norm(lat[0] + lat[1] + lat[2]) * 1.5
            )

        elif key in ('Q', 'Escape'):
            self._canvas.close()
            self._app.quit()

    def _on_resize(self, event):
        """Reposition legend lines and colorbar when the window is resized."""
        if self._legend_lines:
            w, _h = self._canvas.size
            x0 = max(w - 370, 10)
            self._legend_x0 = x0
            for i, tv in enumerate(self._legend_lines):
                tv.pos = (x0, self._legend_y0 + i * self._legend_line_h)
        if self._cbar_visual is not None:
            from vispy.visuals.transforms import STTransform
            y_row = self._legend_y0 + self._dyn_cbar_row * self._legend_line_h
            self._cbar_visual.transform = STTransform(
                translate=(self._legend_x0, y_row)
            )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _compute_arrow_geometry(self, psi_k=None):
        """
        Compute geometry for Vispy ArrowVisual.

        Parameters
        ----------
        psi_k : ndarray, optional
            Pre-computed FFT of psi.  Avoids a redundant FFT when the caller
            (the timer loop) has already computed it.

        Returns
        -------
        segments   : float32 (N*2, 3) — shaft vertex pairs (tail, head)
        seg_colors : float32 (N*2, 4) — per-vertex shaft colour
        arrow_data : float32 (N, 6)   — [tail_xyz, head_xyz] for each arrowhead
        arrow_colors: float32 (N, 4)  — per-arrow head colour
        alpha      : float32 (N,)     — normalized magnitude for each arrow [0,1]
        All arrow values are None if current is essentially zero.
        """
        ds = self.arrow_ds
        psi = self.prop.psi
        lat = self.data.lat
        nx, ny, nz = self.prop.grid   # always spatial (nx, ny, nz), safe for spinor

        j = compute_probability_current(psi, lat, psi_k=psi_k)  # (3, nx, ny, nz)
        jmax = float(np.sqrt(np.sum(j**2, axis=0)).max())
        if jmax < 1e-30:
            return None, None, None, None, None

        # ── downsampled indices ────────────────────────────────────────
        ix = np.arange(0, nx, ds)
        iy = np.arange(0, ny, ds)
        iz = np.arange(0, nz, ds)
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing='ij')
        flat = (gx.ravel(), gy.ravel(), gz.ravel())

        # Tail positions: fractional → Cartesian (N, 3)
        frac = np.stack([
            gx.ravel().astype(np.float64) / nx,
            gy.ravel().astype(np.float64) / ny,
            gz.ravel().astype(np.float64) / nz,
        ], axis=-1)
        pos_tail = (frac @ lat).astype(np.float32)       # (N, 3)

        # Current vectors (N, 3) and magnitudes (N,)
        jvec  = np.stack([j[0][flat], j[1][flat], j[2][flat]], axis=-1)
        jmag_ds = np.linalg.norm(jvec, axis=1)

        # Arrow length proportional to |j|, scaled so the largest arrow
        # has length cell_scale (one grid-spacing step).
        cell_scale = float(np.linalg.norm(lat[0])) / nx * ds * 0.8
        dir_unit = jvec / (jmag_ds[:, None] + 1e-30)

        # ── colours ───────────────────────────────────────────────────
        alpha = np.clip(jmag_ds / (jmax + 1e-30), 0.0, 1.0).astype(np.float32)

        # Scale each arrow by its normalised magnitude so small currents
        # get short arrows and the dominant current gets the full length.
        pos_head = (pos_tail + dir_unit * cell_scale * alpha[:, None]).astype(np.float32)  # (N,3)
        c = np.zeros((len(alpha), 4), dtype=np.float32)
        c[:, 0] = 1.0
        c[:, 1] = np.clip(1.0 - alpha, 0.0, 1.0)   # yellow → red
        c[:, 2] = 0.0
        c[:, 3] = np.clip(alpha * 2.0, 0.2, 1.0)

        # ── shaft segments: (N*2, 3) interleaved tail/head ────────────
        N = len(pos_tail)
        segments = np.empty((N * 2, 3), dtype=np.float32)
        segments[0::2] = pos_tail
        segments[1::2] = pos_head
        seg_colors = np.empty((N * 2, 4), dtype=np.float32)
        seg_colors[0::2] = c
        seg_colors[1::2] = c

        # ── arrow head data: (N, 6) = [tail_xyz, head_xyz] ───────────
        # ArrowVisual uses tail for direction, head as tip position.
        arrow_data = np.concatenate([pos_tail, pos_head], axis=1)  # (N, 6)

        return segments, seg_colors, arrow_data, c, alpha

    def _grid_to_world_transform(self):
        """
        Return a Vispy affine transform that maps the volume's local voxel
        coordinates to the lattice parallelepiped in Cartesian bohr.

        Vispy Volume (shape D,H,W) builds a box spanning local coords:
            x: 0 .. shape[2]-1  (= nz-1)   ← array dim 2, the c-axis / a3
            y: 0 .. shape[1]-1  (= ny-1)   ← array dim 1, the b-axis / a2
            z: 0 .. shape[0]-1  (= nx-1)   ← array dim 0, the a-axis / a1

        So row 0 of M must point along lat[2]/nz, row 2 along lat[0]/nx.
        (Confirmed from vispy/visuals/volume.py: x0,x1 = -0.5, shape[2]-0.5)
        """
        from vispy.visuals.transforms import MatrixTransform

        # Use spatial grid only (last 3 dims for spinor, all 3 for scalar)
        nx, ny, nz = self.prop.grid
        lat = self.data.lat   # rows = a1, a2, a3

        M = np.eye(4, dtype=np.float32)
        M[0, :3] = lat[2] / nz   # local x (array dim 2, size nz) → a3
        M[1, :3] = lat[1] / ny   # local y (array dim 1, size ny) → a2
        M[2, :3] = lat[0] / nx   # local z (array dim 0, size nx) → a1
        M[3, :3] = 0.0           # origin at (0,0,0)

        t = MatrixTransform()
        t.matrix = M
        return t

    @staticmethod
    def _cell_corners(lat):
        """Return the 8 corners of the parallelepiped defined by lattice vectors."""
        o = np.zeros(3)
        a, b, c = lat[0], lat[1], lat[2]
        return np.array([
            o, a, b, c,
            a + b, a + c, b + c, a + b + c,
        ], dtype=np.float32)

    @staticmethod
    def _cell_edges(corners):
        """Return the 12 edges of a parallelepiped as pairs of (start, end) points."""
        o, a, b, c, ab, ac, bc, abc = corners
        edges = [
            # Bottom face
            [o, a], [o, b], [a, ab], [b, ab],
            # Top face
            [c, ac], [c, bc], [ac, abc], [bc, abc],
            # Verticals
            [o, c], [a, ac], [b, bc], [ab, abc],
        ]
        return [np.array(e, dtype=np.float32) for e in edges]

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @staticmethod
    def _print_controls():
        print()
        print("=" * 60)
        print("  Wavefunction 3D Visualizer — Keyboard Controls")
        print("=" * 60)
        print("  SPACE     play / pause animation")
        print("  + / -     double / halve steps per frame")
        print("  D         toggle density volume")
        print("  J         toggle probability current arrows")
        print("  A         toggle atom markers")
        print("  B         toggle cell box")
        print("  I / K     raise / lower density threshold by 25%")
        print("  O / L     decrease / increase volume opacity by 10%")
        print("  C         cycle colormap")
        print("  R         reset camera")
        print("  Q / Esc   quit")
        print()
        print("  Mouse: left-drag=rotate, scroll=zoom, middle-drag=pan")
        print("=" * 60)
        print()

    def __repr__(self):
        return (
            f"WavefunctionVisualizer("
            f"grid={self.prop.grid}, "
            f"spf={self.steps_per_frame}, "
            f"ds={self.arrow_ds})"
        )
