"""Top-level simulator orchestrator.

The :class:`Simulator` owns the source, channel array, geometry,
sub-aperture model and feedback controller. It provides the time-stepping
loop and convenience accessors for the figures of merit and the focal-plane
intensity used by the interactive apps.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from . import geometry as _geom
from . import channels as _chan
from . import propagation as _prop
from . import metrics as _metrics
from .source import LaserSource
from .apertures import SubAperture, GaussianAperture
from .controllers import Controller, OpenLoop


@dataclass
class Simulator:
    """Top-level CBC simulator.

    Parameters
    ----------
    N : int
        Number of channels.
    geometry_kind : {'hexagonal', 'square', 'linear'}
        Tile geometry generator.
    pitch : float
        Centre-to-centre sub-aperture spacing in meters.
    source : LaserSource
        Source parameters. Default 1064 nm, zero linewidth.
    sub_aperture : SubAperture
        Sub-aperture amplitude profile. Default Gaussian with w₀ = 2.5 mm.
    focal_length : float
        Combining lens focal length in meters.
    controller : Controller
        Feedback algorithm. Default open-loop.
    dt : float
        Per-substep time increment in seconds (used by the linewidth model).
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    """

    N: int = 19
    geometry_kind: str = "hexagonal"
    pitch: float = 5.5e-3
    source: Optional[LaserSource] = None
    sub_aperture: Optional[SubAperture] = None
    focal_length: float = 1.0
    controller: Optional[Controller] = None
    dt: float = 1e-6
    rng: Optional[np.random.Generator] = None

    # Built state (set in __post_init__)
    positions: np.ndarray = field(default=None, init=False)
    channels: _chan.ChannelArray = field(default=None, init=False)
    iter: int = field(default=0, init=False)
    x_grid: np.ndarray = field(default=None, init=False)
    y_grid: np.ndarray = field(default=None, init=False)
    envelope: np.ndarray = field(default=None, init=False)
    _focal_waist: float = field(default=0.0, init=False)

    def __post_init__(self):
        if self.source is None:
            self.source = LaserSource()
        if self.sub_aperture is None:
            self.sub_aperture = GaussianAperture(w0=2.5e-3)
        if self.controller is None:
            self.controller = OpenLoop()
        if self.rng is None:
            self.rng = np.random.default_rng()
        self._build_positions()
        self._build_channels()
        self.configure_grid()

    # ------------------------------------------------------------------ setup
    def _build_positions(self):
        if self.geometry_kind == "hexagonal":
            self.positions = _geom.hexagonal(self.N, self.pitch)
        elif self.geometry_kind == "square":
            self.positions = _geom.square_grid(self.N, self.pitch)
        elif self.geometry_kind == "linear":
            self.positions = _geom.linear(self.N, self.pitch)
        else:
            raise ValueError(f"Unknown geometry_kind: {self.geometry_kind}")

    def _build_channels(self):
        self.channels = _chan.ChannelArray(N=self.N)

    def configure_grid(self, n_pixels: int = 128, extent_factor: float = 4.0):
        """Build (or rebuild) the focal-plane grid and envelope.

        ``extent_factor`` is in units of the characteristic focal spot size.
        Larger values give more sky around the central lobe.
        """
        wf = self.sub_aperture.focal_waist_estimate(
            self.source.wavelength, self.focal_length)
        self._focal_waist = wf
        extent = extent_factor * wf
        self.x_grid = np.linspace(-extent, extent, n_pixels)
        self.y_grid = np.linspace(-extent, extent, n_pixels)
        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        self.envelope = self.sub_aperture.focal_envelope(
            X, Y, self.source.wavelength, self.focal_length)

    # ------------------------------------------------------------ disturbance
    def randomize_disturbances(self,
                                phase_sigma: float = np.pi,
                                pol_jitter_deg: float = 0.0,
                                tilt_jitter_urad: float = 0.0) -> None:
        """Re-seed the per-channel disturbances and reset corrections.

        Also resets the iteration counter and the controller's internal state.
        Tilt-induced piston (k·x_n·α_n) is added to ``true_phase`` so that the
        feedback can correct it; the uncorrectable envelope-shift loss is
        captured separately as a per-channel amplitude reduction.
        """
        ch = _chan.initialize_disturbances(
            N=self.N,
            phase_sigma=phase_sigma,
            pol_jitter_rad=np.deg2rad(pol_jitter_deg),
            tilt_jitter_rad=tilt_jitter_urad * 1e-6,
            rng=self.rng,
        )
        ch.true_phase += _chan.tilt_piston_coupling(
            ch, self.positions, self.source.wavenumber)
        self.channels = ch
        self.iter = 0
        self.controller.reset()

    # ----------------------------------------------------- coefficient build
    def _coeffs(self):
        """Per-channel complex coefficients (amp · phase) for each Jones component."""
        amp = _chan.tilt_amplitude_loss(
            self.channels, self.focal_length, self._focal_waist)
        c = amp * np.exp(1j * self.channels.residual_phase)
        return c * self.channels.jones_x(), c * self.channels.jones_y()

    # ----------------------------------------------- propagation / metrics
    def on_axis_intensity(self) -> float:
        cx, cy = self._coeffs()
        return abs(cx.sum()) ** 2 + abs(cy.sum()) ** 2

    def focal_intensity(self) -> np.ndarray:
        cx, cy = self._coeffs()
        Ex, Ey = _prop.focal_plane_field(
            self.positions, cx, cy, self.x_grid, self.y_grid,
            self.envelope, self.source.wavelength, self.focal_length)
        return np.abs(Ex) ** 2 + np.abs(Ey) ** 2

    def strehl(self) -> float:
        cx, cy = self._coeffs()
        return _metrics.strehl_ratio(cx, cy)

    def power_in_bucket(self, radius: float) -> float:
        return _metrics.power_in_bucket(
            self.focal_intensity(), self.x_grid, self.y_grid, radius)

    def residual_rms(self) -> float:
        return _metrics.residual_phase_rms(self.channels.residual_phase)

    # ---------------------------------------------------------- time-step
    def step(self) -> None:
        """Advance simulation by one substep."""
        _chan.apply_linewidth_step(self.channels, self.dt,
                                    self.source.linewidth, self.rng)
        self.controller.step(self)
        self.iter += 1

    def run(self, n_substeps: int) -> None:
        """Run several substeps without intermediate accounting."""
        for _ in range(n_substeps):
            self.step()
