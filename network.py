# network.py
# Implements the RF neuromorphic fabric model.

import numpy as np
import config as cfg


class RFNeuromorphicNetwork:
    """
    A planar array of coupled RF oscillator nodes operating as analog
    spiking neurons. Connectivity emerges from electromagnetic coupling
    (spatial + spectral Gaussian kernel). Weights adapt via Hebbian
    trace learning.

    Nodes are indexed flat: node i is at grid position
        x = i // N,  y = i % N
    Reshape to (N, N) only for visualization.
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.N    = cfg.N
        self.n    = cfg.N * cfg.N       # total number of nodes

        # ------------------------------------------------------------------
        # Section 2.1 — Node positions and resonant frequencies
        # ------------------------------------------------------------------
        xs = np.arange(self.N)
        ys = np.arange(self.N)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')

        # Flat arrays of positions
        self.pos_x = grid_x.flatten().astype(float)  # shape (n,)
        self.pos_y = grid_y.flatten().astype(float)  # shape (n,)

        # Section 2.2 — Linear frequency gradient in the x-direction
        # f_i = f_min + (f_max - f_min) * (x_i / N)
        self.freq = cfg.F_MIN + (cfg.F_MAX - cfg.F_MIN) * (self.pos_x / self.N)

        # ------------------------------------------------------------------
        # Section 2.3 — Structural coupling matrix S (fixed at init)
        # ------------------------------------------------------------------
        self.S = self._build_structural_coupling()

        # Locality mask M: 1 where |S_ij| > epsilon (covers exc + inh)
        self.M = (np.abs(self.S) > cfg.EPSILON).astype(float)

        # Excitatory mask: learning only modulates positive connections
        self.M_exc = (self.S > cfg.EPSILON).astype(float)

        # ------------------------------------------------------------------
        # Learned modulation matrix L (starts at zero)
        # Effective coupling W = S * (1 + L)
        # ------------------------------------------------------------------
        self.L = np.zeros((self.n, self.n))
        self.W = self.S.copy()          # initial W = S + 0 = S

        # ------------------------------------------------------------------
        # Section 3 — Dynamic state variables (all start at zero)
        # ------------------------------------------------------------------
        self.V      = np.zeros(self.n)  # membrane voltage
        self.a      = np.zeros(self.n)  # activity trace
        self.r      = np.zeros(self.n)  # firing rate estimate
        self.spikes = np.zeros(self.n)  # spike output at current timestep

        # History for analysis and plotting
        self.spike_history  = []        # list of spike vectors
        self.voltage_history = []       # list of voltage vectors

    # ------------------------------------------------------------------
    # Section 2.3 / Section 8 — Build the structural coupling matrix
    # ------------------------------------------------------------------
    def _build_structural_coupling(self):
        """
        Section 8 — Lateral inhibition via difference of Gaussians (DoG).

        The spatial component is a Mexican hat profile:
            spatial_ij = exp(-d^2 / sigma_exc^2)
                       - alpha * exp(-d^2 / sigma_inh^2)

        Short-range connections are excitatory (positive).
        Medium-range connections are inhibitory (negative).
        Long-range connections decay to zero.

        The spectral Gaussian is unchanged.

        Full structural coupling:
            S_ij = spatial_ij * spectral_ij

        Inhibitory connections are fixed — not modulated by learning.
        Locality mask M covers all nonzero connections (excitatory and
        inhibitory). Learned modulation L is only applied where S_ij > 0.
        """
        # Spatial distances: shape (n, n)
        dx = self.pos_x[:, None] - self.pos_x[None, :]
        dy = self.pos_y[:, None] - self.pos_y[None, :]
        d2 = dx**2 + dy**2

        # Difference of Gaussians — Mexican hat spatial profile
        exc     = np.exp(-d2 / cfg.SIGMA_EXC**2)
        inh     = np.exp(-d2 / cfg.SIGMA_INH**2)
        spatial = exc - cfg.ALPHA_INH * inh

        # Spectral Gaussian — unchanged
        df2      = (self.freq[:, None] - self.freq[None, :])**2
        spectral = np.exp(-df2 / cfg.SIGMA_FREQ**2)

        S = spatial * spectral

        # Zero out self-connections
        np.fill_diagonal(S, 0.0)

        return S

    # ------------------------------------------------------------------
    # Section 5.1 — Compute input current from external RF field
    # ------------------------------------------------------------------
    def compute_input(self, signal):
        """
        I_i = sum_k A_k * exp(-(f_i - f_k_ext)^2 / sigma_input^2)

        Parameters
        ----------
        signal : list of (frequency, amplitude) tuples
            The external RF field components at this timestep.

        Returns
        -------
        I : ndarray, shape (n,)
            Input current for each node.
        """
        I = np.zeros(self.n)
        for f_ext, amplitude in signal:
            I += amplitude * np.exp(
                -(self.freq - f_ext)**2 / cfg.SIGMA_INPUT**2
            )
        return I

    # ------------------------------------------------------------------
    # Section 3 — Single timestep update
    # ------------------------------------------------------------------
    def step(self, signal=None):
        """
        Advance the network by one timestep dt.

        Parameters
        ----------
        signal : list of (frequency, amplitude) tuples, or None
            External RF field. Pass None for no external input.
        """
        dt = cfg.DT

        # Step 1: compute external input
        I = self.compute_input(signal) if signal is not None else np.zeros(self.n)

        # Step 2: update voltages
        # dV/dt = -V/tau + W @ spikes + I
        self.V += dt * (-self.V / cfg.TAU + self.W @ self.spikes + I)

        # Steps 3-5: identify spikes and reset
        fired        = self.V >= cfg.THETA
        self.spikes  = fired.astype(float)
        self.V[fired] = 0.0

        # Step 6: update activity trace
        # da/dt = -a/tau_trace + spikes
        self.a += dt * (-self.a / cfg.TAU_TRACE + self.spikes)

        # Step 6b: update firing rate estimate (used by readout)
        # dr/dt = -r/tau_rate + spikes
        self.r += dt * (-self.r / cfg.TAU_RATE + self.spikes)

        # Record history
        self.spike_history.append(self.spikes.copy())
        self.voltage_history.append(self.V.copy())

    # ------------------------------------------------------------------
    # Section 4 — Hebbian trace learning step
    # ------------------------------------------------------------------
    def learn(self):
        """
        Apply one learning update after a timestep.

        L <- L + eta * outer(a, spikes)   [Hebbian strengthening]
        L <- L * (1 - lambda * dt)         [weight decay]
        L <- L * M_exc                     [locality — excitatory only]
        L <- clip(L, 0, L_max)             [physical bounds]
        W <- S + L                         [additive coupling]

        Additive formulation: learned weights add to structural coupling
        rather than scaling it. This allows L to activate regions that S
        does not strongly drive, giving learning genuine representational
        freedom beyond the structural baseline.
        """
        dt = cfg.DT

        # Hebbian strengthening
        self.L += cfg.ETA * np.outer(self.a, self.spikes)

        # Weight decay
        self.L *= (1.0 - cfg.LAMBDA * dt)

        # Locality constraint — only excitatory connections are plastic
        self.L *= self.M_exc

        # Clip to physical bounds — lower bound 0 since L is purely additive
        self.L = np.clip(self.L, 0.0, cfg.L_MAX)

        # Recompute effective coupling — additive formulation
        # W = S + L allows learned weights to activate regions
        # that S does not strongly drive, giving learning genuine
        # freedom to reshape representations beyond structural constraints.
        self.W = self.S + self.L

    # ------------------------------------------------------------------
    # Convenience: run for T timesteps with optional learning
    # ------------------------------------------------------------------
    def run(self, signal, T, learn=True):
        """
        Present a signal for T timesteps.

        Parameters
        ----------
        signal  : list of (frequency, amplitude) tuples
        T       : int, number of timesteps
        learn   : bool, whether to apply learning each step
        """
        for _ in range(T):
            self.step(signal)
            if learn:
                self.learn()

    # ------------------------------------------------------------------
    # Readout: mean firing rate over the last T timesteps
    # ------------------------------------------------------------------
    def mean_firing_rate(self, T):
        """
        Returns the mean spike count per node over the last T timesteps.
        Shape: (n,) — can be reshaped to (N, N) for plotting.
        """
        if len(self.spike_history) < T:
            T = len(self.spike_history)
        recent = np.array(self.spike_history[-T:])  # shape (T, n)
        return recent.mean(axis=0)

    # ------------------------------------------------------------------
    # Reset dynamic state (keep learned weights)
    # ------------------------------------------------------------------
    def reset_state(self):
        """
        Reset voltage, traces, and spike history.
        Learned weights L and W are preserved.
        """
        self.V      = np.zeros(self.n)
        self.a      = np.zeros(self.n)
        self.r      = np.zeros(self.n)
        self.spikes = np.zeros(self.n)
        self.spike_history  = []
        self.voltage_history = []

    # ------------------------------------------------------------------
    # Save / load (Section on reproducibility)
    # ------------------------------------------------------------------
    def save(self, path):
        """Save learned weights and config to a .npz file."""
        np.savez(path,
                 L=self.L,
                 W=self.W,
                 freq=self.freq,
                 pos_x=self.pos_x,
                 pos_y=self.pos_y)
        print(f"Network saved to {path}.npz")

    def load(self, path):
        """Load learned weights from a .npz file."""
        data    = np.load(path)
        self.L  = data['L']
        self.W  = data['W']
        print(f"Network loaded from {path}")