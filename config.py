# config.py
# All model parameters in one place.
# Tune these once the simulation is running.

# --- Grid ---
N = 20                  # NxN node grid (400 nodes total)

# --- Frequency gradient ---
F_MIN = 1.0             # minimum resonant frequency (arbitrary units)
F_MAX = 5.0             # maximum resonant frequency

# --- Coupling ---
SIGMA_EXC   = 2.0       # excitatory spatial length scale (node radii) — short range
SIGMA_INH   = 4.0       # inhibitory spatial length scale — broader than excitatory
ALPHA_INH   = 0.5       # inhibitory strength relative to excitatory (0 < alpha < 1)
SIGMA_FREQ  = 1.0       # spectral coupling length scale (frequency units)
EPSILON     = 0.001      # locality mask cutoff

# --- Node dynamics ---
THETA       = 3.0       # firing threshold
TAU         = 10.0      # membrane time constant (ms)
TAU_TRACE   = 50.0      # activity trace time constant (ms) — should be >> TAU
TAU_RATE    = 100.0     # firing rate estimate time constant (ms)
DT          = 0.1       # simulation timestep (ms)

# --- Learning ---
ETA         = 0.1      # learning rate — higher needed for additive formulation
LAMBDA      = 0.01     # weight decay rate
L_MAX       = 5.0       # maximum additive learned weight (W = S + L)

# --- Input ---
SIGMA_INPUT = 0.5       # input tuning width (frequency units)
T_PRESENT   = 500       # timesteps per input presentation