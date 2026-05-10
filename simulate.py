# simulate.py
# Runs the Phase 1 experiment:
#   1. Initialize the network
#   2. Visualize the structural coupling and frequency gradient
#   3. Present two distinct input signals, let the network learn
#   4. Test recall — do distinct inputs produce distinct activity patterns?
#   5. Plot results

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import config as cfg
from network import RFNeuromorphicNetwork


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def reshape(v, N):
    """Reshape flat node vector to NxN grid for plotting."""
    return v.reshape(N, N)


def plot_grid(ax, data, title, cmap='viridis', vmin=None, vmax=None):
    """Plot a single NxN heatmap on ax."""
    im = ax.imshow(data, cmap=cmap, origin='lower',
                   vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('y', fontsize=8)
    ax.set_ylabel('x', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# -----------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------

def main():
    print("Initializing network...")
    net = RFNeuromorphicNetwork(seed=42)
    N   = cfg.N

    # -------------------------------------------------------------------
    # Figure 1: Structural properties
    # -------------------------------------------------------------------
    fig1, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig1.suptitle("Structural Properties (with Lateral Inhibition)", fontsize=12)

    # Frequency gradient
    plot_grid(axes[0],
              reshape(net.freq, N),
              "Resonant Frequency Gradient",
              cmap='plasma')

    # Mexican hat profile — coupling from center node as a function of distance
    center = (N // 2) * N + (N // 2)           # index of center node
    center_row = net.S[center].copy()
    center_row[center] = 0
    plot_grid(axes[1],
              reshape(center_row, N),
              "Coupling Profile\n(from center node)",
              cmap='RdBu',
              vmin=-np.abs(center_row).max(),
              vmax= np.abs(center_row).max())

    # Full S matrix
    lim_S = np.abs(net.S).max()
    im = axes[2].imshow(net.S, cmap='RdBu', aspect='auto',
                        vmin=-lim_S, vmax=lim_S)
    axes[2].set_title("Structural Coupling Matrix S", fontsize=10)
    axes[2].set_xlabel("Node j", fontsize=8)
    axes[2].set_ylabel("Node i", fontsize=8)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Excitatory locality mask
    im2 = axes[3].imshow(net.M_exc, cmap='Greys', aspect='auto')
    axes[3].set_title("Excitatory Mask M_exc\n(plastic connections)", fontsize=10)
    axes[3].set_xlabel("Node j", fontsize=8)
    axes[3].set_ylabel("Node i", fontsize=8)
    plt.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

    fig1.tight_layout()
    fig1.savefig("fig1_structure.png", dpi=150)
    print("Saved fig1_structure.png")

    # -------------------------------------------------------------------
    # Define two input signals
    # Signal A: low frequency component  (f=2.0)
    # Signal B: high frequency component (f=4.0)
    # -------------------------------------------------------------------
    signal_A = [(1.5, 1.5)]
    signal_B = [(4.5, 1.5)]

    # -------------------------------------------------------------------
    # Phase 1: Verify dynamics without learning
    # Present signal A, record spike activity
    # -------------------------------------------------------------------
    print("\nPhase 1: Testing dynamics (no learning)...")
    net.reset_state()

    spike_counts = []
    for t in range(cfg.T_PRESENT):
        net.step(signal_A)
        spike_counts.append(net.spikes.sum())

    total_spikes = sum(spike_counts)
    active_nodes = (net.mean_firing_rate(cfg.T_PRESENT) > 0).sum()
    print(f"  Total spikes over {cfg.T_PRESENT} steps: {total_spikes:.0f}")
    print(f"  Active nodes: {active_nodes} / {net.n}")

    if total_spikes == 0:
        print("  WARNING: No spikes produced. Consider lowering THETA or")
        print("  raising input amplitude in config.py.")
    elif active_nodes == net.n:
        print("  WARNING: All nodes firing. Network may be too excitable.")
        print("  Consider raising THETA or lowering input amplitude.")
    else:
        print("  Dynamics look healthy.")

    # Plot spike rate from Phase 1
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle("Phase 1: Dynamics Under Signal A (No Learning)",
                  fontsize=11)

    plot_grid(axes2[0],
              reshape(net.mean_firing_rate(cfg.T_PRESENT), N),
              "Mean Firing Rate — Signal A\n(no learning)",
              cmap='hot')

    axes2[1].plot(spike_counts, color='steelblue', linewidth=0.8)
    axes2[1].set_title("Total Spikes Per Timestep", fontsize=10)
    axes2[1].set_xlabel("Timestep", fontsize=8)
    axes2[1].set_ylabel("Spike count", fontsize=8)

    fig2.tight_layout()
    fig2.savefig("fig2_dynamics.png", dpi=150)
    print("Saved fig2_dynamics.png")

    # -------------------------------------------------------------------
    # Phase 2: Training with learning
    # Alternate presentations of signal A and signal B
    # -------------------------------------------------------------------
    print("\nPhase 2: Training (with learning)...")
    net.reset_state()

    n_epochs = 10   # number of alternating A/B presentations
    for epoch in range(n_epochs):
        net.run(signal_A, cfg.T_PRESENT, learn=True)
        net.reset_state()           # reset dynamic state, keep weights
        net.run(signal_B, cfg.T_PRESENT, learn=True)
        net.reset_state()

        # Progress report
        mean_L  = np.abs(net.L[net.M > 0]).mean()
        print(f"  Epoch {epoch+1}/{n_epochs} — "
              f"mean |L| in active connections: {mean_L:.4f}")

    print("Training complete.")

    # -------------------------------------------------------------------
    # Phase 3: Recall — do signals A and B produce distinct patterns?
    # -------------------------------------------------------------------
    print("\nPhase 3: Testing recall...")

    # Present A (no learning), record activity
    net.reset_state()
    net.run(signal_A, cfg.T_PRESENT, learn=False)
    rate_A = net.mean_firing_rate(cfg.T_PRESENT)

    # Present B (no learning), record activity
    net.reset_state()
    net.run(signal_B, cfg.T_PRESENT, learn=False)
    rate_B = net.mean_firing_rate(cfg.T_PRESENT)

    # Discriminability: correlation between patterns
    # Low correlation = distinct patterns = good
    if rate_A.std() > 0 and rate_B.std() > 0:
        correlation = np.corrcoef(rate_A, rate_B)[0, 1]
        print(f"  Pattern correlation (A vs B): {correlation:.4f}")
        print(f"  (closer to 0 = more distinct patterns)")
    else:
        print("  WARNING: One or both patterns are flat — "
              "network may not be producing meaningful responses.")

    # -------------------------------------------------------------------
    # Figure 3: Recall patterns
    # -------------------------------------------------------------------
    vmax = max(rate_A.max(), rate_B.max(), 1e-6)

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
    fig3.suptitle("Phase 3: Recall After Training", fontsize=12)

    plot_grid(axes3[0],
              reshape(rate_A, N),
              "Activity Pattern — Signal A\n(after training)",
              cmap='hot', vmin=0, vmax=vmax)

    plot_grid(axes3[1],
              reshape(rate_B, N),
              "Activity Pattern — Signal B\n(after training)",
              cmap='hot', vmin=0, vmax=vmax)

    # Difference map
    diff = rate_A - rate_B
    lim  = np.abs(diff).max() or 1.0
    plot_grid(axes3[2],
              reshape(diff, N),
              "Difference (A minus B)",
              cmap='RdBu', vmin=-lim, vmax=lim)

    fig3.tight_layout()
    fig3.savefig("fig3_recall.png", dpi=150)
    print("Saved fig3_recall.png")

    # -------------------------------------------------------------------
    # Figure 4: Learned weight changes
    # -------------------------------------------------------------------
    fig4, axes4 = plt.subplots(1, 2, figsize=(10, 4))
    fig4.suptitle("Learned Modulation Matrix L", fontsize=12)

    im_L = axes4[0].imshow(net.L, cmap='RdBu',
                            vmin=-cfg.L_MAX, vmax=cfg.L_MAX, aspect='auto')
    axes4[0].set_title("Full L matrix", fontsize=10)
    axes4[0].set_xlabel("Node j", fontsize=8)
    axes4[0].set_ylabel("Node i", fontsize=8)
    plt.colorbar(im_L, ax=axes4[0], fraction=0.046, pad=0.04)

    # Distribution of nonzero learned weights
    active_L = net.L[net.M > 0]
    axes4[1].hist(active_L, bins=50, color='steelblue', edgecolor='white',
                  linewidth=0.4)
    axes4[1].set_title("Distribution of L in active connections", fontsize=10)
    axes4[1].set_xlabel("L_ij value", fontsize=8)
    axes4[1].set_ylabel("Count", fontsize=8)
    axes4[1].axvline(0, color='black', linewidth=0.8, linestyle='--')

    fig4.tight_layout()
    fig4.savefig("fig4_weights.png", dpi=150)
    print("Saved fig4_weights.png")

    # -------------------------------------------------------------------
    # Phase 5: Generalization curve
    # How does completion quality degrade as the cue gets weaker
    # and more frequency-shifted?
    # -------------------------------------------------------------------
    print("\nPhase 5: Generalization curve...")

    # Two degradation axes:
    #   1. Amplitude — how weak can the cue be before recall fails?
    #   2. Frequency offset — how far off can the cue be before recall fails?

    amplitudes   = [1.5, 1.2, 0.8, 0.5, 0.3, 0.15, 0.05]
    freq_offsets = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    # --- Axis 1: amplitude degradation (fixed offset = 0.2) ---
    print("  Amplitude degradation (freq offset = 0.2):")
    completion_vs_amplitude_trained   = []
    completion_vs_amplitude_untrained = []

    net_untrained = RFNeuromorphicNetwork(seed=42)

    for amp in amplitudes:
        cue = [(1.5 + 0.2, amp)]   # fixed small frequency offset

        net.reset_state()
        net.run(cue, cfg.T_PRESENT, learn=False)
        r_trained = net.mean_firing_rate(cfg.T_PRESENT)

        net_untrained.reset_state()
        net_untrained.run(cue, cfg.T_PRESENT, learn=False)
        r_untrained = net_untrained.mean_firing_rate(cfg.T_PRESENT)

        if rate_A.std() > 0 and r_trained.std() > 0:
            c_trained = np.corrcoef(rate_A, r_trained)[0, 1]
        else:
            c_trained = 0.0

        if rate_A.std() > 0 and r_untrained.std() > 0:
            c_untrained = np.corrcoef(rate_A, r_untrained)[0, 1]
        else:
            c_untrained = 0.0

        completion_vs_amplitude_trained.append(c_trained)
        completion_vs_amplitude_untrained.append(c_untrained)
        print(f"    amp={amp:.2f}  trained={c_trained:.4f}  "
              f"untrained={c_untrained:.4f}  "
              f"improvement={c_trained - c_untrained:+.4f}")

    # --- Axis 2: frequency offset degradation (fixed amp = 0.4) ---
    print("  Frequency offset degradation (amp = 0.4):")
    completion_vs_offset_trained   = []
    completion_vs_offset_untrained = []

    for offset in freq_offsets:
        cue = [(1.5 + offset, 0.4)]

        net.reset_state()
        net.run(cue, cfg.T_PRESENT, learn=False)
        r_trained = net.mean_firing_rate(cfg.T_PRESENT)

        net_untrained.reset_state()
        net_untrained.run(cue, cfg.T_PRESENT, learn=False)
        r_untrained = net_untrained.mean_firing_rate(cfg.T_PRESENT)

        if rate_A.std() > 0 and r_trained.std() > 0:
            c_trained = np.corrcoef(rate_A, r_trained)[0, 1]
        else:
            c_trained = 0.0

        if rate_A.std() > 0 and r_untrained.std() > 0:
            c_untrained = np.corrcoef(rate_A, r_untrained)[0, 1]
        else:
            c_untrained = 0.0

        completion_vs_offset_trained.append(c_trained)
        completion_vs_offset_untrained.append(c_untrained)
        print(f"    offset={offset:.1f}  trained={c_trained:.4f}  "
              f"untrained={c_untrained:.4f}  "
              f"improvement={c_trained - c_untrained:+.4f}")

    # --- Figure 5: Generalization curves ---
    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4))
    fig5.suptitle("Generalization Curve: Completion Quality vs Cue Degradation",
                  fontsize=11)

    # Amplitude axis
    axes5[0].plot(amplitudes, completion_vs_amplitude_trained,
                  'o-', color='steelblue', linewidth=1.5, label='Trained')
    axes5[0].plot(amplitudes, completion_vs_amplitude_untrained,
                  's--', color='coral', linewidth=1.5, label='Untrained')
    axes5[0].fill_between(amplitudes,
                           completion_vs_amplitude_untrained,
                           completion_vs_amplitude_trained,
                           alpha=0.15, color='steelblue',
                           label='Learning improvement')
    axes5[0].set_xlabel("Cue amplitude", fontsize=9)
    axes5[0].set_ylabel("Completion quality (r)", fontsize=9)
    axes5[0].set_title("Amplitude degradation\n(freq offset = 0.2)", fontsize=10)
    axes5[0].legend(fontsize=8)
    axes5[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes5[0].set_ylim(-0.1, 1.05)

    # Frequency offset axis
    axes5[1].plot(freq_offsets, completion_vs_offset_trained,
                  'o-', color='steelblue', linewidth=1.5, label='Trained')
    axes5[1].plot(freq_offsets, completion_vs_offset_untrained,
                  's--', color='coral', linewidth=1.5, label='Untrained')
    axes5[1].fill_between(freq_offsets,
                           completion_vs_offset_untrained,
                           completion_vs_offset_trained,
                           alpha=0.15, color='steelblue',
                           label='Learning improvement')
    axes5[1].set_xlabel("Frequency offset (units)", fontsize=9)
    axes5[1].set_ylabel("Completion quality (r)", fontsize=9)
    axes5[1].set_title("Frequency offset degradation\n(amp = 0.4)", fontsize=10)
    axes5[1].legend(fontsize=8)
    axes5[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes5[1].set_ylim(-0.1, 1.05)

    fig5.tight_layout()
    fig5.savefig("fig5_generalization.png", dpi=150)
    print("Saved fig5_generalization.png")

    # -------------------------------------------------------------------
    # Save the trained network
    # -------------------------------------------------------------------
    net.save("trained_network")

    print("\nDone. Output files:")
    print("  fig1_structure.png       — frequency gradient, S matrix, locality mask")
    print("  fig2_dynamics.png        — spiking dynamics before learning")
    print("  fig3_recall.png          — activity patterns after training")
    print("  fig4_weights.png         — learned weight matrix")
    print("  fig5_generalization.png  — generalization curve")
    print("  trained_network.npz      — saved network weights")


if __name__ == "__main__":
    main()