#!/usr/bin/env python3
"""
Generate visualization of Brownian motion properties for blog post.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

def generate_brownian_motion(n_steps=1000, dt=0.01, n_paths=5):
    """Generate multiple Brownian motion paths."""
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
    W = np.zeros((n_paths, n_steps + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    return t, W

def main():
    np.random.seed(42)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # ============================================================
    # Panel 1: Multiple Brownian motion paths
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :])
    t, W = generate_brownian_motion(n_steps=1000, dt=0.01, n_paths=8)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, W.shape[0]))
    for i in range(W.shape[0]):
        ax1.plot(t, W[i], alpha=0.7, linewidth=1.5, color=colors[i], label=f'Path {i+1}')
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('$W(t)$')
    ax1.set_title('Property 1 & 4: Continuous Paths Starting from $W(0) = 0$', 
                  fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=4, loc='upper left', framealpha=0.9)
    
    # Add annotation
    ax1.text(0.02, 0.98, 'Each path is continuous but nowhere differentiable',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============================================================
    # Panel 2: Independent increments
    # ============================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Generate one path and show increments
    t_inc, W_inc = generate_brownian_motion(n_steps=200, dt=0.05, n_paths=1)
    W_inc = W_inc[0]
    
    # Select specific time intervals
    intervals = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    
    ax2.plot(t_inc, W_inc, color='steelblue', linewidth=2, label='$W(t)$')
    
    colors_inc = ['red', 'green', 'orange', 'purple', 'brown']
    for idx, (t_start, t_end) in enumerate(intervals):
        i_start = np.argmin(np.abs(t_inc - t_start))
        i_end = np.argmin(np.abs(t_inc - t_end))
        
        ax2.plot([t_inc[i_start], t_inc[i_end]], 
                [W_inc[i_start], W_inc[i_end]], 
                'o-', color=colors_inc[idx], linewidth=2.5, 
                markersize=6, alpha=0.8,
                label=fr'$\Delta W_{{{idx+1}}}$')
        
        # Add arrow annotation
        mid_t = (t_inc[i_start] + t_inc[i_end]) / 2
        mid_w = (W_inc[i_start] + W_inc[i_end]) / 2
        increment = W_inc[i_end] - W_inc[i_start]
        ax2.annotate(f'{increment:.2f}', 
                    xy=(mid_t, mid_w), 
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors_inc[idx], alpha=0.3))
    
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('$W(t)$')
    ax2.set_title('Property 2: Independent Increments', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8)
    ax2.text(0.02, 0.98, '$W(t_2) - W(t_1)$ independent of $W(t_4) - W(t_3)$',
             transform=ax2.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # ============================================================
    # Panel 3: Gaussian increments distribution
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Generate many paths to show distribution at specific time
    n_samples = 10000
    t_points = [1, 2, 4]
    colors_gauss = ['crimson', 'darkgreen', 'navy']
    
    for idx, t_val in enumerate(t_points):
        # Generate Brownian motion values at time t_val
        dW = np.random.randn(n_samples, int(t_val / 0.01)) * np.sqrt(0.01)
        W_t = np.sum(dW, axis=1)
        
        # Plot histogram
        counts, bins, _ = ax3.hist(W_t, bins=50, alpha=0.4, 
                                    color=colors_gauss[idx], 
                                    density=True, 
                                    label=f'$W({t_val})$ (samples)')
        
        # Overlay theoretical Gaussian
        x = np.linspace(bins[0], bins[-1], 200)
        ax3.plot(x, norm.pdf(x, 0, np.sqrt(t_val)), 
                color=colors_gauss[idx], linewidth=2.5,
                linestyle='--', label=fr'$\mathcal{{N}}(0, {t_val})$')
    
    ax3.set_xlabel('$W(t)$ value')
    ax3.set_ylabel('Probability density')
    ax3.set_title(r'Property 3: Gaussian Distribution $W(t) \sim \mathcal{N}(0, t)$', 
                  fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, f'$\\mathbb{{E}}[W(t)] = 0$\n$\\text{{Var}}[W(t)] = t$',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # ============================================================
    # Panel 4: Scaling property
    # ============================================================
    ax4 = fig.add_subplot(gs[2, 0])
    
    t_scale, W_scale = generate_brownian_motion(n_steps=1000, dt=0.01, n_paths=1)
    W_scale = W_scale[0]
    
    # Original path
    ax4.plot(t_scale, W_scale, color='blue', linewidth=1.5, 
            alpha=0.7, label='$W(t)$')
    
    # Scaled versions: W(ct) vs sqrt(c) * W(t)
    c_values = [0.5, 2.0]
    colors_scale = ['green', 'red']
    
    for c, color in zip(c_values, colors_scale):
        # Time-scaled version: resample at different rate
        if c < 1:
            # Slower time: sample at ct
            t_scaled_idx = (t_scale * c).astype(int)
            t_scaled_idx = np.clip(t_scaled_idx, 0, len(W_scale) - 1)
            W_ct = W_scale[t_scaled_idx]
            ax4.plot(t_scale, W_ct, color=color, linewidth=1.5, 
                    alpha=0.7, label=f'$W({c}t)$', linestyle='-')
        else:
            # Faster time
            t_scaled_idx = (t_scale * c).astype(int)
            valid_mask = t_scaled_idx < len(W_scale)
            t_plot = t_scale[valid_mask]
            W_ct = W_scale[t_scaled_idx[valid_mask]]
            ax4.plot(t_plot, W_ct, color=color, linewidth=1.5, 
                    alpha=0.7, label=f'$W({c}t)$ (truncated)', linestyle='-')
        
        # Amplitude-scaled version: sqrt(c) * W(t)
        W_scaled = np.sqrt(c) * W_scale
        ax4.plot(t_scale, W_scaled, color=color, linewidth=1.5, 
                alpha=0.7, label=fr'$\sqrt{{{c}}} W(t)$', linestyle='--')
    
    ax4.set_xlabel('Time $t$')
    ax4.set_ylabel('$W(t)$')
    ax4.set_title(r'Scaling Property: $W(ct)$ has same distribution as $\sqrt{c} \, W(t)$', 
                  fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.text(0.02, 0.02, 'Time scaling = Amplitude scaling',
             transform=ax4.transAxes, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))
    
    # ============================================================
    # Panel 5: Quadratic variation
    # ============================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Compute quadratic variation for increasingly fine partitions
    t_qv, W_qv = generate_brownian_motion(n_steps=1000, dt=0.01, n_paths=1)
    W_qv = W_qv[0]
    T = t_qv[-1]
    
    partition_sizes = np.logspace(0, 3, 20).astype(int)
    quad_variations = []
    
    for n in partition_sizes:
        if n > len(t_qv):
            break
        indices = np.linspace(0, len(t_qv)-1, n).astype(int)
        increments = np.diff(W_qv[indices])
        qv = np.sum(increments ** 2)
        quad_variations.append(qv)
    
    partition_sizes = partition_sizes[:len(quad_variations)]
    
    ax5.semilogx(partition_sizes, quad_variations, 'o-', 
                color='darkviolet', linewidth=2, markersize=5,
                label=r'Computed $\sum [W(t_{i+1}) - W(t_i)]^2$')
    ax5.axhline(T, color='red', linestyle='--', linewidth=2,
               label=f'Theoretical limit = $T$ = {T:.1f}')
    
    ax5.set_xlabel('Number of partition points')
    ax5.set_ylabel('Quadratic variation')
    ax5.set_title('Quadratic Variation: $(dW)^2 = dt$', fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3, which='both')
    ax5.text(0.02, 0.98, 'As partition gets finer,\nquadratic variation → $T$',
             transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Overall title
    fig.suptitle('Brownian Motion: Mathematical Properties Visualized', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'assets', 'images', 'brownian-motion-properties.png')
    output_path = os.path.normpath(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to {output_path}")
    plt.close()

if __name__ == '__main__':
    main()
