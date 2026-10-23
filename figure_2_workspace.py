"""
=============================================================================
  AgriRob-SN — Figure 2 — reachable workspace (Monte-Carlo sampling)
  Vector (PDF) output, no embedded titles (captions live in LaTeX).
=============================================================================
"""
import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from agrirob_model import workspace_sample

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 15,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'legend.framealpha': 0.92,
    'lines.linewidth': 2.2,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.28,
    'grid.linewidth': 0.7,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
})

from paths import FIGURES
OUT = str(FIGURES)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}")


def fig2():
    P = workspace_sample(n_samples=4000)
    fig = plt.figure(figsize=(13.5, 5.8))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2], cmap='viridis',
                s=3.5, alpha=0.55, depthshade=False)
    from matplotlib.ticker import MaxNLocator
    for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax1.set_xlabel('$X$ (m)', labelpad=14)
    ax1.set_ylabel('$Y$ (m)', labelpad=14)
    ax1.set_zlabel('$Z$ (m)', labelpad=12)
    ax1.set_title('(a) Isometric view', pad=6)
    ax1.view_init(elev=22, azim=48)
    ax1.set_box_aspect((1, 1, 0.9))
    ax1.grid(False)
    ax1.tick_params(labelsize=12, pad=4)

    ax2 = fig.add_subplot(122)
    ax2.scatter(P[:, 0], P[:, 2], s=3.5, alpha=0.30, color='#1f77b4',
                edgecolors='none')
    r = np.hypot(P[:, 0], P[:, 1]).max()
    ax2.set_xlabel('$X$ (m)')
    ax2.set_ylabel('$Z$ (m)')
    ax2.set_title('(b) Sagittal section ($X$–$Z$ plane)', pad=6)
    ax2.set_aspect('equal')
    ax2.annotate(f'max. reach $\\approx$ {r:.2f} m',
                 xy=(0.97, 0.04), xycoords='axes fraction', ha='right',
                 fontsize=13,
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6'))
    fig.subplots_adjust(wspace=0.18)
    save(fig, 'fig2_workspace')



if __name__ == "__main__":
    fig2()
