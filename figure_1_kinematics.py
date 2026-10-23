"""Figure 1 redesign — planar dimensioned schematic + 3D DH frames."""
import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, Circle

from agrirob_model import forward_kinematics

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'pdf.fonttype': 42,
})
from paths import FIGURES
OUT = str(FIGURES)

DARK, ACC = '#2b2b2b', '#1f77b4'


def joint_marker(ax, p, r=0.022):
    ax.add_patch(Circle(p, r, fc='white', ec=DARK, lw=2.0, zorder=9))
    ax.add_patch(Circle(p, r * 0.30, fc=DARK, ec='none', zorder=10))


def dim(ax, p1, p2, label, off, rot=0, side=1):
    """Dimension line with arrows between p1 and p2, offset perpendicular."""
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d = p2 - p1
    n = np.array([-d[1], d[0]])
    n = n / (np.linalg.norm(n) + 1e-12) * off * side
    a, b = p1 + n, p2 + n
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='<->', mutation_scale=13,
                                 color='#8a6d3b', lw=1.5, zorder=6))
    for p, q in ((p1, a), (p2, b)):
        ax.plot([p[0], q[0]], [p[1], q[1]], '-', color='#8a6d3b',
                lw=0.9, alpha=0.75, zorder=5)
    m = (a + b) / 2
    ax.text(m[0], m[1], label, fontsize=14.5, color='#8a6d3b', rotation=rot,
            ha='center', va='center', zorder=11,
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none'))


def build():
    fig = plt.figure(figsize=(14.6, 6.4))

    # ---------------------------------------------------------------- (a) 2D
    ax = fig.add_subplot(121)
    q = np.array([0.0, -1.0, 0.4, 0.0, 0.8, 0.0])
    _, TT = forward_kinematics(q, return_all=True)
    P = np.array([T[:3, 3] for T in TT])
    X, Z = P[:, 0], P[:, 2]

    # platform
    ax.add_patch(Rectangle((-0.20, -0.105), 0.46, 0.075, fc='#e8e8e8',
                           ec='#6f6f6f', lw=1.8, zorder=2))
    for wx in (-0.135, 0.195):
        ax.add_patch(Circle((wx, -0.115), 0.048, fc='#d4d4d4', ec='#4a4a4a',
                            lw=1.8, zorder=3))
        ax.add_patch(Circle((wx, -0.115), 0.013, fc='#4a4a4a', ec='none', zorder=4))
    ax.text(0.03, -0.068, 'mobile platform', fontsize=12.5, ha='center',
            color='#4a4a4a', zorder=6)
    # ground
    ax.plot([-0.30, 0.74], [-0.163, -0.163], '-', color='#7a5c2e', lw=2.2, zorder=1)
    for gx in np.arange(-0.28, 0.74, 0.055):
        ax.plot([gx, gx - 0.028], [-0.163, -0.197], '-', color='#7a5c2e',
                lw=1.1, zorder=1)
    ax.text(0.715, -0.150, 'sandy soil', fontsize=12.5, ha='right',
            va='top', color='#7a5c2e', style='italic')

    # links
    ax.plot(X, Z, '-', color=DARK, lw=7, solid_capstyle='round', zorder=7)
    ax.plot(X, Z, '-', color='#9fbcd8', lw=3.2, solid_capstyle='round', zorder=8)
    for p in P:
        joint_marker(ax, (p[0], p[2]))

    # end-effector
    ax.plot(X[-1], Z[-1], '*', color='#d62728', ms=22, mec='0.15', mew=1.0, zorder=12)
    ax.annotate('end-effector', xy=(X[-1], Z[-1]), xytext=(0.255, 0.028),
                fontsize=13, color='#d62728', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5,
                                shrinkB=8))

    # frame origin labels
    names = {0: '$O_0$', 1: '$O_1\\!=\\!O_2$', 3: '$O_3$',
             4: '$O_4\\!=\\!O_5$', 6: '$O_6$'}
    offs = {0: (-0.062, 0.000), 1: (-0.105, 0.028), 3: (0.062, 0.042),
            4: (0.098, 0.036), 6: (0.085, -0.028)}
    for i, txt in names.items():
        ax.text(X[i] + offs[i][0], Z[i] + offs[i][1], txt, fontsize=15.5,
                fontweight='bold', ha='center', va='center', zorder=13)

    # joint function callouts
    calls = [(1, 'base rotation ($\\theta_1$)\n+ shoulder ($\\theta_2$)', (-0.175, 0.335)),
             (3, 'elbow ($\\theta_3$)', (0.045, 0.582)),
             (4, 'spherical wrist\n($\\theta_4,\\theta_5,\\theta_6$)', (0.605, 0.425))]
    for i, txt, xy in calls:
        ax.annotate(txt, xy=(X[i], Z[i]), xytext=xy, fontsize=12.5,
                    color='#1a4d7a', ha='center', va='center', zorder=14,
                    bbox=dict(boxstyle='round,pad=0.32', fc='#eef5fb',
                              ec='#9fbcd8', lw=1.0),
                    arrowprops=dict(arrowstyle='->', color='#1a4d7a',
                                    lw=1.4, shrinkB=6))

    # dimensions
    dim(ax, (X[0], Z[0]), (X[1], Z[1]), '$d_1$', 0.075, rot=90, side=-1)
    dim(ax, (X[2], Z[2]), (X[3], Z[3]), '$a_3$', 0.058, side=1)
    dim(ax, (X[3], Z[3]), (X[4], Z[4]), '$d_4$', 0.058, side=1)
    dim(ax, (X[4], Z[4]), (X[6], Z[6]), '$d_6$', 0.055, side=1)

    ax.set_xlabel('$X$ (m)')
    ax.set_ylabel('$Z$ (m)')
    ax.set_title('(a) Dimensioned schematic in the sagittal plane', pad=9)
    ax.set_xlim(-0.32, 0.76)
    ax.set_ylim(-0.22, 0.64)
    ax.set_aspect('equal')
    ax.grid(alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)

    # ---------------------------------------------------------------- (b) 3D
    ax3 = fig.add_subplot(122, projection='3d')
    q3 = np.array([0.55, -0.95, 0.45, 0.35, 0.75, 0.25])
    _, T3 = forward_kinematics(q3, return_all=True)
    Q = np.array([T[:3, 3] for T in T3])

    ax3.plot(Q[:, 0], Q[:, 1], Q[:, 2], '-', color=DARK, lw=6,
             solid_capstyle='round', zorder=3)
    ax3.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='white', s=115, edgecolor=DARK,
                linewidths=1.7, zorder=6, depthshade=False)

    L = 0.10
    for T in T3:
        o = T[:3, 3]
        for k, c in enumerate(['#d62728', '#2ca02c', '#1f77b4']):
            a = T[:3, k] * L
            ax3.quiver(o[0], o[1], o[2], a[0], a[1], a[2], color=c,
                       arrow_length_ratio=0.32, lw=2.2, zorder=7)

    o3 = {0: (-0.11, 0.02, -0.055), 1: (-0.13, 0.02, 0.040),
          3: (0.02, 0.02, 0.090), 4: (0.03, 0.10, 0.060), 6: (0.03, 0.06, -0.09)}
    for i, txt in names.items():
        o = T3[i][:3, 3]
        d = o3[i]
        ax3.text(o[0] + d[0], o[1] + d[1], o[2] + d[2], txt, fontsize=14.5,
                 fontweight='bold', zorder=14)

    pe = Q[-1]
    ax3.scatter([pe[0]], [pe[1]], [pe[2]], color='#d62728', s=280, marker='*',
                edgecolor='0.15', linewidths=1.0, zorder=13, depthshade=False)

    c = np.array([[-.40, -.35, -.05], [.40, -.35, -.05], [.40, .35, -.05],
                  [-.40, .35, -.05], [-.40, -.35, -.05]])
    ax3.plot(c[:, 0], c[:, 1], c[:, 2], '-', color='0.5', lw=2.0)
    th = np.linspace(0, 2 * pi, 40)
    for wx in (-0.34, 0.34):
        for wy in (-0.30, 0.30):
            ax3.plot(wx + 0.072 * np.cos(th), [wy] * 40,
                     -0.05 + 0.072 * np.sin(th), '-', color='0.35', lw=1.9)

    for c_, l_ in zip(['#d62728', '#2ca02c', '#1f77b4'],
                      ['$x_i$', '$y_i$', '$z_i$ (joint axis)']):
        ax3.plot([], [], '-', color=c_, lw=2.6, label=l_)
    ax3.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.90), ncol=1,
               handlelength=1.5, borderpad=0.5, fontsize=12)

    ax3.set_xlabel('$X$ (m)', labelpad=11)
    ax3.set_ylabel('$Y$ (m)', labelpad=11)
    ax3.set_zlabel('$Z$ (m)', labelpad=6)
    ax3.set_xlim(-0.46, 0.62); ax3.set_ylim(-0.46, 0.60); ax3.set_zlim(-0.10, 0.66)
    ax3.set_box_aspect((1.0, 1.0, 0.80))
    ax3.view_init(elev=20, azim=-58)
    ax3.grid(False)
    ax3.tick_params(labelsize=11.5, pad=1)
    ax3.set_title('(b) DH frames in a spatial configuration', pad=0, y=0.99)

    fig.subplots_adjust(wspace=0.16)
    fig.savefig(f"{OUT}/fig1_kinematic_diagram.pdf")
    fig.savefig(f"{OUT}/fig1_kinematic_diagram.png", dpi=200)
    plt.close(fig)
    print("fig1 regenerated")


if __name__ == "__main__":
    build()
