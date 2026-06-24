import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk


# draws the spin configuration as arrows on a colored background
def plot_spin_configuration(parent, spins,
                            title="Skyrmion Spin Configuration"):
    N, M = spins.shape[:2]
    Sx, Sy, Sz = spins[:, :, 0], spins[:, :, 1], spins[:, :, 2]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)

    # show out-of-plane component as color map
    im = ax.imshow(Sz, cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[0, M - 1, N - 1, 0], aspect="equal",
                   interpolation="bilinear")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Sz  (out-of-plane)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # subsample arrows so the plot is readable
    step = max(1, min(N, M) // 18)
    X, Y = np.meshgrid(np.arange(M), np.arange(N))

    ax.quiver(X[::step, ::step], Y[::step, ::step],
              Sx[::step, ::step], Sy[::step, ::step],
              scale=1.5, scale_units="xy", angles="xy",
              color="black", alpha=0.55,
              width=0.003, headwidth=3, headlength=4)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X  (lattice units)", fontsize=11)
    ax.set_ylabel("Y  (lattice units)", fontsize=11)
    ax.tick_params(labelsize=10)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    return canvas


# plots the path the skyrmion center took during the simulation
def plot_com_trajectory(parent, positions, lattice_size):
    if len(positions) < 2:
        raise ValueError("Need at least 2 positions for a trajectory.")

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)

    # color points by step number to show direction
    pts = ax.scatter(xs, ys, c=range(len(xs)), cmap="coolwarm",
                     s=35, alpha=0.85, zorder=5)

    # connecting line
    ax.plot(xs, ys, "k-", alpha=0.3, linewidth=1, zorder=3)

    # start and end markers
    ax.plot(xs[0], ys[0], "go", ms=10, label="Start", zorder=6)
    ax.plot(xs[-1], ys[-1], "ro", ms=10, label="End",   zorder=6)

    cbar = fig.colorbar(pts, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Timestep", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title("Skyrmion Centre-of-Mass Trajectory",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("X  (lattice units)", fontsize=11)
    ax.set_ylabel("Y  (lattice units)", fontsize=11)

    ax.set_xlim(0, lattice_size)
    ax.set_ylim(lattice_size, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    return canvas


# shows the physics equations as labels in the info panel
def display_equations(parent):
    frame = tk.Frame(parent, relief=tk.GROOVE, bd=2, padx=10, pady=10)

    tk.Label(frame, text="Physics Equations",
             font=("Arial", 13, "bold")).pack(anchor=tk.W, pady=(0, 8))

    # skyrmion spin texture equations
    tk.Label(frame, text="Skyrmion Spin Texture:",
             font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(5, 2))
    for line in [
        "theta(r) = pi * exp(-r / R)",
        "Sx = sin(theta) * cos(phi)",
        "Sy = sin(theta) * sin(phi)",
        "Sz = cos(theta)",
        "phi = arctan2(y - y0, x - x0)",
    ]:
        tk.Label(frame, text=line,
                 font=("Arial", 10)).pack(anchor=tk.W)

    # energy equations
    tk.Label(frame, text="Magnetic Energies:",
             font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(10, 2))
    for line in [
        "Heisenberg Exchange:  E_ex = -J * sum(Si . Sj)",
        "DMI:  E_DM = D * sum(dij . (Si x Sj))",
        "Zeeman:  E_Z = -B * sum(Sz_i)",
    ]:
        tk.Label(frame, text=line,
                 font=("Arial", 10)).pack(anchor=tk.W)

    # motion equation
    tk.Label(frame, text="Motion (Thiele equation):",
             font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(10, 2))
    for line in [
        "G x v + alpha*D . v + F = 0",
        "Longitudinal  +  Hall effect",
    ]:
        tk.Label(frame, text=line,
                 font=("Arial", 10)).pack(anchor=tk.W)

    frame.pack(fill=tk.X)
    return frame


# updates the results panel with simulation data
def display_results(parent, initial_pos, final_pos, distance, num_steps):
    for w in parent.winfo_children():
        w.destroy()

    frame = tk.Frame(parent, relief=tk.GROOVE, bd=2, padx=10, pady=10)
    frame.pack(fill=tk.X, expand=False)

    tk.Label(frame, text="Simulation Results",
             font=("Arial", 13, "bold")).pack(anchor=tk.W, pady=(0, 8))

    items = [
        ("Initial Position:",   f"({initial_pos[0]:.2f}, {initial_pos[1]:.2f})"),
        ("Final Position:",     f"({final_pos[0]:.2f}, {final_pos[1]:.2f})"),
        ("Distance Travelled:", f"{distance:.2f} lattice units"),
        ("Number of Steps:",    str(num_steps)),
    ]

    for label, value in items:
        row = tk.Frame(frame)
        row.pack(anchor=tk.W, fill=tk.X, pady=2)
        tk.Label(row, text=label,   font=("Arial", 10, "bold"),
                 width=20, anchor=tk.W).pack(side=tk.LEFT)
        tk.Label(row, text=value,   font=("Arial", 10),
                 anchor=tk.W).pack(side=tk.LEFT, padx=(8, 0))
