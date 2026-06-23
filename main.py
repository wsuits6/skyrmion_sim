#!/usr/bin/env python3
"""
main.py -- Entry point for the Hybrid Quantum-Classical Simulation
           of Skyrmion Dynamics.

This Tkinter desktop application lets you:

  1. Set simulation parameters (lattice size, exchange constant,
     DMI strength, magnetic field, current, steps).
  2. Generate a skyrmion spin texture.
  3. Run a motion simulation (skyrmion driven by spin-polarised current).
  4. Plot the centre-of-mass trajectory.
  5. Export results to a text file.

Usage
-----
    python main.py

Dependencies:  numpy, matplotlib  (see requirements.txt)
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import os

from simulation import (
    generate_skyrmion,
    compute_total_energy,
    simulate_motion,
)
from visualization import (
    plot_spin_configuration,
    plot_com_trajectory,
    display_equations,
    display_results,
)


class SkyrmionApp:
    """Main GUI application."""

    def __init__(self, root):
        self.root = root
        root.title("Hybrid Quantum-Classical Simulation of Skyrmion Dynamics")
        root.geometry("1400x900")
        root.minsize(1200, 800)

        # ---- internal state ----
        self.spins     = None          # current spin configuration
        self.centre    = None          # current centre (cx, cy)
        self.positions = None          # trajectory list
        self.energies  = None          # energy history

        self._build_ui()
        self._initialise()

    # ------------------------------------------------------------------
    #  Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Create the three-panel layout: controls | plot | info."""
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)   # fixed-width left
        self.root.grid_columnconfigure(1, weight=1)   # stretches
        self.root.grid_columnconfigure(2, weight=0)   # fixed-width right

        self._build_control_panel()   # column 0
        self._build_plot_panel()      # column 1
        self._build_info_panel()      # column 2

    def _build_control_panel(self):
        """Left panel: parameter inputs and action buttons."""
        frame = tk.Frame(self.root, padx=15, pady=15,
                         relief=tk.RIDGE, bd=2)
        frame.grid(row=0, column=0, sticky=tk.NS,
                   padx=(10, 5), pady=10)

        tk.Label(frame, text="Simulation Controls",
                 font=("Arial", 15, "bold")).pack(anchor=tk.CENTER,
                                                   pady=(0, 18))

        # -----  Input fields  -----
        inp = tk.Frame(frame, relief=tk.GROOVE, bd=2, padx=10, pady=10)
        inp.pack(fill=tk.X, pady=(0, 15))

        tk.Label(inp, text="Parameters",
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 8))

        self._vars = {}               # label -> StringVar
        fields = [
            ("Lattice Size:",       "30"),
            ("Exchange Constant J:", "1.0"),
            ("DMI Strength D:",     "0.5"),
            ("Magnetic Field B:",   "0.3"),
            ("Current Strength:",   "0.5"),
            ("Simulation Steps:",   "100"),
        ]

        for label, default in fields:
            row = tk.Frame(inp)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, width=20, anchor=tk.W,
                     font=("Arial", 10)).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            tk.Entry(row, textvariable=var, width=10,
                     font=("Arial", 10)).pack(side=tk.RIGHT)
            self._vars[label] = var

        # -----  Buttons  -----
        btn_frame = tk.Frame(frame, padx=10, pady=10)
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(btn_frame, text="Actions",
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 8))

        buttons = [
            ("Generate Skyrmion",   self._on_generate,  "#4CAF50"),
            ("Run Motion Simulation", self._on_motion,  "#2196F3"),
            ("Plot Center of Mass", self._on_plot_com,  "#FF9800"),
            ("Export Results",      self._on_export,    "#9C27B0"),
        ]

        for text, cmd, colour in buttons:
            tk.Button(btn_frame, text=text, command=cmd,
                      font=("Arial", 11, "bold"),
                      bg=colour, fg="white",
                      padx=10, pady=7,
                      cursor="hand2",
                      relief=tk.RAISED, bd=3).pack(fill=tk.X, pady=4)

        # -----  Status  -----
        st = tk.Frame(frame, relief=tk.GROOVE, bd=2, padx=10, pady=10)
        st.pack(fill=tk.X)

        tk.Label(st, text="Status",
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 8))

        self._status = tk.Label(st, text="Ready",
                                font=("Arial", 10), wraplength=220,
                                justify=tk.LEFT)
        self._status.pack(anchor=tk.W)

    def _build_plot_panel(self):
        """Centre panel: hosts the matplotlib figure(s)."""
        frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=2)
        frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=10)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self._plot_container = tk.Frame(frame, bg="white")
        self._plot_container.grid(row=0, column=0, sticky=tk.NSEW)

        # placeholder message
        msg = tk.Message(
            self._plot_container,
            text="Generate a skyrmion to begin\n\n"
                 "1.  Set parameters (or keep defaults)\n"
                 "2.  Click  'Generate Skyrmion'\n"
                 "3.  Click  'Run Motion Simulation'\n"
                 "4.  Click  'Plot Center of Mass'",
            font=("Arial", 14), bg="white", fg="grey",
            justify=tk.CENTER, width=500,
        )
        msg.pack(expand=True, fill=tk.BOTH)

    def _build_info_panel(self):
        """Right panel: equations and results."""
        frame = tk.Frame(self.root, padx=15, pady=15,
                         relief=tk.RIDGE, bd=2)
        frame.grid(row=0, column=2, sticky=tk.NS,
                   padx=(5, 10), pady=10)

        # equations
        self._eq_frame = tk.Frame(frame)
        self._eq_frame.pack(fill=tk.X, pady=(0, 20))
        display_equations(self._eq_frame)

        # results
        res_frame = tk.Frame(frame, relief=tk.GROOVE, bd=2,
                             padx=10, pady=10)
        res_frame.pack(fill=tk.X)

        tk.Label(res_frame, text="Results",
                 font=("Arial", 12, "bold")).pack(anchor=tk.W,
                                                   pady=(0, 8))

        self._res_container = tk.Frame(res_frame)
        self._res_container.pack(fill=tk.X)

        self._res_label = tk.Label(
            self._res_container,
            text="No results yet.\nRun a simulation first.",
            font=("Arial", 10), fg="grey", justify=tk.CENTER,
        )
        self._res_label.pack(pady=20)

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def _initialise(self):
        """Generate a skyrmion with the default parameters."""
        self._set_status("Generating initial skyrmion ...")
        try:
            sz = int(self._vars["Lattice Size:"].get())
            self.spins, self.centre = generate_skyrmion(sz)
            self._refresh_plot()
            self._set_status("Ready")
        except Exception as exc:
            self._set_status(f"Init error: {exc}")

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg):
        self._status.config(text=msg)
        self.root.update_idletasks()

    def _read_params(self):
        """Read and validate all inputs; return a dict or raise."""
        try:
            return {
                "lattice_size":     int(self._vars["Lattice Size:"].get()),
                "J":               float(self._vars["Exchange Constant J:"].get()),
                "D":               float(self._vars["DMI Strength D:"].get()),
                "B":               float(self._vars["Magnetic Field B:"].get()),
                "current_strength": float(self._vars["Current Strength:"].get()),
                "num_steps":        int(self._vars["Simulation Steps:"].get()),
            }
        except ValueError:
            raise ValueError("All fields must contain valid numbers.")

    def _check_params(self, p):
        if p["lattice_size"] < 4:
            raise ValueError("Lattice size must be >= 4.")
        if p["J"] < 0:
            raise ValueError("Exchange constant J must be >= 0.")
        if p["num_steps"] < 1:
            raise ValueError("Steps must be >= 1.")

    def _clear_plot(self):
        for w in self._plot_container.winfo_children():
            w.destroy()

    def _refresh_plot(self):
        if self.spins is None:
            return
        self._clear_plot()
        c = plot_spin_configuration(self._plot_container, self.spins)
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------

    def _on_generate(self):
        """Generate a fresh skyrmion."""
        self._set_status("Generating skyrmion ...")
        try:
            p = self._read_params()
            self._check_params(p)

            self.spins, self.centre = generate_skyrmion(p["lattice_size"])
            E = compute_total_energy(self.spins, p["J"], p["D"], p["B"])

            self._refresh_plot()

            msg = (f"Skyrmion generated\n"
                   f"E_ex = {E['exchange']:.1f}  "
                   f"E_dm = {E['dmi']:.1f}  "
                   f"E_ze = {E['zeeman']:.1f}")
            self._set_status(msg)

        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            self._set_status("Error: invalid parameters")
        except Exception as exc:
            messagebox.showerror("Error", f"Generation failed:\n{exc}")
            self._set_status("Error: generation failed")

    def _on_motion(self):
        """Run the motion simulation."""
        if self.spins is None:
            messagebox.showwarning("No Skyrmion",
                                   "Generate a skyrmion first.")
            return

        self._set_status("Running motion simulation ...")
        try:
            p = self._read_params()
            self._check_params(p)

            copy = self.spins.copy()
            self.spins, self.positions, self.energies = simulate_motion(
                copy,
                p["current_strength"],
                p["num_steps"],
                J=p["J"],
                D=p["D"],
                B=p["B"],
            )

            self._refresh_plot()

            init = self.positions[0]
            final = self.positions[-1]
            dist = np.sqrt((final[0] - init[0]) ** 2 +
                           (final[1] - init[1]) ** 2)

            self._show_results(init, final, dist, len(self.positions) - 1)
            self._set_status(f"Simulation done  |  distance = {dist:.2f}")

        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            self._set_status("Error: invalid parameters")
        except Exception as exc:
            messagebox.showerror("Error", f"Simulation failed:\n{exc}")
            self._set_status("Error: simulation failed")

    def _on_plot_com(self):
        """Plot the centre-of-mass trajectory."""
        if not self.positions or len(self.positions) < 2:
            messagebox.showwarning("No Trajectory",
                                   "Run a motion simulation first.")
            return

        self._set_status("Plotting trajectory ...")
        try:
            p = self._read_params()
            self._clear_plot()
            c = plot_com_trajectory(self._plot_container,
                                    self.positions,
                                    p["lattice_size"])
            c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._set_status("Trajectory plotted")
        except Exception as exc:
            messagebox.showerror("Error", f"Plot failed:\n{exc}")
            self._set_status("Error: plot failed")

    def _on_export(self):
        """Export trajectory and parameters to a text file."""
        if not self.positions:
            messagebox.showwarning("No Data",
                                   "Run a motion simulation first.")
            return

        try:
            p   = self._read_params()
            init  = self.positions[0]
            final = self.positions[-1]
            dist  = np.sqrt((final[0] - init[0]) ** 2 +
                            (final[1] - init[1]) ** 2)

            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            path    = os.path.join(desktop, "skyrmion_results.txt")

            lines = [
                "=" * 58,
                "  Hybrid Quantum-Classical Simulation of Skyrmion Dynamics",
                "  Results Export",
                "=" * 58,
                "",
                "PARAMETERS",
                f"  Lattice Size:         {p['lattice_size']} x {p['lattice_size']}",
                f"  Exchange Constant J:  {p['J']}",
                f"  DMI Strength D:       {p['D']}",
                f"  Magnetic Field B:     {p['B']}",
                f"  Current Strength:     {p['current_strength']}",
                f"  Simulation Steps:     {p['num_steps']}",
                "",
                "RESULTS",
                f"  Initial Position:     ({init[0]:.2f}, {init[1]:.2f})",
                f"  Final Position:       ({final[0]:.2f}, {final[1]:.2f})",
                f"  Distance Travelled:   {dist:.2f}",
                f"  Steps Recorded:       {len(self.positions)}",
                "",
            ]

            if self.energies:
                lines.append("ENERGY (first -> last, delta)")
                first, last = self.energies[0], self.energies[-1]
                for key in ("exchange", "dmi", "zeeman", "total"):
                    d = last[key] - first[key]
                    lines.append(
                        f"  {key.capitalize():10s}:  "
                        f"{first[key]:10.2f} -> {last[key]:10.2f}  "
                        f"(Delta = {d:+.2f})"
                    )
                lines.append("")

            lines.append("TRAJECTORY")
            lines.append(f"{'Step':>6}  {'X':>8}  {'Y':>8}")
            for i, (cx, cy) in enumerate(self.positions):
                lines.append(f"{i:6d}  {cx:8.2f}  {cy:8.2f}")
            lines.append("")
            lines.append("=" * 58)
            lines.append("End of report")
            lines.append("=" * 58)

            with open(path, "w") as fh:
                fh.write("\n".join(lines))

            self._set_status(f"Exported to {path}")
            messagebox.showinfo("Export Successful",
                                f"Results saved to:\n{path}")

        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))
            self._set_status("Error: export failed")

    def _show_results(self, init, final, dist, steps):
        """Update the results panel."""
        for w in self._res_container.winfo_children():
            w.destroy()
        display_results(self._res_container, init, final, dist, steps)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = SkyrmionApp(root)          # noqa: F841  (kept alive by root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
