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
    def __init__(self, root):
        self.root = root
        root.title("Skyrmion Dynamics Simulator")
        root.geometry("1500x950")
        root.minsize(1300, 850)

        # internal state
        self.spins     = None
        self.centre    = None
        self.positions = None
        self.energies  = None

        self._build_ui()
        self._initialise()

    # builds the main layout
    def _build_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=0)

        self._build_control_panel()
        self._build_plot_panel()
        self._build_info_panel()

    # left panel with parameter inputs and buttons
    def _build_control_panel(self):
        frame = tk.Frame(self.root, padx=20, pady=20,
                         relief=tk.RIDGE, bd=2)
        frame.grid(row=0, column=0, sticky=tk.NS,
                   padx=(10, 5), pady=10)

        tk.Label(frame, text="Simulation Controls",
                 font=("Arial", 16, "bold")).pack(anchor=tk.CENTER,
                                                   pady=(0, 20))

        # input fields
        inp = tk.Frame(frame, relief=tk.GROOVE, bd=2, padx=15, pady=15)
        inp.pack(fill=tk.X, pady=(0, 15))

        tk.Label(inp, text="Parameters",
                 font=("Arial", 13, "bold")).pack(anchor=tk.W, pady=(0, 10))

        self._vars = {}
        # (label, default, min, max, max_chars)
        fields = [
            ("Lattice Size:",       "30",  4,  200, 3),
            ("Exchange Constant J:", "1.0", 0,  10,  4),
            ("DMI Strength D:",     "0.5", 0,  5,   4),
            ("Magnetic Field B:",   "0.3", 0,  5,   4),
            ("Current Strength:",   "0.5", 0.01, 5, 4),
            ("Simulation Steps:",   "100", 1,  5000, 4),
        ]

        def _make_validator(limit):
            def validate(P):
                return len(P) <= limit
            return self.root.register(validate)

        self._defaults = {}
        for label, default, vmin, vmax, maxlen in fields:
            self._defaults[label] = default
            row = tk.Frame(inp)
            row.pack(fill=tk.X, pady=4)
            tk.Label(row, text=label, width=22, anchor=tk.W,
                     font=("Arial", 11)).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            vcmd = _make_validator(maxlen)
            entry = tk.Entry(row, textvariable=var, width=14,
                             font=("Arial", 11),
                             validate="key", validatecommand=(vcmd, "%P"))
            entry.pack(side=tk.RIGHT)
            self._vars[label] = var

        # buttons
        btn_frame = tk.Frame(frame, padx=15, pady=15)
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(btn_frame, text="Actions",
                 font=("Arial", 13, "bold")).pack(anchor=tk.W, pady=(0, 10))

        buttons = [
            ("Generate Skyrmion",   self._on_generate,  "#4CAF50"),
            ("Run Motion Simulation", self._on_motion,  "#2196F3"),
            ("Plot Center of Mass", self._on_plot_com,  "#FF9800"),
            ("Export Results",      self._on_export,    "#9C27B0"),
            ("Reset Parameters",    self._on_reset,     "#607D8B"),
        ]

        for text, cmd, colour in buttons:
            tk.Button(btn_frame, text=text, command=cmd,
                      font=("Arial", 12, "bold"),
                      bg=colour, fg="white",
                      padx=15, pady=10,
                      cursor="hand2",
                      relief=tk.RAISED, bd=3).pack(fill=tk.X, pady=5)

        # status display
        st = tk.Frame(frame, relief=tk.GROOVE, bd=2, padx=15, pady=15)
        st.pack(fill=tk.X)

        tk.Label(st, text="Status",
                 font=("Arial", 13, "bold")).pack(anchor=tk.W, pady=(0, 10))

        self._status = tk.Label(st, text="Ready",
                                font=("Arial", 11), wraplength=250,
                                justify=tk.LEFT)
        self._status.pack(anchor=tk.W)

    # center panel for the plot
    def _build_plot_panel(self):
        frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=2)
        frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=10)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self._plot_container = tk.Frame(frame, bg="white")
        self._plot_container.grid(row=0, column=0, sticky=tk.NSEW)

        # placeholder message when no plot is shown
        msg = tk.Message(
            self._plot_container,
            text="Generate a skyrmion to begin\n\n"
                 "1.  Set parameters (or keep defaults)\n"
                 "2.  Click  'Generate Skyrmion'\n"
                 "3.  Click  'Run Motion Simulation'\n"
                 "4.  Click  'Plot Center of Mass'",
            font=("Arial", 14), bg="white", fg="grey",
            justify=tk.CENTER, width=600,
        )
        msg.pack(expand=True, fill=tk.BOTH)

    # right panel showing equations and results
    def _build_info_panel(self):
        frame = tk.Frame(self.root, padx=20, pady=20,
                         relief=tk.RIDGE, bd=2)
        frame.grid(row=0, column=2, sticky=tk.NS,
                   padx=(5, 10), pady=10)

        # equations section
        self._eq_frame = tk.Frame(frame)
        self._eq_frame.pack(fill=tk.X, pady=(0, 20))
        display_equations(self._eq_frame)

        # results section
        res_frame = tk.Frame(frame, relief=tk.GROOVE, bd=2,
                             padx=15, pady=15)
        res_frame.pack(fill=tk.X)

        tk.Label(res_frame, text="Results",
                 font=("Arial", 13, "bold")).pack(anchor=tk.W,
                                                   pady=(0, 10))

        self._res_container = tk.Frame(res_frame)
        self._res_container.pack(fill=tk.X)

        self._res_label = tk.Label(
            self._res_container,
            text="No results yet.\nRun a simulation first.",
            font=("Arial", 11), fg="grey", justify=tk.CENTER,
        )
        self._res_label.pack(pady=20)

    # generates the initial skyrmion with default params
    def _initialise(self):
        self._set_status("Generating initial skyrmion ...")
        try:
            sz = int(self._vars["Lattice Size:"].get())
            self.spins, self.centre = generate_skyrmion(sz)
            self._refresh_plot()
            self._set_status("Ready")
        except Exception as exc:
            self._set_status(f"Init error: {exc}")

    # helpers
    def _set_status(self, msg):
        self._status.config(text=msg)
        self.root.update_idletasks()

    # reads all parameter values from the input fields
    def _read_params(self):
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

    # checks that parameters are within allowed ranges
    def _check_params(self, p):
        if p["lattice_size"] < 4 or p["lattice_size"] > 200:
            raise ValueError("Lattice size must be between 4 and 200.")
        if p["J"] < 0 or p["J"] > 10:
            raise ValueError("Exchange constant J must be between 0 and 10.")
        if p["D"] < 0 or p["D"] > 5:
            raise ValueError("DMI strength D must be between 0 and 5.")
        if p["B"] < 0 or p["B"] > 5:
            raise ValueError("Magnetic field B must be between 0 and 5.")
        if p["current_strength"] < 0.01 or p["current_strength"] > 5:
            raise ValueError("Current strength must be between 0.01 and 5.")
        if p["num_steps"] < 1 or p["num_steps"] > 5000:
            raise ValueError("Steps must be between 1 and 5000.")

    def _clear_plot(self):
        for w in self._plot_container.winfo_children():
            w.destroy()

    def _refresh_plot(self):
        if self.spins is None:
            return
        self._clear_plot()
        c = plot_spin_configuration(self._plot_container, self.spins)
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # button callbacks

    # generates a fresh skyrmion
    def _on_generate(self):
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

    # runs the motion simulation
    def _on_motion(self):
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

    # plots the center of mass trajectory
    def _on_plot_com(self):
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

    # exports results to a text file on the desktop
    def _on_export(self):
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
                "  Skyrmion Simulation Results",
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

    # resets all parameters and simulation state to defaults
    def _on_reset(self):
        for label, default in self._defaults.items():
            self._vars[label].set(default)
        self.spins = None
        self.centre = None
        self.positions = None
        self.energies = None
        self._clear_plot()
        msg = tk.Message(
            self._plot_container,
            text="Generate a skyrmion to begin\n\n"
                 "1.  Set parameters (or keep defaults)\n"
                 "2.  Click  'Generate Skyrmion'\n"
                 "3.  Click  'Run Motion Simulation'\n"
                 "4.  Click  'Plot Center of Mass'",
            font=("Arial", 14), bg="white", fg="grey",
            justify=tk.CENTER, width=600,
        )
        msg.pack(expand=True, fill=tk.BOTH)
        for w in self._res_container.winfo_children():
            w.destroy()
        tk.Label(
            self._res_container,
            text="No results yet.\nRun a simulation first.",
            font=("Arial", 11), fg="grey", justify=tk.CENTER,
        ).pack(pady=20)
        self._set_status("Ready")

    # updates the results display panel
    def _show_results(self, init, final, dist, steps):
        for w in self._res_container.winfo_children():
            w.destroy()
        display_results(self._res_container, init, final, dist, steps)


def main():
    root = tk.Tk()
    app = SkyrmionApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
