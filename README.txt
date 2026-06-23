============================================================================
  Hybrid Quantum-Classical Simulation of Skyrmion Dynamics
============================================================================

A desktop GUI application that simulates and visualises magnetic
skyrmions -- topologically protected spin textures that behave like
particles and can be moved with electric currents.

The program implements a classical atomistic spin model with:

  * Heisenberg Exchange Energy
  * Dzyaloshinskii-Moriya Interaction (DMI)
  * Zeeman Energy

and visualises the spin texture and centre-of-mass motion using
matplotlib quiver plots embedded in a Tkinter window.


--------------------------------------------------------------------------
  Project Structure
--------------------------------------------------------------------------

  main.py            -- Tkinter GUI entry point
  simulation.py      -- physics engine (skyrmion generation, energies,
                        motion simulation)
  visualization.py   -- plotting and GUI display helpers
  requirements.txt   -- Python dependencies
  README.txt         -- this file


--------------------------------------------------------------------------
  Installation
--------------------------------------------------------------------------

**Prerequisites**

  Python 3.7 or later must be installed on your system.

**Step 1: Create a virtual environment (recommended)**

  On Linux / macOS:

      python3 -m venv skyrmion_env
      source skyrmion_env/bin/activate

  On Windows:

      python -m venv skyrmion_env
      skyrmion_env\Scripts\activate

**Step 2: Install dependencies**

      pip install -r requirements.txt

  This will install numpy and matplotlib.


--------------------------------------------------------------------------
  Running the Application
--------------------------------------------------------------------------

  Make sure you are in the project directory (skyrmion_sim/) and your
  virtual environment is active, then run:

      python main.py

  A window of 1400 x 900 pixels will open.  If your screen resolution
  is smaller, the window can be resized (minimum 1200 x 800).


--------------------------------------------------------------------------
  How to Use
--------------------------------------------------------------------------

  1.  Adjust the parameters (Lattice Size, Exchange Constant J,
      DMI Strength D, Magnetic Field B, Current Strength,
      Simulation Steps) in the left panel.

  2.  Click **Generate Skyrmion** to create the spin texture.
      A quiver plot of the skyrmion appears in the centre panel.

  3.  Click **Run Motion Simulation** to drive the skyrmion with
      a spin-polarised current.  The final spin state is shown.

  4.  Click **Plot Center of Mass** to see the trajectory traced
      by the skyrmion centre during the simulation.

  5.  Click **Export Results** to save the parameters, energies
      and trajectory to a text file on the Desktop.


--------------------------------------------------------------------------
  Physics Summary
--------------------------------------------------------------------------

  A skyrmion is a swirling magnetic texture.  Each little arrow
  (spin) on the grid represents a magnetic moment.  The three
  energy terms that stabilise the skyrmion are:

  * Exchange (J):  neighbouring spins prefer to point the same way.
  * DMI (D):       neighbouring spins prefer to twist.
  * Zeeman (B):    an external field pushes spins up or down.

  When a current flows through the skyrmion, the conduction
  electrons transfer angular momentum and push the skyrmion
  along (like a leaf carried by a stream).  The skyrmion moves
  both along the current and sideways (skyrmion Hall effect).


--------------------------------------------------------------------------
  Troubleshooting
--------------------------------------------------------------------------

  **No module named 'tkinter'**
      On Linux:  sudo apt install python3-tk   (Debian/Ubuntu)
                 sudo dnf install python3-tkinter  (Fedora)
      On macOS:  tkinter is included with the standard Python
                 installer from python.org.
      On Windows: tkinter is included by default.

  **ImportError: No module named numpy/matplotlib**
      Run:  pip install -r requirements.txt
