Skyrmion Dynamics Simulator
===========================

A desktop app that simulates magnetic skyrmions. Basically little
swirly magnetic patterns that can be pushed around with electric
currents. Uses tkinter for the GUI and matplotlib for plotting.

Files
-----
main.py          -- the GUI
simulation.py    -- the physics stuff
visualization.py -- plotting and display
requirements.txt -- what you need to install
README.txt       -- this file

Setup
-----
You need Python 3.7 or newer. Its a good idea to use a virtual
environment:

  python3 -m venv skyrmion_env
  source skyrmion_env/bin/activate      # linux/mac
  # skyrmion_env\Scripts\activate       # windows

Then install the required packages:

  pip install -r requirements.txt

Running
-------
Make sure the virtual env is active and run:

  python main.py

A window should pop up. If something is missing it will tell you.

How to use
----------
1. Type in your parameters (or just use the defaults)
2. Click "Generate Skyrmion" to make the spin texture
3. Click "Run Motion Simulation" to watch it move
4. Click "Plot Center of Mass" to see the path it took
5. Click "Export Results" to save everything to a text file

Physics (briefly)
-----------------
Skyrmions are these cool magnetic whirlpools. The three energy
terms that keep them stable are exchange (wants spins aligned),
DMI (wants spins twisted), and Zeeman (from the external field).
When you run a current through them they move like leaves on water,
including a sideways drift called the Hall effect.

Troubleshooting
---------------
If you get "No module named tkinter":
  Linux: sudo apt install python3-tk
  Mac/Windows: should come with Python

If numpy or matplotlib are missing:
  pip install -r requirements.txt
