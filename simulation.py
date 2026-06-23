"""
simulation.py -- Physics simulation module for Skyrmion dynamics.

This module handles all the physics calculations:

  * Generation of skyrmion spin textures
  * Calculation of magnetic energies (Exchange, DMI, Zeeman)
  * Simulation of skyrmion motion under spin-polarized current
  * Center-of-mass tracking

Physics background
------------------
A skyrmion is a topologically protected magnetic texture that
looks like a little whirlpool of spins (magnetic moments).  It
can be thought of as a particle that can be moved around with
electric currents -- this makes it interesting for future
data-storage technologies.

The three energy contributions that stabilize a skyrmion are:

  1.  Heisenberg Exchange Energy  E_ex = -J  S_i . S_j
      This favours neighbouring spins pointing in the same
      direction (ferromagnetic alignment).

  2.  Dzyaloshinskii-Moriya Interaction  E_DM = D  d_ij . (S_i x S_j)
      This twists neighbouring spins relative to each other,
      which is what gives the skyrmion its characteristic swirl.

  3.  Zeeman Energy  E_Z = -B  S_z_i
      This couples each spin to an external magnetic field that
      points out of the plane.
"""

import numpy as np


# ---------------------------------------------------------------------------
#  Skyrmion generation
# ---------------------------------------------------------------------------

def generate_skyrmion(lattice_size, R=None):
    """
    Create a smooth skyrmion spin texture on a square lattice.

    Physics
    -------
    We place a skyrmion at the centre of the lattice.  Each spin
    is described by two angles:

      * theta(r) = pi * exp(-r / R)   -- radial profile
      * phi     = arctan2(y, y0, x - x0)  -- vortex winding

    The spin components are then:

      Sx = sin(theta) * cos(phi)
      Sy = sin(theta) * sin(phi)
      Sz = cos(theta)

    At the centre (r = 0):   theta = pi   ->  Sz = -1  (spin down)
    Far away   (r >> R):     theta ~ 0    ->  Sz = +1  (spin up)

    The smooth transition from down to up is what gives the
    skyrmion its characteristic "bubble" shape.

    Parameters
    ----------
    lattice_size : int
        Side length of the square lattice (must be >= 4).
    R : float or None
        Skyrmion radius in lattice units.  If None it defaults
        to lattice_size / 6.

    Returns
    -------
    spins : ndarray, shape (lattice_size, lattice_size, 3)
        Sx, Sy, Sz components at every lattice site.
    centre : tuple (float, float)
        (cx, cy) position of the skyrmion centre.

    Raises
    ------
    ValueError
        If lattice_size is not a positive integer >= 4.
    """
    if not isinstance(lattice_size, int) or lattice_size < 4:
        raise ValueError("Lattice size must be an integer >= 4.")

    if R is None:
        R = lattice_size / 6.0

    # Centre of the lattice
    cx = cy = lattice_size / 2.0

    # 2-D coordinate grid
    x = np.arange(lattice_size)
    y = np.arange(lattice_size)
    X, Y = np.meshgrid(x, y)

    # Distance from centre
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Winding angle around the centre
    phi = np.arctan2(Y - cy, X - cx)

    # Radial profile -- the key skyrmion shape function
    theta = np.pi * np.exp(-r / R)

    # Cartesian spin components
    Sx = np.sin(theta) * np.cos(phi)
    Sy = np.sin(theta) * np.sin(phi)
    Sz = np.cos(theta)

    spins = np.stack([Sx, Sy, Sz], axis=-1)

    return spins, (cx, cy)


# ---------------------------------------------------------------------------
#  Energy calculations
# ---------------------------------------------------------------------------

def compute_exchange_energy(spins, J):
    """
    Heisenberg exchange energy.

    E_ex = -J * sum_{<i,j>} (S_i . S_j)

    Only nearest-neighbour pairs on the square lattice are
    considered.  Positive J favours parallel alignment.

    Parameters
    ----------
    spins : ndarray, shape (N, M, 3)
        Spin configuration.
    J : float
        Exchange constant (J > 0 for ferromagnet).

    Returns
    -------
    float
    """
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")

    # Dot products along the x-direction  (bonds (i,j) -> (i,j+1))
    dot_x = np.sum(spins[:, :-1, :] * spins[:, 1:, :], axis=-1)

    # Dot products along the y-direction  (bonds (i,j) -> (i+1,j))
    dot_y = np.sum(spins[:-1, :, :] * spins[1:, :, :], axis=-1)

    return -J * (np.sum(dot_x) + np.sum(dot_y))


def compute_dmi_energy(spins, D):
    """
    Dzyaloshinskii-Moriya interaction (interfacial / Néel type).

    E_DM = D * sum_{<i,j>} [ d_ij . (S_i x S_j) ]

    For interfacial DMI in a 2-D square lattice the DMI vector
    on a bond from site *i* to site *j* is:

        d_ij = r_ij x z

    where r_ij is the unit vector along the bond and z = (0,0,1).

    This translates to:

      +x bond:  d = (0, -1, 0)   ->  contribution = Six*Sjz - Siz*Sjx
      +y bond:  d = (1,  0, 0)   ->  contribution = Siy*Sjz - Siz*Sjy

    Parameters
    ----------
    spins : ndarray, shape (N, M, 3)
    D : float
        DMI strength.

    Returns
    -------
    float
    """
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")

    Sx, Sy, Sz = spins[:, :, 0], spins[:, :, 1], spins[:, :, 2]

    # +x bonds: d = (0, -1, 0)
    # d . (S_i x S_j) = -(Siz*Sjx - Six*Sjz) = Six*Sjz - Siz*Sjx
    dmi_x = Sx[:, :-1] * Sz[:, 1:] - Sz[:, :-1] * Sx[:, 1:]

    # +y bonds: d = (1, 0, 0)
    # d . (S_i x S_j) = Siy*Sjz - Siz*Sjy
    dmi_y = Sy[:-1, :] * Sz[1:, :] - Sz[:-1, :] * Sy[1:, :]

    return D * (np.sum(dmi_x) + np.sum(dmi_y))


def compute_zeeman_energy(spins, B):
    """
    Zeeman energy (out-of-plane magnetic field).

    E_Z = -B * sum_i Sz_i

    A positive B favours spins pointing up (Sz > 0).

    Parameters
    ----------
    spins : ndarray, shape (N, M, 3)
    B : float
        Magnetic field strength.

    Returns
    -------
    float
    """
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")

    return -B * np.sum(spins[:, :, 2])


def compute_total_energy(spins, J, D, B):
    """
    Sum of exchange, DMI and Zeeman energies.

    Returns a dict with keys ``exchange``, ``dmi``, ``zeeman``
    and ``total``.
    """
    E_ex = compute_exchange_energy(spins, J)
    E_dm = compute_dmi_energy(spins, D)
    E_ze = compute_zeeman_energy(spins, B)
    return {"exchange": E_ex, "dmi": E_dm, "zeeman": E_ze,
            "total": E_ex + E_dm + E_ze}


# ---------------------------------------------------------------------------
#  Centre-of-mass tracking
# ---------------------------------------------------------------------------

def compute_centre_of_mass(spins):
    """
    Locate the skyrmion by weighting each site with (1 - Sz) / 2.

    The skyrmion core has Sz close to -1, giving it a large weight,
    while the uniform background (Sz ~ +1) receives almost no
    weight.  The centre of mass is the weighted average of the
    lattice coordinates.

    Parameters
    ----------
    spins : ndarray, shape (N, M, 3)

    Returns
    -------
    (cx, cy) : tuple of float
    """
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")

    N, M = spins.shape[:2]
    Sz = spins[:, :, 2]

    X, Y = np.meshgrid(np.arange(M), np.arange(N))

    # Weight:  0  when Sz = +1 (background),  1  when Sz = -1 (core)
    w = (1.0 - Sz) / 2.0
    total_w = np.sum(w)

    if total_w < 1e-12:
        return M / 2.0, N / 2.0

    cx = np.sum(X * w) / total_w
    cy = np.sum(Y * w) / total_w
    return cx, cy


# ---------------------------------------------------------------------------
#  Motion simulation
# ---------------------------------------------------------------------------

def simulate_motion(spins, current_strength, num_steps, J=1.0, D=0.5, B=0.3):
    """
    Drive the skyrmion with a spin-polarised current.

    Physics
    -------
    A spin-polarised current transfers angular momentum to the
    skyrmion via the s-d exchange interaction, exerting a torque
    that makes it move.  The motion is described (at a coarse
    level) by the Thiele equation:

        G x v  +  a D . v  +  F  =  0

    where G is the gyrocoupling vector, D the dissipation tensor,
    a the damping and F the force from the current.

    In this simplified model we shift the skyrmion centre by a
    small amount at each timestep.  The displacement has both a
    longitudinal component (along the current, here the +x-axis)
    and a transverse component (the skyrmion Hall effect).

    Parameters
    ----------
    spins : ndarray, shape (N, M, 3)
        Initial spin configuration (will be modified in place).
    current_strength : float
        Strength of the applied current.
    num_steps : int
        Number of simulation timesteps.
    J, D, B : float
        Material constants (passed through for energy recording).

    Returns
    -------
    spins : ndarray, shape (N, M, 3)
        Final spin configuration.
    positions : list of (float, float)
        Centre-of-mass coordinates at every step (including
        the initial position).
    energies : list of dict
        Energy dict at every step.
    """
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")
    if not np.isfinite(current_strength):
        raise ValueError("current_strength must be finite.")
    if not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")

    N, M = spins.shape[:2]
    cx, cy = compute_centre_of_mass(spins)

    # Displacement per step  (tunable scale factor)
    scale = current_strength * 0.02
    hall_angle = np.pi / 6          # ~30 degrees (skyrmion Hall effect)

    positions = [(cx, cy)]
    energies  = [compute_total_energy(spins, J, D, B)]
    R = max(N, M) / 6.0

    for _ in range(num_steps):
        # Longitudinal and transverse displacement
        dx = scale * np.cos(hall_angle)
        dy = scale * np.sin(hall_angle)

        cx += dx
        cy += dy

        # Keep the centre safely away from the lattice edges
        cx = np.clip(cx, 2, M - 3)
        cy = np.clip(cy, 2, N - 3)

        # Rebuild the spin texture around the new centre
        X, Y = np.meshgrid(np.arange(M), np.arange(N))
        r   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        phi = np.arctan2(Y - cy, X - cx)
        theta = np.pi * np.exp(-r / R)

        spins[:, :, 0] = np.sin(theta) * np.cos(phi)
        spins[:, :, 1] = np.sin(theta) * np.sin(phi)
        spins[:, :, 2] = np.cos(theta)

        positions.append((cx, cy))
        energies.append(compute_total_energy(spins, J, D, B))

    return spins, positions, energies
