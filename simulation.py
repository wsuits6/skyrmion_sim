import numpy as np


# generates a skyrmion spin texture on a square lattice
# uses theta(r) = pi * exp(-r/R) for the radial profile
def generate_skyrmion(lattice_size, R=None):
    if not isinstance(lattice_size, int) or lattice_size < 4:
        raise ValueError("Lattice size must be an integer >= 4.")

    if R is None:
        R = lattice_size / 6.0

    # center of the lattice
    cx = cy = lattice_size / 2.0

    # coordinate grid
    x = np.arange(lattice_size)
    y = np.arange(lattice_size)
    X, Y = np.meshgrid(x, y)

    # distance from center and winding angle
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    phi = np.arctan2(Y - cy, X - cx)

    # radial profile - this creates the skyrmion shape
    theta = np.pi * np.exp(-r / R)

    # spin components
    Sx = np.sin(theta) * np.cos(phi)
    Sy = np.sin(theta) * np.sin(phi)
    Sz = np.cos(theta)

    spins = np.stack([Sx, Sy, Sz], axis=-1)

    return spins, (cx, cy)


# Heisenberg exchange energy: E_ex = -J * sum(Si . Sj)
# only nearest neighbor pairs on the square lattice
def compute_exchange_energy(spins, J):
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")

    # dot products along x direction
    dot_x = np.sum(spins[:, :-1, :] * spins[:, 1:, :], axis=-1)
    # dot products along y direction
    dot_y = np.sum(spins[:-1, :, :] * spins[1:, :, :], axis=-1)

    return -J * (np.sum(dot_x) + np.sum(dot_y))


# Dzyaloshinskii-Moriya interaction energy: E_DM = D * sum(dij . (Si x Sj))
# interfacial DMI on a 2D square lattice
def compute_dmi_energy(spins, D):
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")

    Sx, Sy, Sz = spins[:, :, 0], spins[:, :, 1], spins[:, :, 2]

    # +x bonds give d = (0, -1, 0) -> contribution: Six*Sjz - Siz*Sjx
    dmi_x = Sx[:, :-1] * Sz[:, 1:] - Sz[:, :-1] * Sx[:, 1:]
    # +y bonds give d = (1, 0, 0) -> contribution: Siy*Sjz - Siz*Sjy
    dmi_y = Sy[:-1, :] * Sz[1:, :] - Sz[:-1, :] * Sy[1:, :]

    return D * (np.sum(dmi_x) + np.sum(dmi_y))


# Zeeman energy from out-of-plane magnetic field: E_Z = -B * sum(Sz_i)
def compute_zeeman_energy(spins, B):
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")

    return -B * np.sum(spins[:, :, 2])


# sums up all three energy contributions
def compute_total_energy(spins, J, D, B):
    E_ex = compute_exchange_energy(spins, J)
    E_dm = compute_dmi_energy(spins, D)
    E_ze = compute_zeeman_energy(spins, B)
    return {"exchange": E_ex, "dmi": E_dm, "zeeman": E_ze,
            "total": E_ex + E_dm + E_ze}


# finds the skyrmion center using (1 - Sz)/2 weighting
# the core has Sz ~ -1 so it gets more weight
def compute_centre_of_mass(spins):
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")

    N, M = spins.shape[:2]
    Sz = spins[:, :, 2]

    X, Y = np.meshgrid(np.arange(M), np.arange(N))

    # weight: 0 when Sz = +1 (background), 1 when Sz = -1 (core)
    w = (1.0 - Sz) / 2.0
    total_w = np.sum(w)

    if total_w < 1e-12:
        return M / 2.0, N / 2.0

    cx = np.sum(X * w) / total_w
    cy = np.sum(Y * w) / total_w
    return cx, cy


# simulates skyrmion motion under a spin-polarized current
# uses a simple model with longitudinal and transverse (hall effect) displacement
def simulate_motion(spins, current_strength, num_steps, J=1.0, D=0.5, B=0.3):
    if spins.ndim != 3 or spins.shape[2] != 3:
        raise ValueError("spins must have shape (N, M, 3).")
    if not np.isfinite(current_strength):
        raise ValueError("current_strength must be finite.")
    if not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")

    N, M = spins.shape[:2]
    cx, cy = compute_centre_of_mass(spins)

    # displacement per step
    scale = current_strength * 0.02
    hall_angle = np.pi / 6

    positions = [(cx, cy)]
    energies  = [compute_total_energy(spins, J, D, B)]
    R = max(N, M) / 6.0

    for _ in range(num_steps):
        # longitudinal and transverse motion
        dx = scale * np.cos(hall_angle)
        dy = scale * np.sin(hall_angle)

        cx += dx
        cy += dy

        # keep the center away from the edges
        cx = np.clip(cx, 2, M - 3)
        cy = np.clip(cy, 2, N - 3)

        # rebuild the spin texture around the new center
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
