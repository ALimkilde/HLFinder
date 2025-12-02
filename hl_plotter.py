import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def hlheight(l):
    return 0.08 * l + 12

@njit
def hlheight_atpos(x, l):
    hl = hlheight(l)
    a = -4.0*(hl - 8)/l**2
    b = 4.0*(hl - 8)/l
    c = 8.0
   
    return a*x**2 + b*x + c


@njit
def get_score(terrain, surface, r0, c0, r1, c1, px_size_m, h_min, l):
    """
    Returns:
        d_m     – distance along the line in meters (1D float array)
        terr    – terrain heights along the line (1D array)
        surf    – surface heights along the line (1D array)
    """

    # Line length in pixels
    dr = r1 - r0
    dc = c1 - c0
    length_px = int(np.hypot(dr, dc)) + 1

    # Parameter t in [0,1]
    t = np.linspace(0, 1, length_px)

    # Pixel coordinates along the line (float → round to nearest)
    rr = r0 + t * dr
    cc = c0 + t * dc

    r_int = np.rint(rr).astype(np.int32)
    c_int = np.rint(cc).astype(np.int32)

    # Clip to valid image bounds
    # r_int = np.clip(r_int, 0, terrain.shape[0] - 1)
    # c_int = np.clip(c_int, 0, terrain.shape[1] - 1)

    # Distance array (meters)
    d_m = t * np.hypot(dr * px_size_m, dc * px_size_m)

    # Extract curves (fast, vectorized)
    terr = np.empty(len(rr))
    for i in range(len(rr)):
        terr[i] = h_min - hlheight_atpos(d_m[i], l) - terrain[r_int[i], c_int[i]]

    score = float(np.sum(np.greater(terr,0)))/length_px
    return score


def extract_line_profiles(terrain, surface, r0, c0, r1, c1, px_size_m):
    """
    Returns:
        d_m     – distance along the line in meters (1D float array)
        terr    – terrain heights along the line (1D array)
        surf    – surface heights along the line (1D array)
    """

    # Line length in pixels
    dr = r1 - r0
    dc = c1 - c0
    length_px = int(np.hypot(dr, dc)) + 1

    # Parameter t in [0,1]
    t = np.linspace(0, 1, length_px)

    # Pixel coordinates along the line (float → round to nearest)
    rr = r0 + t * dr
    cc = c0 + t * dc

    r_int = np.rint(rr).astype(np.int32)
    c_int = np.rint(cc).astype(np.int32)

    # Clip to valid image bounds
    r_int = np.clip(r_int, 0, terrain.shape[0] - 1)
    c_int = np.clip(c_int, 0, terrain.shape[1] - 1)

    # Distance array (meters)
    d_m = t * np.hypot(dr * px_size_m, dc * px_size_m)

    # Extract curves (fast, vectorized)
    terr = terrain[r_int, c_int]
    if (surface is not None):
       surf = surface[r_int, c_int]
    else:
       surf = terr

    return d_m, terr, surf


def plot_line_profiles(d_m, terr, surf, score, l, h_min):
    plt.figure(figsize=(8, 4))
    plt.plot(d_m, terr, label="Terrain", lw=2)
    # plt.plot(d_m, surf, label="Surface", lw=2)

    ideal_height = np.empty(len(d_m))
    for i, d in enumerate(d_m):
        ideal_height[i] = h_min - hlheight_atpos(d, l)

    plt.plot(d_m, ideal_height, 'k--', label="hlheight_atpos", lw=2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title(f"Height Profiles, score = {score}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

