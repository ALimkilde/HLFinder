import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from config import TREE_DIST

@njit
def hlheight(l):
    return 0.08 * l + 8

@njit
def hlheight_walk_atpos(x, l):
    hl = hlheight(l)
    a = -4.0*(hl - 8)/l**2
    b = 4.0*(hl - 8)/l
    c = 1.0
   
    return a*x**2 + b*x + c

@njit
def hlheight_leash_atpos(x, l):
    hl = hlheight(l)
    a = -4.0*(hl - 8)/l**2
    b = 4.0*(hl - 8)/l
    c = 5.0
   
    return a*x**2 + b*x + c

@njit
def hlheight_atpos(x, l):
    hl = hlheight(l)
    a = -4.0*(hl - 8)/l**2
    b = 4.0*(hl - 8)/l
    c = 8.5
   
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

    # Distance array (meters)
    d_m = t * np.hypot(dr * px_size_m, dc * px_size_m)

    # Extract curves (fast, vectorized)
    terr = np.empty(len(rr))
    terr_leash = np.empty(len(rr))
    surf = np.empty(len(rr))

    viol = 0  # number of points violating the walk-height test

    for i in range(len(rr)):
        d = d_m[i]
        r = r_int[i]
        c = c_int[i]

        terr[i] = h_min - hlheight_atpos(d, l) - terrain[r, c]
        terr_leash[i] = h_min - hlheight_leash_atpos(d, l) - terrain[r, c]
        if(surface is not None):
            surf[i] = h_min - hlheight_leash_atpos(d, l) - surface[r, c]
        
        # Only check inside the valid walking interval
        if d > TREE_DIST and d < l - TREE_DIST:
            limit = h_min - hlheight_walk_atpos(d, l)
            if terrain[r,c] > limit :
                viol += 1

            if surface is not None:
                if surface[r,c] > limit:
                    viol += 1

    dont_hit_tree = viol == 0

    walkable_terr = np.sum(np.greater(terr,0))
    walkable_surf = np.sum(np.greater(surf,0))

    score_terr = float(walkable_terr)/length_px
    score_surf = float(walkable_surf)/length_px
    
    if (walkable_terr > 0):
        hmean_terr = np.sum(np.maximum(terr_leash,0))/walkable_terr
    else:
        hmean_terr = -1

    if (walkable_surf > 0):
        hmean_surf = np.sum(np.maximum(surf,0))/walkable_surf
    else:
        hmean_surf = -1

    walkable = l*min(walkable_terr, walkable_surf)/length_px
    
    return min(score_terr, score_surf), dont_hit_tree, hmean_terr, hmean_surf, walkable


def extract_line_profiles(terrain, surface, anchor, r0, c0, r1, c1, px_size_m):
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

    if (anchor is not None):
       anch = anchor[r_int, c_int]
    else:
       anch = terr

    return d_m, terr, surf, anch


def plot_line_profiles(d_m, terr, surf, anch, score, l, h_min, hanchor):
    plt.figure(figsize=(8, 4))

    ideal_height_backup = np.empty(len(d_m))
    ideal_height_leash = np.empty(len(d_m))
    ideal_height_walk = np.empty(len(d_m))
    terr = terr.astype(np.float64)
    surf = surf.astype(np.float64)
    for i, d in enumerate(d_m):
        ideal_height_backup[i] = - hlheight_atpos(d, l)
        ideal_height_leash[i] = - hlheight_leash_atpos(d, l)
        ideal_height_walk[i] = - hlheight_walk_atpos(d, l)
        terr[i] = terr[i] - hanchor
        surf[i] = surf[i] - hanchor
        anch[i] = anch[i] - hanchor

    plt.plot(d_m, terr, label="Terrain", lw=2)
    plt.plot(d_m, surf, label="Surface", lw=2)
    plt.plot(d_m, anch, label="Anchors", lw=2)
    plt.plot(d_m, ideal_height_backup, 'k--', label="backup", lw=2)
    plt.plot(d_m, ideal_height_leash, 'b--', label="leash", lw=2)
    plt.plot(d_m, ideal_height_walk, 'r--', label="walk", lw=2)

    plt.axvline(TREE_DIST); plt.axvline(l - TREE_DIST)

    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title(f"Height Profiles, score = {score}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

