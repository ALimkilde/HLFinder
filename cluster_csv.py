import numpy as np
from sklearn.cluster import DBSCAN

def two_stage_cluster(
    results,
    px_size_m,
    radius=100,
    a1_index=(0, 1),   # (rm, cm)
    a2_index=(2, 3),   # (ra1, ca1) or change if needed
):
    if not results:
        return []

    radius_px = radius / px_size_m

    # ------------------------
    # Stage 1: cluster A1
    # ------------------------
    coords_a1 = np.array(
        [(r[a1_index[0]] * px_size_m,
          r[a1_index[1]] * px_size_m)
         for r in results],
        dtype=float
    )

    db1 = DBSCAN(
        eps=radius,
        min_samples=1,
        algorithm="kd_tree"
    )
    labels_a1 = db1.fit_predict(coords_a1)

    final_labels = np.full(len(results), -1, dtype=int)
    next_label = 0

    # ------------------------
    # Stage 2: cluster A2 inside each A1 cluster
    # ------------------------
    for lbl in np.unique(labels_a1):
        idxs = np.where(labels_a1 == lbl)[0]

        if len(idxs) == 1:
            final_labels[idxs[0]] = next_label
            next_label += 1
            continue

        coords_a2 = np.array(
            [(results[i][a2_index[0]] * px_size_m,
              results[i][a2_index[1]] * px_size_m)
             for i in idxs],
            dtype=float
        )

        db2 = DBSCAN(
            eps=radius,
            min_samples=1,
            algorithm="kd_tree"
        )
        labels_a2 = db2.fit_predict(coords_a2)

        for sub_lbl in np.unique(labels_a2):
            sub_idxs = idxs[labels_a2 == sub_lbl]
            final_labels[sub_idxs] = next_label
            next_label += 1

    return final_labels

def cluster_and_extract(
    results,
    search_pic,
    px_size_m,
    radius=100,
    keep=["hmean"],   # <-- user controls what to keep
):
    if not results:
        return results

    coords = np.array(
        [(rm * px_size_m, cm * px_size_m)
         for (rm, cm, *_rest) in results],
        dtype=float
    )

    labels = two_stage_cluster(
        results,
        px_size_m=px_size_m,
        radius=radius,
        a1_index=(2, 3),     # (rm, cm)
        a2_index=(4, 5),     # adjust if you want (ra2, ca2)
    )
    n_clusters = labels.max() + 1

    # ---------------------------------------
    # Define how each metric is computed
    # ---------------------------------------
    def compute_hmean(item):
        (rm, cm, ra1, ca1, ra2, ca2, h_min, l, h_mid, h0, h, hgoal,
         score, hmean_terr, hmean_surf, walkable) = item
        return min(hmean_terr, hmean_surf)
    def compute_rigging_height(item):
        (rm, cm, ra1, ca1, ra2, ca2, h_min, l, h_mid, h0, h, hgoal,
         score, hmean_terr, hmean_surf, walkable) = item
        rigging_height_a1 = h_min - search_pic.im[ra1, ca1]
        rigging_height_a2 = h_min - search_pic.im[ra2, ca2]
        return -min(rigging_height_a1, rigging_height_a2)

    metric_funcs = {
        "score":     lambda item: item[12],
        "walkable":  lambda item: item[15],
        "hmean":     compute_hmean,
        "rigging_height":     compute_rigging_height,
        "height":     lambda item: item[6],
    }

    # sanity filter
    keep = [k for k in keep if k in metric_funcs]

    selected = []
    labels = np.asarray(labels)

    for cl in range(n_clusters):
        idxs = np.where(labels == cl)[0]


        # Find the best item for each metric requested
        for metric_name in keep:
            metric_func = metric_funcs[metric_name]
            best_idx = max(idxs, key=lambda i: metric_func(results[i]))
            selected.append(results[best_idx])

    return selected
