import numpy as np
from sklearn.cluster import DBSCAN

def cluster_and_extract(
    results,
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

    db = DBSCAN(eps=radius, min_samples=1, algorithm='kd_tree')
    labels = db.fit_predict(coords)
    n_clusters = labels.max() + 1

    # ---------------------------------------
    # Define how each metric is computed
    # ---------------------------------------
    def compute_hmean(item):
        (_, _, _, _, _, _, h_min, l, h_mid, h0, h, hgoal,
         score, hmean_terr, hmean_surf, walkable) = item
        return min(hmean_terr, hmean_surf)

    metric_funcs = {
        "score":     lambda item: item[12],
        "walkable":  lambda item: item[15],
        "hmean":     compute_hmean,
        "height":     lambda item: item[7],
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
