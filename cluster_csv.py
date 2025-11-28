import numpy as np
from sklearn.cluster import DBSCAN

def cluster_and_extract(results, px_size_m, radius=100):
    """
    Cluster results using DBSCAN instead of manual KD-tree BFS.
    Much faster for large datasets.

    Score = h_min - htree - hgoal.

    Returns:
        A list of the best item per cluster.
    """

    if not results:
        return results

    # ---------------------------------------
    # Step 1: Prepare coordinates (scaled to meters)
    # ---------------------------------------
    coords = np.array([(rm * px_size_m, cm * px_size_m)
                       for (rm, cm, *_rest) in results],
                      dtype=float)

    # ---------------------------------------
    # Step 2: Run DBSCAN clustering
    # ---------------------------------------
    # eps = distance threshold (radius)
    # min_samples=1 ensures every point belongs to a cluster
    db = DBSCAN(eps=radius, min_samples=1, algorithm='kd_tree')
    labels = db.fit_predict(coords)

    # Number of clusters
    n_clusters = labels.max() + 1

    # ---------------------------------------
    # Step 3: For each cluster, pick best scoring item
    # ---------------------------------------
    selected = []
    labels = np.asarray(labels)

    for cl in range(n_clusters):
        # indices of items in this cluster
        idxs = np.where(labels == cl)[0]

        # Compute scores for these indices
        best_idx = None
        best_score = -np.inf

        for idx in idxs:
            (
                rm, cm, r0, c0, r, c,
                h_min, l, h_mid, h0, h,
                htree, hgoal
            ) = results[idx]

            score = h_min - htree - hgoal

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(results[best_idx])

    return selected



