import pandas as pd
import numpy as np
import sys
from scipy.spatial import KDTree

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


import numpy as np
from scipy.spatial import KDTree


def cluster_and_extract(results, px_size_m, radius=100):
    """
    Cluster the results based on pixel positions (rm, cm) converted to meters,
    and for each cluster select the tuple with the highest score.

    score = h_min - htree - hgoal

    Returns:
        A list of tuples (same structure as input).
    """

    if (len(results) == 0):
        return results

    # ---------------------------------------
    # Step 1: KDTree clustering on (rm, cm) in meters
    # ---------------------------------------
    coords = np.array([(rm * px_size_m, cm * px_size_m)
                       for (rm, cm, *_rest) in results])
    tree = KDTree(coords)

    n = len(results)
    visited = np.zeros(n, dtype=bool)
    clusters = []

    # BFS clustering
    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        cluster = []
        visited[i] = True

        while queue:
            idx = queue.pop()
            cluster.append(idx)

            neighbors = tree.query_ball_point(coords[idx], r=radius)
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        clusters.append(cluster)

    # ---------------------------------------
    # Step 2: For each cluster select the best scoring tuple
    # score = h_min - htree - hgoal
    # ---------------------------------------
    selected = []

    for cluster in clusters:
        best_item = None
        best_score = -np.inf

        for idx in cluster:
            item = results[idx]
            (
                rm, cm, r0, c0, r, c,
                h_min, l, h_mid, h0, h,
                htree, hgoal
            ) = item

            score = h_min - htree - hgoal

            if score > best_score:
                best_score = score
                best_item = item

        selected.append(best_item)

    # Remove duplicates (just in case)
    selected = list(dict.fromkeys(selected))

    return selected


