import pandas as pd
import numpy as np
import sys
from scipy.spatial import KDTree

def cluster_and_extract(df, ranges, radius=100):
    """
    Cluster rows by spatial proximity (midx, midy) within `radius`,
    then for each cluster and for each length range, select the highest row.
    """

    coords = df[['midx', 'midy']].values
    tree = KDTree(coords)

    # ----- Step 1: Find clusters -----
    n = len(df)
    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        cluster = []
        queue = [i]
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

    # ----- Step 2: For each cluster and each range, pick the highest -----
    result_rows = []

    for cluster in clusters:
        sub = df.iloc[cluster]

        for _, r in ranges.iterrows():
            lo = r["min_hl_length"]
            hi = r["max_hl_length"]

            # filter cluster rows that fall inside this length range
            sub_range = sub[(sub["length"] >= lo) & (sub["length"] < hi)]

            if len(sub_range) == 0:
                continue  # no row in this cluster fits in the range

            # pick highest
            highest = sub_range.loc[sub_range["height"].idxmax()]
            result_rows.append(highest)

    # ----- Step 3: Build result dataframe -----
    result_df = pd.DataFrame(result_rows).drop_duplicates().reset_index(drop=True)
    return result_df



if __name__ == "__main__":
    file_in = sys.argv[1]
    file_out = sys.argv[2]

    ranges = pd.DataFrame([
        {"min_hl_length": 0,   "max_hl_length": 50,  "H": 10, "pxsize": 5},
        {"min_hl_length": 50,  "max_hl_length": 150, "H": 15, "pxsize": 7},
        {"min_hl_length": 150, "max_hl_length": 250, "H": 20, "pxsize": 10},
        {"min_hl_length": 200, "max_hl_length": 350, "H": 25, "pxsize": 10},
        {"min_hl_length": 350, "max_hl_length": 500, "H": 30, "pxsize": 15}
    ])

    df = pd.read_csv(file_in, sep=" ")
    clustered_df = cluster_and_extract(df, ranges, radius=50)
    clustered_df.to_csv(file_out, sep=' ')
