from tokenize import tabsize
import numpy as np
from sklearn.metrics import (adjusted_rand_score, fowlkes_mallows_score,
                             v_measure_score)
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from tabulate import tabulate
import matplotlib.pyplot as plt


def main():
    """Main method"""

    specimen = 'TUD_1_2'
    ulimit = 0.0255
    llimit = 0.02
    ubound = 6000
    lbound = 1000

    # Load segments and points
    if os.path.isfile("points.pkl") and os.path.isfile("segments.pkl"):
        print("loading segments and points")
        with open(f"points.pkl", "rb") as points_file:
            points = pickle.load(points_file)
        with open(f"segments.pkl", "rb") as segments_file:
            segments = pickle.load(segments_file)
    else:
        print("parsing points and segments")
        points, segments = parse_xml(
            f'data/fiber/{specimen}.xml', ubound, lbound)
        with open(f"points.pkl", "wb") as points_file:
            pickle.dump(points, points_file)
        with open(f"segments.pkl", "wb") as segments_file:
            pickle.dump(segments, segments_file)

    # Load strain
    if os.path.isfile("strain.npy"):
        print("loading strain")
        strain = np.load("strain.npy")
    else:
        print("parsing strain")
        strain = np.genfromtxt(
            f'data/strain/{specimen}/Ezz03.dat',
            delimiter=' ',
            skip_header=4,
            usecols=(0, 1, 2, 3)
        )
        np.save("strain.npy", strain)

    # Remove specific entries
    print("removing specific entries")
    strain = strain[np.where(strain[:, 2] >= lbound // 1000)]
    strain = strain[np.where(strain[:, 2] <= ubound // 1000)]
    strain = strain[np.where(strain[:, 3] < ulimit)]
    strain = strain[np.where(strain[:, 3] > llimit)]

    strain_scaler = MinMaxScaler()
    points_scaler = MinMaxScaler()

    points_s = points_scaler.fit_transform(points)
    strain_s = strain_scaler.fit_transform(strain)

    # Get nearest points
    strain_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    strain_nn.fit(strain_s[:, :3])

    strain_neighbors = strain_nn.kneighbors(points_s, return_distance=True)[1]

    clusterings_dir = "results/clusterings"

    clustering = "Birch"
    key = "threshold"
    values = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25]

    data = []

    points_s = np.transpose(points_s)

    for value in values:

        scores = []

        strain_clustering_path = (f"{clusterings_dir}/strain/{clustering}/"
                                  f"{clustering}-{key}_{value}.pkl")

        point_clustering_path = (f"{clusterings_dir}/point/{clustering}/"
                                 f"{clustering}-{key}_0.1.pkl")

        with open(strain_clustering_path, "rb") as strain_clustering_file:
            strain_clustering = pickle.load(strain_clustering_file)

        with open(point_clustering_path, "rb") as point_clustering_file:
            point_clustering = pickle.load(point_clustering_file)

        # Select clustering of strains which correspond to points
        point_clusters = point_clustering.labels_
        strain_clusters = strain_clustering.labels_[strain_neighbors][:, 0]

        # plotting.compare3d_clustering(
        #     point_clusters,
        #     strain_clusters,
        #     points_s,
        #     clustering,
        #     key,
        #     value
        # )

        scores.append(value)

        scores.append(adjusted_rand_score(
            point_clusters,
            strain_clusters
        ))
        scores.append(fowlkes_mallows_score(
            point_clusters,
            strain_clusters
        ))
        scores.append(v_measure_score(
            point_clusters,
            strain_clusters
        ))

        data.append(scores)

    headers = [
        "threshold",
        "adjusted_rand_score",
        "fowlkes_mallows_score",
        "v_measure_score"
    ]

    with open("results/similarity_scores.txt", "w+") as score_file:
        score_file.write("Comparison between strain clustering over"
                         "different birch-thresholds \nand point clustering "
                         "with birch-threshold 0.1\n\n")
        score_file.write(tabulate(data, headers=headers))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.getcwd())
    from polycorr.parse_xml import parse_xml
    import polycorr.plotting as plotting
    main()
