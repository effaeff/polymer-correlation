"""Script for parsing and plotting fibers"""

# from plot_utils import hist
import os
from tkinter import W
from typing import Collection
from matplotlib import projections
from matplotlib.cm import ScalarMappable

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import hdbscan
from sklearn import cluster
from sklearn import neighbors
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors, NeighborhoodComponentsAnalysis

import pickle
from collections import Counter
from tqdm import tqdm
import time

from sklearn.cluster import AgglomerativeClustering, Birch, OPTICS, DBSCAN
import active_semi_clustering.semi_supervised.pairwise_constraints as pw_c
# from active_semi_clustering.semi_supervised.pairwise_constraints import (
#     PCKMeans)
from pckmeans_empty import PCKMeans
from itertools import combinations
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


def save_clustering(clustering, filename):

    path = f"results/clusterings/{clustering.__class__.__name__}"

    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}/{filename}.pkl", "wb") as out_file:
        pickle.dump(clustering, out_file)


def eval_fiber_clustering(segments, clustering):

    most_common_clusters = []
    probabilities = []

    for segment in segments:

        clusters = clustering[segment[np.where(segment != -1)]]
        if clusters.size > 0:
            most_common_cluster = Counter(clusters).most_common(1)[0][0]
            probability = Counter(
                clusters).most_common(1)[0][1] / len(clusters)
            most_common_clusters.append(most_common_cluster)
            probabilities.append(probability)
        else:
            most_common_clusters.append(-1)
            probabilities.append(0.0)

    return most_common_clusters, probabilities


def plot_results(strain, segments, segments_strain, points, clustering,
                 clustering_name, param_key, param_val):

    line_scaler = MinMaxScaler()

    lines = []
    for segment in segments:
        line = np.vstack([points[idx] for idx in segment])
        line_scaler.partial_fit(line)
        lines.append(line)

    scales_lines = []
    for line in lines:
        line = line_scaler.transform(line)
        scales_lines.append(line)

    lines = scales_lines

    most_common_clusters, prob = eval_fiber_clustering(
        segments_strain, clustering)

    strain = np.transpose(strain)

    plotting.compare3d_fibers(
        strain,
        clustering,
        lines,
        most_common_clusters,
        points,
        clustering_name,
        param_key,
        param_val
    )


def main():
    """Main method"""

    specimen = 'TUD_1_2'
    ulimit = 0.0255
    llimit = 0.02
    ubound = 6000
    lbound = 1000

    link = "all"  # ['all', 'start-end']

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

    must_link = []
    if link == 'start-end':
        for segment in segments:
            must_link.append((segment[0], segment[-1]))
    elif link == 'all':
        for segment in segments:
            for pair in combinations(segment, 2):
                must_link.append(pair)

    must_link = np.unique(must_link, axis=0)

    print("scale strain")
    scaler = MinMaxScaler()
    points = scaler.fit_transform(points)

    clusterings = [
        # PCKMeans()
        Birch(n_clusters=None),
        # OPTICS(min_samples=8, n_jobs=-1),  # min_samples = 2*dim
        # DBSCAN(min_samples=8, n_jobs=-1),  # (Sander et al., 1998)
        # hdbscan.HDBSCAN(core_dist_n_jobs=-1)
    ]

    params = [
        # {"n_clusters": [190625]}
        {"threshold": np.arange(0.05, 0.8, 0.01)},
        # {"max_eps": np.arange(0.25, 1.1, 0.25)},
        # {"eps": np.arange(0.05, 0.19, 0.05)},
        # {"min_cluster_size": np.arange(5, 51, 5)}
    ]

    sys.setrecursionlimit(10000)

    pbar_clustering = tqdm(clusterings)
    for idx, clustering in enumerate(pbar_clustering):
        probabilities = []
        num_clusters = []
        param = params[idx]

        pbar_clustering.set_postfix_str(
            f"{clustering.__class__.__name__}")

        if param != {}:
            key = list(param)[0]

            pbar_params = tqdm(
                np.round(param[key], 2)
                if clustering.__class__.__name__ != "HDBSCAN"
                else [int(k) for k in param[key]],
                leave=False)
            for value in pbar_params:
                if hasattr(clustering, key):
                    setattr(
                        clustering,
                        key,
                        value
                    )
                pbar_params.set_postfix_str(
                    f"{key} = {getattr(clustering, key)}")

                clustering_path = (f"results/clusterings/"
                                   f"{clustering.__class__.__name__}/"
                                   f"{clustering.__class__.__name__}-"
                                   f"{key}_{value}.pkl")

                if os.path.isfile(clustering_path):

                    with open(clustering_path, "rb") as clustering_file:
                        clustering = pickle.load(clustering_file)
                    print("loaded clustering file")

                else:

                    if clustering.__class__.__name__ in dir(pw_c):
                        print("cluster semi-supervised")
                        clustering.fit(points, ml=must_link)
                    else:
                        clustering.fit(points)
                    save_clustering(
                        clustering,
                        f"{clustering.__class__.__name__}"
                        f"-{key}_{value}"
                    )

            #     _, prob = eval_fiber_clustering(
            #         points, clustering.labels_
            #     )

            #     probabilities.append(prob)
            #     num_clusters.append(len(np.unique(clustering.labels_)))

            #     plot_results(strain, segments, points, points,
            #                  clustering.labels_, clustering.__class__.__name__,
            #                  key, value)

            # plotting.plot_confidence(
            #     probabilities,
            #     np.round(param[key], 2),
            #     key,
            #     clustering.__class__.__name__
            # )

            plotting.plot_num_clusters(
                num_clusters,
                np.round(param[key], 2),
                key,
                clustering.__class__.__name__
            )

        else:

            clustering_path = (f"results/clusterings/"
                               f"{clustering.__class__.__name__}/"
                               f"{clustering.__class__.__name__}.pkl")

            if os.path.isfile(clustering_path):

                with open(clustering_path, "rb") as clustering_file:
                    clustering = pickle.load(clustering_file)
                print("loaded clustering file")

            else:

                clustering.fit(strain)
                save_clustering(
                    clustering,
                    f"{clustering.__class__.__name__}"
                )

            _, prob = eval_fiber_clustering(
                points, clustering.labels_
            )

            num_clusters.append(len(np.unique(clustering.labels_)))

            plot_results(strain, segments, points, points,
                         clustering.labels_,
                         clustering.__class__.__name__, "", "")

            plotting.plot_confidence(
                prob,
                [""],
                "",
                clustering.__class__.__name__
            )

            plotting.plot_num_clusters(
                num_clusters, [""], "", clustering.__class__.__name__)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.getcwd())
    from polycorr.parse_xml import parse_xml
    import polycorr.plotting as plotting
    main()
