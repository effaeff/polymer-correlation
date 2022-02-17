"""Script for parsing and plotting fibers"""

# from plot_utils import hist
import os
from typing import Collection
from matplotlib import projections
from matplotlib.cm import ScalarMappable

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import hdbscan
from sklearn import cluster
from sklearn.manifold import TSNE

import pickle
from collections import Counter
from tqdm import tqdm
import time

from sklearn.cluster import AgglomerativeClustering, Birch, OPTICS, DBSCAN
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

        clusters = clustering[segment]
        most_common_cluster = Counter(clusters).most_common(1)[0][0]
        probability = Counter(clusters).most_common(1)[0][1] / len(clusters)
        most_common_clusters.append(most_common_cluster)
        probabilities.append(probability)

    return most_common_clusters, probabilities


def main():
    """Main method"""

    specimen = 'TUD_1_2'
    ulimit = 0.0255
    llimit = 0.02
    ubound = 6000
    lbound = 1000

    # 1. Load Data
    # 2. Cluster Data with different
    #       a) Clustering algorithms
    #       b) Hyperparameters
    # 3. Save clusterings
    # 4. Evaluate fiber clustering
    # 5. Plot
    #       a) 3D comparison for points and fibers
    #       b) confidence of fiber clusterings
    #       c) Num clusters

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

    print("scale strain")
    scaler = MinMaxScaler()
    strain = scaler.fit_transform(strain)

    clusterings = [
        # Birch(n_clusters=None),
        # OPTICS(min_samples=8, n_jobs=-1),  # min_samples = 2*dim
        # DBSCAN(min_samples=8, n_jobs=-1),  # (Sander et al., 1998)
        hdbscan.HDBSCAN(core_dist_n_jobs=-1)
    ]

    params = [
        # {"threshold": np.arange(0.05, 0.26, 0.05)},
        # {},
        # {"eps": np.arange(0.05, 0.19, 0.05)},
        {"min_cluster_size": np.arange(5, 51, 5)}
    ]

    sys.setrecursionlimit(10000)

    pbar_clustering = tqdm(clusterings)
    for idx, clustering in enumerate(pbar_clustering):
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
                clustering.fit(strain)
                save_clustering(
                    clustering,
                    f"{clustering.__class__.__name__}"
                    f"-{key}_{value}"
                )
        else:
            clustering.fit(strain)
            save_clustering(
                clustering,
                f"{clustering.__class__.__name__}"
            )

    quit()

    # --- Plotting --- #

    pbar_clustering = tqdm(clusterings)
    for idx, clustering in enumerate(pbar_clustering):
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

    with open(
     f"results/1try/Birch/Birch_0.1_skaled.pkl", "rb") as clusterer_file:
        clusterer = pickle.load(clusterer_file)

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
        segments, clusterer.labels_)

    strain = np.transpose(strain)

    plotting.compare3d_fibers(
        strain,
        clusterer.labels_,
        lines,
        most_common_clusters,
        points,
        "BIRCH",
        "threshold",
        0.1
    )

    quit()

    quit()
    thresholds = np.round(np.linspace(0.05, 0.2, 4), 2)
    probablities = []
    for threshold in tqdm(thresholds):

        with open(f"results/Birch/Birch_{threshold}_skaled.pkl", "rb") as clusterer_file:
            clusterer = pickle.load(clusterer_file)

        most_common_clusters, prob = eval_fiber_clustering(
            segments, clusterer.labels_)
        probablities.append(prob)

    plotting.plot_confidence(probablities, thresholds)
    quit()

    print(max(probablities))
    print(min(probablities))
    print(np.mean(probablities))
    quit()

    # for i in np.unique(most_common_clusters):
    #     idxs = np.where(np.array(most_common_clusters) == i)[0]
    #     segments_c = np.array(segments)[np.array(idxs)]
    #     clustering_c = np.array(most_common_clusters)[idxs]
    plotting.plot_fibers_clustering(
        points,
        segments,
        most_common_clusters
    )

    quit()
    
    for cluster in clusterer.labels_:

        idxs = np.where(clusterer.labels_ == cluster)[0]
        print(idxs)

        plotting.plot_fibers_clustering(
            points[idxs],
            segments,
            clusterer.labels_[idxs]
        )
    quit()

    print(len(points))
    print(len(segments))
    quit()
    
    # with open(f"points.pkl", "wb") as out_file:
    #     pickle.dump(points, out_file)
    # with open(f"segments.pkl", "wb") as out_file:
    #     pickle.dump(segments, out_file)
    quit()
    plotting.plot_fibers(points, segments)
    quit()




    # projection = TSNE(verbose=1, n_jobs=-1).fit_transform(strain)
    # plt.scatter(*projection.T)
    # plt.show()
    # quit()
    # Random sub sample for plot
    # print("start random sampling...")
    # s_samples = 100
    # indices = np.arange(0, len(strain))
    # rand_ids = np.random.choice(indices, s_samples, replace=False)
    # strain = strain[rand_ids]

    #metrics = list(hdbscan.dist_metrics.METRIC_MAPPING.keys())

    # if not os.path.isfile("clusterer.pkl"):
    #     print("start clustering...")

    scores = np.empty(49)
    num_clusters = np.empty(49)
    variances = np.empty(49)

    # loop = tqdm(range(2, 51))
    # for i, min_size in enumerate(loop):


    #     loop.set_postfix({"min_size": min_size})



    #for threshold in tqdm(np.arange(0.05, 0.2, 0.05)):
    # clusterer = Birch(threshold=threshold, n_clusters=None)
    # clusterer.fit(strain)
    
    with open(f"results/Birch/Birch_{threshold}_skaled.pkl", "rb") as clusterer_file:
        clusterer = pickle.load(clusterer_file)
        plotting.compare3d(
            np.transpose(strain), clusterer.labels_, title=threshold)

    

    # with open(f"results/Birch/Birch_{threshold}_skaled.pkl", "wb") as out_file:
    #     pickle.dump(clusterer, out_file)


    # idxs = np.where(strain[:, -1] > 0.99)
    # strain = strain[idxs]


    quit()

    min_size = 7

    clusterer_path = f"results/clusterings/hdbscan_{min_size}.pkl"
    with open(clusterer_path, "rb") as clusterer_file:
        clusterer = pickle.load(clusterer_file)
    fig = plt.figure()
    axs = fig.add_subplot(projection='3d')
    print(Counter(clusterer.labels_).most_common(10))

    idxs = np.where((clusterer.labels_ != -1) & (clusterer.labels_ != 46))[0]
    strain = np.transpose(strain[idxs])
    plotting.compare3d(strain, clusterer.labels_[idxs], 7)
    quit()

        # scores[i] = len(
        #     np.where(clusterer.probabilities_ < 0.05)[0]) / len(
        #         strain
        #     )
        # num_clusters[i] = len(
        #     np.unique(clusterer.labels_)
        # )

        # #labels, values = zip(*Counter(clusterer.labels_))
        # c = Counter(clusterer.labels_)
        # variances[i] = np.var(list(c.values()))

        # plotting.compare3d(
        #     np.transpose(strain), clusterer.labels_, min_clusters=min_size)
    quit()

    #plot_metrics(scores, num_clusters, variances)
    #plot_num_clusters_vs_scores(scores, num_clusters)
    #plot_num_clusters(num_clusters)
    #plot_scores(scores, list(range(2, 51))[np.argmin(scores)])

        # with open(f"results/hdbscan_{min_size}.pkl", "wb") as out_file:
        #     pickle.dump(clusterer, out_file)
    # else:
    #     with open("clusterer.pkl", "rb") as clusterer_file:
    #         clusterer = pickle.load(clusterer_file)

    quit()

    # print(Counter(clusterer.labels_).most_common(10))
    # quit()
    # labels, values = zip(*Counter(clusterer.labels_).most_common(10))

    # indexes = np.arange(len(labels))
    # width = 1

    # plt.bar(indexes, values, width)
    # plt.xticks(indexes + width * 0.5, labels)
    # plt.show()
    # quit()

    # idxs = [np.where(clusterer.labels_ == i) for i in range(300)]
    # idxs = np.array(np.hstack(idxs)).flatten()

    idxs = np.where((clusterer.labels_ != -1) & (clusterer.labels_ != 238))[0]
    idxs = np.where(clusterer.labels_ == -1)[0]
    fig = plt.figure()
    axs = fig.add_subplot(projection='3d')

    org_strain = strain
    #org_strain = np.transpose(org_strain)
    #plot2 = axs.scatter(org_strain[0], org_strain[1], org_strain[2], c=org_strain[3])

    strain = strain[idxs]
    strain = np.transpose(strain)

    plot = axs.scatter(strain[0], strain[1], strain[2], c=org_strain[idxs, 3])
    axs.view_init(elev=0, azim=-90)
    print(np.shape(strain))
    print(np.shape(org_strain))
    plt.show()
    quit()

    strain = np.transpose(strain)

    # hist(strain[3])
    print("start plotting...")
    fig = plt.figure()
    axs = fig.add_subplot(projection='3d')
    plot = axs.scatter(strain[0], strain[1], strain[2], c=clusterer.labels_, s=5)
    fig.colorbar(plot)
    plt.show()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.getcwd())
    from polycorr.parse_xml import parse_xml
    import polycorr.plotting as plotting
    main()
