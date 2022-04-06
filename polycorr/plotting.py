"""Plot fibers and strains"""

from cProfile import label
from matplotlib import projections
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mtick
from random import randint
import os
from matplotlib.ticker import MaxNLocator

from collections import Counter

DARK2 = [(217 / 255, 95 / 255, 2 / 255),
         (27 / 255, 158 / 255, 119 / 255),
         (117 / 255, 112 / 255, 179 / 255),
         (231 / 255, 41 / 255, 138 / 255),
         (102 / 255, 166 / 255, 30 / 255),
         (230 / 255, 171 / 255, 2 / 255),
         (166 / 255, 118 / 255, 29 / 255),
         (102 / 255, 102 / 255, 102 / 255)]


def plot_fibers(points, segments):
    """Method to plot fibers based on points and segments"""

    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')

    lines = []
    colors = []
    for segment in segments:
        lines.append([points[idx] for idx in segment])
        colors.append(f'#{randint(0, 0xFFFFFF):06x}')

    collection = mplot3d.art3d.Line3DCollection(lines, colors=colors)
    axs.add_collection(collection)
    points = np.transpose(points)
    axs.set_xticks(
        np.arange
        (min(points[0]), max(points[0]), (max(points[0])-min(points[0]))//5))
    axs.set_yticks(
        np.arange(
            min(points[1]), max(points[1]),
            (max(points[1])-min(points[1]))//5))
    axs.set_zticks(
        np.arange(
            min(points[2]), max(points[2]),
            (max(points[2])-min(points[2]))//5))

    plt.show()


def plot_fibers_clustering(points, segments, clustering):
    """Method to plot fibers based on points and segments"""

    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')

    lines = []
    for segment in segments:
        lines.append([points[idx] for idx in segment])

    sm = ScalarMappable(cmap="gist_ncar")
    colors = sm.to_rgba(clustering)

    collection = mplot3d.art3d.Line3DCollection(
        lines, colors=colors)
    axs.add_collection(collection)
    points = np.transpose(points)
    axs.set_xticks(
        np.arange
        (min(points[0]), max(points[0]), (max(points[0])-min(points[0]))//5))
    axs.set_yticks(
        np.arange(
            min(points[1]), max(points[1]),
            (max(points[1])-min(points[1]))//5))
    axs.set_zticks(
        np.arange(
            min(points[2]), max(points[2]),
            (max(points[2])-min(points[2]))//5))

    plt.show()


def plot_most_common_clusters(clustering, metric):
    labels, values = zip(*Counter(clustering).most_common(10))

    indexes = np.arange(len(labels))
    width = 0.5

    plt.bar(indexes, values, width)
    plt.xticks(indexes, labels)
    plt.title(
        f"HDBSCAN with {metric} disctance. Num. Clusters:"
        f"{len(np.unique(clustering))}")
    plt.savefig(f"results/hdbscan_{metric}.png", dpi=300)


def plot_scores(scores, optimal_min_size):

    plt.plot(np.arange(2, len(scores) + 2), scores, 'o', ms=4, c="black")
    plt.plot(np.arange(2, len(scores) + 2), scores, c="black")
    plt.xticks(
        np.arange(10, len(scores) + 2, 10), np.arange(10, len(scores) + 2, 10))
    plt.ylabel("score")
    plt.xlabel("min size of clusters")
    plt.title(f"optimal min size of clusters: {optimal_min_size}")
    plt.grid(True)
    plt.savefig("results/scores.png", dpi=300)


def plot_num_clusters(nums, params, param_key, clustering_name):

    fig, ax = plt.subplots()
    ax.plot(nums, 'o', ms=4, c=DARK2[0])
    ax.plot(nums, c=DARK2[0])

    ax.set_xticks(np.arange(len(nums)))
    ax.set_xticklabels(params)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("number of clusters")
    ax.set_xlabel(param_key)
    plt.title(f"number of clusters for {clustering_name}")

    plt.grid(True)
    plt.savefig(
        f"results/plots/"
        f"{clustering_name}/{clustering_name}_num_clusters.png", dpi=300)
    plt.close()


def plot_num_clusters_vs_scores(scores, nums):

    plt.plot(np.arange(2, len(nums) + 2), nums, 'o', ms=4, c=DARK2[0])
    plt.plot(
        np.arange(2, len(nums) + 2), nums, c=DARK2[0], label="num of clusters")
    plt.plot(
        np.arange(2, len(scores) + 2), scores * 1000, 'o', ms=4, c=DARK2[1])
    plt.plot(
        np.arange(2, len(scores) + 2),
        scores * 1000, c=DARK2[1], label="score e+2")
    plt.xticks(
        np.arange(10, len(nums) + 2, 10), np.arange(10, len(nums) + 2, 10))
    plt.xlabel("min size of clusters")
    plt.title(f"number of clusters vs score")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/num_clusters_vs_scores.png", dpi=300)


def plot_metrics(scores, nums, variances):

    plt.plot(np.arange(2, len(nums) + 2), nums, 'o', ms=4, c=DARK2[0])
    plt.plot(
        np.arange(2, len(nums) + 2), nums, c=DARK2[0], label="num of clusters")
    plt.plot(
        np.arange(2, len(scores) + 2), scores * 1000, 'o', ms=4, c=DARK2[1])
    plt.plot(
        np.arange(2, len(scores) + 2),
        scores * 1000, c=DARK2[1], label="score * 2e+2")
    plt.plot(
        np.arange(2, len(variances) + 2),
        variances * 1e-9, 'o', ms=4, c=DARK2[2])
    plt.plot(
        np.arange(2, len(variances) + 2),
        variances * 1e-9, c=DARK2[2], label="variances * 1e-9")
    plt.xticks(
        np.arange(10, len(nums) + 2, 10), np.arange(10, len(nums) + 2, 10))
    plt.xlabel("min size of clusters")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/metrics.png", dpi=300)


def compare3d(strain, clustering, idxs=None, title=""):

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    ax1.scatter(*strain[:3], c=strain[-1])
    ax1.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False)
    ax1.set_title("strain", y=-0.1)

    if idxs is not None:
        strain = strain[:, idxs]
    ax2.scatter(*strain[:3], c=clustering, cmap="gist_ncar")
    ax2.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False)
    ax2.set_title("clustering", y=-0.1)

    plt.tight_layout()
    fig.suptitle(
        f"BIRCH clustering with threshold = {title}", y=0.9)
    plt.savefig(f"results/Birch/compare3d{title}.png", dpi=300)


def compare3d_fibers(strain, point_clustering, fibers, fiber_clustering,
                     points,  clustering_name, param_key, param_val,
                     idxs=None):

    sm = ScalarMappable(cmap="gist_ncar")
    clusters = np.unique(point_clustering)
    colors = sm.to_rgba(clusters)
    clusters = np.insert(clusters, 0, -1, axis=0)
    colors = np.insert(colors, 0, [0., 0., 0., 0.], axis=0)

    point_colors = [
        colors[np.where(clusters == cluster)][0]
        for cluster in point_clustering]
    fiber_colors = [
        colors[np.where(clusters == cluster)][0]
        for cluster in fiber_clustering]

    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    ax1.scatter(*strain[:3], c=strain[-1])
    ax1.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False)
    ax1.set_title("strain", y=-0.1)

    if idxs is not None:
        strain = strain[:, idxs]
    ax2.scatter(*strain[:3], c=point_colors)
    ax2.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False)
    ax2.set_title("point clustering", y=-0.1)

    collection = mplot3d.art3d.Line3DCollection(
        fibers, colors=fiber_colors)
    ax3.add_collection(collection)
    ax3.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False)
    ax3.set_xticks(ax2.get_xticks())
    ax3.set_yticks(ax2.get_yticks())
    ax3.set_zticks(ax2.get_zticks())
    ax3.set_title("fiber clustering", y=-0.1)
    plt.axis('auto')

    plt.tight_layout()
    fig.suptitle(
        f"{clustering_name} with {param_key} = {param_val}", y=0.9)
    path = f"results/plots/{clustering_name}"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(
        f"{path}"
        f"/{clustering_name}_-{param_key}_{param_val}"
        "_compare3d.png", dpi=300)


def compare3d_fibers_point_clustering(strain, point_clustering, fibers,
                                      fiber_clustering, points,
                                      clustering_name, param_key, param_val,
                                      idxs=None):

    sm = ScalarMappable(cmap="gist_ncar")
    clusters = np.unique(point_clustering)
    colors = sm.to_rgba(clusters)
    clusters = np.insert(clusters, 0, -1, axis=0)
    colors = np.insert(colors, 0, [0., 0., 0., 0.], axis=0)

    point_colors = [
        colors[np.where(clusters == cluster)][0]
        for cluster in point_clustering]
    fiber_colors = [
        colors[np.where(clusters == cluster)][0]
        for cluster in fiber_clustering]

    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    ax1.scatter(*strain[:3], c=strain[-1])
    ax1.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False)
    ax1.set_title("strain", y=-0.1)

    points = np.transpose(points)

    if idxs is not None:
        points = points[:, idxs]
    ax2.scatter(*points[:3], c=point_colors)
    ax2.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False)
    ax2.set_title("point clustering", y=-0.1)

    collection = mplot3d.art3d.Line3DCollection(
        fibers, colors=fiber_colors)

    ax3.add_collection(collection)
    # ax3.set_xlim([0, np.max(points[0])])
    # ax3.set_ylim([0, np.max(points[1])])
    # ax3.set_zlim([0, np.max(points[2])])
    ax3.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False)
    ax3.set_xlim([0, np.max(points[0])])
    ax3.set_ylim([0, np.max(points[1])])
    ax3.set_zlim([0, np.max(points[2])])
    # ax3.set_xticks(ax2.get_xticks())
    # ax3.set_yticks(ax2.get_yticks())
    # ax3.set_zticks(ax2.get_zticks())

    ax3.set_title("fiber clustering", y=-0.1)

    plt.tight_layout()
    fig.suptitle(
        f"{clustering_name} with {param_key} = {param_val}", y=0.9)
    path = f"results/plots/{clustering_name}"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(
        f"{path}"
        f"/{clustering_name}_-{param_key}_{param_val}"
        "_compare3d_point_clustering.png", dpi=300)


def plot_confidence(probabilities, params, param_key, clustering_name):

    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
    axs.boxplot(
        probabilities,
        boxprops=dict(color=DARK2[0]),
        medianprops=dict(color=DARK2[1], linewidth=2),
        meanprops=dict(marker="*", markerfacecolor="w", markeredgecolor='k'),
        showfliers=False,
        showmeans=True)
    axs.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    axs.set_xlabel(param_key)
    axs.set_ylabel("Confidence of fiber clustering")
    axs.set_xticklabels(params)

    fig.suptitle(f"Confidence of {clustering_name} fiber clustering")
    plt.savefig(
        f"results/plots/"
        f"{clustering_name}/{clustering_name}_confidence.png", dpi=300)


def plot_point_clustering(points, point_clustering, clustering_name, param_key,
                          param_val):
    sm = ScalarMappable(cmap="gist_ncar")
    clusters = np.unique(point_clustering)
    colors = sm.to_rgba(clusters)
    clusters = np.insert(clusters, 0, -1, axis=0)
    colors = np.insert(colors, 0, [0., 0., 0., 0.], axis=0)

    point_colors = [
        colors[np.where(clusters == cluster)][0]
        for cluster in point_clustering]

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")

    ax1.scatter(*points, c=point_colors)
    ax1.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False)
    ax1.set_title("point clustering", y=-0.1)

    plt.axis('auto')

    plt.tight_layout()
    fig.suptitle(
        f"{clustering_name} with {param_key} = {param_val}")
    path = f"results/plots/{clustering_name}"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(
        f"{path}"
        f"/{clustering_name}_-{param_key}_{param_val}"
        "_points.png", dpi=300)
