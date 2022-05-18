import tabnanny
import numpy as np
import os
import pickle
from collections import Counter
from tabulate import tabulate


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

    point_clustering_files = os.listdir("results/clusterings/point/Birch")
    entity_scores = []
    for file in point_clustering_files:

        if file.endswith(".pkl"):
            with open(f"results/clusterings/point/Birch/{file}", "rb") as cl_f:
                birch_clustering = pickle.load(cl_f)
                num_clusters = len(np.unique(birch_clustering.labels_))

            constrained_clustering_file = (f"results/clusterings/PCKMeans/"
                                           f"PCKMeans-n_clusters_"
                                           f"{num_clusters}.pkl")
            if os.path.exists(constrained_clustering_file):
                with open(constrained_clustering_file, "rb") as cl_f:
                    constrained_clustering = pickle.load(cl_f)

                all_same_cluster_constrained = [
                    0 if len(
                        np.unique(
                            constrained_clustering.labels_[segment])) == 1
                    else 1 for segment in segments
                ]

                all_same_cluster_birch = [
                    0 if len(
                        np.unique(
                            birch_clustering.labels_[segment])) == 1
                    else 1 for segment in segments
                ]

                entity_scores.append(
                    [
                        file.split("_")[-1][:-4],
                        num_clusters,
                        (1 - sum(all_same_cluster_constrained) /
                         len(all_same_cluster_constrained)),
                        (1 - sum(all_same_cluster_birch) /
                         len(all_same_cluster_birch))
                    ]

                )

    print(
        tabulate(
            [[score[0], f"{score[2]:.2%}", f"{score[3]:.2%}"]
             for score in entity_scores],
            headers=("Threshold", "PCKMeans", "Birch")
        )
    )


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.getcwd())
    from polycorr.parse_xml import parse_xml
    import polycorr.plotting as plotting
    main()
