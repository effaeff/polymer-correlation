from cv2 import threshold
import numpy as np
from sklearn.metrics import adjusted_rand_score
import os
import pickle


def main():

    clusterings_dir = "results/clusterings"

    clustering = "Birch"
    key = "threshold"
    value = 0.1

    strain_clustering_path = (f"{clusterings_dir}/strain/{clustering}/"
                              f"{clustering}-{key}_{value}.pkl")

    point_clustering_path = (f"{clusterings_dir}/point/{clustering}/"
                             f"{clustering}-{key}_{value}.pkl")

    with open(strain_clustering_path, "rb") as strain_clustering_file:
        strain_clustering = pickle.load(strain_clustering_file)

    with open(point_clustering_path, "rb") as point_clustering_file:
        point_clustering = pickle.load(point_clustering_file)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.getcwd())
    from polycorr.parse_xml import parse_xml
    import polycorr.plotting as plotting
    main()
