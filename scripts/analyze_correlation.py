"""Script for parsing and plotting fibers"""

from plot_utils import hist

import numpy as np
from polycorr.parse_xml import parse_xml
from polycorr.plotting import plot_fibers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

def main():
    """Main method"""
    specimen = 'TUD_1_2'
    ulimit = 0.0255
    llimit = 0.02
    ubound = 6000
    lbound = 1000

    # points, segments = parse_xml(f'data/fiber/{specimen}.xml', ubound, lbound)
    # plot_fibers(points, segments)

    strain = np.genfromtxt(
        f'data/strain/{specimen}/Ezz03.dat',
        delimiter=' ',
        skip_header=4,
        usecols=(0, 1, 2, 3)
    )

    # Remove specific entries
    strain = strain[np.where(strain[:, 2] >= lbound // 1000)]
    strain = strain[np.where(strain[:, 2] <= ubound // 1000)]
    strain = strain[np.where(strain[:, 3] < ulimit)]
    strain = strain[np.where(strain[:, 3] > llimit)]
    print(strain.shape)

    # Random sub sample for plot
    # s_samples = 100000
    # indices = np.arange(0, len(strain))
    # rand_ids= np.random.choice(indices, s_samples, replace=False)
    # strain = strain[rand_ids]

    # clustering = AgglomerativeClustering().fit(strain)

    strain = np.transpose(strain)

    # hist(strain[3])

    fig = plt.figure()
    axs = fig.add_subplot(projection='3d')
    plot = axs.scatter(strain[0], strain[1], strain[2], c=strain[3], s=5)
    fig.colorbar(plot)
    plt.show()

if __name__ == '__main__':
    main()
