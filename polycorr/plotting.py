"""Plot fibers and strains"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from random import randint

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
    axs.set_xticks(np.arange(min(points[0]), max(points[0]), (max(points[0])-min(points[0]))//5))
    axs.set_yticks(np.arange(min(points[1]), max(points[1]), (max(points[1])-min(points[1]))//5))
    axs.set_zticks(np.arange(min(points[2]), max(points[2]), (max(points[2])-min(points[2]))//5))

    plt.show()
