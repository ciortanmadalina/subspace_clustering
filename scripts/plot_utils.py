import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score


def silhouette_plot(data, truth, numClusters=None, cmap='jet', ax=None):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        truth {[type]} -- [description]

    Keyword Arguments:
        numClusters {[type]} -- [description] (default: {None})
        cmap {str} -- [description] (default: {'jet'})
        ax {[type]} -- [description] (default: {None})
    """
    plt.figure()
    silSamples = silhouette_samples(data, truth)
    sscore = silhouette_score(data, truth)

    if numClusters is None:
        numClusters = len(np.unique(truth))
    if ax is None:
        ax = plt.gca()

    clusterSpace = 10
    curY = 5
    for i in range(numClusters):
        color = cm.get_cmap(cmap)(float(i) / numClusters)

        oneClusterValues = silSamples[truth == i]
        clusterSize = oneClusterValues.shape[0]

        oneClusterValues = sorted(silSamples[truth == i])

        maxY = curY + clusterSize
        ax.fill_between(np.arange(curY, maxY),
                        0,
                        oneClusterValues,
                        facecolor=color,
                        edgecolor=color)
        ax.set_title(f'Silhouette:{sscore:.2f}')
        ax.axhline(sscore, linestyle='--', c='r')
        #         ax.text(-0.05, curY + 0.5 * clusterSize, str(i))
        curY = maxY + clusterSpace
