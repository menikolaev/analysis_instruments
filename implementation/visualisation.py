import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
import numpy as np
from sklearn import metrics
from implementation.methods import get_data
import matplotlib.pyplot as pl

__author__ = 'mihailnikolaev'


def plot_cluster(data, plt):
    kmeans = KMeans(n_clusters=6, max_iter=1000)
    kmeans.fit(data)

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.scatter(data[:, 0], data[:, 1], c='g')
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())


def accurate_clusterisation_statistics():
    data, PARAMETERS = get_data()
    # # ----------- K-Means -----------
    # titles = []
    # for n_clusters in [x for x in xrange(2, 10)]:
    #     iterations = []
    #     for iters in [x for x in xrange(500, 5000, 500)]:
    #         cls = KMeans(n_clusters, max_iter=iters).fit(data)
    #         labels = cls.labels_
    #         iterations.append(metrics.silhouette_score(data, labels, metric='euclidean'))
    #     print "Num: %d" % n_clusters
    #     titles.append("Num: %d" % n_clusters)
    #     pl.plot([x for x in xrange(500, 5000, 500)], iterations)
    # # pl.legend([x for x in xrange(2, 10)], titles)
    # pl.show()

    # ------------ AgglomerativeClustering ------------
    titles = []
    iterations = []
    for n_clusters in [x for x in xrange(2, 10)]:
        cls = AgglomerativeClustering(n_clusters).fit(data)
        labels = cls.labels_
        iterations.append(metrics.silhouette_score(data, labels, metric='euclidean'))
        print "Num: %d" % n_clusters
        titles.append("Num: %d" % n_clusters)
    pl.plot([x for x in xrange(2, 10)], iterations)
    # pl.legend([x for x in xrange(2, 10)], titles)
    pl.show()


if __name__ == '__main__':
    accurate_clusterisation_statistics()