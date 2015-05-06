from __future__ import division
import time
import itertools
from pandas import DataFrame
from sklearn import metrics, cluster, linear_model
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.mixture import DPGMM, GMM, VBGMM
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from clusterisation_module.utils import get_data
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'mihailnikolaev'


def compare_scaler_and_non_scaler():
    ids, data, params = get_data()
    scl = StandardScaler()
    scaled_data = scl.fit_transform(data)
    for k in range(5, 6):
        cls = DPGMM(n_components=k, alpha=100, n_iter=1000)
        cls.fit(data)
        print "DPGMM", cls.aic(data), cls.bic(data)
        plt.subplot(121)
        plot(cls, data, cls.predict(data), data.columns)
        cls = DPGMM(n_components=k, alpha=100, n_iter=1000)
        cls.fit(scaled_data)
        print "DPGMM scaled", cls.aic(scaled_data), cls.bic(scaled_data)
        plt.subplot(122)
        plot(cls, scl.inverse_transform(scaled_data), cls.predict(scaled_data), data.columns)
        plt.show()


def get_optimal_parameters_DPGMM():
    ids, data, params = get_data()
    for a in np.arange(1, 100, 10):
        for n_iter in np.arange(100, 1000, 100):
            cls = DPGMM(n_components=5, alpha=a, n_iter=n_iter)
            cls.fit(data)
            print "DPGMM, a = {}, n_iter = {}, aic = {}, bic = {}".format(a, n_iter, cls.aic(data), cls.bic(data))


def get_clusters_for_zero_payments():
    ids, data, params = get_data()
    data = data[(data.payments_sum > 0) & (data.number_of_quests > 0)]
    cls = DPGMM(n_components=10, alpha=5, n_iter=100)
    cls.fit(data)
    plot(cls, data, cls.predict(data), data.columns)
    plt.show()


def plot(cls, data, predicted, columns):
    data = DataFrame(data, columns=columns)
    print set(predicted)
    means = []
    stds = []
    for i in set(predicted):
        d = DataFrame(data[predicted==i])
        # print "Cluster: ", i, "Count: ", len(d)
        print d.describe()
        means.append(d.payments_sum.mean(axis=0))
        stds.append(d.payments_sum.std(axis=0))
        # print DataFrame({'means': d.mean(axis=0), 'std': d.std(axis=0)})
    print "l2 norm: {}".format(calc_l2_norm(np.array(means), data.payments_sum.mean(axis=0)))
    print "l2 norm: {}".format(calc_l2_norm(np.array(stds), data.payments_sum.std(axis=0)))
    data = np.array(data)
    pca = PCA(2)
    pdata = pca.fit(data).transform(data)
    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
    for i, k in enumerate(set(predicted)):
        if not np.any(predicted == i):
            continue
        plt.scatter(pdata[predicted == k, 0], pdata[predicted == k, 1], .8, color=color_iter.next())

    plt.xlim()
    plt.ylim()
    plt.title("DPGMM")
    plt.xticks(())
    plt.yticks(())


def calc_l2_norm(probs, glob_prob):
    return np.sqrt(np.linalg.norm(probs - glob_prob) / probs.shape[0])


def hier_clustering():
    ids, data, params = get_data()
    data = data[(data.payments_sum > 0)]
    X = data
    # Plot the distances
    n_clusters = 8

    # Plot clustering results
    for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
        model = AgglomerativeClustering(n_clusters=n_clusters,
                                        linkage="average", affinity=metric)
        model.fit(X)
        plt.figure()
        plt.axes([0, 0, 1, 1])
        for l, c in zip(np.arange(model.n_clusters), 'rgbk'):
            plt.plot(X[model.labels_ == l].T, c=c, alpha=.5)
        plt.axis('tight')
        plt.axis('off')
        plt.suptitle("AgglomerativeClustering(affinity=%s)" % metric, size=20)

    for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
        model = AgglomerativeClustering(n_clusters=n_clusters,
                                        linkage="average", affinity=metric)
        model.fit(X)
        plt.figure()
        plt.axes([0, 0, 1, 1])
        plot(model, data, model.labels_, data.columns)
        plt.axis('tight')
        plt.axis('off')
        plt.suptitle("AgglomerativeClustering(affinity=%s)" % metric, size=20)
    # plot(model, data, model.labels_, data.columns)
    plt.show()


def test_methods_on_pca_data():
    ids, data, params = get_data()
    data = data.drop(['viral', 'organic', 'commercial'], axis=1)
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    clustering_names = [
        'MiniBatchKMeans', 'MeanShift', 'DBSCAN', 'Birch', 'GMM', 'DPGMM', 'VBGMM']

    plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1

    datasets = [data]
    for i_dataset, dataset in enumerate(datasets):
        X = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
        sctl = PCA(2).fit(X)
        print sctl.explained_variance_, sctl.explained_variance_ratio_, sctl.noise_variance_

        X = sctl.transform(X)
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # create clustering estimators
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=5)
        # ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
        #                                        connectivity=connectivity)
        # spectral = cluster.SpectralClustering(n_clusters=2,
        #                                       eigen_solver='arpack',
        #                                       affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=.2)
        gmm = GMM(n_components=5)
        dpgmm = DPGMM(n_components=10, alpha=5, n_iter=50)
        vbgmm = VBGMM(n_components=10, alpha=5, n_iter=50)
        # affinity_propagation = cluster.AffinityPropagation(damping=.9,
        #                                                    preference=-200)

        # average_linkage = cluster.AgglomerativeClustering(
        #     linkage="average", affinity="cityblock", n_clusters=2,
        #     connectivity=connectivity)

        birch = cluster.Birch(n_clusters=5)
        clustering_algorithms = [
            two_means, ms, gmm, birch,
            dpgmm, vbgmm, dbscan]

        for name, algorithm in zip(clustering_names, clustering_algorithms):
            # predict cluster memberships
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            # plot
            plt.subplot(1, len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                center_colors = colors[:len(centers)]
                plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1
        X = sctl.inverse_transform(X)

    plt.show()


def ridge_model():
    ids, data, params = get_data()
    data = data.drop(['viral', 'organic', 'commercial'], axis=1)
    X = np.array(data)
    X = StandardScaler().fit_transform(X)
    cls = DPGMM(n_components=10, alpha=4)
    cls.fit(X)
    y = cls.predict(X)
    ###############################################################################
    # Compute paths

    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas)
    clf = linear_model.Ridge(fit_intercept=False)

    coefs = []
    for a in alphas:
        clf.set_params(alpha=a)
        clf.fit(X, y)
        coefs.append(clf.coef_)

    ###############################################################################
    # Display results

    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()


if __name__ == '__main__':
    # compare_scaler_and_non_scaler()
    # # get_optimal_parameters_DPGMM()
    # get_clusters_for_zero_payments()
    # hier_clustering()
    # test_methods_on_pca_data()
    ridge_model()