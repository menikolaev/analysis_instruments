from __future__ import division
import itertools
from pandas import DataFrame
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.mixture import DPGMM, GMM, VBGMM
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
    cls = DPGMM(n_components=5, alpha=10, n_iter=100)
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
    for i, (mean, covar, color) in enumerate(
            zip(cls.means_, cls._get_covars(), color_iter)):
        if not np.any(predicted == i):
            continue
        plt.scatter(pdata[predicted == i, 0], pdata[predicted == i, 1], .8, color=color)

    plt.xlim()
    plt.ylim()
    plt.title("DPGMM")
    plt.xticks(())
    plt.yticks(())


def calc_l2_norm(probs, glob_prob):
    return np.sqrt(np.linalg.norm(probs - glob_prob) / probs.shape[0])


if __name__ == '__main__':
    compare_scaler_and_non_scaler()
    # # get_optimal_parameters_DPGMM()
    # get_clusters_for_zero_payments()