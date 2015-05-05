# coding=utf-8
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from pandas import DataFrame
from pandas.util.testing import Series
from sklearn import manifold, metrics
from sklearn.cluster import MeanShift, KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, \
    SpectralClustering, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.mixture import GMM, VBGMM, DPGMM
from sklearn.preprocessing import StandardScaler, scale
from clusterisation_module.test import calc_l2_norm
from utils import get_data, correlation_matrix, get_params_matrix

__author__ = 'mihailnikolaev'


STRICT_CLUSTERING_METHODS = [MeanShift(min_bin_freq=3, bin_seeding=True), KMeans(8), MiniBatchKMeans(8)]
# DBSCAN(), AgglomerativeClustering(8)

SOFT_CLUSTERING_METHODS = [GMM(n_components=8, n_iter=200), DPGMM(n_components=8, alpha=100., n_iter=1000),
                           VBGMM(n_components=8, alpha=1000, n_iter=100)]


class Clustering(object):
    """
        Non-probability clustering class
        Examples:
            AgglomerativeClustering
            DBSCAN
            Birch
            K-Means
            MiniBatchKMeans
            AffinityPropagation

    """

    def __init__(self, data, raw_rules=None, query='', **kwargs):
        """
            data - данные из файлов
            raw_rules - правила для данных (если нет query)
            query - строка, которая содержит в себе фильтр для данных
            kwargs - словарь для признаков данных
        """

        self.data = data
        for key, value in kwargs.iteritems():
            setattr(self, key, self.data['%s' % key] >= value if key in self.data else Series())
        self.cleansed_data = self.get_cleansed_data(raw_rules, query)
        self.classifier = None

    def clean_rules(self, raw_rules):
        """
            Обработать данные согласно правилам
            возможные значения:
                -1: инвертировать правило (>= в <)
                любое неотрицательное число: оставить правило неизменным
        """
        parameters = self.data.columns
        query = ''
        if not raw_rules:
            return

        for i in xrange(len(raw_rules)):
            if raw_rules[i] and hasattr(self, parameters[i]):
                if raw_rules[i] == -1:
                    setattr(self, parameters[i], ~getattr(self, parameters[i]))
                elif raw_rules[i] < -1:
                    raise AttributeError('Invalid attribute %s' % str(raw_rules[i]))

                if not getattr(getattr(self, parameters[i]), 'empty'):
                    query = ' & '.join([query, '@self.%s' % parameters[i]])
        return query[3:] if query else query

    def get_cleansed_data(self, raw_rules, query=''):
        """
            Применить правила на данные
            Если есть query просто применить этот фильтр над данными
            Иначе просмотреть правила и создать фильтр на их основе
        """

        if query:
            return self.data.query(query)
        else:
            query = self.clean_rules(raw_rules)
            cleansed_data = self.data.query(query) if query else self.data
            return cleansed_data

    @staticmethod
    def normalise(cleansed_data):
        """
            normalise to 2D space
        """
        normalised_data = StandardScaler().fit_transform(cleansed_data)
        pca_data = PCA(n_components=2).fit_transform(normalised_data)
        return pca_data

    def fit(self, classifier=None):
        """
            Обучить кластеризатор (классификация без учителя)

        """

        if classifier is None:
            self.classifier = MeanShift(min_bin_freq=3, bin_seeding=True)
        else:
            self.classifier = classifier
        stdcsl = StandardScaler()
        normalised_data = stdcsl.fit_transform(self.cleansed_data)
        self.classifier.fit(normalised_data)
        all_data = stdcsl.inverse_transform(normalised_data)
        if hasattr(self.classifier, 'labels_'):
            all_data = np.hstack((stdcsl.inverse_transform(normalised_data), self.classifier.labels_[:, None]))
        classified_data = pd.DataFrame(all_data)
        return np.array(classified_data)

    def get_tsne(self):
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        trans_data = tsne.fit_transform(self.data).T
        plt.scatter(trans_data[0], trans_data[1], cmap=plt.cm.rainbow)
        plt.axis('tight')

    def plot(self, classifier=None):
        if isinstance(classifier, DBSCAN) or isinstance(classifier, AgglomerativeClustering):
            print("DBSCAN and AgglomerativeClustering needs another way of plotting")
            return
        if not classifier:
            classifier = MeanShift(min_bin_freq=3, bin_seeding=True)
        pca_data = self.normalise(self.cleansed_data)
        cls = classifier
        cls.fit(pca_data)

        x_min, x_max = pca_data[:, 0].min() - 1, pca_data[:, 0].max() + 1
        y_min, y_max = pca_data[:, 1].min() - 1, pca_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = cls.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.scatter(pca_data[:, 0], pca_data[:, 1], c='g')
        # Plot the centroids as a white X
        centroids = cls.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='b', zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)




class ProbabilityClustering(Clustering):
    """
        Probability class for EM algorithm and its realisations
        Examples:
            GMM
            DPGMM
            VBGMM
    """

    def __init__(self, data, raw_rules=None, query='', **kwargs):
        super(ProbabilityClustering, self).__init__(data, **kwargs)
        self.data = data
        for key, value in kwargs.iteritems():
            setattr(self, key, self.data['%s' % key] >= value if key in self.data else Series())
        self.data = self.get_cleansed_data(raw_rules, query)
        self.classifier = None

    def fit(self, probability_classifier=None):
        if not probability_classifier:
            self.classifier = GMM(n_components=5, n_iter=1000, n_init=2)
        else:
            self.classifier = probability_classifier
        self.classifier.fit(self.data)
        y = self.classifier.predict(self.data)
        all_data = np.hstack((self.data, y[:, None]))
        return all_data

    def plot_probs(self, predicted, title):
        color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
        for i, (mean, covar, color) in enumerate(
                zip(self.classifier.means_, self.classifier._get_covars(), color_iter)):
            if not np.any(predicted == i):
                continue
            plt.scatter(self.data[predicted == i, 0], self.data[predicted == i, 1], .8, color=color)

        plt.xlim()
        plt.ylim()
        plt.title(title)
        plt.xticks(())
        plt.yticks(())

    def estimate(self, columns, predicted=None):
        print set(predicted)
        if predicted is None:
            print DataFrame(self.data, columns=columns).describe()
        else:
            for i in set(predicted):
                print DataFrame(self.data[predicted == i], columns=columns).describe()

    def eval_method(self, method_name=None, **kwargs):
        func = getattr(self.classifier, method_name)
        if not func:
            raise AttributeError(u"Нет атрибута с таким именем")

        return func(**kwargs)

# TODO: Clustering boosting realisation


def bar_charts(cls_data, columns):
    fig, ax = plt.subplots()

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%d' % int(height),
                    ha='center', va='bottom')
    ind = np.arange(len(set(cls_data[:, -1])))  # the x locations for the groups
    width = 0.1       # the width of the bars
    colors = itertools.cycle('bgrcmykw')
    labels = []
    rects = []
    means = []
    stds = []
    for item in set(cls_data[:, -1]):
        clusterMeans = []
        clusterStd = []
        dt = DataFrame(cls_data, columns=columns)[cls_data[:, -1] == item]
        for col in columns:
            clusterMeans.append(np.array(dt[col]).mean())
            clusterStd.append(np.array(dt[col]).std())
        means.append(clusterMeans)
        stds.append(clusterStd)
        labels.append(item)
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Scores')
        ax.set_title('Scores by clusters')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(labels)

    pca = PCA(3)
    means = pca.fit(np.array(means).T)
    stds = pca.fit(scale(np.array(stds).T))

    for item in xrange(means.components_.shape[0]):
        rects.append(ax.bar(ind + width * item, tuple(means.components_[item]), width, color=colors.next(),
                            yerr=tuple(stds.components_[item])))
    ax.legend(tuple(rects), tuple())

    for item in rects:
        autolabel(item)

if __name__ == '__main__':
    # Get dataset and PARAMETERS for some of strict clustering methods
    ids, data, columns = get_data()

    # Strict clustering
    cls = Clustering(data)

    # Soft clustering
    pc = ProbabilityClustering(data)

    columns = list(columns)
    columns += ['class']

    # correlation matrix
    # correlation_matrix(data, np.corrcoef)
    get_params_matrix(data)
    plt.show()


    strict_data = []
    soft_data = []
    print("------------ Strict clustering ---------------\n")
    for i, method in enumerate(STRICT_CLUSTERING_METHODS, 1):
        print("Method number: {}".format(i))
        print("Method name: {}".format(method.__class__.__name__))
        plt.subplot('23' + str(i))
        cls_data = cls.fit(method)
        strict_data.append(cls_data)
        predicted = cls.classifier.predict(data)
        print("Adjusted rand score: {}".format(metrics.adjusted_rand_score(cls_data[:, -1], predicted)))
        print("Adjusted mutual info score: {}".format(metrics.adjusted_mutual_info_score(cls_data[:, -1], predicted)))
        print("Homogeneity score: {}".format(metrics.homogeneity_score(cls_data[:, -1], predicted)))
        print("Cluster description")
        means = []
        stds = []
        for item in set(cls_data[:, -1]):
            d = DataFrame(cls_data[cls_data[:, -1] == item], columns=columns)
            # print(d.describe())
            means.append(d.payments_sum.mean(axis=0))
            stds.append(d.payments_sum.std(axis=0))
        print "l2 norm: {}".format(calc_l2_norm(np.array(means), data.payments_sum.mean(axis=0)))
        print "l2 norm: {}".format(calc_l2_norm(np.array(stds), data.payments_sum.std(axis=0)))
        # pc.plot_probs(cls_data[:, -1][:20000], method.__class__.__name__)
        print("Clustering score: {}".format(metrics.silhouette_score(cls_data[:10000, :-1], cls_data[:10000, -1])))
        print("----------------------------------------\n")
    # plt.show()

    for i in xrange(len(strict_data)):
        bar_charts(strict_data[i], columns)
    plt.show()

    print("\n------------ Soft clustering ---------------\n")
    for i, method in enumerate(SOFT_CLUSTERING_METHODS, 1):
        print("Method number: {}".format(i))
        print("Method name: {}".format(method.__class__.__name__))
        plt.subplot('13' + str(i))
        cls_data = pc.fit(method)
        predicted = pc.eval_method('predict', **{'X': data})
        soft_data.append(cls_data)
        print("Adjusted rand score: {}".format(metrics.adjusted_rand_score(cls_data[:, -1], predicted)))
        print("Adjusted mutual info score: {}".format(metrics.adjusted_mutual_info_score(cls_data[:, -1], predicted)))
        print("Homogeneity score: {}".format(metrics.homogeneity_score(cls_data[:, -1], predicted)))
        print("Cluster description")
        means = []
        stds = []
        for item in set(cls_data[:, -1]):
            d = DataFrame(cls_data[cls_data[:, -1] == item], columns=columns)
            # print(d.describe())
            means.append(d.payments_sum.mean(axis=0))
            stds.append(d.payments_sum.std(axis=0))
        print "l2 norm: {}".format(calc_l2_norm(np.array(means), data.payments_sum.mean(axis=0)))
        print "l2 norm: {}".format(calc_l2_norm(np.array(stds), data.payments_sum.std(axis=0)))
        # pc.plot_probs(cls_data[:, -1][:20000], method.__class__.__name__)
        print("Clustering score: {}".format(metrics.silhouette_score(cls_data[:10000, :-1], cls_data[:10000, -1])))
        print("----------------------------------------\n")
    # plt.show()

    for i in xrange(len(soft_data)):
        bar_charts(soft_data[i], columns)
    plt.show()
