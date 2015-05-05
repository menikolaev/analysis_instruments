# coding=utf-8
from datetime import datetime, time
import itertools
import matplotlib as mpl
from scipy import linalg
from sklearn import manifold
import numpy as np
from pandas.util.testing import DataFrame, Series
from sklearn.cluster import DBSCAN, MiniBatchKMeans, MeanShift, estimate_bandwidth
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.gaussian_process.gaussian_process import l1_cross_distances
from sklearn.metrics import roc_curve, auc
from sklearn.mixture import GMM, DPGMM
from sklearn.preprocessing import StandardScaler
from implementation.methods import get_data
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'mihailnikolaev'

PARAMETERS = ['payments_count', 'payments_sum', 'number_of_quests', 'overall_time', 'from_last_session',
              'count_of_sessions']


class ProbabilityClusterisation(object):
    """
        Probability class for EM algorithm and its realisations
        Examples:
            GMM
            DPGMM
            VBGMM
    """
    def __init__(self, data, *kwargs):
        self.data = np.array(data)
        self.classifier = None

    def fit(self, probability_classifier=None):
        if not probability_classifier:
            self.classifier = GMM(n_components=5, n_iter=1000, n_init=2)
        else:
            self.classifier = probability_classifier
        self.classifier.fit(self.data)
        y = self.classifier.predict(self.data)
        return y

    def plot(self, predicted, title):
        color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
        for i, (mean, covar, color) in enumerate(zip(self.classifier.means_, self.classifier._get_covars(), color_iter)):
            # v, w = linalg.eigh(covar)
            # u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(predicted == i):
                continue
            plt.scatter(self.data[predicted == i, 0], self.data[predicted == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            # angle = np.arctan(u[1] / u[0])
            # angle = 180 * angle / np.pi  # convert to degrees
            # ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
            # ell.set_clip_box(splot.bbox)
            # ell.set_alpha(0.5)
            # splot.add_artist(ell)

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


class Clusterisation(object):
    """
        Non-probability clusterisation class
        Examples:
            AglomerativeClusterisation
            DBSCAN
            Birch
            K-Means
            MiniBatchKMeans
    """
    def __init__(self, data, raw_rules=None, **kwargs):
        self.data = data
        for key, value in kwargs.iteritems():
            setattr(self, key, self.data['%s' % key] >= value if key in self.data else Series())
        self.cleansed_data = self.get_cleansed_data(raw_rules)

    def clean_rules(self, raw_rules):
        query = ''
        if not raw_rules:
            return

        for i in xrange(len(raw_rules)):
            if raw_rules[i] and hasattr(self, PARAMETERS[i]):
                if raw_rules[i] == -1:
                    setattr(self, PARAMETERS[i], ~getattr(self, PARAMETERS[i]))
                elif raw_rules[i] < -1:
                    raise AttributeError('Invalid attribute %s' % str(raw_rules[i]))

                if not getattr(getattr(self, PARAMETERS[i]), 'empty'):
                    query = ' & '.join([query, '@self.%s' % PARAMETERS[i]])
        return query[3:] if query else query

    def get_cleansed_data(self, raw_rules):
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
        if not classifier:
            classifier = MeanShift(min_bin_freq=3, bin_seeding=True)
        stdcsl = StandardScaler()
        normalised_data = stdcsl.fit_transform(self.cleansed_data)
        cls = classifier
        cls.fit(normalised_data)
        all_data = stdcsl.inverse_transform(normalised_data)
        if hasattr(cls, 'labels_'):
            all_data = np.hstack((stdcsl.inverse_transform(normalised_data), cls.labels_[:, None]))
        classified_data = pd.DataFrame(all_data)
        return np.array(classified_data)

    def get_tsne(self):
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        trans_data = tsne.fit_transform(self.data).T
        plt.scatter(trans_data[0], trans_data[1], cmap=plt.cm.rainbow)
        plt.axis('tight')

    def plot(self, classifier=None):
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
        # fig = plt.figure()
        # ax = Axes3D(fig)

        # plt.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c='g')
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c='g')
        # Plot the centroids as a white X
        centroids = cls.cluster_centers_
        # plt.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
        #             marker='x', s=169, linewidths=3,
        #             color='b', zorder=10)
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='b', zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)


if '__main__' == __name__:
    data, PARAMETERS = get_data()
    print l1_cross_distances(np.array(data))
    fpr = [0]*2
    tpr = [0]*2
    area = [0]*2

    print str(PARAMETERS)
    # data = data.drop(labels=['payments_count', 'count_of_sessions', 'number_of_quests', 'count_of_sessions'], axis=1)
    print data.describe()
    plt.subplot(141)
    cls = Clusterisation(data, payments_count=1, payments_sum=0, number_of_quests=0, overall_time=0, from_last_session=0,
                         count_of_sessions=0, raw_rules=[1, 1, 1, 1, 1, 1, 1, 1])
    # cls.get_tsne()
    cls.plot(MeanShift(min_bin_freq=4))

    plt.subplot(142)
    time_start = datetime.now()
    cls = Clusterisation(data, payments_count=0, payments_sum=0, number_of_quests=0, overall_time=0, from_last_session=0,
                         count_of_sessions=0, raw_rules=[1, 1, 1, 1, 1, 1, 1, 1])
    cls_data = cls.fit()
    delta_time = datetime.now() - time_start
    print "Time: %s" % str(delta_time.microseconds)
    cls.plot()
    columns = list(data.columns)
    columns.insert(0, 'class')
    df = DataFrame(cls_data, columns=columns)
    for i in xrange(0, 7):
        print len(df[df['class'] == i])
        print df[df['class'] == i].describe()

    plt.subplot(143)
    pc = ProbabilityClusterisation(data)
    y = pc.fit()
    expected = np.array([0 if x > 0 else 1 for x in data['payments_sum']])
    fpr[0], tpr[0], _ = roc_curve(expected, y)
    area[0] = auc(fpr[0], tpr[0])
    pc.plot(y, "GMM")

    plt.subplot(144)
    pc = ProbabilityClusterisation(data)
    y = pc.fit(DPGMM(n_components=2, covariance_type='diag', alpha=200,
                       n_iter=10000, random_state=0))
    print y
    expected = np.array([0 if x > 0 else 1 for x in data['payments_sum']])
    fpr[1], tpr[1], _ = roc_curve(expected, y)
    area[1] = auc(fpr[1], tpr[1])
    pc.estimate(list(data.columns), y)
    pc.plot(y, "DPGMM")
    plt.show()

    plt.figure()
    plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % area[0])
    plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % area[1])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('DPGMM')
    plt.legend(loc="lower right")
    plt.show()
