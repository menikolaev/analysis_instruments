from datetime import datetime
import numpy as np
from pandas.util.testing import DataFrame, Series
from sklearn.cluster import DBSCAN, MiniBatchKMeans, MeanShift, estimate_bandwidth
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from implementation.methods import get_data
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'mihailnikolaev'

PARAMETERS = ['payments_count', 'payments_sum', 'number_of_quests', 'overall_time', 'from_last_session',
              'count_of_sessions']


class Clusterisation(object):
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
        all_data = np.hstack((stdcsl.inverse_transform(normalised_data), cls.labels_[:, None]))
        classified_data = pd.DataFrame(all_data)
        return np.array(classified_data)

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
    data = get_data()
    data = data.drop(labels=['payments_count'], axis=1)
    print data.describe()
    plt.subplot(121)
    cls = Clusterisation(data, payments_count=0, payments_sum=1, number_of_quests=0, overall_time=0, from_last_session=0,
                         count_of_sessions=0, raw_rules=[1, 1, 1, 1, 1, 1])
    cls.plot(MeanShift(min_bin_freq=15, bin_seeding=True))
    plt.subplot(122)
    time_start = datetime.now()
    cls = Clusterisation(data, payments_count=0, payments_sum=0, number_of_quests=0, overall_time=0, from_last_session=0,
                         count_of_sessions=0, raw_rules=[1, 1, 1, 1, 1, 1])
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
    plt.show()
