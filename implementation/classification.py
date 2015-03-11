from matplotlib.colors import ListedColormap
from sklearn.cluster import MeanShift
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from implementation.clusterisation import Clusterisation
from implementation.methods import get_data
from implementation.preprocessing import get_correlations

__author__ = 'mihailnikolaev'


class Classifier(object):
    def __init__(self, full_data, classifier=None):
        self.data = np.array(full_data)
        self.classifier = classifier

    def fit(self, train_data=None, train_labels=None):
        if not self.classifier:
            self.classifier = GradientBoostingClassifier()

        return self.classifier.fit(self.data[:, :-1], self.data[:, -1]) if not len(train_labels) else \
            self.classifier.fit(train_data, train_labels)

    def score(self, test_data, test_labels):
        return self.classifier.score(test_data, test_labels)

    def __normalise_data(self):
        normalised_data = StandardScaler().fit_transform(self.data[:, :-1])
        pca_data = PCA(n_components=2).fit_transform(normalised_data) if normalised_data.shape[1] > 2 else normalised_data
        return pca_data

    def plot(self):
        pca_data = self.__normalise_data()

        clf = self.fit(pca_data, self.data[:, -1])

        h = .02

        X = pca_data
        y = self.data[:, -1]

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())


def main(cls_data):
    train, test = train_test_split(cls_data, test_size=0.3)
    classifier = Classifier(train, SVC(C=40, gamma=0))
    classifier.plot()


if '__main__' == __name__:
    data = get_data()
    cov_matrix = get_correlations(data)
    for item in cov_matrix:
        print item
    data = data.drop(labels=['from_last_session', 'number_of_quests', 'count_of_sessions'], axis=1)
    cls = Clusterisation(data, payments_count=0, payments_sum=0, number_of_quests=0, overall_time=1,
                         from_last_session=0,
                         count_of_sessions=0, raw_rules=[1, 1, 1, 1, 1, 1])
    cls_data = np.array(cls.fit(MeanShift(min_bin_freq=1, bin_seeding=True)))
    main(cls_data)
    plt.show()