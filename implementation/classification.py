# coding=utf-8
from sklearn import cross_validation, metrics
from sklearn.cluster import MeanShift
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.mixture import GMM, DPGMM
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from implementation import clusterisation
from implementation.clusterisation import Clusterisation, ProbabilityClusterisation
from implementation.methods import get_data, set_RFM_classes

__author__ = 'mihailnikolaev'


class Classifier(object):
    def __init__(self, full_data, classifier=None):
        self.data = np.array(full_data)
        self.classifier = classifier

    def fit(self, train_data=None, train_labels=None):
        if not self.classifier:
            self.classifier = GradientBoostingClassifier()

        if train_labels:
            return self.classifier.fit(train_data, train_labels)
        else:
            return self.classifier.fit(self.data[:, :-1], self.data[:, -1])

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
    classifier = Classifier(cls_data, GradientBoostingClassifier())
    classifier.fit()
    scores = cross_validation.cross_val_score(GradientBoostingClassifier(), cls_data[:, :-1], cls_data[:, -1], cv=20)
    print scores.mean(), scores.std()

    predicted = classifier.classifier.predict(cls_data[:, :-1])
    print predicted
    print cls_data[:, -1]
    print metrics.classification_report(cls_data[:, -1], predicted)
    classifier.plot()


if '__main__' == __name__:
    data, clusterisation.PARAMETERS = get_data()
    # data = data.drop(labels=['payments_count'], axis=1)
    # data = data[data.payments_sum >= 1]
    cls = ProbabilityClusterisation(data)
    preds = np.array(cls.fit(DPGMM(n_components=5, covariance_type='diag', alpha=10.,
                       n_iter=10000, random_state=0)))
    print cls.classifier.score(np.array(data))

    cls_data = np.hstack((cls.data, preds[:, None], ))
    # cls = Clusterisation(data, payments_count=0, payments_sum=1, number_of_quests=0, overall_time=0,
    #                      from_last_session=0,
    #                      count_of_sessions=0, raw_rules=[1, 1, 1, 1, 1, 1, 1, 1])
    # cls_data = np.array(cls.fit(GMM(n_components=3)))
    # ses_p = np.array(data)
    # classes = set_RFM_classes(ses_p, 7, 2)
    # ses_p = np.array([np.hstack((ses_p[i, :], classes[i])) for i in range(len(classes))])
    main(cls_data)

    plt.show()