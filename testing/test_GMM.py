import itertools
from pandas import DataFrame
from scipy import linalg
from implementation.methods import get_data

__author__ = 'mihailnikolaev'

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import mixture, metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM


color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

X, columns = get_data()
X = np.array(X)
plt.figure(figsize=(14, 10))
for i, (clf, title) in enumerate([
        (mixture.GMM(n_components=10, covariance_type='full', n_iter=1000),
        "Expectation-maximization"),
        (mixture.DPGMM(n_components=5, covariance_type='diag', alpha=10.,
                       n_iter=10000, random_state=0),
         "Dirichlet Process,alpha=100.")]):

    clf.fit(X)
    splot = plt.subplot(1, 2, 1 + i)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        print DataFrame(X[Y_ == i], columns=columns).describe()

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim()
    plt.ylim()
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

plt.show()