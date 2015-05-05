# coding=utf-8
from datetime import time
from matplotlib.ticker import NullFormatter
from pandas import Series
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as pl
from implementation.methods import get_data

__author__ = 'mihailnikolaev'

PARAMETERS = ['payments_sum', 'number_of_quests', 'overall_time', 'from_last_session', 'count_of_sessions',
              'paymnets_count']


def plot_corr_matrix(data, corr_method):
    headers = data.columns.values
    corr_matrix = corr_method(data.T)
    pl.pcolor(corr_matrix, cmap='bwr', vmin=-1, vmax=1)
    pl.colorbar()
    pl.yticks(np.arange(0.5, 24.5), headers)
    pl.xticks(np.arange(0.5, 24.5), headers, rotation=20)
    pl.xlim(xmax=corr_matrix.shape[0])
    pl.ylim(ymax=corr_matrix.shape[1])
    pl.title(u"Корреляции между признаками")


if '__main__' == __name__:
    data, PARAMETERS = get_data()
    print data.avg_payment
    # data = data.drop(labels=['payments_count', 'payments_sum'], axis=1)
    pl.figure(figsize=(16, 8))
    # pl.subplot(121)
    plot_corr_matrix(data, np.corrcoef)
    pl.show()