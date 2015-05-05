# coding=utf-8
import os
from pandas.util.testing import DataFrame
from sklearn.cluster import MeanShift, KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, \
    AffinityPropagation
from sklearn.mixture import DPGMM, GMM, VBGMM
from clustering import Clustering, ProbabilityClustering
from utils import get_data
import cPickle as pickle
import numpy as np

__author__ = 'mihailnikolaev'

STRICT_CLUSTERING_METHODS = [MeanShift, KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering,
                             SpectralClustering, AffinityPropagation]

SOFT_CLUSTERING_METHODS = [GMM, DPGMM, VBGMM]

STRICT_CLUSTERING_METHODS_INST = {
    'MeanShift': MeanShift(min_bin_freq=3, bin_seeding=True),
    'KMeans': KMeans(n_clusters=8, max_iter=1000),
    'MiniBatchKMeans': MiniBatchKMeans(),
    'DBSCAN': DBSCAN(),
    'AgglomerativeClustering': AgglomerativeClustering(8)
}

SOFT_CLUSTERING_METHODS_INST = {
    'GMM': GMM(n_components=5, n_iter=200),
    'DPGMM': DPGMM(n_components=3, alpha=100., n_iter=1000),
    'VBGMM': VBGMM(n_components=5, alpha=1000, n_iter=100)
}


class UserProcessing(object):
    """
        Основной класс утилиты для кластеризации данных.
        Позволяет обучить модель, сохранить и восстановить ее состояние, а также
        сохранить результаты работы в csv формат
    """

    def __init__(self, data_file):
        """ Получение первичных данных """
        self.ids, self.data, self.parameters = get_data(data_file)
        self.model = None

    def fit_model(self, clustering_method=None, clusters=5, **kwargs):
        """
            Метод обучает модель в соответствии с выбранным методом кластеризации
            ОСТРОЖНО: метод пока плохо работает с некоторыми видами кластеризации, которые не имеют метода fit
        """
        if clustering_method is not None:
            if clustering_method in SOFT_CLUSTERING_METHODS_INST:
                model = SOFT_CLUSTERING_METHODS_INST[clustering_method]
            elif clustering_method in STRICT_CLUSTERING_METHODS_INST:
                model = STRICT_CLUSTERING_METHODS_INST[clustering_method]
            else:
                raise NotImplementedError('No clustering method called {}'.format(clustering_method))
        else:
            model = DPGMM(n_components=clusters, alpha=10., n_iter=1000)

        if isinstance(model, tuple(STRICT_CLUSTERING_METHODS)):
            self.model = Clustering(self.data, **kwargs)
        elif isinstance(model, tuple(SOFT_CLUSTERING_METHODS)):
            self.model = ProbabilityClustering(self.data, **kwargs)
        else:
            raise TypeError("There is no clustering method called {}".format(str(model)))

        return self.model.fit(model)

    def predict(self, X):
        """
            Метод предсказывает метки в соответствии с выбранным методом кластеризации
            ОСТРОЖНО: метод пока плохо работает с некоторыми видами кластеризации, которые не имеют метода predict
        """
        if isinstance(self.model.classifier, tuple(STRICT_CLUSTERING_METHODS)):
            # TODO: predict method for model (hide realization, avoid problem with different kinds of predict methods)
            predicted = self.model.classifier.predict(X)
        elif isinstance(self.model.classifier, tuple(SOFT_CLUSTERING_METHODS)):
            predicted = self.model.eval_method('predict', **{'X': X})
        else:
            raise TypeError("There is no clustering method called {}".format(str(self.model.classifier)))

        return predicted

    def describe_results(self, fitted_data, prdeiction_mode=False):
        """
            Вывод результатов
            В зависимости от того, какой режим выбран, будут записываться разные данные
            Для обучения:
                Будет выведена статистическая информация о каждом кластере
                count - число объектов
                mean - выборочное среднее
                std - стандартное отклонение
                min - минимаьное значение
                25% - 25% перцентиль
                50% - 50% перцентиль
                75% - 75% перцентиль
                max - максимальное значение
            Для предсказания:
                Первичные данные + последний столбец метки кластеров
        """
        if self.model is None:
            raise NotImplementedError("Model is not exists or not fitted")

        columns = list(self.parameters)
        columns += ['class']

        if prdeiction_mode:
            columns.insert(0, 'id')
            return DataFrame(np.hstack((np.array(self.ids)[:, None], fitted_data)), columns=columns)

        described_list = []
        for item in set(fitted_data[:, -1]):
            described_list.append(DataFrame(fitted_data, columns=columns)[fitted_data[:, -1] == item].describe())
        return described_list

    def serialize(self):
        """
            Сохранение обученной модели в файл 'proc_obj'
        """
        with open('proc_obj', 'wb') as f:
            pickle.dump(self.model, f, 2)

    def deserialize(self, obj_file):
        """
            Дессериализация обученной модели из указанного файла
        """
        with open(obj_file, 'rb') as f:
            self.model = pickle.load(f)

    @staticmethod
    def save_to_file(data, path='', mode='a'):
        """
            Сохранение результатов работы в указанный файл
            Если mode = a, то данные сохраняются в конец файла
            Если mode = w, то данные перезаписываются
        """
        if isinstance(data, list):
            if mode == 'w':
                try:
                    os.remove(path)
                except OSError as ex:
                    print(str(ex) + ' File will be created.')

            for item in data:
                item.to_csv(path, sep=';', mode='a')
        else:
            data.to_csv(path, sep=';', mode=mode)