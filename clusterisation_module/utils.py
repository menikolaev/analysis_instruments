# coding=utf-8
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'mihailnikolaev'


def get_data(path='../data/data.csv', data_format='file'):
    """
        Получить исходные данные из файлов
        data:
            id
            number_of_quests - number of ended quests
            payments_sum - overall sum of payments by player
            payments_count - number of transactions
            overall_time - all time in game (days)
            count_of_sessions - number of sessions in game
            from_last_session - time from last session (days)
            traffic - how player came to game
            acc_lifetime - how many time account live
    """

    if data_format == 'file':
        gc.disable()
        data = pd.read_csv(path, delimiter=';', header=0,
                           names=['id', 'number_of_quests', 'payments_sum', 'payments_count', 'overall_time',
                                  'count_of_sessions', 'from_last_session', 'traffic', 'acc_lifetime'])
        gc.enable()
        ids = data.id
        data.traffic[(data.traffic == 'Viral_traffic') | (data.traffic == 'Viral traffic')] = 'viral'
        data.traffic[data.traffic == 'Organic traffic'] = 'organic'
        data.traffic[(data.traffic != 'viral') & (data.traffic != 'organic')] = 'commercial'
        data = data.drop(data.columns[[0, 3, 5]], axis=1)
        str_types = data.columns[data.dtypes == object]
        for item in str_types:
            dummies = pd.get_dummies(data[item])
            data = data.join(dummies)
        data = data.drop(str_types, axis=1)
        # TODO: these 3 parameters should be in data file
        # data['avg_payment'] = pd.Series([x/y for x, y in zip(data.payments_sum, data.payments_count)],
        #                          index=data.index)
        # data['avg_time_per_ses'] = pd.Series([x/y for x, y in zip(data.overall_time, data.count_of_sessions)],
        #                          index=data.index)
        data['game_time_per_lifetime'] = pd.Series([x/y for x, y in zip(data.overall_time, data.acc_lifetime)],
                                 index=data.index)

        return ids, data, data.columns
    elif data_format == 'db':
        pass
    else:
        raise AttributeError("data_format should be 'file' or 'db'")


def from_y_to_x(numb, k):
    """
        Перевод из k системы в 10ную
    """
    n = 0
    numb = str(numb)
    for i in range(len(numb)-1, 0, -1):
        n += int(numb[i])*(k ** i)
    return n


def set_RFM_classes(data, col_len, n):
    """
        Расставить метки объектам согласно RFM разделению данных
    """

    classes = []
    for i in range(col_len):
        maxi = data[:, i].max()
        mini = data[:, i].min()
        class_i = np.linspace(mini, maxi, n)
        classes.append(class_i)
    print classes

    labels = []
    for item in data:
        curr_class = ''
        for i in range(col_len):
            class_i = classes[i]
            for x in range(0, len(class_i)+1):
                if x == 0 and item[i] <= class_i[x]:
                    curr_class += '1'
                    continue
                elif x == len(class_i) and item[i] > class_i[x-1]:
                    curr_class += str(len(class_i)+1)
                    continue
                else:
                    if class_i[x-1] < item[i] <= class_i[x]:
                        curr_class += str(x+1)
                        continue
        labels.append(from_y_to_x(int(curr_class), n+1))
    return np.array(labels)


def correlation_matrix(data, corr_method):
    headers = data.columns.values
    corr_matrix = corr_method(data.T)
    plt.pcolor(corr_matrix, cmap='bwr', vmin=-1, vmax=1)
    plt.colorbar()
    plt.yticks(np.arange(0.5, 24.5), headers)
    plt.xticks(np.arange(0.5, 24.5), headers, rotation=20)
    plt.xlim(xmax=corr_matrix.shape[0])
    plt.ylim(ymax=corr_matrix.shape[1])
    plt.title(u"Корреляции между признаками")


def get_params_matrix(data):
    indexes = data.columns
    print indexes
    npdata = np.array(data)
    fig, ax = plt.subplots(len(indexes), len(indexes), figsize=(16, 8))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    for i in xrange(len(indexes)):
        for j in xrange(len(indexes)):
            ax[i, j].scatter(npdata[:, i], npdata[:, j], .8)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlabel(indexes[i])
            ax[i, j].set_ylabel(indexes[j])
