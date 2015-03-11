import numpy as np
__author__ = 'mihailnikolaev'

PARAMETERS = ['payments_sum', 'number_of_quests', 'overall_time', 'from_last_session', 'count_of_sessions',
              'paymnets_count']


def get_correlations(data):
    correlation = np.cov(data['payments_count'], data['payments_sum'])
    return correlation