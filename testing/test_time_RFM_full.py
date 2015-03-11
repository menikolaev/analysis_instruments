from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

__author__ = 'mihailnikolaev'


def predict(data, minT, maxT, stp, classifier, plt):
    scores = []
    r = np.linspace(minT, maxT, stp)
    for x in r[1:-1]:
        train, test = train_test_split(data, train_size=x)
        dt = classifier
        dt.fit(train[:, :-1], train[:, -1])
        score = dt.score(test[:, :-1], test[:, -1])
        scores.append(score)
    plt.plot(r[1:-1], scores)


def get_data(data_format='file'):
    if data_format == 'file':
        sessions = pd.read_csv('sessions.csv', delimiter=';', names=['id', 'overall_time', 'count_of_sessions', 'from_last_session'])
        payments = pd.read_csv('payments.csv', delimiter=';', names=['id', 'payments_sum', 'payments_count'])
        quests = pd.read_csv('quests.csv', delimiter=';', names=['id', 'number_of_quests'])
        ses = pd.merge(sessions, payments, 'outer')
        ses = pd.merge(ses, quests, 'outer')
        ses = ses.drop(ses.columns[[0]], axis=1)
        return ses
    elif data_format == 'db':
        pass
    else:
        raise AttributeError("data_format should be 'file' or 'db'")


def from_y_to_x(numb, k):
    n = 0
    numb = str(numb)
    for i in range(len(numb)-1, 0, -1):
        n += int(numb[i])*(k ** i)
    return n


def set_RFM_classes(data, col_len, n):
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

ses_p = np.array(get_data())
time_start = datetime.now()
classes = set_RFM_classes(ses_p, 6, 5)
ses_p = np.array([np.hstack((ses_p[i, :], classes[i])) for i in range(len(classes))])
delta_time = datetime.now() - time_start
print "Time: %s" % str(delta_time.microseconds)

# predict(ses_p, 0, 1, 200, DecisionTreeClassifier(), plt)
# plt.show()