from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from implementation.methods import get_data
from implementation.visualisation import plot_cluster

__author__ = 'mihailnikolaev'
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale, StandardScaler
import matplotlib.pyplot as plt


def from_x_to_y(numb, k):
    n = ''
    while numb > 0:
        y = str(numb % k)
        n = y + n
        numb = int(numb / 2)
    return int(n)


def from_y_to_x(numb, k):
    n = 0
    numb = str(numb)
    for i in range(len(numb)-1, 0, -1):
        n += int(numb[i])*(k ** i)
    return n


def get_classes(data, col_len, n):
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


def predict(data, minT, maxT, stp):
    scores = []
    r = np.linspace(minT, maxT, stp)
    print r
    for x in r[1:-1]:
        train, test = train_test_split(data, train_size=x)
        dt = DecisionTreeClassifier()
        dt.fit(train[:, :-1], train[:, -1])
        score = dt.score(test[:, :-1], test[:, -1])
        scores.append(score)
    plt.plot(r[1:-1], scores)

"""
sessions:
    id
    overall_time - all time in game (days)
    count_of_sessions - number of sessions in game
    from_last_session - time from last session (days)

payments:
    id
    payments_sum - overall sum of payments by player
    payments_count - number of transactions

quests:
    id
    number_of_quests - number of ended quests
"""
ses = get_data()

ses_p = ses[ses['payments_sum'] != 0]
ses_with_zero = ses[ses['payments_sum'] == 0]
ses_p_more20_ses = ses_p[ses_p['count_of_sessions'] >= 20]
ses_p_less20_ses = ses_p[ses_p['count_of_sessions'] < 20]
ses_with_zero_more20_ses = ses_with_zero[(ses_with_zero['count_of_sessions'] >= 20) & (ses_with_zero['number_of_quests'] > 20)]
ses_with_zero_less20_ses = ses_with_zero[(ses_with_zero['count_of_sessions'] < 20) & (ses_with_zero['number_of_quests'] > 20)]
print ses_p

ses_p_scaler = StandardScaler()
ses_p = ses_p_scaler.fit_transform(np.array(ses_p))
ses_with_zero = StandardScaler().fit_transform(np.array(ses_with_zero))
ses_p_more20_ses = StandardScaler().fit_transform(np.array(ses_p_more20_ses))
ses_p_less20_ses = StandardScaler().fit_transform(np.array(ses_p_less20_ses))
ses_with_zero_more20_ses = StandardScaler().fit_transform(np.array(ses_with_zero_more20_ses))
ses_with_zero_less20_ses = StandardScaler().fit_transform(np.array(ses_with_zero_less20_ses))

full_kmeans = KMeans(n_clusters=5)
full_kmeans.fit(ses_p)
defined_classes = np.hstack((ses_p_scaler.inverse_transform(ses_p), full_kmeans.labels_[:, None]))
print defined_classes

classes = get_classes(ses_p, 6, 2)
ses_p = np.array([np.hstack((ses_p[i, :], classes[i])) for i in range(len(classes))])
pred_ses = ses_p
train, test = train_test_split(ses_p, test_size=0.3)
dt = DecisionTreeClassifier()


pca = PCA(n_components=2)
pred_ses_pca = PCA(n_components=2)
print pred_ses[:, :-1]
reduced_sessions = pred_ses_pca.fit_transform(pred_ses[:, :-1])
reduced_sessions_test = pca.fit_transform(test[:, :-1])
reduced_sessions_with_zeros = pca.fit_transform(ses_with_zero)
reduced_sessions_more20 = pca.fit_transform(ses_p_more20_ses)
reduced_sessions_less20 = pca.fit_transform(ses_p_less20_ses)
reduced_sessions_with_zeros_more20 = pca.fit_transform(ses_with_zero_more20_ses)
reduced_sessions_with_zeros_less20 = pca.fit_transform(ses_with_zero_less20_ses)
pred = pred_ses_pca.inverse_transform(reduced_sessions)
print pred
print ses_p_scaler.inverse_transform(pred)


ses_p = pca.fit_transform(ses_p[:, :-1])
dt.fit(train[:, :-1], train[:, -1])
print dt.score(test[:, :-1], test[:, -1])
plt.figure(figsize=(20, 10))
plt.subplot(321)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'payments_sum = 0')
plot_cluster(reduced_sessions, plt)
plt.subplot(322)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'payments_sum != 0')
plot_cluster(reduced_sessions_with_zeros, plt)
plt.subplot(323)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'payments_sum = 0, count_of_sessions >= 20')
plot_cluster(reduced_sessions_more20, plt)
plt.subplot(324)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'payments_sum = 0, count_of_sessions < 20')
plot_cluster(reduced_sessions_less20, plt)
plt.subplot(325)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'payments_sum != 0, count_of_sessions >= 20')
plot_cluster(reduced_sessions_with_zeros_more20, plt)
plt.subplot(326)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'payments_sum != 0, count_of_sessions < 20')
plot_cluster(reduced_sessions_with_zeros_less20, plt)

plt.show()
