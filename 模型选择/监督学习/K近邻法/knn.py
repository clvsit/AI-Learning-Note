%matplotlib inline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

x1 = np.round(np.random.random_sample(20) * 5)
x2 = np.round(np.random.random_sample(20) * 5)
y = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0])
dataset = np.column_stack((x1, x2, y))
data_1 = dataset[dataset[:, 2] == 1]
data_0 = dataset[dataset[:, 2] == 0]
fig = plt.figure(figsize=(10, 3))
ax1 = fig.add_subplot(121)
ax1.scatter(data_1[:, 0], data_1[:, 1], color='r')
ax1.scatter(data_0[:, 0], data_0[:, 1], color='b')
ax2 = fig.add_subplot(122)
ax2.scatter(data_1[:, 0], data_1[:, 1], color='r')
ax2.scatter(data_0[:, 0], data_0[:, 1], color='b')
ax2.scatter([1], [2], color='g')
plt.show()

x = dataset[:, 0:2]
y = dataset[:, 2]

class KNN:
    
    def __init__(self, k=5, dist_method='euclidean', vote_method='count'):
        if k <= 0:
            raise Exception('The param k is not less then or equal to zero!')
        
        dist_method_list = ['euclidean', 'city_block', 'chebyshev']
        if dist_method not in dist_method_list:
            raise Exception('The distance method not exist!')
            
        vote_method_list = ['count', 'distance']
        if vote_method not in vote_method_list:
            raise Exception('The vote method not exist!')
            
        self.k = k
        self.dist_method = dist_method
        self.vote_method = vote_method
        
    def set_k(self, k):
        if k <= 0:
            raise Exception('The param k is not less then or equal to zero!')
        self.k = k
        
    def set_dist_method(self, dist_method):
        dist_method_list = ['euclidean', 'city_block', 'chebyshev']
        if dist_method not in dist_method_list:
            raise Exception('The distance method not exist!')
        self.dist_method = dist_method
        
    def set_vote_method(self, vote_method):
        vote_method_list = ['count', 'distance']
        if vote_method not in vote_method_list:
            raise Exception('The vote method not exist!')
        self.vote_method = vote_method
    
    def _get_distance(self, XA, XB, method='euclidean'):
        return cdist(XA, [XB], method)
    
    def _find_k_nearest(self, x, y, target):
        dists = self._get_distance(x, target)
        nearests = sorted([[dists[i], y[i]] for i in range(x.shape[0])], key=lambda data:data[0])
        return np.array(nearests[:self.k])
    
    def _vote(self, nearests):
        if self.vote_method == 'count':
            labels = nearests[:, 1].tolist()            
            return sorted([(labels.count(label), label) for label in set(labels)])[-1][1]
        elif self.vote_method == 'distance':
            labels = {}
            dist_mean = np.mean(nearests[:, 0])
            for data in nearests:
                dist = dist_mean / data[0]
                label = data[1]
                if label not in labels:
                    labels[label] = dist
                else:
                    labels[label] += dist
            return sorted([(labels[label], label) for label in set(labels)])[-1][1]
        
    def fit(self, x, y):
        self.x = x
        self.y = y
    
    def predict(self, target):
        nearests = self._find_k_nearest(self.x, self.y, target)
        label = self._vote(nearests)
        return label

knn = KNN(vote_method='distance')
knn.fit(x, y)
knn.predict([1, 2])