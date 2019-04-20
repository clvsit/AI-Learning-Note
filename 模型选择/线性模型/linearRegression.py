%matplotlib inline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset = pd.read_csv('data/quiz.csv')
array = dataset.values
x = array[:, 1:3]
y = array[:, 3]

fig = plt.figure(figsize=(10, 3))
ax1 = fig.add_subplot(121)
ax1.scatter(x[:, 0], y)
ax1.set_xlabel('Last Score')
ax1.set_ylabel('Score')
ax2 = fig.add_subplot(122)
ax2.scatter(x[:, 1], y)
ax2.set_xlabel('Hours Spent')
ax2.set_ylabel('score')
plt.show()

# 存在异常点 id 4，将该数据去除
x = np.delete(x, 3, axis=0)
y = np.delete(y, 3, axis=0)

class LinearRegression:
    
    def __init__(self, learning_rate=0.01, stop_criterion='threshold', n_iters=1000, threshold=0.001):
        self.learning_rate_ = learning_rate
        stop_criterionp_list = ['iter_count', 'threshold']
        if stop_criterion not in stop_criterionp_list:
            raise Exception('The param stop_criterion is invalid!')
        self.stop_criterion_ = stop_criterion        
        self.n_iters_ = n_iters
        self.threshold_ = threshold
        self._loop = 0
    
    def _min_max_scaler(self, dataset):
        cols = dataset.shape[1]
        new_dataset = []
        for col in range(cols):
            max_value, min_value = max(dataset[:, col]), min(dataset[:, col])            
            data_col = (dataset[:, col] - min_value) / (max_value - min_value)
            new_dataset.append(data_col)
        return np.array(new_dataset).T
    
    def _stop_criterion_handle(self):
        if self.stop_criterion_ == 'iter_count':
            self._loop += 1
            return True if self._loop <= self.n_iters_ else False
        else:
            if self._last_cost == 0:
                return True
            return True if self._last_cost - self._cur_cost > self.threshold_ else False
    
    def fit(self, dataset, labels):
        dataset_transformed = self._min_max_scaler(np.column_stack((dataset, labels)))
        x, y = dataset_transformed[:, 0:-1], dataset_transformed[:, -1]
        length, features = x.shape[0], x.shape[1] + 1
        weight = np.zeros(features)
        x = np.column_stack((x, np.ones((length, 1))))
        self._loop, self._last_cost = 0, 0
        
        while self._stop_criterion_handle():
            new_weight = weight.copy()
            self._last_cost = np.sum((y - np.dot(x, new_weight))**2) / length
            for feature in range(features):
                new_weight[feature] += self.learning_rate_ * np.sum((y - np.dot(x, weight)) * x[:, feature]) / length
            self._cur_cost = np.sum((y - np.dot(x, new_weight))**2) / length
            if (weight == new_weight).all():
                break
            weight = new_weight
        self.coef_, self.intercept_ = weight[:-1], weight[-1]
    
    def predict(self, dataset):
        if len(dataset.shape) == 0:
            x = np.array([dataset])
        else:
            x = dataset
        return np.dot(x, self.coef_) + self.intercept_


model = LinearRegression(learning_rate=0.05, stop_criterion='iter_count', n_iters=5000)
model.fit(x, y)
print(model.coef_, model.intercept_)
print(model.predict(x))