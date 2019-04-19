%matplotlib inline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df_dataset = pd.read_csv('data/testSet.txt', header=None, delimiter='\t')
dataset = df_dataset.values


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataset[:, 0], dataset[:, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()


class PCA:
    
    def __init__(self, n_components=None):
        if n_components and isinstance(n_components, int):
            self.n_components = n_components
        else:
            self.n_components = None
        
    def _decentration(self, dataset):
        dataset_mean = np.mean(dataset, axis=0)
        return dataset - dataset_mean
    
    def fit(self, dataset):
        if self.n_components is None:
            self.n_components = np.min(dataset.shape)
        # 去中心化
        dataset_removed = self._decentration(dataset)
        # 求协方差矩阵
        # xTx = np.dot(dataset_removed.T, dataset_removed)
        xTx = np.cov(dataset_removeda.T)
        # 分解协方差矩阵
        feature_values, feature_vectors = np.linalg.eig(xTx)
        # 对 feature_values 进行升序排序
        feature_values_ind = np.argsort(feature_values)
        # 选择其中最大的 d 个特征向量
        self.transform_vectors = feature_vectors[:, feature_values_ind[-self.n_components:]]
        
    def transform(self, dataset):
        dataset_removed = self._decentration(dataset)
        return np.dot(dataset_removed, self.transform_vectors)
    
    def fit_transform(self, dataset):
        if self.n_components is None:
            self.n_components = np.min(dataset.shape)
        dataset_removed = self._decentration(dataset)
        # xTx = np.dot(dataset_removed.T, dataset_removed)
        xTx = np.cov(dataset_removed.T)
        feature_values, feature_vectors = np.linalg.eig(xTx)
        feature_values_ind = np.argsort(feature_values)
        self.transform_vectors = feature_vectors[:, feature_values_ind[-self.n_components:]]
        return np.dot(dataset_removed, self.transform_vectors)


pca = PCA(n_components=1)
pca.fit_transform(dataset)