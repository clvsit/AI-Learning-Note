import numpy as np
from scipy.spatial.distance import cdist


class MDS:
    
    def __init__(self):
        from scipy.spatial.distance import cdist
    
    def _cal_dist(self, matrix, row, col):
        if row == '*' and col == '*':
            return np.sum(matrix**2)
        elif row == '*' and col != '*':
            return np.sum(matrix[:, col]**2)
        elif row != '*' and col == '*':
            return np.sum(matrix[row, :]**2)
        else:
            return matrix[row, col]**2
    
    def fit(self, dataset):
        length = dataset.shape[0]
        # 计算原始空间的距离矩阵
        original_matrix = cdist(dataset, dataset, 'euclidean')
        # 计算 dist_i.、dist_.j 以及 dist_..
        dist_matrix = np.matrix(np.zeros(original_matrix.shape))
        rows, cols = dist_matrix.shape
        # 获得矩阵 B
        for row in range(rows):
            for col in range(cols):
                distij = self._cal_dist(original_matrix, row, col)
                dist_i = self._cal_dist(original_matrix, row, '*') / length
                dist_j = self._cal_dist(original_matrix, '*', col) / length
                dist_all = self._cal_dist(original_matrix, '*', '*') / (length**2)
                # print(distij, dist_i, dist_j, dist_all)
                dist_matrix[row, col] = -(distij - dist_i - dist_j + dist_all) / 2
        # 计算特征值和特征向量
        feature_values, feature_vectors = np.linalg.eig(dist_matrix)
        # print(feature_values)
        # print(feature_vectors)
        select_feature_values = []
        for i in range(len(feature_values) - 1, -1, -1):
            # print(np.round(feature_values[i]))
            if np.round(feature_values[i]) != 0:
                select_feature_values.append(feature_values[i])
            else:
                feature_vectors = np.delete(feature_vectors, i, axis=1)
        # print(select_feature_values)
        # print(feature_vectors)
        eye_matrix = np.eye(len(select_feature_values))
        for i in range(len(select_feature_values)):
            eye_matrix[i, i] = select_feature_values[i]
        return np.dot(feature_vectors, eye_matrix**0.5)


mds = MDS()
mds.fit(np.array([
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3]
]))

mds.fit(np.array([
    [-1, 1, 0],
    [-4, 3, 0],
    [1, 0, 2]
]))