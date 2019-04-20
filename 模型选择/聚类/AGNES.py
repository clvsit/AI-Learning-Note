import numpy
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

def load_dataset():
    return numpy.array([
        [0.697, 0.460],
        [0.774, 0.376],
        [0.634, 0.264],
        [0.608, 0.318],
        [0.556, 0.215],
        [0.403, 0.237],
        [0.481, 0.149],
        [0.437, 0.211],
        [0.666, 0.091],
        [0.243, 0.267],
        [0.245, 0.057],
        [0.343, 0.099],
        [0.639, 0.161],
        [0.657, 0.198],
        [0.360, 0.370],
        [0.593, 0.042],
        [0.719, 0.103],
        [0.359, 0.188],
        [0.339, 0.241],
        [0.282, 0.257],
        [0.748, 0.232],
        [0.714, 0.346],
        [0.483, 0.312],
        [0.478, 0.437],
        [0.525, 0.369],
        [0.751, 0.489],
        [0.532, 0.472],
        [0.473, 0.376],
        [0.725, 0.445],
        [0.446, 0.459]
    ])

data_mat = load_dataset()

class AGNES:
    
    def __init__(self, k, dist_method='avg'):
        self.k = k
        self.dist_method = dist_method
        
    def _get_dist(self, XA, XB, type='min'):
        if len(XA.shape) == 1:
            XA = numpy.array([XA])
        if len(XB.shape) == 1:
            XB = numpy.array([XB])
        dist = 0
        if type == 'min':
            dist = cdist(XA, XB, 'euclidean').min()
        elif type == 'max':
            dist = cdist(XA, XB, 'euclidean').max()
        else:
            dist = cdist(XA, XB, 'euclidean').sum() / XA.shape[0] / XB.shape[0]
        return dist
    
    def fit(self, dataset):
        length = dataset.shape[0]
        clusters, dist_matrix = [], numpy.mat(numpy.zeros((length, length)))
        
        for data in dataset:
            clusters.append(data)
            
        for i in range(length):
            for j in range(length):
                if i == j:
                    dist = numpy.inf
                else:
                    dist = self._get_dist(clusters[i], clusters[j], self.dist_method)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # 设置当前聚类簇的个数
        cluster_count = length
        
        while cluster_count > self.k:
            # 找出距离最近的两个聚类簇
            first, second = numpy.where(dist_matrix == dist_matrix.min())[0]
            # 合并这两个聚类簇
            clusters[first] = numpy.vstack((clusters[first], clusters[second]))
            # 重新编号聚类簇
            for i in range(second + 1, cluster_count):
                clusters[i - 1] = clusters[i]
            clusters.pop()
            # 删除距离矩阵的第 second 行与列
            dist_matrix = numpy.delete(dist_matrix, second, axis=0)
            dist_matrix = numpy.delete(dist_matrix, second, axis=1)        
            # 重新计算距离矩阵第 first 簇与其他簇之间距离
            for i in range(cluster_count - 1):
                if first == i:
                    dist = numpy.inf
                else:
                    dist = self._get_dist(clusters[first], clusters[i], self.dist_method)
                dist_matrix[first, i] = dist
                dist_matrix[i, first] = dist
            cluster_count -= 1
        return clusters

agnes = AGNES(7, dist_method='max')
agnes.fit(data_mat)