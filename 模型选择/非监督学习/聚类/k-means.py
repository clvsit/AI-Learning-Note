import numpy as np

def load_dataset():
    return np.array([
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

class KMeans:
    
    def __init__(self, k, stop_criterion=True, max_iter=500, min_threshold=1):
        self.k = k
        self.stop_criterion = stop_criterion
        self.max_iter = max_iter
        self.iter = 0
        self.min_threshold = min_threshold
        self.data_cluster_info = None
    
    def _random_select(self, data, k):
        length = data.shape[0]
        step, start = length // k, numpy.random.randint(length)
        data_select = []
        for i in range(k):        
            data_select.append(data[start])
            start += step
            start = start % length
        return data_select
    
    def fit(self, data):
        # 初始化
        length = data.shape[0]
        # 随机从样本集中挑选 k 个样本向量作为簇的均值向量
        cluster_vectors = self._random_select(data, self.k)
        self.data_cluster_info = numpy.array(numpy.zeros((length, 2)))
        flag = True
        self.iter = 0

        while self._stop_criterion():
            cluster = []
            flag = False
            # 循环每一个样本，将其划分到相应的簇内
            for i in range(length):
                min_dist = numpy.inf
                cluster_index = -1
                for index in range(self.k):
                    dist = numpy.linalg.norm(data[i] - cluster_vectors[index], 2)**2
                    if dist < min_dist:
                        min_dist = dist
                        cluster_index = index
                self.data_cluster_info[i] = (cluster_index, min_dist)

            # 更新每个簇的均值向量
            for i in range(self.k):
                cluster_data = data[self.data_cluster_info[:, 0] == i]
                cluster.append(cluster_data)
                cluster_new = numpy.mean(cluster_data, axis=0)
                if (cluster_new != cluster_vectors[i]).all():
                    flag = True
                    cluster_vectors[i] = cluster_new
            
            self.iter += 1
            if not flag:
                break
        return cluster
    
    def _stop_criterion(self):
        # 最大运行轮数
        if self.stop_criterion:
            return True if self.iter <= self.max_iter else False
        # 最小调整幅度阈值
        else:
            if self.iter == 0:
                error = numpy.infty
            else:
                error = numpy.sum(self.data_cluster_info[1], axis=0)
            print(error)
            return True if error > self.min_threshold else False

import numpy
kmeans = KMeans(2, stop_criterion=True)
result = kmeans.fit(data_mat)
cluster_1 = result[0]
cluster_2 = result[1]

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(cluster_1[:, 0], cluster_1[:, 1], c='red')
ax.scatter(cluster_2[:, 0], cluster_2[:, 1], c='blue')
ax.set_xlabel('density')
ax.set_ylabel('sugar content')
plt.show()