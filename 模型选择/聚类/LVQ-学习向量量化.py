import numpy
from matplotlib import pyplot as plt

def load_dataset():
    return numpy.array([
        [0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0],
        [0.359, 0.188, 0],
        [0.339, 0.241, 0],
        [0.282, 0.257, 0],
        [0.748, 0.232, 0],
        [0.714, 0.346, 1],
        [0.483, 0.312, 1],
        [0.478, 0.437, 1],
        [0.525, 0.369, 1],
        [0.751, 0.489, 1],
        [0.532, 0.472, 1],
        [0.473, 0.376, 1],
        [0.725, 0.445, 1],
        [0.446, 0.459, 1]
    ])

data_mat = load_dataset()

class LVQ:
    
    def __init__(self, k, n_iters=100, learning_rate=1):
        self.k = k
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.prototypes = None
    
    def _random_select(self, data):
        length = data.shape[0]
        step, start = length // self.k, numpy.random.randint(length)
        data_select = []
        for i in range(self.k):        
            data_select.append(data[start])
            start += step
            start = start % length
        return numpy.array(data_select)
    
    def fit(self, dataset):
        length = dataset.shape[0]
        prototypes = self._random_select(dataset)

        for i in range(self.n_iters):        
            data = dataset[numpy.random.randint(length)]
            min_dist = numpy.inf
            cluster_index = -1
            for j in range(self.k):
                dist = numpy.linalg.norm(data[:2] - prototypes[j, :2], 2)**2
                if dist < min_dist:
                    min_dist = dist
                    cluster_index = j
            if data[2] == prototypes[cluster_index, 2]:
                prototypes[cluster_index, :2] += self.learning_rate * (data[:2] - prototypes[cluster_index, :2])
            else:
                prototypes[cluster_index, :2] -= self.learning_rate * (data[:2] - prototypes[cluster_index, :2])
        self.prototypes = prototypes
        return prototypes

lvq = LVQ(5, n_iters=200, learning_rate=0.5)
prototypes = lvq.fit(data_mat)
print(prototypes)