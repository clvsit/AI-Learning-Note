import numpy as np


dataset = np.array([
    ['A', ''], ['B', '香蕉'], ['AB', '苹果'], ['O', '西瓜']
])
dataset2 = np.array([
    [2, 1], [3, 2], [1, 4], [5, 3], [3, 5]
])


class Preprocessing:
    
    def __init__(self):
        pass
    
    def _one_hot_encoding(self, dataset):
        length, num_feature = dataset.shape
        result = []
        for feature in range(num_feature):
            num_values = len(np.unique(dataset[:, feature]))
            result_feature = []
            one_hot_dict = {}
            one_hot_index = 0
            for i in range(length):
                data = dataset[i, feature]                
                if data not in one_hot_dict:
                    vector = [0] * num_values
                    vector[one_hot_index] = 1
                    one_hot_dict[data] = vector
                    one_hot_index += 1
                result_feature.append(one_hot_dict[data])                
            result.append(np.array(result_feature))
        return np.array(result)
    
    def _binary_encoding(self, dataset):
        length, num_feature = dataset.shape
        result = []
        for feature in range(num_feature):
            num_values = len(np.unique(dataset[:, feature]))
            vector_length = len(self._num2binMatrix(num_values))
            result_feature = []
            binary_dict = {}
            binary_index = 1
            for i in range(length):
                data = dataset[i, feature]                
                if data not in binary_dict:                                    
                    binary_dict[data] = self._num2binMatrix(binary_index, vector_length)
                    binary_index += 1                
                result_feature.append(binary_dict[data])                
            result.append(np.array(result_feature))
        return np.array(result)
    
    def _num2binMatrix(self, num, length=0):        
        bin_matrix = list(map(lambda x:int(x), list(bin(num).replace('0b', ''))))        
        if length != 0:
            vector = [0] * length
            diff = length - len(bin_matrix)
            for i in range(diff, length):
                vector[i] = bin_matrix[i - diff]
        else:
            vector = bin_matrix
        return vector
    
    def categorical2numerical(self, dataset, type='oneHot'):
        if type == 'oneHot':
            return self._one_hot_encoding(dataset)
        else:
            return self._binary_encoding(dataset)

    def min_max_scaling(self, dataset):
        dataset_min, dataset_max = np.min(dataset, axis=0), np.max(dataset, axis=0)
        dataset_range = dataset_max - dataset_min
        return (dataset - dataset_min) / dataset_range
    
    def standard_scaling(self, dataset):
        dataset_mean = np.mean(dataset, axis=0)
        dataset_var = np.sum((dataset - np.mean(dataset, axis=0))**2 / len(dataset), axis=0)**0.5
        return (dataset - dataset_mean) / dataset_var


p_test = Preprocessing()
p_test.categorical2numerical(dataset[:,:], 'binary')