%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt


def load_dataset(filename):
    data_mat = []
    with open(filename) as file:
        for line in file.readlines():
            cur_line = line.strip().split('\t')
            flt_line = list(map(float, cur_line))
            data_mat.append(flt_line)
    return data_mat


class CART():
    
    def __init__(self, leaf_type='default', err_type='default', ops=(1, 4)):
        if leaf_type == 'default':
            self.leaf_type_ = self._reg_leaf
        if err_type == 'default':
            self.err_type_ = self._reg_err
        self.ops_ = ops
    
    def _bin_split_dataset(self, dataset, feature, value):
        mat_0 = dataset[np.nonzero(dataset[:, feature] > value)[0], :]
        mat_1 = dataset[np.nonzero(dataset[:, feature] <= value)[0], :]
        return mat_0, mat_1
    
    def _reg_leaf(self, dataset):
        return np.mean(dataset[:, -1])
    
    def _reg_err(self, dataset):
        return np.var(dataset[:, -1]) * np.shape(dataset)[0]
    
    def _choose_best_split(self, dataset):
        tol_s, tol_n = self.ops_
        # 如果所有标签的值都相等，则无须切分可直接退出
        if dataset[:, -1].shape[0] == 1:
            return None
        m, n = np.shape(dataset)
        s = self.err_type_(dataset)
        best_s, best_index, best_value = np.inf, 0, 0
        for feat_index in range(n - 1):
            for split_val in set(dataset[:, feat_index].T.tolist()[0]):            
                mat_0, mat_1 = self._bin_split_dataset(dataset, feat_index, split_val)
                if (np.shape(mat_0)[0] < tol_n) or (np.shape(mat_1)[0] < tol_n):
                    continue
                new_s = self.err_type_(mat_0) + self.err_type_(mat_1)
                if new_s < best_s:
                    best_index = feat_index
                    best_value = split_val
                    best_s = new_s
        # 如果误差减少不大则退出
        if (s - best_s) < tol_s:
            return None, self.leaf_type_(dataset)
        mat_0, mat_1 = self._bin_split_dataset(dataset, best_index, best_value)
        # 如果切分出的数据集很小则退出
        if (np.shape(mat_0)[0] < tol_n) or (np.shape(mat_1)[0] < tol_n):
            return None, self.leaf_type_(dataset)
        return best_index, best_value
    
    def create_tree(self, dataset):
        feat, val = self._choose_best_split(dataset)
        if feat == None:
            return val
        ret_tree = {}
        ret_tree['spInd'] = feat
        ret_tree['spVal'] = val
        lset, rset = self._bin_split_dataset(dataset, feat, val)
        ret_tree['left'] = self.create_tree(lset)
        ret_tree['right'] = self.create_tree(rset)
        return ret_tree
    
    def _is_tree(self, obj):
        return type(obj).__name__ == 'dict'
    
    def _get_mean(self, tree):
        if self._is_tree(tree['right']):
            tree['right'] = self._get_mean(tree['right'])
        if self._is_tree(tree['left']):
            tree['left'] = self._get_mean(tree['left'])
        return (tree['left'] + tree['right']) / 2.0
    
    def prune(self, tree, test_data):
        # 没有测试数据则对树进行塌陷处理
        if np.shape(test_data)[0] == 0:
            return self._get_mean(tree)
        if (self._is_tree(tree['left'])) or (self._is_tree(tree['right'])):
            lset, rset = self._bin_split_dataset(test_data, tree['spInd'], tree['spVal'])
        if self._is_tree(tree['left']):
            tree['left'] = self.prune(tree['left'], lset)
        if self._is_tree(tree['right']):
            tree['right'] = self.prune(tree['right'], rset)
        if not self._is_tree(tree['left']) and not self._is_tree(tree['right']):
            lset, rset = self._bin_split_dataset(test_data, tree['spInd'], tree['spVal'])
            error_no_merge = np.sum(np.power(lset[:, -1] - tree['left'], 2)) + np.sum(np.power(rset[:, -1] - tree['right'], 2))
            tree_mean = (tree['left'] + tree['right']) / 2.0
            error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
            if error_merge < error_no_merge:
                print('merging')
                return tree_mean
            else:
                return tree
        else:
            return tree


my_dat = load_dataset('data/ex00.txt')
my_dat = np.mat(my_dat)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(my_dat[:, 0].tolist(), my_dat[:, 1].tolist())
ax.set_title('ex00.txt dataset')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

model = CART()
model.create_tree(my_dat)

my_dat2 = load_dataset('data/ex2.txt')
my_dat2 = np.mat(my_dat2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(my_dat2[:, 0].tolist(), my_dat2[:, 1].tolist())
ax.set_title('ex2.txt dataset')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

model.prune(tree, my_dat2_test)