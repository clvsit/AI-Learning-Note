%matplotlib inline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class TreeNode:
    
    def __init__(self, value, type='decision'):
        self.value = value
        self.type = type
        self.children = {}


class DecisionTreeClassifier:
    
    def __init__(self, feature_select='gain', pruning='pre'):
        feature_select_list = ['gain', 'gain_ratio']
        if feature_select not in feature_select_list:
            raise Exception('The param feature_select is invalid!')
        pruning_list = ['pre', 'post']
        if pruning not in pruning_list:
            raise Exception('The param pruning is invalid!')
        self.root = None
        self.feature_select = feature_select
        self.pruning = pruning
    
    def set_feature_select(self, feature_select):
        self.feature_select = feature_select
    
    def _cal_shannon_entropy(self, dataset):
        labels, length = set(dataset[:, -1]), dataset.shape[0]
        shannon_sum = 0
        for label in labels:
            p_label = dataset[dataset[:, -1] == label].shape[0] / length
            shannon_sum += -p_label * np.log(p_label)
        return shannon_sum
    
    def _split_dataset(self, dataset, feature, feature_value):
        split_data = dataset[dataset[:, feature] == feature_value]
        return np.delete(split_data, feature, axis=1)
    
    def _choose_best_feature(self, dataset):
        length = dataset.shape[0]
        features = dataset.shape[1] - 1
        base_shannon = self._cal_shannon_entropy(dataset)
        split_shannon = []
        for feature in range(features):
            feature_values = set(dataset[:, feature])
            shannon = 0
            ratio = 0
            for feature_value in feature_values:
                dataset_feature = self._split_dataset(dataset, feature, feature_value)
                dataset_feature_p = (dataset_feature.shape[0] / length)
                shannon += dataset_feature_p * self._cal_shannon_entropy(dataset_feature)
                ratio += -dataset_feature_p * np.log(dataset_feature_p)
            if self.feature_select == 'gain':
                split_shannon.append(base_shannon - shannon)
            elif self.feature_select == 'gain_ratio':
                split_shannon.append((base_shannon - shannon) / ratio)
        best_feature = np.argmax(split_shannon)
        return best_feature
    
    def _vote(self, dataset):
        labels = set(dataset[:, -1])
        label_max, label_max_count = 0, 0
        for label in labels:
            label_count = dataset[dataset[:, -1] == label].shape[0]
            if label_count > label_max_count:
                label_max, label_max_count = label, label_count
        return label_max
    
    def _create_decision_tree(self, dataset):
        labels, features = set(dataset[:, -1]), dataset.shape[1] - 1
        if features == 0:
            return TreeNode(self._vote(dataset), type='leaf')
        if len(labels) == 1:
            return TreeNode(dataset[0, -1], type='leaf')
        best_feature = self._choose_best_feature(dataset)
        best_feature_values = set(dataset[:, best_feature])
        node = TreeNode(best_feature, 'decision')
        for best_feature_value in best_feature_values:
            split_feature_data = self._split_dataset(dataset, best_feature, best_feature_value)
            if split_feature_data.shape[0] == 0:
                node.children[best_feature_value] = TreeNode(self._vote(dataset), type='leaf')
            else:
                node.children[best_feature_value] = self._create_decision_tree(split_feature_data)
        return node
    
    def _post_pruning(self, node, dataset, validation):
        for child in node.children:
            if node.children[child].type != 'leaf':
                self._post_pruning(node.children[child], self._split_dataset(dataset, node.value, child), validation)
        v_data = validation[:, 0:-1].tolist()
        v_labels = validation[:, -1].tolist()        
        score_before = self.score(v_data, v_labels)
        temp_value, temp_children = node.value, node.children.copy()
        node.children = {}
        score_after = self.score(v_data, v_labels)
        if score_after < score_before:
            node.children = temp_children
        else:
            node.value = self._vote(dataset)
            node.type = 'leaf'

    
    def _iter_predict(self, dt, data):
        feature, node_type = dt.value, dt.type
        if node_type == 'leaf':
            return feature
        feature_value = data[feature]
        for child in dt.children:
            if child == feature_value:
                return self._iter_predict(dt.children[child], data[:feature] + data[feature+1:])
    
    def fit(self, dataset, validation):
        self.root = self._create_decision_tree(dataset)
        if self.pruning == 'post':
            self._post_pruning(self.root, dataset, validation)
        
    def predict(self, dataset):
        result = []
        for data in dataset:
            result.append(self._iter_predict(self.root, data))
        return result
    
    def score(self, dataset, labels):
        length = len(labels)
        if length != len(dataset):
            raise Exception('The count of  dataset is not equal to the count of labels!')
        predicted_labels = self.predict(dataset)
        true_count = 0
        for i in range(length):
            if predicted_labels[i] == labels[i]:
                true_count += 1
        return true_count / length


watermelon_train = pd.read_csv('data/watermelon_train.csv')
watermelon_validation = pd.read_csv('data/watermelon_validation.csv')

train_data = watermelon_train.values[:, 1:]
train_labels = watermelon_train.values[:, -1]
validation_data = watermelon_validation.values[:, 1:]
validation_labels = watermelon_validation.values[:, -1]

model = DecisionTreeClassifier(pruning='post')
model.fit(train_data, validation_data)
model.predict(validation_data.tolist())
model.score(validation_data.tolist(), validation_labels.tolist())