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
    
    # 定义文本框和箭头格式
    decision_node = dict(boxstyle='sawtooth', fc='1')
    leaf_node = dict(boxstyle='round4', fc='1')
    arrow_args = dict(arrowstyle='<-')
    plot_tree_model = {}
    
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
    
    def _create_decision_tree_with_pre_pruning(self, dataset, validation):
        labels, features = set(dataset[:, -1]), dataset.shape[1] - 1
        if features == 0:
            return TreeNode(self._vote(dataset), type='leaf')
        if len(labels) == 1:
            return TreeNode(dataset[0, -1], type='leaf')
        best_feature = self._choose_best_feature(dataset)
        best_feature_values = set(dataset[:, best_feature])
        if not self.root:
            self.root = TreeNode(self._vote(dataset), type='leaf')
            if not self._pre_pruning(self.root, dataset, validation, best_feature):
                return
            node = TreeNode(best_feature, 'decision')
            self.root = node
            for best_feature_value in best_feature_values:
                split_feature_data = self._split_dataset(dataset, best_feature, best_feature_value)
                if split_feature_data.shape[0] == 0:
                    node.children[best_feature_value] = TreeNode(self._vote(dataset), type='leaf')
                else:
                    node.children[best_feature_value] = self._create_decision_tree_with_pre_pruning(split_feature_data, validation)
        else:
            node = TreeNode(self._vote(dataset), type='leaf')
            if not self._pre_pruning(node, dataset, validation, best_feature):
                return node
            node = TreeNode(best_feature, 'decision')
            for best_feature_value in best_feature_values:
                split_feature_data = self._split_dataset(dataset, best_feature, best_feature_value)
                if split_feature_data.shape[0] == 0:
                    node.children[best_feature_value] = TreeNode(self._vote(dataset), type='leaf')
                else:
                    node.children[best_feature_value] = self._create_decision_tree_with_pre_pruning(split_feature_data, validation)
        return node
    
    def _pre_pruning(self, node, dataset, validation, best_feature):
        v_data, v_labels = validation[:, 0:-1].tolist(), validation[:, -1].tolist()
        best_feature_values = set(dataset[:, best_feature])
        score_before = self.score(v_data, v_labels)
        temp_value = node.value
        node.value, node.type = best_feature, 'decision'
        for best_feature_value in best_feature_values:
            split_feature_data = self._split_dataset(dataset, best_feature, best_feature_value)
            node.children[best_feature_value] = TreeNode(self._vote(split_feature_data), type='leaf')
        score_after = self.score(v_data, v_labels)
        node.children = {}
        if score_after <= score_before:
            node.value, node.type = temp_value, 'leaf'
            return False 
        else: 
            return True
    
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
    
    def fit(self, dataset, validation):
        if self.pruning == 'pre':
            self._create_decision_tree_with_pre_pruning(dataset, validation)
        elif self.pruning == 'post':
            self.root = self._create_decision_tree(dataset)
            self._post_pruning(self.root, dataset, validation)
    
    def _iter_predict(self, dt, data):
        feature, node_type = dt.value, dt.type
        if node_type == 'leaf':
            return feature
        feature_value = data[feature]
        for child in dt.children:
            if child == feature_value:
                return self._iter_predict(dt.children[child], data[:feature] + data[feature+1:])
        
    def tree2dict(self):
        return self._pre_order_tree2dict(self.root)
        
    def _pre_order_tree2dict(self, node):
        if node.type == 'leaf':
            return node.value
        tree_dict = {}
        tree_dict[node.value] = {}
        for child in node.children:
            tree_dict[node.value][child] = self._pre_order_tree2dict(node.children[child])
        return tree_dict
    
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
    
    # 绘图代码
    
    def _get_num_leafs(self, node):
        """
        获取决策树的叶子节点数目
        """
        num_leafs = 0
        node_list = [node]
        while len(node_list) > 0:
            node = node_list.pop(0)
            if node.type == 'leaf':
                num_leafs += 1
            else:
                for child in node.children:
                    node_list.append(node.children[child])
        return num_leafs
    
    def _get_tree_depth(self, node):
        """
        获取决策树的深度
        """
        return self._pre_order_get_tree_depth(node, 0)
    
    def _pre_order_get_tree_depth(self, node, depth):
        if node.type == 'leaf':
            return depth
        depth_result = []
        for child in node.children:
            depth_result.append(self._pre_order_get_tree_depth(node.children[child], depth + 1))
        return max(depth_result)
    
    def _plot_node(self, node_txt, center_pt, parent_pt, node_type):
        # 绘制节点
        self.plot_tree_model['ax1'].annotate(node_txt, xy=parent_pt, xycoords='axes fraction', 
                            xytext=center_pt, textcoords='axes fraction', va='center', ha='center', 
                             bbox=node_type, arrowprops=self.arrow_args)
    
    def _plot_mid_text(self, cntr_pt, parent_pt, txt_string):
        # 绘制带箭头的注解
        x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
        y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
        self.plot_tree_model['ax1'].text(x_mid, y_mid, txt_string, fontdict={'size': 12})
        
    def _plot_tree(self, node, parent_pt, node_txt):
        # 计算树的宽和高
        num_leafs, depth = self._get_num_leafs(node), self._get_tree_depth(node)
        # 计算树节点的摆放位置，绘制在水平方向和垂直方向的中心位置
        cntr_pt = (self.plot_tree_model['xOff'] + (1.0 + float(num_leafs)) / 2.0 / self.plot_tree_model['totalW'], self.plot_tree_model['yOff'])
        # 进行文本和节点绘制
        self._plot_mid_text(cntr_pt, parent_pt, node_txt)
        self._plot_node(node.value, cntr_pt, parent_pt, self.decision_node)        
        # 按比例减少 plot_tree.yOff
        self.plot_tree_model['yOff'] = self.plot_tree_model['yOff'] - 1.0 / self.plot_tree_model['totalD']
        for child in node.children:
            if node.children[child].type == 'decision':
                self._plot_tree(node.children[child], cntr_pt, child)
            else:
                self.plot_tree_model['xOff'] = self.plot_tree_model['xOff'] + 1.0 / self.plot_tree_model['totalW']
                self._plot_node(node.children[child].value, (self.plot_tree_model['xOff'], self.plot_tree_model['yOff']), cntr_pt, self.leaf_node)
                self._plot_mid_text((self.plot_tree_model['xOff'], self.plot_tree_model['yOff']), cntr_pt, child)
        self.plot_tree_model['yOff'] = self.plot_tree_model['yOff'] + 1.0 / self.plot_tree_model['totalD']
        
    def create_plot(self):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.plot_tree_model['ax1'] = plt.subplot(111, frameon=False, **axprops)
        self.plot_tree_model['totalW'] = float(self._get_num_leafs(self.root))
        self.plot_tree_model['totalD'] = float(self._get_tree_depth(self.root))
        self.plot_tree_model['xOff'] = -0.5 / self.plot_tree_model['totalW']
        self.plot_tree_model['yOff'] = 1.0
        self._plot_tree(self.root, (0.5, 1.0), '')
        plt.show()


watermelon_train = pd.read_csv('data/watermelon_train.csv')
watermelon_validation = pd.read_csv('data/watermelon_validation.csv')

train_data = watermelon_train.values[:, 1:]
train_labels = watermelon_train.values[:, -1]
validation_data = watermelon_validation.values[:, 1:]
validation_labels = watermelon_validation.values[:, -1]

model = DecisionTreeClassifier(pruning='pre')
model.fit(train_data, validation_data)
model.predict(validation_data.tolist())
model.score(validation_data.tolist(), validation_labels.tolist())
model.create_plot()