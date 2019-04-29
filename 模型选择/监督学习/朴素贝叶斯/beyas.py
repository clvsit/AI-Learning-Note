import numpy as np
import pandas as pd

df_data = pd.read_csv('dataset.csv')

# 数据预处理
dataset = df_data.values
dataset_985 = [1 if i == 'Yes' else 0 for i in dataset[:, 0]]
dataset_skill = [1 if i == 'C++' else 0 for i in dataset[:, 2]]
dataset_enroll = [1 if i == 'Yes' else 0 for i in dataset[:, 3]]
dataset_degree = []
for data in dataset[:, 1]:
    if data == '本科':
        dataset_degree.append(1)
    elif data == '硕士':
        dataset_degree.append(2)
    else:
        dataset_degree.append(3)
df_dataset_wished = pd.DataFrame(data={
    '985': dataset_985, 
    'degree': dataset_degree, 
    'skill': dataset_skill, 
    'enroll': dataset_enroll
})

dataset_wished = df_dataset_wished.values
x = dataset_wished[:, 0:3]
y = dataset_wished[:, 3]

class Bayes:
    
    def __init__(self):
        pass
    
    def fit(self, dataset, labels):
        labels_count, feature_count = len(labels), dataset.shape[1]
        datasets = np.hstack((dataset, labels.reshape((labels_count, 1))))
        p_labels, p_conditions = {}, {}
        # 类别数
        labels_type_count = len(set(labels))
        
        for label in set(labels):            
            dataset_label = datasets[datasets[:, 3] == label]
            dataset_label_length = dataset_label.shape[0]
            # 计算先验概率
            p_label = (dataset_label_length + 1) / (labels_count + labels_type_count)
            p_labels[label] = p_label
            p_conditions[label] = {}
            
            # 计算条件概率
            for feature in range(feature_count):
                p_conditions[label][feature] = {}
                dataset_label_feature = dataset_label[:, feature]
                # 当前特征的取值数
                feature_value_count = len(set(dataset_label_feature))
                for feature_value in set(dataset_label_feature):
                    p_conditions[label][feature][feature_value] = (dataset_label_feature[dataset_label_feature==feature_value].shape[0] + 1) / (dataset_label_length + feature_value_count)
        self.p_labels_ = p_labels
        self.p_conditions_ = p_conditions
    
    def predict(self, dataset):
        result = []
        
        for data in dataset:
            p = []
            for label in set(self.p_labels_):
                p_label = np.log(self.p_labels_[label])
                p_condition = 0
                feature_index = 0
                for feature_value in data:
                    p_condition += np.log(self.p_conditions_[label][feature_index][feature_value])
                    feature_index += 1
                p.append(p_label + p_condition)
            result.append(np.argmax(p))
        return result

bayes = Bayes()
bayes.fit(x, y)
print(bayes.predict([[1, 1, 1], [0, 1, 1]]))