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
        labels_length, features = len(labels), dataset.shape[1]
        datasets = np.hstack((dataset, labels.reshape(labels_length, 1)))
        
        p_labels = {}
        p_conditions = {}
        for label in set(labels):
            label_length = labels[labels == label].shape[0]
            p_labels[label] = label_length / labels_length
            dataset_label = datasets[datasets[:, 3] == label]
            p_conditions[label] = {}
            condition_label = p_conditions[label]
            for feature in range(features):
                condition_label[feature] = {}
                condition_feature = condition_label[feature]
                for feature_value in set(dataset[:, feature]):
                    condition_feature[feature_value] = dataset_label[dataset_label[:, feature] == feature_value].shape[0] / label_length        
        self.p_labels = p_labels
        self.p_conditions = p_conditions
    
    def predict(self, dataset):
        result = []
        for data in dataset:
            p = []
            for label in p_labels:
                p_label = self.p_labels[label]
                condition_label = self.p_conditions[label]
                feature_index = 0
                p_condition = 1
                for feature_value in data:
                    p_condition *= condition_label[feature_index][feature_value]
                    feature_index += 1
                p.append(p_label * p_condition)
            p = np.array(p)
            result.append(np.argmax(p))
        return result

bayes = Bayes()
bayes.fit(x, y)
print(bayes.predict([[1, 1, 1], [0, 1, 1]]))