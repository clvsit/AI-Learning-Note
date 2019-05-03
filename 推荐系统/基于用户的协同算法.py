%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt


class Recommand:
    
    def __init__(self):
        self.items_dict_ = None
        self.user_dict_ = None
        self.simi_matrix_ = None
    
    def _convert2Items(self, dataset):
        items_dict = {}
        users_list = []
        for data in dataset:
            user, records = data['user'], data['record']
            users_list.append(user)
            for record in records:
                if record not in items_dict:
                    items_dict[record] = []
                items_dict[record].append(user)
        return items_dict, sorted(users_list)
    
    def _get_user_purchased_count(self, records):
        user_purchased_dict = {}
        for record in records:
            user_purchased_dict[record['user']] = len(record['record'])
        return user_purchased_dict
    
    def _generate_user_matrix(self, items_dict, users_list):
        user_num = len(users_list)
        # 初始化用户矩阵
        user_matrix = np.mat(np.zeros((user_num, user_num)))
        # 建立用户与下标的对应关系
        user_dict, ind = {}, 0
        for user in users_list:
            user_dict[user] = ind
            user_dict[ind] = user
            ind += 1
        # 构建用户矩阵
        for item in items_dict:
            user_list = items_dict[item]
            user_list_length = len(user_list)
            for i in range(user_list_length - 1):            
                user_A_ind = user_dict[user_list[i]]
                for j in range(i + 1, user_list_length):                
                    user_B_ind = user_dict[user_list[j]]
                    user_matrix[user_A_ind, user_B_ind] += 1
                    user_matrix[user_B_ind, user_A_ind] += 1
        self.user_dict_ = user_dict
        return user_matrix
    
    def _user_similarity(self, user_matrix, user_purchased_dict):
        user_num = len(user_purchased_dict)
        # 初始化用户相似度矩阵
        simi_matrix = np.mat(np.zeros((user_num, user_num)))
        # 计算每对用户的相似度
        for user_A, user_A_num in user_purchased_dict.items():
            user_A_ind = self.user_dict_[user_A]
            other_users = list(user_purchased_dict.keys())
            for user_B in other_users:
                user_B_ind = self.user_dict_[user_B]
                similarity = user_matrix[user_A_ind, user_B_ind] / np.sqrt(user_purchased_dict[user_A] * user_purchased_dict[user_B])
                simi_matrix[user_A_ind, user_B_ind] = similarity
                simi_matrix[user_B_ind, user_A_ind] = similarity
        self.simi_matrix_ = simi_matrix
    
    def fit(self, dataset):
        items_dict, user_lict = self._convert2Items(dataset)
        self.items_dict_ = items_dict
        user_purchased_dict = self._get_user_purchased_count(dataset)
        user_matrix = self._generate_user_matrix(items_dict, user_lict)
        self._user_similarity(user_matrix, user_purchased_dict)
        
    def get_user_similarity(self, user, user_num):
        simi_user_list = self.simi_matrix_[user_dict[user]]
        user_ind_list = np.argsort(simi_user_list).tolist()[0][-user_num:]
        user_list = []
        for user_ind in user_ind_list:
            user_list.append(self.user_dict_[user_ind])
        return user_list
    
    def cal_score(self, user, user_num, records, user_score):
        simi_user_list = set(self.get_user_similarity(user, user_num,))
        score_dict = {}
        for good in records:
            user_list = set(self.items_dict_[good])
            user_set = simi_user_list & user_list
            if good not in score_dict:
                score_dict[good] = 0
            for simi_user in user_set:
                similarity = self.simi_matrix_[user_dict[user], self.user_dict_[simi_user]]
                score_dict[good] += similarity * user_score[simi_user][good]
        return score_dict


dataset = np.array([
    {'user': 'A', 'record': [1, 2, 4]},
    {'user': 'B', 'record': [2, 4]},
    {'user': 'C', 'record': [1, 2, 5]},
    {'user': 'D', 'record': [2, 3]}
])
records = np.array([1, 2, 3])
user_score = {
    'A': {1: 4, 2: 3, 4: 4},
    'B': {2: 3, 4: 5},
    'C': {1: 5, 2: 3, 5: 3},
    'D': {2: 4, 3: 2}
}

model = Recommand()
model.fit(dataset)
model.get_user_similarity('C', 2)
model.cal_score('C', 2, records, user_score)