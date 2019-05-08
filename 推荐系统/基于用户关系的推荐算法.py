%matplotlib inline
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Recommand(object):
    
    def __init__(self, method='out'):
        method_list = ['out', 'in', 'out-in']
        if method not in method_list:
            raise Exception('The parameter method is invalid!')
        self.method_ = method
        self.user_similarity_ = None
    
    def _build_relation_graph(self, dataset, user_list):
        user_focus_dict, user_fans_dict = {}, {}
        for user in user_list:
            user_focus_dict[user] = []
            user_fans_dict[user] = []
        for data in dataset:            
            user_focus_dict[data[0]].append(data[1])
            user_fans_dict[data[1]].append(data[0])
        for user in user_focus_dict:
            user_focus_dict[user] = frozenset(user_focus_dict[user])
        for user in user_fans_dict:
            user_fans_dict[user] = frozenset(user_fans_dict[user])
        return user_focus_dict, user_fans_dict
    
    def _get_similarity_matrix(self, user_focus_dict, user_fans_dict, user_list):
        user_num = len(user_list)
        user_similarity = {}
        for i in range(user_num):
            user_A = user_list[i]
            user_similarity[user_A] = {}
            for j in range(user_num):
                if i == j:
                    continue
                user_B = user_list[j]
                if self.method_ == 'out':
                    user_similarity[user_A][user_B] = len(set(user_focus_dict[user_A]) & set(user_focus_dict[user_B])) / \
                    np.sqrt((len(user_focus_dict[user_A]) + 1) * (len(user_focus_dict[user_B]) + 1))
                elif self.method_ == 'in':
                    user_similarity[user_A][user_B] = len(set(user_fans_dict[user_A]) & set(user_fans_dict[user_B])) / \
                    np.sqrt((len(user_fans_dict[user_A]) + 1) * (len(user_fans_dict[user_B]) + 1))
                elif self.method_ == 'out-in':
                    user_similarity[user_A][user_B] = len(set(user_focus_dict[user_A]) & set(user_fans_dict[user_B])) / \
                    np.sqrt((len(user_focus_dict[user_A]) + 1) * (len(user_fans_dict[user_B]) + 1))                
        self.user_similarity_ = user_similarity
    
    def fit(self, dataset, user_list=None):
        user_all_list = sorted(list(set(dataset[:, 0]) | set(dataset[:, 1])))
        user_list = user_list if user_list else user_all_list
        user_focus_dict, user_fans_dict = self._build_relation_graph(dataset, user_all_list)
        self._get_similarity_matrix(user_focus_dict, user_fans_dict, user_list)
    
    def _format_records(self, records):
        item_list = sorted(list(set(records[:, 1])))
        item_record = {}
        for item in item_list:
            item_record[item] = {}
        user_record = {}
        for record in records:
            if record[0] not in user_record:
                user_record[record[0]] = {}
            item_record[record[1]][record[0]] = record[2]
            user_record[record[0]][record[1]] = record[2]
        return item_record, user_record
    
    def _choose_similarity_user(self, user, k):
        simi_users = self.user_similarity_[user]
        rank_users = []
        for user in simi_users:
            rank_users.append((user, simi_users[user]))
        rank_users.sort(key=lambda x:x[1], reverse=True)
        return rank_users[:k]
        
    def recommand(self, dataset, user, user_num=3, item_num=3, is_score=False):
        print('正在规范化数据集......')
        item_record, user_record = self._format_records(dataset)
        print(item_record, user_record)
        print('规范完成')
        print('获取相似的用户集合')
        simi_users = self._choose_similarity_user(user, user_num)
        print('获取相似用户集合完成')
        print('开始进行物品推荐')
        # 去除已经购买过的物品
        item_list = list(set(dataset[:, 1]) - set(user_record[user]))
        score = {}
        for item in item_list:
            score[item] = 0
            item_score = item_record[item]
            # print(item_score)
            for user_info in simi_users:
                user = user_info[0]
                if user in item_score:
                    # print('score:', item_record[item][user])
                    # print('similarity:', user_info[1])
                    score[item] += item_record[item][user] * user_info[1]
        rank_score = []
        for item in score:
            rank_score.append((item, score[item]))
        rank_score.sort(key=lambda x:x[1], reverse=True)
        return rank_score[:item_num]


records = pd.read_csv('data/purchase_records.csv')
focus = pd.read_csv('data/focus.csv')
model = Recommand('out')
model.fit(focus.values)
model.recommand(records.values, 'A', item_num=1)
model.user_similarity_['A']