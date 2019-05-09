%matplotlib inline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gensim.models import Word2Vec


class Node(object):
    
    def __init__(self, val):
        self.value = val
        self.neighbors = {}
        
    def __str__(self):
        return self.value


class Recommand(object):
    
    def __init__(self, iter_count=5, step=5, back=0.5, forward=0.5):
        self.iter_count_ = iter_count
        self.step_ = step
        self.back_ = back
        self.forward_ = forward
        self.model_ = None
    
    def _build_neighbors_table(self, dataset):
        header_table = {}
        for data in dataset:            
            user, focus = data[0], data[1]
            if user not in header_table:
                node_user = Node(user)
                header_table[user] = Node(user)
            if focus not in header_table:
                node_focus = Node(focus)
                header_table[focus] = node_focus
            header_table[user].neighbors[focus] = header_table[focus]
        return header_table
    
    def _random_choose(self, neighbors, node_cur, node_last):
        if node_last is None:
            random = int(np.ceil(np.random.random() * len(node_cur.neighbors)))
            ind = 1
            for node_user in neighbors:
                if ind == random:
                    return neighbors[node_user]
                ind += 1
        prob = {}
        for node_user in neighbors:
            node = neighbors[node_user]
            if node == node_last:
                prob[node] = 1 / self.back_
            elif node in node_last.neighbors or node_last in node.neighbors:
                prob[node] = 1
            else:
                prob[node] = 1 / self.forward_

        total = 0
        for key in prob:
            total += prob[key]
        random = np.random.random() * total
        total_prob = 0
        for key in prob:
            total_prob += prob[key]
            if total_prob > random:
                return key
    
    def _random_walk(self, header_table): 
        path = []
        for user in header_table:
            for i in range(self.iter_count_):
                node_last = None
                node_cur = header_table[user]            
                path_iter = [node_cur.value]
                for j in range(self.step_):
                    neighbors = node_cur.neighbors
                    if len(neighbors) == 0:
                        break
                    node_next = self._random_choose(neighbors, node_cur, node_last)
                    path_iter.append(node_next.value)
                    node_last = node_cur
                    node_cur = node_next
                path.append(path_iter)
        return path
    
    def fit(self, dataset):
        path = self._random_walk(self._build_neighbors_table(dataset))
        self.model_ = Word2Vec(path, min_count=2)
    
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
    
    def _choose_similarity_user(self, user, user_list, user_num):
        similarity_user = []
        for user_ in user_list:
            if user != user_:
                similarity_user.append((user_, self.model_.wv.similarity(user, user_)))
        return sorted(similarity_user, key=lambda x:x[1], reverse=True)[:user_num]
        
    def recommand(self, dataset, user, user_num=3, item_num=3, is_score=False):
        print('正在规范化数据集......')
        item_record, user_record = self._format_records(dataset)
        print(item_record, user_record)
        print('规范完成')
        print('获取相似的用户集合')
        simi_users = self._choose_similarity_user(user, list(user_record.keys()), user_num)
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


focus = pd.read_csv('data/focus.csv')
focus_dataset = focus.values
records = pd.read_csv('data/purchase_records.csv')

model = Recommand()
model.fit(focus_dataset)
model.recommand(records.values, 'A')