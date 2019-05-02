%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt


class Recommand:
    
    def __init__(self, alpha=0.3):
        self.alpha_ = alpha
        self.good_dict_ = None
        self.simi_matrix_ = None
    
    def _get_matrix(self, records):
        # 获取共有多少种不同的商品
        goods = []
        for record in records:
            for good_info in record:
                good = list(good_info)[0]
                if good not in goods:
                    goods.append(good)
        # 对商品进行排序
        goods.sort()
        # 初始化共现矩阵
        goods_num = len(goods)
        matrix = np.mat(np.zeros((goods_num, goods_num)))
        # 建立商品与矩阵坐标的映射
        good_dict, index = {}, 0
        for good in goods:
            good_dict[good] = index
            index += 1
        # 填充共现矩阵
        for record in records:
            goods_length = len(record)
            for i in range(goods_length - 1):
                good_A_info = record[i]
                good_A_ind, good_A_grade = good_dict[list(good_A_info.keys())[0]], list(good_A_info.values())[0]
                for j in range(i + 1, goods_length):
                    good_B_info = record[j]
                    good_B_ind, good_B_grade = good_dict[list(good_B_info.keys())[0]], list(good_B_info.values())[0]
                    matrix[good_A_ind, good_B_ind] += good_A_grade * good_B_grade
                    matrix[good_B_ind, good_A_ind] += good_A_grade * good_B_grade
        return matrix, good_dict
    
    def _get_good_grade(self, records):
        # 获取共有多少种不同的商品
        goods = []
        for record in records:
            for good_info in record:
                good = list(good_info)[0]
                if good not in goods:
                    goods.append(good)
        # 对商品进行排序
        goods.sort()
        # 初始化商品评分字典以及商品购买次数字典
        good_grade_dict = {}
        good_count_dict = {}
        for good in goods:
            good_grade_dict[good] = 0
            good_count_dict[good] = 0
        # 开始统计商品的评分
        for record in records:
            for good_info in record:            
                good, grade = list(good_info.keys())[0], list(good_info.values())[0]
                good_grade_dict[good] += grade ** 2
                good_count_dict[good] += 1
        return good_grade_dict
    
    def _good_similarity(self, occu_matrix, good_grade_dict):
        # 获取商品种类以及初始化相似度矩阵
        goods = list(good_grade_dict.keys())
        goods_num = len(good_grade_dict)
        simi_matrix = np.mat(np.zeros((goods_num, goods_num)))
        # 开始计算商品之间的相似度
        for i in range(goods_num - 1):
            good_A = goods[i]
            good_A_ind = self.good_dict_[good_A]
            for j in range(i + 1, goods_num):
                good_B = goods[j]
                good_B_ind = self.good_dict_[good_B]
                # 计算相似度
                good_A_count, good_B_count = good_grade_dict[good_A], good_grade_dict[good_B]
                if good_A_count > good_B_count:
                    similarity = occu_matrix[good_A_ind, good_B_ind] / (good_A_count**(1 - self.alpha_) * good_B_count**self.alpha_)
                else:
                    similarity = occu_matrix[good_A_ind, good_B_ind] / (good_A_count**self.alpha_ * good_B_count**(1 - self.alpha_))
                simi_matrix[good_A_ind, good_B_ind] = similarity
                simi_matrix[good_B_ind, good_A_ind] = similarity
        self.simi_matrix_ = simi_matrix
        
    def fit(self, dataset):
        occu_matrix, good_dict = self._get_matrix(dataset)
        goods_grade_dict = self._get_good_grade(dataset)
        self.good_dict_ = good_dict
        self._good_similarity(occu_matrix, goods_grade_dict)
    
    def recommand(self, user_records, num=3):
        # 获取已购买物品列表以及初始化物品评分字典
        purchased_goods = list(user_records.keys())
        goods_grade = {}
        for good in self.good_dict_:
            goods_grade[good] = 0
        # 遍历所有物品列表
        for good in self.good_dict_:
            # 对于已购买的物品不再进行推荐
            if good not in user_records:
                good_ind = self.good_dict_[good]
                for good_i in good_dict:
                    if good_i in purchased_goods:
                        goods_grade[good] += user_records[good_i] * self.simi_matrix_[good_ind, self.good_dict_[good_i]]
                    else:
                        goods_grade[good] += 1 * self.simi_matrix_[good_ind, self.good_dict_[good_i]]
        # 获取当前物品的其他相似度最高的 num 个物品
        top_simi_goods = list(goods_grade.items())
        top_simi_goods.sort(key=lambda x:x[1], reverse=True)
        return top_simi_goods[:num]


dataset = np.array([
    [{1: 5}, {2: 4}, {4: 1}],
    [{1: 4}, {4: 3}],
    [{1: 5}, {2: 3}, {5: 3}],
    [{2: 3}, {3: 5}],
    [{3: 4}, {5: 2}],
    [{2: 4}, {4: 3}]
])
user_records = {1: 4, 2: 3}


model = Recommand()
model.fit(dataset)
model.recommand(user_records)