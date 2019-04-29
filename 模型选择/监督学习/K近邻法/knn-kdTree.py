import numpy as np


data_source = [
    [34, 38],
    [43, 11],
    [37, 40],
    [42, 43],
    [34, 31]
]
type_source = [1, 2, 2, 1, 1]
data_transform = np.array(data_source, dtype=int)
y_transform = np.array(type_source, dtype=int)


class KdNode(object):
    data: 0
    split: 0
    parent: None
    left: None
    right: None
    type: None

    def __init__(self, data, split, type):
        self.data, self.split, self.type = data, split, type
        self.flag = False


class KNN(object):

    def __init__(self, k=3):
        self.k = k
        self.kd_tree = None

    def fit(self, x, y):
        y = y.reshape((len(y), 1))
        data = np.hstack((x, y))
        self.kd_tree = self.create_kd_tree(data)

    def predict(self, x):
        neighbors = []
        for i in range(self.k):
            neighbors.append(self.find_nearest(x))

        print(neighbors)
        classes = []
        for item in neighbors:
            classes.append(item['nearest'][-1])
        classes = np.array(classes, dtype=int)
        print('classes', classes)
        l = sorted([(np.sum(classes == i), i) for i in set(classes.flat)])
        return l[-1][1]

    def create_kd_tree(self, data):
        if len(data) == 0:
            return None

        # 获取方差最大的维度
        squared = np.var(data, 0)
        split = np.argmax(squared[:-1])

        # 对数据进行排序
        ind = np.argsort(data, 0)
        data = data[ind[:, split]]

        # 计算中位数并划分数据
        middle_index = int(np.floor(len(data) / 2))
        data_left, data_middle, data_right = np.vsplit(data, (middle_index, middle_index + 1))
        kd_point = KdNode(data_middle, split, int(data_middle[:, -1]))

        kd_left = self.create_kd_tree(data_left)
        if kd_left:
            kd_left.parent = kd_point
        kd_point.left = kd_left

        kd_right = self.create_kd_tree(data_right)
        if kd_right:
            kd_right.parent = kd_point
        kd_point.right = kd_right

        return kd_point

    def find_nearest(self, target):
        if self.kd_tree is None:
            return {
                'nearest': None,
                'dist': -1
            }

        kd_point = self.kd_tree
        nearest_node = None
        min_dist = 0
        search_path = []

        while kd_point:
            data_node = kd_point.data.ravel()
            dist = np.linalg.norm(data_node[:-1] - target)
            split = kd_point.split
            search_path.append(kd_point)

            if kd_point.flag is False:
                if nearest_node is None:
                    nearest_node = kd_point
                    nearest = data_node
                    min_dist = dist
                else:
                    if dist < min_dist:
                        nearest_node = kd_point
                        nearest = data_node
                        min_dist = dist

            if target[split] < data_node[split]:
                kd_point = kd_point.left
            else:
                kd_point = kd_point.right

        while len(search_path) > 0:
            kd_point = search_path.pop()

            if kd_point is None:
                continue
            data_node = kd_point.data.ravel()
            dist = np.linalg.norm(data_node[:-1] - target)
            split = kd_point.split
            print('data_node:', data_node)
            print('min_dist:', min_dist)
            print('dist:', dist)

            if dist >= min_dist:
                if kd_point.flag is False:
                    if nearest_node is None:
                        nearest_node = kd_point
                        nearest = data_node
                        min_dist = dist

                if target[split] < data_node[split]:
                    search_path.append(kd_point.right)
                else:
                    search_path.append(kd_point.left)
            elif dist < min_dist:
                if kd_point.flag is False:
                    if nearest_node is None:
                        nearest_node = kd_point
                        nearest = data_node
                        min_dist = dist
                    else:
                        if dist < min_dist:
                            nearest_node = kd_point
                            nearest = data_node
                            min_dist = dist
                if target[split] < data_node[split]:
                    search_path.append(kd_point.left)
                else:
                    search_path.append(kd_point.right)

        nearest_node.flag = True
        print({
            'nearest': nearest,
            'dist': min_dist
        })
        return {
            'nearest': nearest,
            'dist': min_dist
        }


def test_kd_tree():
    model = KNN(k=4)
    model.fit(data_transform, y_transform)
    result = model.predict([32, 13])
    print(result)


if __name__ == '__main__':
    test_kd_tree()