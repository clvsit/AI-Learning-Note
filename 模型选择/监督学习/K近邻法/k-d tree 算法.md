# k-d tree 算法
k-d 树（k-dimensional 树的简称），是一种分割 k 维数据空间的数据结构。主要应用于多维空间关键数据的搜索（如：范围搜索和最近邻搜索）。

## 应用背景

![二维数据k-d树空间划分示意图](https://images2018.cnblogs.com/blog/1220093/201804/1220093-20180406133236538-1408820839.png)

## 过程
k-d tree 算法主要分为两部分：
- k-d 树的构建算法；
- 基于 k-d 树的最邻近查找算法。

### k-d 树的构建算法
k-d 树是一个二叉树，每个节点表示一个空间范围，下表给出 k-d 树节点的数据结构。

属性名 | 数据类型 | 描述
---|---|---
data | 数据向量 | 数据集中某个数据点，是 k 维向量
range | 空间向量 | 该节点所代表的空间范围（暂时未用）
split | 待定 | 垂直于分割超平面的方向轴序号
left | k-d 树 | 由位于该节点分割超平面左子空间内所有数据点构成的 k-d 树
right | k-d 树 | 由位于该节点分割超平面右子空间内所有数据点构成的 k-d 树
parent | k-d 树 | 父节点

【说明】：split 属性的类型为待定是因为这取决于数据的组织形式。例如：
```JS
const data = [
    [10, 25],
    [53, 26],
    // ...
    [82, 35]
]
```
在这种形式下，split 可以是整型，用数字来代表每一个方向轴的序号。
```JS
const data =[
    { x: 10, y: 25 },
    { x: 53, y: 26 },
    // ...
    { x: 82, y: 35 }
]
```
在上述组织形式下，split 建议为字符串型。

【代码-k-d 树节点】：
```JS
function KdNode (data, split) {
    this.data = data;
    this.split = split;
    this.left = null;
    this.right = null;
    this.parent = null;
}
```

在明确了 k-d 树节点的数据结构后，我们可以开始构建我们的 k-d 树。

【代码】：以二维数据为例。
```JS
const data = [
    [10, 25],
    [53, 26],
    [25, 14],
    [57, 49],
    [82, 35],
    [24, 47],
    [67, 19],
    [31, 58]
];

function createKdTree (data) {
    // 如果没有数据，则返回 null
    if (data.length === 0) {
        return null;
    }
    
    // 获取各维度的数据
    let dataDim1 = data.map(item => { return item[0]; }),
        dataDim2 = data.map(item => { return item[1]; });
    
    // 确定分割的维度
    let dim1 = calVariance(dataDim1),
        dim2 = calVariance(dataDim2),
        split = dataDim1 > dataDim2 ? 0 : 1;
    
    // 对数据进行排序
    data.sort((a, b) => { return a[split] - b[split]; });
    
    // 获取中位数 - 即 k-d 树节点的 data 属性
    let dataIndex = Math.floor(data.length / 2),
        nodeData = data[dataIndex];
    
    // 创建 k-d 树节点
    let kdNode = new KdNode(nodeData, split);
    
    // 将剩余数据划分到左右子树
    let dataLeft = data.slice(0, dataIndex),
        dataRight = data.slice(dataIndex + 1);
    
    let leftNode = createKdTree(dataLeft);
    // 非 null，否则会报错
    if (leftNode) {
        leftNode.parent = kdNode;    
    }
    kdNode.left = leftNode;
    
    let rightNode = createKdTree(dataRight);
    // 同 leftNode
    if (rightNode) {
        rightNode.parent = kdNode;
    }
    kdNode.right = rightNode;
    
    return kdNode;
}

/**
 * 计算方差
 * @param data array 数据数组
 */
function calVariance (data) {
    const mean = data.reduce((a, b) => {
        return a + b;
    }) / data.length;
    
    return data.reduce((a, b) => {
        return a + Math.pow((b - mean), 2);
    }, 0);
}
```

【结合实例说明】：
- 由于此例比较简单，数据维度只有二维，所以计算分割的维度时比较容易，只需要分别计算 x 轴和 y 轴的方差，并进行比对即可。
- 第一轮计算完成后可以发现：在 x 轴上的方差要大于在 y 轴上的方差，因此将第一个节点的 split 设置为 x。
- 接下来就是确定 Node-data，先将数据按 x 轴升序排序（2, 4, 5, 9, 8, 9），并取其中的中位数 7，所以 Node-data 即为（7, 2）。于是，该点的分割超平面就是通过（7, 2）并垂直于 split = 0（x 轴）的直线 x = 7。
- 确定左子空间和右子空间。分割超平面 x = 7 将整个空间分为两部分，将 x < 7 的划分到左子空间，x >= 7 的划分到右子空间。
- 最后对左子空间和右子空间内的数据重复根节点的过程，直到无法生成 k-d 树节点。
- 从代码中可以看出，k-d 树的构建是一个逐级展开的递归过程。

【问题】：为什么要用方差来确定节点的划分维度？

【回答】：因为方差越大，代表这组数据分散程度越大，那么对该维度进行数据划分的效果会更好。

【问题】：如果有和 Node-data 相同 x 值的点该怎么划分呢？例如数据 [1, 2, 4, 4, 5, 6]，中位数为 4，那么另外一个 4 该何去何从呢？

【回答】：在确定左子空间和右子空间的步骤中，将 < Node-data[split] 的数据划分到左子空间，其余的数据都归为右子空间。因此，与 Node-data 相同轴值的数据都被划分到右子空间中。

### 最邻近查找算法
【目的】：检索在 k-d 树中与检索点距离最近的数据点。

【思想】：通过检索二叉树找到与检索点匹配的空间区域。但此区域的坐标点不一定是检索点的最近邻点，因此还需要进行回溯操作，对所有的可能进行一次遍历。

【代码】：
```JS
function findNearest (kdTree, target) {
    // 如果 kdTree 不存在，则返回最邻近点为 null，距离为 -1
    if (!kdTree) {
        return {
            nearest: null,
            dict: -1
        }
    }
    
    // 初始化
    let kdPoint = kdTree,
        nearest = kdTree.data,
        minDist = calDist(nearest, target),
        searchPath = [];
        
    // 检索二叉树，直到叶子结点
    while (kdPoint) {
        let split = kdPoint.split,
            dist = calDist(kdPoint.data, target);
        
        // 将当前结点添加到搜索路径中，以便于后续的回溯操作
        searchPath.push(kdPoint);
        
        // 计算当前节点与检索点的距离，若小于先前的最短距离，则进行数据更新
        if (dist < minDist) {
            nearest = kdPoint.data;
            minDist = dist;
        }
        
        // 继续搜索二叉树
        if (target[split] < kdPoint.data[split]) {
            kdPoint = kdPoint.left;
        } else {
            kdPoint = kdPoint.right;
        }
    }
    
    // 二叉树检索完成后，开始进行回溯操作
    while (searchPath.length !== 0) {
        let searchPoint = searchPath.pop(),
            split = searchPoint ? searchPoint.split : "";
        
        if (!searchPoint) {
            continue;
        }
        // 计算当前节点与检索点的距离
        let dist = calDist(searchPoint.data,  target);
        
        // 如果当前距离小于等于最小距离，表示仍然有可能取得最小值
        if (dist <= minDist) {
            
            // 进入相反的空间（因为满足条件的空间在检索二叉树时已经进入）
            if (target[split] < searchPoint.data[split]) {
                searchPath.push(searchPoint.right);
            } else {
                searchPath.push(searchPoint.left);
            }
            
            // 更新数据
            nearest = searchPoint.data;
            minDist = dist;
        }
    }
    
    // 返回结果
    return {
        nearest: nearest,
        minDist: minDist
    }
}

/**
 * 计算两点之间的距离
 * @param posA {Object|Array} 位置A 数据
 * @param posB {Object|Array} 位置B 数据
 * @param type String 数据组织类型
 */
function calDist (posA, posB, type) {
    let powSum = 0;
    
    // 对象形式组织数据
    if (type === "key") {
        for (let key in posA) {
            powSum += Math.pow((posB[key] - posA[key]), 2);
        }
    } 
    // 数组形式组织数据
    else {
        for (let i = posA.length; i--;) {
            powSum += Math.pow((posB[i] - posA[i]), 2);
        }
    }
    return Math.sqrt(powSum);
}
```
【说明】：
- 最邻近查找算法主要可分为两部分：检索二叉树和回溯操作。
- 检索二叉树：
    - 在检索过程（从根节点沿着某条路径到叶子节点）中获取最短路径，此时的最短路径并非最终的最短路径。
    - 记录搜索路径——根节点到叶子节点的节点路径。需要注意的是，先记录搜索路径，后继续搜索二叉树。
- 回溯操作：依次回顾搜索路径的每一个节点，若找到路径小于或等于检索二叉树过程中得到的最短路径，则进入该空间并重复先前的操作。

#### 回溯操作
对于回溯操作我个人认为可以分为两部分：
1. 对搜索路径中已有节点的处理；
2. 在回溯过程中新增节点的处理。

一般来说，在回溯已有节点时，计算所得路径是不会出现小于最短路径的情况。最有可能的情况是，找到路径等于最短路径的节点，然后去查看另一个空间是否能够找到比当前最短路径更短的路径。
```JS
// 进入相反的空间（因为满足条件的空间在检索二叉树时已经进入）
if (target[split] < searchPoint.data[split]) {
    searchPath.push(searchPoint.right);
} else {
    searchPath.push(searchPoint.left);
}
```
上述代码就起到这样的作用，同时也往搜索路径数组中新增了节点。

但新增节点的处理过程应该有所不同，因为当前节点先前检索二叉树时未曾进入过，因此不应该是进入相反的空间。
```JS
if (dist < minDist) {
    if (target[split] < searchPoint.data[split]) {
        searchPath.push(searchPoint.left);
    } else {
        searchPath.push(searchPoint.left);
    }
            
    // 更新数据
    nearest = searchPoint.data;
    minDist = dist;
}
```
当该节点进入搜索路径数组后，对它的下一次操作则等同于“对已有节点的处理”，换言之，进入相反的空间。于是，我们就实现了对所有可能节点的遍历。

【修改后的代码】：
```JS
if (dist = minDist) {
            
    // 进入相反的空间（因为满足条件的空间在检索二叉树时已经进入）
    if (target[split] < searchPoint.data[split]) {
        searchPath.push(searchPoint.right);
    } else {
        searchPath.push(searchPoint.left);
    }
} else if (dist < minDist) {
    if (target[split] < searchPoint.data[split]) {
        searchPath.push(searchPoint.left);
    } else {
        searchPath.push(searchPoint.left);
    }
    
    // 更新数据
    nearest = searchPoint.data;
    minDist = dist;
}
```

#### Python 代码
```python
import numpy as np

data_source = [
    [34, 38],
    [43, 11],
    [37, 40],
    [42, 43],
    [34, 31]
]
data = np.array(data_source, dtype=int)

# K-D Tree 节点
class KdNode:
    data: 0
    split: 0
    parent: None
    left: None
    right: None

    def __init__(self, data, split):
        self.data = data
        self.split = split


def create_kd_tree(data):
    if len(data) == 0:
        return None

    # 获取方差最大的维度
    split = np.argmax(np.var(data, 0))
    ind = np.argsort(data, 0)
    data = data[ind[:, split]]
    middle_index = int(np.floor(len(data) / 2))
    data_left, data_node, data_right = np.vsplit(data, (middle_index, middle_index + 1))
    kd_point = KdNode(data_node, split)

    node_left = create_kd_tree(data_left)
    if node_left is not None:
        node_left.parent = kd_point
    kd_point.left = node_left

    node_right = create_kd_tree(data_right)
    if node_right is not None:
        node_right.parent = kd_point
    kd_point.right = node_right

    return kd_point


def find_nearest(kd_tree, target):
    if kd_tree is None:
        return {
            "nearest": None,
            "dist": -1
        }
    target = np.array(target)
    kd_point = kd_tree
    nearest = kd_tree.data.ravel()
    min_dist = np.linalg.norm(nearest - target)
    search_path = []

    while kd_point:
        data_node = kd_point.data.ravel()
        dist = np.linalg.norm(data_node - target)
        split = kd_point.split

        if dist < min_dist:
            nearest = data_node
            min_dist = dist

        search_path.append(kd_point)

        print(target[split])
        print(data_node)
        if target[split] < data_node[split]:
            kd_point = kd_point.left
        else:
            kd_point = kd_point.right

    while len(search_path) > 0:
        kd_point = search_path.pop()
        
        if kd_point is None:
            continue
        data_node = kd_point.data.ravel()
        dist = np.linalg.norm(data_node - target)
        split = kd_point.split

        if dist == min_dist:
            if target[split] < data_node[split]:
                search_path.append(kd_point.right)
            else:
                search_path.append(kd_point.left)
        elif dist < min_dist:
            nearest = data_node
            min_dist = dist
            if target[split] < data_node[split]:
                search_path.append(kd_point.left)
            else:
                search_path.append(kd_point.right)

    return {
        "nearest": nearest,
        "dist": min_dist
    }

kd_tree = create_kd_tree(data)
result = find_nearest(kd_tree, [10, 34])
print(result)

# 输出
{'nearest': array([34, 31]), 'dist': 24.186773244895647}
```

【注意事项】：
若 KdNode 节点的 left、right 为 None，则实例化后的节点对象不会有 left 和 right 属性。

```python
class KdNode(object):
    data: 0
    split: 0
    parent: None
    left: None
    right: None
    
    def __init__(self, data, split):
        self.data = data
        self.split = split


kd_node = KdNode(1, 1)
print(kd_node.left)
Traceback (most recent call last):
  File "<input>", line 1, in <module>
AttributeError: 'KdNode' object has no attribute 'left'
```
故而，在创建 K-d 树时，代码如下所示：
```python
node_left = create_kd_tree(data_left)
if node_left:
    node_left.parent = kd_point
kd_point.left = node_left
```
目的在于：不管 node\_left 是否存在，都会给 kd\_point 添加 left 属性。

## 参考
- k-d tree算法：https://www.cnblogs.com/eyeszjwang/articles/2429382.html