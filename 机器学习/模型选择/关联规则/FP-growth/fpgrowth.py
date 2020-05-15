class treeNode:
    
    def __init__(self, name_value, num_occur, parent_node):
        self.name = name_value
        self.count = num_occur
        self.node_link = None
        self.parent = parent_node
        self.children = {}
    
    def inc(self, num_occur):
        self.count += num_occur
        
    def disp(self, ind=1):
        print('  ' * ind, self.name, '  ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def load_simp_data():
    simp_data = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]
    return simp_data


def create_init_set(dataset):
    ret_dict = {}
    for trans in dataset:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


simp_dat = load_simp_data()
init_set = create_init_set(simp_dat)


class FPGrowth():
    
    def __init__(self):
        pass
    
    def create_tree(self, dataset, min_sup=1):
        header_table = {}
        for trans in dataset:
            for item in trans:
                header_table[item] = header_table.get(item, 0) + dataset[trans]
        temp_table = {}
        # 移除不满足最小支持度的项集
        for k in header_table.keys():
            if header_table[k] >= min_sup:
                temp_table[k] = header_table[k]
        del(header_table)
        header_table = temp_table
        freq_item_set = set(header_table.keys())
        # 如果没有项集满足要求，则退出
        if len(freq_item_set) == 0:
            return None, None
        for k in header_table:
            header_table[k] = [header_table[k], None]
        ret_tree = treeNode('Null Set', 1, None)
        for tran_set, count in dataset.items():
            # 根据全局频率对每个事务中的元素进行排序
            local_D = {}
            for item in tran_set:
                if item in freq_item_set:
                    local_D[item] = header_table[item][0]
            if len(local_D) > 0:
                ordered_items = [v[0] for v in sorted(local_D.items(), key=lambda p:p[1], reverse=True)]
                self._update_tree(ordered_items, ret_tree, header_table, count)
        return ret_tree, header_table
    
    def _update_tree(self, items, in_tree, header_table, count):
        if items[0] in in_tree.children:
            in_tree.children[items[0]].inc(count)
        else:
            in_tree.children[items[0]] = treeNode(items[0], count, in_tree)
            if header_table[items[0]][1] == None:
                header_table[items[0]][1] = in_tree.children[items[0]]
            else:
                self._update_header(header_table[items[0]][1], in_tree.children[items[0]])
        if len(items) > 1:
            # 对剩下的项集迭代调用 update_tree 函数
            self._update_tree(items[1::], in_tree.children[items[0]], header_table, count)
            
    def _update_header(self, node_to_test, target_node):
        while node_to_test.node_link != None:
            node_to_test = node_to_test.node_link
        node_to_test.node_link = target_node
        
    def _ascend_tree(self, leaf_node, prefix_path):
        if leaf_node.parent != None:
            prefix_path.append(leaf_node.name)
            self._ascend_tree(leaf_node.parent, prefix_path)
        
    def _find_prefix_path(self, base_pat, tree_node):
        cond_pats = {}
        while tree_node != None:
            prefix_path = []
            self._ascend_tree(tree_node, prefix_path)
            if len(prefix_path) > 1:
                cond_pats[frozenset(prefix_path[1:])] = tree_node.count
            tree_node = tree_node.node_link
        return cond_pats
    
    def mine_tree(self, in_tree, header_table, min_sup, prefix, freq_item_list):
        big_l = [v[0] for v in sorted(header_table.items(), key=lambda p:p[1][0])]
        for base_pat in big_l:
            new_freq_set = prefix.copy()
            new_freq_set.add(base_pat)
            freq_item_list.append(new_freq_set)
            cond_patt_bases = self._find_prefix_path(base_pat, header_table[base_pat][1])
            my_cond_tree, my_head = self.create_tree(cond_patt_bases, min_sup)
            if my_head != None:
                # print('conditional tree for: ', new_freq_set)
                # my_cond_tree.disp(1)
                self.mine_tree(my_cond_tree, my_head, min_sup, new_freq_set, freq_item_list)


fp = FPGrowth()
my_fptree, my_header_tab = create_tree(init_set, 3)
freq_items = []
mine_tree(my_fptree, my_header_tab, 3, set([]), freq_items)