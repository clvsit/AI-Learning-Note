import numpy as np


def load_dataset():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


class Apriori:
    
    def __init__(self):
        pass
    
    def _create_c1(self, dataset):
        c1 = []
        for transaction in dataset:
            for item in transaction:
                if not [item] in c1:
                    c1.append([item])
        c1.sort()
        return list(map(frozenset, c1))
    
    def _scan_D(self, D, ck, min_support):
        ss_cnt = {}
        for tid in D:
            for can in ck:
                if can.issubset(tid):
                    if not ss_cnt.__contains__(can): 
                        ss_cnt[can] = 1
                    else:
                        ss_cnt[can] += 1
        num_items = float(len(D))
        ret_list = []
        support_data = {}
        for key in ss_cnt:
            support = ss_cnt[key] / num_items
            if support >= min_support:
                ret_list.insert(0, key)
            support_data[key] = support
        return ret_list, support_data
    
    def _apriori_gen(self, lk, k):
        # creates CK
        ret_list = []
        len_lk = len(lk)
        for i in range(len_lk):
            for j in range(i + 1, len_lk):
                l1 = list(lk[i])[:k-2]
                l2 = list(lk[j])[:k-2]
                l1.sort()
                l2.sort()
                if l1 == l2:
                    ret_list.append(lk[i] | lk[j])
        return ret_list
    
    def apriori(self, dataset, min_support=0.5):
        c1 = self._create_c1(dataset)
        D = list(map(set, dataset))
        l1, support_data = self._scan_D(D, c1, min_support)
        l = [l1]
        k = 2
        while len(l[k-2]) > 0:
            ck = self._apriori_gen(l[k-2], k)
            lk, supk = self._scan_D(D, ck, min_support)
            support_data.update(supk)
            l.append(lk)
            k += 1
        return l, support_data
    
    def generate_rules(self, l, support_data, min_conf=0.7):
        big_rule_list = []
        for i in range(1, len(l)):
            for freq_set in l[i]:
                h1 = [frozenset([item]) for item in freq_set]
                if i > 1:
                    self._rules_from_conseq(freq_set, h1, support_data, big_rule_list, min_conf)
                else:
                    self._calc_conf(freq_set, h1, support_data, big_rule_list, min_conf)
        return big_rule_list
    
    def _calc_conf(self, freq_set, h, support_data, br1, min_conf=0.7):
        pruned_h = []
        for conseq in h:
            conf = support_data[freq_set] / support_data[freq_set - conseq]
            if conf >= min_conf:
                print(freq_set - conseq, '-->', conseq, 'conf:', conf)
                br1.append((freq_set - conseq, conseq, conf))
                pruned_h.append(conseq)
        return pruned_h
    
    def _rules_from_conseq(self, freq_set, h, support_data, br1, min_conf=0.7):
        m = len(h[0])
        if len(freq_set) > (m + 1):
            hmp1 = self._apriori_gen(h, m + 1)
            hmp1 = self._calc_conf(freq_set, hmp1, support_data, br1, min_conf)
            if len(hmp1) > 1:
                self._rules_from_conseq(freq_set, hmp1, support_data, br1, min_conf)


dataset = load_dataset()
apriori = Apriori()
l, support_data = apriori.apriori(dataset, min_support=0.7)
rules = apriori.generate_rules(l, support_data)