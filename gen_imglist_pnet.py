# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 1:34
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : test.py
# @Software: PyCharm
import numpy as np
npr=np.random
import os
data_dir='./data/'

def combine_pnet_train_data():
    '''将pos,part,neg,landmark四者混在一起'''
    with open(os.path.join(data_dir, '12/pos_12.txt'), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(data_dir, '12/neg_12.txt'), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(data_dir, '12/part_12.txt'), 'r') as f:
        part = f.readlines()
    with open(os.path.join(data_dir, '12/landmark_12_aug.txt'), 'r') as f:
        landmark = f.readlines()
    dir_path = os.path.join(data_dir, '12')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, 'train_pnet_landmark.txt'), 'w') as f:
        nums = [len(neg), len(pos), len(part)]
        base_num = 250000
        print('neg数量：{} pos数量：{} part数量:{} 基数:{}'.format(len(neg), len(pos), len(part), base_num))
        if len(neg) > base_num * 3:
            neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
        else:
            neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
        sum_p = len(neg_keep) // 3
        pos_keep = npr.choice(len(pos), sum_p, replace=True)
        part_keep = npr.choice(len(part), sum_p, replace=True)
        print('neg数量：{} pos数量：{} part数量:{}'.format(len(neg_keep), len(pos_keep), len(part_keep)))
        for i in pos_keep:
            f.write(pos[i])
        for i in neg_keep:
            f.write(neg[i])
        for i in part_keep:
            f.write(part[i])
        for item in landmark:
            f.write(item)

    pass

if __name__ == '__main__':
    combine_pnet_train_data()
    pass