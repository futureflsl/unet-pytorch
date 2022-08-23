# -*- coding: utf-8 -*-
# Copyright (C) 2022 FIRC. All Rights Reserved
# @Time    : 2022/8/23 上午7:36
# @Author  : FIRC
# @Email   : 1623863129@qq.com
# @File    : split_train_data.py
# @Software: PyCharm
# @ Function Description:
'''
function as follows:

'''
import os
import random
import shutil


class SplitManager(object):
    def __init__(self, mask_dir, train_ratio=0.9):
        self.mask_dir = mask_dir
        self.train_ratio = train_ratio

    def split_train_val_test_dataset(self, file_list, train_ratio=0.9, trainval_ratio=0.9, need_test_dataset=False,
                                     shuffle_list=True):
        if shuffle_list:
            random.shuffle(file_list)
        total_file_count = len(file_list)
        train_list = []
        val_list = []
        test_list = []
        if need_test_dataset:
            trainval_count = int(total_file_count * trainval_ratio)
            trainval_list = file_list[:trainval_count]
            test_list = file_list[trainval_count:]
            train_count = int(train_ratio * len(trainval_list))
            train_list = trainval_list[:train_count]
            val_list = trainval_list[train_count:]
        else:
            train_count = int(train_ratio * total_file_count)
            train_list = file_list[:train_count]
            val_list = file_list[train_count:]
        return train_list, val_list, test_list

    def start_split(self):
        files = []
        for file in os.listdir(self.mask_dir):
            if file.endswith('.png'):
                files.append(os.path.splitext(file)[0])
        train_list, val_list, _ = self.split_train_val_test_dataset(files, self.train_ratio)
        with open('./mydata/train.txt', 'w') as f:
            f.write('\n'.join(train_list))
        with open('./mydata/val.txt', 'w') as f:
            f.write('\n'.join(val_list))


if __name__ == '__main__':
    mask_dir = './mydata/masks'
    sm = SplitManager(mask_dir=mask_dir, train_ratio=0.9)
    sm.start_split()
