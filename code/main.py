#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import datetime
# import numpy as np
# import sys


class Model(object):
    """模型"""

    def __init__(self, args):
        # Initialize with args
        self.debug = args.debug
        self.path_to_dataset_a = args.path_to_dataset_a
        self.path_to_dataset_b = args.path_to_dataset_b
        # Initialize fields
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

    def log(self, *args, **kwargs):
        """用于调试"""
        if self.debug:
            return print(*args, **kwargs)

    def read_data(self):
        """读入数据"""
        self.log("===== READING DATA START =====")
        self.log(self.path_to_dataset_a)
        self.log(self.path_to_dataset_b)
        self.log("===== READING DATA EMD =====")

    def train(self):
        pass

    def predict(self):
        pass

    def save_result(self, path_to_submit_file):
        pass
        # self.testing_label.to_csv(path_to_submit_file, header=None, index=False)


def parse_args():
    """设置程序参数"""
    parser = argparse.ArgumentParser(
        description = 'Process short videos and answer questions')
    parser.add_argument('--path-to-dataset-a',
                        default = './data/DatasetA',
                        help = 'Path to dataset A')
    parser.add_argument('--path-to-dataset-b',
                        default = './data/DatasetB',
                        help = 'Path to dataset B')
    # parser.add_argument('path_to_log_file',
    #                     help = 'path to log file which is to be analyzed')
    # parser.add_argument('-d', '--date-range',
    #                     default = '-',
    #                     help = 'count records within [YYYYmmdd]-[YYYYmmdd], '
    #                            'the default start and end date are 00010101 '
    #                            'and 99991231')
    # parser.add_argument('os_name',
    #                     nargs = '*',
    #                     help = 'show statistics for the provided OS. '
    #                            'Default to show statistics for all OS')
    parser.add_argument("-d", "--debug", action="store_true",
                        help="show debug message")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = Model(args)
    model.read_data()
    model.train()
    model.predict()
    # Save to submit folder
    filename = "submit_%s.txt" % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save_result("./submit/" + filename)


if __name__ == '__main__':
    main()
