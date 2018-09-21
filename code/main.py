#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import datetime
import sys

class Model(object):
    """模型"""

    def __init__(self, args):
        self.debug = args.debug
        self.training_data = []
        self.training_label = []
        self.testing_data = []
        self.testing_label = []

    def read_data(self):
        pass

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
    filename = "submit_%s.csv" % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save_result("../submit/" + filename)


if __name__ == '__main__':
    main()