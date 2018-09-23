#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import datetime
import os
# import numpy as np
import subprocess


class Model(object):
    """模型"""

    def __init__(self, args):
        # Initialize with args
        self.debug = args.debug
        self.path_to_images = args.path_to_images
        self.path_to_features = args.path_to_features
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

    def pre_process(self):
        """数据预处理，将输入数据（视频）转换为图片"""
        self.log('===== PRE-PROCESSING START =====')
        dataset_paths = [self.path_to_dataset_a, self.path_to_dataset_b]
        data_types = ['train', 'test']

        for data_type in data_types:
            image_type_path = self.path_to_images + data_type + '/'
            subprocess.call(['mkdir', '-p', image_type_path])
            for dataset_path in dataset_paths:
                dataset_type_path = dataset_path + data_type + '/'
                if os.path.isdir(dataset_type_path):
                    dataset_files = os.listdir(dataset_type_path)
                    for file_name in dataset_files:
                        if os.path.isdir(file_name):
                            self.log('[WARN] ignore directories under training data')
                            continue
                        if file_name == '.DS_Store':
                            self.log('[INFO] ignore %s under %s' % (file_name, dataset_type_path))
                            continue
                        file_name_no_ext = file_name.split('.')[0]
                        image_processed_path = image_type_path + file_name_no_ext + '.success'
                        if os.path.isfile(image_processed_path):
                            self.log('[INFO] video %s already processed, skipped' % file_name_no_ext)
                            continue
                        file_path = dataset_type_path + file_name
                        image_path_pattern = '%s%s_%%3d.jpg' % (image_type_path, file_name_no_ext)
                        self.log('[INFO] Processing %s' % file_path)
                        returncode = subprocess.call(['ffmpeg', '-v', 'error',
                                                      '-i', file_path, '-r', '1', image_path_pattern])
                        if returncode == 0:
                            subprocess.call(['touch', image_processed_path])
                        else:
                            self.log('[WARN] video %s not processed successfully!' % file_name_no_ext)
        self.log('===== PRE-PROCESSING END =====')

    def extract_feature(self):
        """提取视频特征"""
        self.log('\n===== FEATURE EXTRACTION START =====')
        data_types = ['train', 'test']
        subprocess.call(['mkdir', '-p', self.path_to_features])
        for data_type in data_types:
            image_type_path = self.path_to_images + data_type + '/'
            out_file = self.path_to_features + data_type + '.h5'
            returncode = subprocess.call(['python', './code/TF_FeatureExtraction/example_feat_extract.py',
                                          '--network', 'resnet_v2_152', '--checkpoint',
                                          '/Users/hanjunx/workspace/tensorflow/checkpoints/resnet_v2_152_2017_04_14/resnet_v2_152.ckpt',
                                          '--image_path', image_type_path,
                                          '--out_file', out_file,
                                          '--layer_names', 'resnet_v2_152/logits',
                                          '--preproc_func', 'inception'])
            if returncode == 0:
                self.log('[INFO] feature extraction for %s succeeded!' % data_type)
            else:
                self.log('[ERROR] feature extraction for %s failed!' % data_type)
        self.log('===== FEATURE EXTRACTION END =====')

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
    parser.add_argument('--path-to-images',
                        default = './data/images_sample/',
                        help = 'Path to images')
    parser.add_argument('--path-to-features',
                        default = './data/features_sample/',
                        help = 'Path to features')
    parser.add_argument('--path-to-dataset-a',
                        default = './data/DatasetA/',
                        help = 'Path to dataset A')
    parser.add_argument('--path-to-dataset-b',
                        default = './data/DatasetB/',
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
    # model.pre_process()
    model.extract_feature()
    model.train()
    model.predict()
    # Save to submit folder
    filename = 'submit_%s.txt' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save_result('./submit/' + filename)


if __name__ == '__main__':
    main()
