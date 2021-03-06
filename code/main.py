#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import argparse
import datetime
import h5py
import numpy as np
import os
import subprocess


class Model(object):
    """模型"""

    def __init__(self, args):
        # Initialize with args
        self.debug = args.debug
        self.partition = args.partition
        self.path_to_submit_file = args.path_to_submit_file
        self.path_to_checkpoints = args.path_to_checkpoints
        self.path_to_images = args.path_to_images
        self.path_to_features = args.path_to_features
        self.path_to_dataset_a = args.path_to_dataset_a
        self.path_to_dataset_b = args.path_to_dataset_b
        # Initialize fields
        self.pool_layer = 'global_pool'
        self.id_cutoff = 4285
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
        self.log('\n===== PRE-PROCESSING START =====')

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

    def sample(self, limit, path_to_sampled_images):
        """为每个视频采样指定张数的图片"""
        self.log('\n===== SAMPLING START =====')

        data_types = ['train', 'test']
        for data_type in data_types:
            image_type_path = self.path_to_images + data_type + '/'
            if not os.path.isdir(image_type_path):
                self.log('[ERROR] images folder not existed!')
                return
            image_files = os.listdir(image_type_path)
            success_files = [x for x in image_files if x.endswith('.success')]
            self.log('[INFO]', success_files)
            for video_success in success_files:
                video_id = video_success.split('.')[0]
                sampled_image_path = path_to_sampled_images + data_type + '/' + video_id + '/'
                if os.path.isdir(sampled_image_path):
                    self.log('[INFO] sampled images for video %s already existed. skipped.' % video_id)
                    continue
                subprocess.call(['mkdir', '-p', sampled_image_path])
                video_images = [x for x in image_files if x.startswith(video_id + '_')]
                video_images.sort()
                frame_cnt = len(video_images)
                self.log('[INFO]', video_id, frame_cnt)
                block_size = frame_cnt // (limit + 1)
                indexs = [block_size * x for x in range(1, limit + 1)]
                # self.log('[DEBUG]', indexs)
                sampled_video_images = [video_images[idx] for idx in indexs]
                self.log('[INFO]', sampled_video_images)
                for img in sampled_video_images:
                    path_to_img = image_type_path + img
                    subprocess.call(['cp', path_to_img, sampled_image_path])

        self.log('===== SAMPLING END =====')


    def extract_feature(self):
        """提取视频特征"""
        self.log('\n===== FEATURE EXTRACTION START =====')

        subprocess.call(['mkdir', '-p', self.path_to_features])
        data_types = ['train', 'test']
        for data_type in data_types:
            out_type_path = self.path_to_features + data_type + '/'
            subprocess.call(['mkdir', '-p', out_type_path])
            image_type_path = self.path_to_images + data_type + '/'
            image_dirs = os.listdir(image_type_path)
            for video_id in image_dirs:
                image_path = image_type_path + video_id + '/'
                if not os.path.isdir(image_path):
                    continue
                # exclude dataset A
                int_id = int(video_id[3:])
                partition = 'B_'
                if int_id < self.id_cutoff:
                    partition = 'A_'
                out_file = out_type_path + video_id + '.h5'
                self.log('[INFO]', out_file)
                if os.path.isfile(out_file):
                    self.log('[INFO] feature for %s already extracted, skipped' % data_type)
                    continue
                partition += data_type
                if data_type == 'train':
                    if partition.startswith('B'):
                        partition += '_' + str(int_id % 2)
                    else:
                        partition += '_' + str(int_id % 4)
                if partition != self.partition:
                    self.log('[INFO] video %s is from another partition %s, skipped' % (video_id, partition))
                    continue
                network = 'resnet_v2_152'
                path_to_resnet_checkpoint = self.path_to_checkpoints + 'resnet_v2_152_2017_04_14/resnet_v2_152.ckpt'
                layer_names = self.pool_layer
                preproc_func = 'inception'
                cmd = ['python', './code/TF_FeatureExtraction/example_feat_extract.py',
                       '--network', network,
                       '--checkpoint', path_to_resnet_checkpoint,
                       '--image_path', image_path,
                       '--out_file', out_file,
                       '--layer_names', layer_names,
                       '--preproc_func', preproc_func]
                self.log('[INFO] [CMD]', ' '.join(cmd))
                returncode = subprocess.call(cmd)
                if returncode == 0:
                    self.log('[INFO] feature extraction for %s succeeded!' % image_path)
                else:
                    self.log('[ERROR] feature extraction for %s failed!' % image_path)

        self.log('===== FEATURE EXTRACTION END =====')

    def combine(self):
        """组合特征"""
        self.log('\n===== FEATURE COMBINATION START =====')

        data_types = ['train', 'test']
        for data_type in data_types:
            feature_combined_file = data_type + '.h5'
            path_to_feature_combined_file = self.path_to_features + feature_combined_file
            # self.log('[INFO] [combine.target]', path_to_feature_combined_file)
            combined_f5 = h5py.File(path_to_feature_combined_file, 'w')
            self.log('[INFO] [combine.target]', combined_f5)
            # self.log('[DEBUG] [combine.target]', dir(combined_f5))
            feature_type_dir = self.path_to_features + data_type + '/'
            self.log('[INFO] [combine.dir]', feature_type_dir)
            for feature_type_file in os.listdir(feature_type_dir):
                path_to_feature_type_file = feature_type_dir + feature_type_file
                if not feature_type_file.endswith('.h5'):
                    self.log('[INFO] h5 file for "%s" not found, skipped' % feature_type_file)
                    continue
                video_id = feature_type_file.split('.')[0]
                # Read h5
                feat_h5 = h5py.File(path_to_feature_type_file, 'r')
                self.log('[INFO] [feat_h5]', video_id, feat_h5)
                # self.log('[INFO] [feat_h5]', feat_h5.keys())
                # self.log('[DEBUG] [feat_h5]', dir(feat_h5))

                # Filenames
                filenames = feat_h5['filenames']
                # self.log('[INFO] [filenames]', filenames)
                np_filenames = np.array(filenames)
                # self.log('[INFO] [np_filenames]', np_filenames)

                # Resnet
                layer = self.pool_layer
                if layer not in feat_h5.keys():
                    self.log('[ERROR] [feat_h5] "%s" not in output layers!' % layer)
                    return
                layer_dset = feat_h5[layer]
                # self.log('[INFO] [resnet_v2_152.layer]', layer_dset)
                # self.log('[DEBUG] [resnet_v2_152.layer]', dir(layer_dset))
                np_res_layer_dset = np.array(layer_dset)
                # self.log('[DEBUG] [np_res_layer_dset]', dir(np_res_layer_dset))
                self.log('[INFO] [np_res_layer_dset.size]', np_res_layer_dset.shape)
                self.log('[INFO] [np_res_layer_dset.dtype]', np_res_layer_dset.dtype)
                combined_f5.create_dataset(video_id, np_res_layer_dset.shape,
                                           np_res_layer_dset.dtype, np_res_layer_dset)

        self.log('===== FEATURE COMBINATION END =====')

    def train(self):
        """训练模型"""
        self.log('\n===== TRAINING PHASE START =====')
        data_type = 'train'
        feature_combined_file = data_type + '.h5'
        path_to_feature_combined_file = self.path_to_features + feature_combined_file
        combined_f5 = h5py.File(path_to_feature_combined_file, 'r')
        self.log('[INFO] [combined]', combined_f5)
        self.log('[INFO] [combined]', combined_f5.keys())
        # self.log('[DEBUG] [combined]', dir(combined_f5))

        for video_id in combined_f5.keys():
            feature_dset = combined_f5[video_id]
            np_feature_dset = np.array(feature_dset)
            self.log('[INFO] [np_feature_dset]', video_id, np_feature_dset.shape)

        self.log('===== TRAINING PHASE END =====')

    def predict(self):
        """预测测试数据"""
        self.log('\n===== PREDICTION PHASE START =====')

        self.log('===== PREDICTION PHASE END =====')

        # 存储结果
        self.log('\nSaving result to %s' % self.path_to_submit_file)
        # self.testing_label.to_csv(self.path_to_submit_file, header=None, index=False)


def parse_args():
    """设置程序参数"""
    parser = argparse.ArgumentParser(
        description = 'Process short videos and answer questions')
    filename = 'submit_%s.txt' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--partition',
                        default='B_test',
                        help='Feature extraction partition')
    parser.add_argument('--path-to-submit-file',
                        default='./submit/' + filename,
                        help='Path to submit file')
    parser.add_argument('--path-to-checkpoints',
                        default = '/Users/hanjunx/workspace/tensorflow/checkpoints/',
                        help = 'Path to checkpoints')
    parser.add_argument('--path-to-sampled-images-1fpv',
                        default = './data/images_1fpv/',
                        help = 'Path to images (1 frames per video)')
    parser.add_argument('--path-to-sampled-images-5fpv',
                        default = './data/images_5fpv/',
                        help = 'Path to images (5 frames per video)')
    parser.add_argument('--path-to-images',
                        default = './data/images/',
                        help = 'Path to images')
    parser.add_argument('--path-to-features',
                        default = './data/features/',
                        help = 'Path to features')
    parser.add_argument('--path-to-dataset-a',
                        default = './data/DatasetA/',
                        help = 'Path to dataset A')
    parser.add_argument('--path-to-dataset-b',
                        default = './data/DatasetB/',
                        help = 'Path to dataset B')
    parser.add_argument("-d", "--debug", action="store_true",
                        help="show debug message")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = Model(args)
    # model.pre_process()
    # model.sample(1, args.path_to_sampled_images_1fpv)
    # model.sample(5, args.path_to_sampled_images_5fpv)
    model.extract_feature()
    model.combine()
    # model.train()
    # model.predict()


if __name__ == '__main__':
    main()
