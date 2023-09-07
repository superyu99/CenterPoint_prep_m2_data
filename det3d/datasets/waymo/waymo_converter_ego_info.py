"""Tool to convert Waymo Open Dataset to pickle files.
    Adapted from https://github.com/WangYueFt/pillar-od
    # Copyright (c) Massachusetts Institute of Technology and its affiliates.
    # Licensed under MIT License
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, argparse, tqdm, pickle, os 

import waymo_decoder 
import tensorflow.compat.v2 as tf
from waymo_open_dataset import dataset_pb2

from multiprocessing import Pool 
import numpy as np

tf.enable_v2_behavior()

fnames = None 
EGO_PATH = None

def convert(idx):
    global fnames
    fname = fnames[idx]
    dataset = tf.data.TFRecordDataset(fname, compression_type='')
    for frame_id, data in enumerate(dataset):
        if 'seq_{}_frame_{}.pkl'.format(idx, frame_id) in PKL_DIR:
            print('seq_{}_frame_{}.pkl'.format(idx, frame_id) + 'exist')
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        ego_info = np.reshape(np.array(frame.pose.transform), [4, 4])
        ego_info = {'ego_info': ego_info}

        with open(os.path.join(EGO_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(ego_info, f)


def main(args):
    global fnames 
    fnames = sorted(list(glob.glob(args.record_path)))

    print("Number of files {}".format(len(fnames)))

    with Pool(128) as p: # change according to your cpu
        r = list(tqdm.tqdm(p.imap(convert, range(len(fnames))), total=len(fnames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo Data Converter')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--record_path', type=str, required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.root_path):
        os.mkdir(args.root_path)

    EGO_PATH = os.path.join(args.root_path, 'ego_info')

    if not os.path.isdir(EGO_PATH):
        os.mkdir(EGO_PATH)

    PKL_DIR = os.listdir(EGO_PATH)

    main(args)
