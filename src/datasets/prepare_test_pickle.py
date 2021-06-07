import math
import os
import random
import sys
import pickle
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm


def prepare_pickle():
    dataroot = os.getenv('GEBD_ROOT', '/mnt/bd/byte-lufficc-data/datasets/GEBD/')
    anno_path = os.path.join(dataroot, 'GEBD_test_info.pkl')
    video_root = os.path.join(dataroot, 'GEBD_test_frames')
    downsample = 3
    # prepare file for multi-frames-GEBD
    # dict_train_ann
    with open(anno_path, 'rb') as f:
        dict_train_ann = pickle.load(f, encoding='lartin1')

    # downsample factor: sample one every `ds` frames
    frame_per_side = 8
    num_global_feats = 10
    thres = -1.0
    dynamic_ds = True
    add_global = False
    min_change_dur = 0.3

    test_file = 'multi-frames-GEBD-test-{}{}.pkl'.format(frame_per_side if not dynamic_ds else f'dynamic{frame_per_side}', '' if thres <= 0 else '_thres_{:.1f}'.format(thres))
    if add_global:
        p1, p2 = test_file.split('.')
        test_file = '{}-{}.{}'.format(p1, 'with_global', p2)
    load_file_path = os.path.join('./DataAssets', test_file)

    SEQ = []
    neg = 0
    pos = 0
    print(len(dict_train_ann.keys()))
    for vname in dict_train_ann.keys():
        vdict = dict_train_ann[vname]
        # vlen = vdict['num_frames']
        fps = vdict['fps']
        if dynamic_ds:
            ds = max(math.ceil(fps / 8), 1)
        else:
            ds = downsample

        # f1_consis = vdict['f1_consis']
        path_frame = vdict['path_frame']
        # print(path_frame.split('/'))
        cls, frame_folder = path_frame.split('/')[:2]
        # if self.mode == 'train':
        video_dir = os.path.join(video_root, cls, frame_folder)
        # elif self.mode == 'val':
        #     video_dir = os.path.join(self.dataroot, self.split_folder, cls, frame_folder[:-14])
        #     frame_folder = frame_folder[:-14]
        if not os.path.exists(video_dir):
            continue
        vlen = len(os.listdir(video_dir))

        # select the annotation with highest f1 score
        # highest = np.argmax(f1_consis)
        # if thres > 0 and f1_consis[highest] < thres:
        #     continue

        # change_idices = vdict['substages_myframeidx'][highest]

        # (float)num of frames with min_change_dur/2
        half_dur_2_nframes = min_change_dur * fps / 2.
        # (int)num of frames with min_change_dur/2
        ceil_half_dur_2_nframes = int(np.ceil(half_dur_2_nframes))

        start_offset = 1
        selected_indices = np.arange(start_offset, vlen, ds)

        global_indices = None
        if add_global:
            global_indices = np.linspace(1, vlen, num_global_feats, dtype=np.int32)

        # idx chosen after from downsampling falls in the time range [change-dur/2, change+dur/2]
        # should be tagged as positive(bdy), otherwise negative(bkg)
        GT = [0] * len(selected_indices)

        for idx, (current_idx, lbl) in enumerate(zip(selected_indices, GT)):
            record = dict()
            shift = np.arange(-frame_per_side, frame_per_side)
            shift[shift >= 0] += 1
            shift = shift * ds
            block_idx = shift + current_idx
            block_idx[block_idx < 1] = 1
            block_idx[block_idx > vlen] = vlen
            block_idx = block_idx.tolist()

            record['folder'] = f'{cls}/{frame_folder}'
            record['current_idx'] = current_idx
            record['block_idx'] = block_idx
            record['label'] = lbl
            if global_indices is not None:
                record['global_indices'] = global_indices.tolist()

            SEQ.append(record)

            if lbl == 0:
                neg += 1
            else:
                pos += 1
    print(f' #bdy-{pos}\n #bkg-{neg}\n #total-{pos + neg}.')
    folder = '/'.join(load_file_path.split('/')[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)
    pickle.dump(SEQ, open(load_file_path, "wb"))
    print(len(SEQ))


if __name__ == '__main__':
    prepare_pickle()
