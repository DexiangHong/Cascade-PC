import math
import os
import random
import sys
import pickle

import h5py
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
try:
    from datasets.augmentation import Scale, ToTensor, Normalize
except:
    from augmentation import Scale, ToTensor, Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor as ToTensor_torch



def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def pil_flow_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


multi_frames_transform = transforms.Compose([
    Scale(size=(224,224)),
    ToTensor(),
    Normalize()
])

multi_flows_transform = transforms.Compose([
    Scale(size=(224,224)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

audio_transform = transforms.Compose([
    # from
    ToTensor_torch(),
    Resize(size=(128, 80))

])


class KineticsGEBDMulFrames(Dataset):
    def __init__(self, mode = 'train', dataroot='/home/tiger/videos-cuts/kinetics-400/Kinetics_GEBD_frame',
                 frames_per_side=8, tmpl='image_{:05d}.jpg',transform = None, args=None, use_min_val=False,
                 use_train_val=False, cascade_label=True, min_change_dur_list=[0.4, 0.3, 0.25], use_audio=False):
        assert mode.lower() in ['train', 'val', 'valnew', 'test', 'minval'], 'Wrong mode for k400'
        # self.downsample_rate = args.downsample_rate
        self.use_flow = getattr(args, 'use_flow', False)
        self.use_audio = use_audio
        self.downsample_rate = getattr(args, 'downsample_rate', 8)
        self.use_min_val = use_min_val
        self.use_train_val = use_train_val
        self.mode = mode
        self.split_folder = mode + '_' + 'split'
        self.train = self.mode.lower() == 'train'
        self.dataroot = dataroot
        self.frame_per_side = frames_per_side
        self.tmpl = tmpl
        self.thres = 0 if mode.lower() != 'train' else 0
        dynamic_ds = True
        add_global = False
        self.train_file = 'multi-frames-GEBD-train-{}{}-{}-{}.pkl'.format(frames_per_side if not dynamic_ds else f'dynamic{frames_per_side}', '' if self.thres <= 0 else '_thres_{:.1f}'.format(self.thres), use_train_val, self.downsample_rate)
        self.val_file = 'multi-frames-GEBD-{}-{}{}-{}-{}.pkl'.format(mode,frames_per_side if not dynamic_ds else f'dynamic{frames_per_side}', '' if self.thres <= 0 else '_thres_{:.1f}'.format(self.thres), use_min_val, self.downsample_rate)
        print(self.val_file)
        self.load_file = self.train_file if self.mode=='train' else self.val_file
        if add_global:
            p1, p2 = self.load_file.split('.')
            self.load_file = '{}-{}.{}'.format(p1, 'with_global', p2)

        if cascade_label:
            p1, p2 = self.load_file.split('.')
            self.load_file = '{}-{}-{}.{}'.format(p1, 'cascade_label', min_change_dur_list, p2)

        self.load_file_path = os.path.join('./DataAssets', self.load_file)
        
        if not (os.path.exists(self.load_file_path) and os.path.isfile(self.load_file_path)):
            if (args is not None and ((not hasattr(args, 'rank')) or args.rank==0)) or args is None:
                print('Preparing pickle file ...')
                if mode == 'train' and self.use_train_val:
                    anno_path = '../GEBD_Raw_Annotation/k400_mr345_trainval_min_change_duration0.3.pkl'
                elif mode == 'val' and self.use_min_val:
                    anno_path = '../GEBD_Raw_Annotation/k400_mr345_minval_min_change_duration0.3.pkl'
                else:
                    anno_path = '../GEBD_Raw_Annotation/k400_mr345_{}_min_change_duration0.3.pkl'.format(mode)

                self._prepare_pickle(
                    anno_path=anno_path,
                    downsample=3,
                    min_change_dur=0.3, 
                    keep_rate=1,
                    load_file_path=self.load_file_path,
                    thres=self.thres,
                    dynamic_ds=dynamic_ds,
                    add_global=add_global,
                    cascade_label=cascade_label,
                    min_change_dur_list=min_change_dur_list
                )
        if transform is not None:
            self.transform = transform
        else:
            self.transform = multi_frames_transform
        self.transform_flow = multi_flows_transform

        self.seqs = pickle.load(open(self.load_file_path, 'rb'), encoding='lartin1')
        print('Load from {}'.format(self.load_file_path))

        self.seqs = np.array(self.seqs, dtype=object)

        self.use_global_feats = True
        if self.use_global_feats:
            self.global_feats = None
            self.vid_to_idx = None
            # path = os.path.join(self.dataroot, f'GEBD_{self.mode}_clip32_csn_global_features.h5')
            # with h5py.File(path, mode='r') as db:
            #     self.global_feats = db['features'][:]
            #     self.vid_to_idx = {vid.decode('utf-8'): idx for idx, vid in enumerate(db['ids'][:])}

        if self.mode == 'train':
            self.train_labels = torch.LongTensor([dta['label'] for dta in self.seqs])
        else:
            self.val_labels = torch.LongTensor([dta['label'] for dta in self.seqs])

    def __getitem__(self, index):
        if self.use_global_feats and self.global_feats is None:
            mode = 'val' if self.mode == 'minval' else self.mode
            path = os.path.join(self.dataroot, f'GEBD_{mode}_clip32_csn_global_features.h5')
            with h5py.File(path, mode='r') as db:
                self.global_feats = db['features'][:]
                self.vid_to_idx = {vid.decode('utf-8'): idx for idx, vid in enumerate(db['ids'][:])}
            if self.mode == 'train' and self.use_train_val:
                path = os.path.join(self.dataroot, 'GEBD_val_clip32_csn_global_features.h5')
                with h5py.File(path, mode='r') as db:
                    length = len(self.global_feats)
                    self.global_feats = np.concatenate([self.global_feats, db['features'][:]], axis=0)
                    vid_to_idx_2 = {vid.decode('utf-8'): idx+length for idx, vid in enumerate(db['ids'][:])}
                    self.vid_to_idx.update(vid_to_idx_2)

        item = self.seqs[index]
        block_idx = item['block_idx']
        folder = item['folder']
        vid = folder.split('/')[-1][:11]

        current_idx = item['current_idx']

        if self.mode == 'train' and self.use_train_val:
            video_dir1 = os.path.join(self.dataroot, self.split_folder, folder)
            # video_dir2 = os.path.join(self.dataroot, 'val_split', folder)
            if os.path.exists(video_dir1):
                split_folder = self.split_folder
            else:
                split_folder = 'val_split'
        else:
            split_folder = self.split_folder

        img = self.transform([pil_loader(
            os.path.join(self.dataroot, split_folder, folder, self.tmpl.format(i))
        ) for i in block_idx])

        img = torch.stack(img, dim=0)

        use_flow = self.use_flow
        if use_flow:
            flow_x = self.transform_flow([pil_flow_loader(
                os.path.join(self.dataroot, split_folder + '_flow', folder, 'flow_x_{:05d}.jpg'.format(max(i - 2, 0)))
            ) for i in block_idx])

            flow_y = self.transform_flow([pil_flow_loader(
                os.path.join(self.dataroot, split_folder + '_flow', folder, 'flow_y_{:05d}.jpg'.format(max(i - 2, 0)))
            ) for i in block_idx])

            flow = [torch.cat([x, y]) for x, y in zip(flow_x, flow_y)]
            flow = torch.stack(flow, dim=0)
        else:
            flow = None

        global_img = None
        if 'global_indices' in item:
            global_indices = item['global_indices']
            global_img = self.transform([pil_loader(
                os.path.join(self.dataroot, self.split_folder, folder, self.tmpl.format(i))
            ) for i in global_indices])

        if global_img is not None:
            global_img = torch.stack(global_img, dim=0)

        global_feats = None
        if self.use_global_feats:
            idx = self.vid_to_idx[vid]
            global_feats = self.global_feats[idx]

        if self.use_audio:
            aud_path = os.path.join(self.dataroot, split_folder+'_audio', folder+'.npy')
            vlen = len(os.listdir(os.path.join(self.dataroot, split_folder, folder)))
            if os.path.exists(aud_path):
                aud_feat_whole = np.load(aud_path)
                aud_len = aud_feat_whole.shape[0]
                aud_single_len = aud_len / vlen
                # aud_feat = np.concatenate([aud_feat_whole[int((i-1)*aud_single_len):min(int(i*aud_single_len), aud_len), :] for i in block_idx], axis=0)
                start = block_idx[0]
                end = block_idx[-1]
                aud_feat = aud_feat_whole[int((start-1)*aud_single_len):min(int(end*aud_single_len), aud_len), :]

                try:
                    aud_feat = audio_transform(aud_feat)
                except Exception as e:
                    print(aud_len, aud_single_len, aud_path, aud_feat.shape, e)
                    aud_feat = torch.zeros([1, 128, 80], dtype=torch.float32)
            else:
                aud_feat = torch.zeros([1, 128, 80], dtype=torch.float32)

        else:
            aud_feat = None

        sample = {
            'inp':img,
            'label':torch.as_tensor(item['label'], dtype=torch.int64),
            'path': os.path.join(self.dataroot, split_folder, folder, self.tmpl.format(current_idx))
        }

        if flow is not None:
            sample['flow'] = flow

        if global_img is not None:
            sample['global_img'] = global_img

        if global_feats is not None:
            sample['global_feats'] = global_feats

        if self.use_audio:
            sample['aud_feats'] = aud_feat

        return sample

    def __len__(self):
        return len(self.seqs)
    
    def _prepare_pickle(self,
        anno_path='../GEBD_Raw_Annotation/k400_mr345_train_min_change_duration0.3.pkl',downsample=3,min_change_dur=0.3,
                        keep_rate=0.8, load_file_path='./data/multi-frames-train.pkl', thres=-1.0, dynamic_ds=False,
                        add_global=False, cascade_label=False, min_change_dur_list=[0.4, 0.3, 0.25]):

        if dynamic_ds:
            print('Using Dynamic downsample!')

        is_train = 'train' in load_file_path
        # prepare file for multi-frames-GEBD
        # dict_train_ann
        with open(anno_path,'rb') as f:
            dict_train_ann = pickle.load(f, encoding='lartin1')

        # downsample factor: sample one every `ds` frames
        ds = downsample
        num_global_feats = 10

        SEQ = []
        neg = 0
        pos = 0
        print(len(dict_train_ann.keys()))

        for vname in tqdm(dict_train_ann.keys()):
            vdict = dict_train_ann[vname]
            # vlen = vdict['num_frames']
            fps = vdict['fps']
            if dynamic_ds:
                ds = max(math.ceil(fps / self.downsample_rate), 1)
            else:
                ds = downsample

            f1_consis = vdict['f1_consis']
            path_frame = vdict['path_frame']
            # print(path_frame.split('/'))
            cls, frame_folder = path_frame.split('/')[:2]
            # if self.mode == 'train':
            video_dir = os.path.join(self.dataroot, self.split_folder, cls, frame_folder)
            if self.use_train_val and self.mode == 'train':
                video_dir1 = os.path.join(self.dataroot, self.split_folder, cls, frame_folder)
                video_dir2 = os.path.join(self.dataroot, 'val_split', cls, frame_folder)
                if os.path.exists(video_dir1):
                    video_dir = video_dir1
                else:
                    video_dir = video_dir2
            # elif self.mode == 'val':
            #     video_dir = os.path.join(self.dataroot, self.split_folder, cls, frame_folder[:-14])
            #     frame_folder = frame_folder[:-14]
            if not os.path.exists(video_dir):
                continue
            vlen = len(os.listdir(video_dir))

            # select the annotation with highest f1 score
            highest = np.argmax(f1_consis)
            if thres > 0 and np.mean(f1_consis) < thres:
                continue

            change_idices = vdict['substages_myframeidx'][highest]
            
            # (float)num of frames with min_change_dur/2 
            half_dur_2_nframes = min_change_dur * fps / 2.
            # (int)num of frames with min_change_dur/2
            if cascade_label:
                half_dur_2_nframes_list = [change_dur * fps / 2. for change_dur in min_change_dur_list]

            start_offset = 1
            selected_indices = np.arange(start_offset, vlen, ds)

            global_indices = None
            if add_global:
                global_indices = np.linspace(1, vlen, num_global_feats, dtype=np.int32)

            
            # idx chosen after from downsampling falls in the time range [change-dur/2, change+dur/2] 
            # should be tagged as positive(bdy), otherwise negative(bkg)
            GT = []
            for i in selected_indices:
                if cascade_label:
                    GT.append([0] * len(min_change_dur_list))
                    for dur_idx, half_dur in enumerate(half_dur_2_nframes_list):
                        for change in change_idices:
                            if i >= change - half_dur and i <= change + half_dur:
                                GT[-1][dur_idx] = 1
                                break
                else:
                    GT.append(0)
                    for change in change_idices:
                        if i >= change - half_dur_2_nframes and i <= change + half_dur_2_nframes:
                            GT.pop()  # pop '0'
                            GT.append(1)
                            break
            assert(len(selected_indices)==len(GT),'length frame indices is not equal to length GT.') 
            
            for idx,(current_idx,lbl) in enumerate(zip(selected_indices, GT)):
                # for multi-frames input
                if self.train and random.random()>keep_rate:
                    continue

                record = dict()
                shift = np.arange(-self.frame_per_side, self.frame_per_side)
                shift[shift>=0]+=1
                shift = shift*ds
                block_idx = shift + current_idx
                block_idx[block_idx<1] = 1
                block_idx[block_idx>vlen] = vlen
                block_idx = block_idx.tolist()

                record['folder'] = f'{cls}/{frame_folder}'
                record['current_idx'] = current_idx
                record['block_idx']= block_idx
                record['label'] = lbl
                if global_indices is not None:
                    record['global_indices'] = global_indices.tolist()

                SEQ.append(record)

                if cascade_label:
                    pos += sum(lbl)
                    neg += (len(min_change_dur_list) - sum(lbl))
                else:
                    if lbl == 0:
                        neg += 1
                    else:
                        pos += 1
        print(f' #bdy-{pos}\n #bkg-{neg}\n #total-{pos+neg}.')
        folder = '/'.join(load_file_path.split('/')[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(SEQ, open( load_file_path, "wb"))
        print(len(SEQ))


class TaposGEBDMulFrames(Dataset):
    def __init__(self, mode = 'train', dataroot='/PATH_TO/TAPOS/instances_frame256', frames_per_side=5, tmpl='image_{:05d}.jpg',transform = None, args=None):
        assert mode.lower() in ['train', 'val'], 'Wrong mode for TAPOS'
        self.mode = mode
        self.train = self.mode.lower() == 'train'
        self.dataroot = dataroot
        self.frame_per_side = frames_per_side
        self.tmpl = tmpl
        self.train_file = 'multi-frames-TAPOS-train-{}.pkl'.format(frames_per_side)
        self.val_file = 'multi-frames-TAPOS-val-{}.pkl'.format(frames_per_side)
        self.load_file = self.train_file if self.mode.lower()=='train' else self.val_file
        self.load_file_path = os.path.join('./DataAssets', self.load_file)
        
        if not (os.path.exists(self.load_file_path) and os.path.isfile(self.load_file_path)):
            if (args is not None and args.rank==0) or args is None:
                print('Preparing pickle file ...')
                self._prepare_pickle(
                    anno_path='PATH_TO/TAPOS_{}_anno.pkl'.format(mode), # FIXME
                    downsample=3,
                    min_change_dur=0.3, 
                    keep_rate=1.,
                    load_file_path=self.load_file_path
                )
        if transform is not None:
            self.transform = transform
        else:
            self.transform = multi_frames_transform
        
        self.seqs = pickle.load(open(self.load_file_path, 'rb'), encoding='lartin1')
        self.seqs = np.array(self.seqs, dtype=object)
        

        if self.mode == 'train':
            self.train_labels = torch.LongTensor([dta['label'] for dta in self.seqs])
        else:
            self.val_labels = torch.LongTensor([dta['label'] for dta in self.seqs])
    

    
    def __getitem__(self, index):
        item = self.seqs[index]
        block_idx = item['block_idx']
        folder = item['folder']
        current_idx = item['current_idx']
        img = self.transform([pil_loader(
            os.path.join(self.dataroot, folder, self.tmpl.format(i))
        ) for i in block_idx])

        img = torch.stack(img, dim=0)
        return {
            'inp':img,
            'label':item['label'],
            'path': os.path.join(self.dataroot, folder, self.tmpl.format(current_idx))
        }
    
    def __len__(self):
        return len(self.seqs)

    def _prepare_pickle(self,
        anno_path='PATH_TO/TAPOS/save_output/TAPOS_train_anno.pkl',downsample=3,min_change_dur=0.3, keep_rate=0.8, load_file_path='./data/multi-frames-train.pkl'):
        # prepare file for multi-frames-GEBD
        # dict_train_ann
        with open(anno_path,'rb') as f:
            dict_train_ann = pickle.load(f, encoding='lartin1')

        # Some fields in anno for reference
        # {'raw': {'action': 11, 'substages': [0, 79, 195], 'total_frames': 195, 'shot_timestamps': [43.36, 53.48], 'subset': 'train'}, 
        # 'path': 'yMK2zxDDs2A/s00004_0_100_7_931', 
        # 'myfps': 25.0, 
        # 'my_num_frames': 197, 
        # 'my_duration': 7.88, 
        # 'my_substages_frameidx': [79]
        # }

        # downsample factor: sample one every `ds` frames
        ds = downsample

        SEQ = []
        neg = 0
        pos = 0

        for vname in dict_train_ann.keys():
            vdict = dict_train_ann[vname]

            vlen = vdict['my_num_frames']
            fps = vdict['myfps']
            path_frame = vdict['path']
            
            # select the annotation with highest f1 score
            change_idices = vdict['my_substages_frameidx']
            
            # (float)num of frames with min_change_dur/2 
            half_dur_2_nframes = min_change_dur * fps / 2.
            # (int)num of frames with min_change_dur/2 
            ceil_half_dur_2_nframes = int(np.ceil(half_dur_2_nframes))

            start_offset = np.random.choice(ds)+1
            selected_indices = np.arange(start_offset, vlen, ds)
            
            # idx chosen after from downsampling falls in the time range [change-dur/2, change+dur/2] 
            # should be tagged as positive(bdy), otherwise negative(bkg)
            GT = []
            for i in selected_indices:
                GT.append(0)
                for change in change_idices:
                    if i >= change-half_dur_2_nframes and i<= change+half_dur_2_nframes:
                        GT.pop()       #pop '0' 
                        GT.append(1)   
                        break
            assert(len(selected_indices)==len(GT),'length frame indices is not equal to length GT.') 
            
            for idx,(current_idx,lbl) in enumerate(zip(selected_indices, GT)):
                # for multi-frames input
                if self.train and random.random()>keep_rate:
                    continue

                record = dict()
                shift = np.arange(-self.frame_per_side, self.frame_per_side)
                shift[shift>=0]+=1
                shift = shift*ds
                block_idx = shift + current_idx
                block_idx[block_idx<1] = 1
                block_idx[block_idx>vlen] = vlen
                block_idx = block_idx.tolist()

                record['folder'] = path_frame
                record['current_idx'] = current_idx
                record['block_idx']= block_idx
                record['label'] = lbl
                SEQ.append(record)
                
                if lbl==0:
                    neg+=1
                else:
                    pos+=1
        print(f' #bdy-{pos}\n #bkg-{neg}\n #total-{pos+neg}.')
        folder = '/'.join(load_file_path.split('/')[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(SEQ, open( load_file_path, "wb"))
        print(len(SEQ))


###################################################################################
# Dummy dataset for debugging
###################################################################################
class MultiFDummyDataSet(Dataset):
    def __init__(self, mode = 'train', transform = None, args=None):
        assert mode.lower() in ['train', 'val', 'test','valnew'], 'Wrong split'
        self.mode = mode
        self.train = self.mode.lower() =='train'
        self.args = args

        
        if transform is not None:
            self.transform = transform

        self.train_labels = torch.LongTensor(np.random.choice([0,1],1000000))
        self.val_labels = torch.LongTensor(np.random.choice([0,1],1000000))
        self.load_file = self.train_labels if self.mode == 'train' else self.val_labels
        self.load_file = self.train_labels if self.mode =='train' else self.val_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, label) where target is class_index of the target class.
        """
        label = self.load_file[index]
        inp = torch.randn(10,3,224,224)


        return {'inp':inp, 'label':label}

    def __len__(self):
        return len(self.load_file)

if __name__ == '__main__':
    # KineticsGEBDMulFrames
    dataset = KineticsGEBDMulFrames(mode='val')
    print(dataset[24511])
