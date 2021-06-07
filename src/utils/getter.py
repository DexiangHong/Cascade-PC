'''
Dataset getter
Dataloader getter
Model selector
'''
import os

import torch
import torch.nn as nn
from torch.utils.data import dataset
import torchvision as tv
from utils.sampler import BalancedBatchSampler, OrderedDistributedSampler, DistBalancedBatchSampler
from utils.augmentation import *
from datasets.MultiFDataset import KineticsGEBDMulFrames, TaposGEBDMulFrames, MultiFDummyDataSet
from modeling.mc3lstmaudioGEBD_cascade import mc3lstmaudioGEBDCascade

transform_series = transforms.Compose([
    Scale(size=(224,224)),
    ToTensor(),
    Normalize() # mean std refer to utils/augmentation.py
])
transform_tv = tv.transforms.Compose([
    tv.transforms.Resize((224,224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PrefetchLoader:
    # prefetch dataloader for kinetics400-GEBD
    def __init__(self, loader):
        self.loader = loader
    
    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for item in self.loader:
            with torch.cuda.stream(stream):
                next_input = item['inp'].cuda(non_blocking=True)
                next_target = item['label'].cuda(non_blocking=True)
            if not first:
                yield {'inp':next_input, 'label':next_target, 'path':item['path']}
            else:
                first = False
            
            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
            path = item['path']

        yield {'inp':input, 'label':target, 'path':path}

    def __len__(self):
        return len(self.loader)
    
    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        keys = ['inp', 'flow', 'global_img', 'global_feats', 'aud_feats']
        try:
            self.nextitem = next(self.loader)
            self.next_input = {}
            for key in keys:
                if key in self.nextitem:
                    self.next_input[key] = self.nextitem[key]
            self.next_target = self.nextitem['label']
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            for key in self.next_input:
                self.next_input[key] = self.next_input[key].cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def getDataset(dataset_name, mode, args):

    if dataset_name.lower() == 'multif_dummy':
        dataset = MultiFDummyDataSet(mode=mode)
    elif dataset_name.lower()=='kinetics_multiframes':
        dataroot = '../../kinetics-400/Kinetics_GEBD_frame' ## FIXME
        # dataroot = os.getenv('GEBD_ROOT', '/mnt/bd/byte-lufficc-data/datasets/GEBD/')
        dataset = KineticsGEBDMulFrames(mode=mode,dataroot=dataroot,frames_per_side=args.frame_per_side,transform=transform_series,
                                        args=args, use_min_val=getattr(args, 'use_min_val', False), use_train_val=getattr(args, 'use_train_val', False),
                                        cascade_label=getattr(args, 'use_cascade', False),
                                        min_change_dur_list=getattr(args, 'min_change_dur_list', [0.4, 0.3, 0.25]),
                                        use_audio=args.use_audio)
    elif dataset_name.lower()=='tapos_multiframes':
        dataroot = '/PATH_TO/TAPOS_instances_frame256'
        dataset = TaposGEBDMulFrames(mode=mode, dataroot=dataroot, frames_per_side=5, tmpl='image_{:05d}.jpg',transform = transform_series, args=args)
    else:
        raise NotImplementedError
    return dataset

def getDataLoader_for_test(dataset, args=None):
    batchsize = args.batch_size
    if hasattr(args, 'distributed') and args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, sampler=sampler, num_workers=8, pin_memory=True, drop_last=False)
    return loader

def getDataLoader(dataset, is_training=True, args=None):
    sampler = None
    if args.distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if args.balance_batch:
                assert args.batch_size == args.n_sample_classes * args.n_samples
                sampler = DistBalancedBatchSampler(dataset, args.num_classes, args.n_sample_classes, args.n_samples, args.seed)
                print(f'rank{args.rank} using balanced batch sampling.')
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            # sampler = OrderedDistributedSampler(dataset)
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)

    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=args.batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=is_training
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    return loader


def getModel(model_name='mc3lstm_cascade_audio',args=None):
    if model_name.lower() == 'mc3lstm_cascade_audio':
        model = mc3lstmaudioGEBDCascade(use_flow=False, frames_per_side=args.frame_per_side, filter_thresh=getattr(args, 'filter_thresh', [0.2, 0.3]))
    else:
        raise NotImplementedError('Model {} not implemented.'.format(model_name))
    return model


