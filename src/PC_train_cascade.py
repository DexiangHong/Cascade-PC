import os
import pickle
import sys
import time
import argparse
from contextlib import suppress

import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.nn.functional as F
import logging
from datetime import datetime
from collections import OrderedDict, defaultdict

from torch.utils.tensorboard import SummaryWriter

from utils.eval import eval_f1
from utils.getter import getModel, getDataset, getDataLoader, data_prefetcher
from utils.optim_factory import create_optimizer
from utils.scheduler import create_scheduler
from utils.checkpoint_saver import CheckpointSaver
from utils.log import setup_default_logging
from utils.cuda import NativeScaler
from utils.model_ema import ModelEmaV2
from utils.clip_grad import dispatch_clip_grad
from utils.helper import resume_checkpoint, load_checkpoint, get_outdir, distribute_bn, update_summary, model_parameters, reduce_tensor
from utils.metric import AverageMeter, accuracy
from utils.distribute import synchronize, all_gather, is_main_process
import json


has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# distributed
parser.add_argument("--distributed", action='store_true', default=False)
parser.add_argument("--local_rank", type=int, default=-1)
# dist_bn
parser.add_argument('--sync-bn', action='store_true', default=False,
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')

# Dataset / Model parameters
parser.add_argument('--data_dir', metavar='DIR', default='/kinetics400/',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--exp', '-e', default='',
                    help='Experiment name')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='val',
                    help='dataset validation split (default: validation)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=2, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')


# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=2, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')



# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Training via mixed precision
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')

# balanced batch sampler
parser.add_argument('--balance-batch', action='store_true', default=False,
                    help='use balanced batch sampler for training')
parser.add_argument('--n-sample-classes', type=int, default=2, 
                    help='#num of classes sampled in one batch')
parser.add_argument('--n-samples', type=int, default=16,
                    help='#samples per class in balanced batch sampling.')
# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=25, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--num-workers', type=int, default=10,
                    help='num workers of dataloader.')
parser.add_argument('--pin-memory', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',           ### config this for output folder
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--use_min_val', default=False, action='store_true')
parser.add_argument('--use_train_val', default=False, action='store_true')
parser.add_argument('--downsample_rate', default=8, type=int)
parser.add_argument('--use_cascade', default=True, action='store_true')
parser.add_argument('--min_change_dur_list', default=[0.4, 0.3, 0.25], type=float, nargs='+')
parser.add_argument('--filter_thresh', default=[0.2, 0.3], type=float, nargs='+')
parser.add_argument('--use_audio', default=False, action='store_true')
parser.add_argument('--use_flow', default=False, action='store_true')
parser.add_argument('--frame_per_side', default=8, type=int)


args = parser.parse_args()

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

GLOBAL_STEP = 0


def writing_scores(abs_path, scores, dict_record, args):
    for path,score in zip(abs_path, scores):
        #/Checkpoint/leiwx/weixian/data/Kinetics_GEBD_frame/val_split/waxing_chest/Swo0mvCOPz4_000235_000245/image_00231.jpg'
        #/Checkpoint/leiwx/weixian/TAPOS/instances_frame256/oxjUeTc_Vag/s00019_2_603_17_620/image_00231.jpg
        frame = path.split('/')[-1]
        frame_idx = int(frame[6:11])
        vid = get_vid_from_path(args=args, path=path)
        if vid not in dict_record.keys():
            dict_record[vid]=dict()
            dict_record[vid]['frame_idx']=[]
            dict_record[vid]['scores']=[]
        dict_record[vid]['frame_idx'].append(frame_idx)
        dict_record[vid]['scores'].append(score)


def get_vid_from_path(args,path):
    if 'kinetics' in args.dataset.lower():
        vid_dir, _ = path.split('/')[-2:]
        vid = vid_dir[:11]
        return vid
    elif 'tapos' in args.dataset.lower():
        vid = '_'.join(path.split('/')[-3:-1])
        return vid
    else:
        raise NotImplementedError


def main():
    args = parser.parse_args()
    setup_default_logging(log_path=os.path.join(args.output, 'log.txt'))

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0


    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp and has_native_amp:
        use_amp = 'native'
        _logger.info(f'Using Pytorch {torch.__version__} amp...')

    torch.manual_seed(args.seed + args.rank)

    model = getModel(model_name=args.model, args=args)

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    # move model to gpu
    model.cuda()
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
     
    # optimizer
    optimizer = create_optimizer(args, model)

    amp_autocast = suppress # do nothing
    loss_scaler = None
    if use_amp=='native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')
   
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank==0
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    #set up distributed training
    if args.distributed:
        if args.local_rank == 0:
            _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[args.local_rank], find_unused_parameters=True)  # can use device str in Torch >= 1.1
    
    # lr schedule
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))


    # create the train and eval dataset
    dataset_train = getDataset(dataset_name=args.dataset, mode=args.train_split,args=args)
    dataset_eval = getDataset(dataset_name=args.dataset, mode=args.val_split, args=args)

    # create loader
    loader_train = getDataLoader(dataset_train, is_training=True, args=args)
    loader_eval = getDataLoader(dataset_eval, is_training=False, args=args)
    
    # set_up loss function
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # set_up checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    summary_writer = None
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            # args.dataset,
            args.model,
            args.exp,
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        args.output_dir = output_dir
        decreasing = True if eval_metric == 'loss' else False
        summary_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            args_dict = args.__dict__
            json.dump(args_dict, f)
        # with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        #     f.write(args_text)
    

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            # eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, epoch=epoch)

            train_metrics = train_one_epoch(epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, summary_writer=summary_writer)

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, epoch=epoch)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)', epoch=epoch)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                lr_scheduler.step(epoch+1, eval_metrics[eval_metric])
            
            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
            
    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, summary_writer=None):
    
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    prefetcher = data_prefetcher(loader)
    input, target = prefetcher.next()
    batch_idx=0
    global GLOBAL_STEP
    #for batch_idx, items in enumerate(loader):
    while input is not None:
        GLOBAL_STEP += 1
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        # input = items['inp']
        # target = items['label'] 
        # if not args.use_prefetcher:
        #     input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        if args.channels_last:
            assert True, 'Should not here!!'
            input = input.contiguous(memory_format=torch.channels_last)

        loss_dict = None
        with amp_autocast():
            loss = model(input, target)
            if isinstance(loss, dict):
                loss_dict = loss
                loss = sum(loss.values()) / len(loss.keys())

        if not args.distributed:
            losses_m.update(loss.item(), input['inp'].size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        if summary_writer is not None and GLOBAL_STEP % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            summary_writer.add_scalar('loss', losses_m.val, global_step=GLOBAL_STEP)
            summary_writer.add_scalar('lr', lr, global_step=GLOBAL_STEP)
            if loss_dict is not None:
                for tag in loss_dict:
                    summary_writer.add_scalar(f'losses/{tag}', loss_dict[tag], global_step=GLOBAL_STEP)

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input['inp'].size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input['inp'].size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input['inp'].size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))


        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        batch_idx += 1

        if args.distributed:
            torch.distributed.barrier()
        input, target = prefetcher.next()
        # end for/while

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', epoch=0):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    model_pred_dict = {}
    with torch.no_grad():
        keys = ['inp', 'flow', 'global_img', 'global_feats', 'aud_feats']
        for batch_idx, items in enumerate(loader):
            inps = {}
            for key in keys:
                if key in items:
                    inps[key] = items[key].float().cuda(non_blocking=True)
            paths = items['path']
            target = items['label'].cuda(non_blocking=True)
            if target.shape[1] > 1:
                target = target[:, -1]

            output = model(inps)
            if isinstance(output,(list, tuple)):
                output = output[0]

            bdy_scores = F.softmax(output, dim=1)[:,1].cpu().numpy()
            writing_scores(paths, bdy_scores, model_pred_dict, args)

            # --------------------------------
            # --------------------------------
            # --------------------------------

            last_batch = batch_idx == last_idx

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target) # FIXME
            acc1 = accuracy(output, target, topk=(1,))
            if isinstance(acc1,(list, tuple)):
                acc1 = acc1[0]

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inps['inp'].size(0))
            top1_m.update(acc1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m)
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

    synchronize()
    data_list = all_gather(model_pred_dict)
    if not is_main_process():
        metrics['F1'] = 0.00
        return metrics

    model_pred_dict = defaultdict(dict)
    for p in data_list:
        for vid in p:
            if 'frame_idx' not in model_pred_dict[vid]:
                model_pred_dict[vid]['frame_idx'] = []
                model_pred_dict[vid]['scores'] = []

            model_pred_dict[vid]['frame_idx'].extend(p[vid]['frame_idx'])
            model_pred_dict[vid]['scores'].extend(p[vid]['scores'])

    for vid in model_pred_dict:
        frame_idx = np.array(model_pred_dict[vid]['frame_idx'])
        scores = np.array(model_pred_dict[vid]['scores'])
        indices = np.argsort(frame_idx)
        model_pred_dict[vid]['frame_idx'] = frame_idx[indices].tolist()
        model_pred_dict[vid]['scores'] = scores[indices].tolist()

    with open(os.path.join(args.output_dir, f'sequence_score_epoch{epoch}.pkl'), 'wb') as f:
        pickle.dump(model_pred_dict, f, pickle.HIGHEST_PROTOCOL)

    gt_path = f'../GEBD_Raw_Annotation/k400_mr345_minval_min_change_duration0.3.pkl'
    f1, rec, prec = eval_f1(model_pred_dict, gt_path)
    print('F1: {:.3f}, Rec: {:.3f}, Prec: {:.3f}'.format(f1, rec, prec))
    metrics['F1'] = f1
    metrics['Rec'] = rec
    metrics['Prec'] = prec
    return metrics


if __name__ == '__main__':
    main()
