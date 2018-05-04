import time

import torch.multiprocessing as mp

"""
if __name__ == '__main__':
    print("forkserver")
    mp.set_start_method('forkserver')
"""

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from lib.utils.target_assigner import Matcher, ImageBalancedPositiveNegativeSampler, batch_box_iou
from lib.utils.box_coder import BoxCoder
from lib.utils.generate_anchors import generate_anchors
from lib.utils.meters import MedianMeter, AverageMeter
from lib.utils.losses import smooth_l1_loss

import logging
import sys

import json


def train_one_epoch(model, optimizer, data_loader, epoch, args):
    logger = logging.getLogger(__name__)
    json_logger = logging.getLogger('json')
    logger.info('Start epoch')

    losses = MedianMeter()
    losses_rpn_bbox = MedianMeter()
    losses_rpn_cls = MedianMeter()
    losses_bbox = MedianMeter()
    losses_cls = MedianMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = MedianMeter()
    fg_accuracies = MedianMeter()
    bg_accuracies = MedianMeter()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    end = time.time()
    epoch_time = time.time()
    for i, batch in enumerate(data_loader):
        data_time.update(time.time() - end)

        # scheduler.step()
        # adjust_learning_rate(optimizer, (epoch - 1) * len(data_loader) + i)

        if torch.cuda.is_available():
            batch = tuple(t.cuda(non_blocking=True) for t in batch)

        imgs, gt_boxes, gt_labels, img_idx_gt, im_shape = batch

        # scores, box_deltas, rpn_scores, rpn_box_deltas, boxes, all_anchors, img_idx_proposal, inds_inside, idxs = model(imgs, im_shape)
        # rpn_scores, rpn_box_deltas, all_anchors, img_idx_proposal, inds_inside = model(imgs, im_shape)
        if args.rpn_only:
            rpn_scores, rpn_box_deltas, rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = model(imgs, im_shape, gt_boxes, img_idx_gt, gt_labels)
        else:
            scores, box_deltas, rpn_scores, rpn_box_deltas, rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, num_pos = model(imgs, im_shape, gt_boxes, img_idx_gt, gt_labels)

            loss_classif = F.cross_entropy(scores, labels, size_average=True)
            loss_box_reg = smooth_l1_loss(box_deltas, bbox_targets, bbox_inside_weights, bbox_outside_weights, 1) / box_deltas.shape[0]

        #rpn_labels, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = model.rpn.prepare_targets(all_anchors, gt_boxes, img_idx_proposal, img_idx_gt, inds_inside)
        #labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, idxs_, num_pos  = model.prepare_targets(boxes, gt_boxes, idxs, img_idx_gt, gt_labels)


        #scores = scores[idxs_]
        #box_deltas = box_deltas[idxs_]


        N, A, H, W = rpn_scores.shape
        rpn_scores = rpn_scores.contiguous().view(-1)
        rpn_box_deltas = rpn_box_deltas.view(N, A, 4, H, W).permute(0, 1, 3, 4, 2).contiguous().view(-1, 4)
        is_useful = rpn_labels >= 0
        rpn_classif_loss = F.binary_cross_entropy_with_logits(rpn_scores[is_useful], rpn_labels[is_useful].float())
        rpn_loss_box_reg = smooth_l1_loss(rpn_box_deltas, rpn_box_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights) / N

        if args.rpn_only:
            loss = rpn_classif_loss + rpn_loss_box_reg
            losses.update(loss.item(), is_useful.float().sum().item())
        else:
            loss = loss_classif + loss_box_reg + rpn_classif_loss + rpn_loss_box_reg

            losses_bbox.update(loss_box_reg.item(), scores.size(0))
            losses_cls.update(loss_classif.item(), scores.size(0))

            losses.update(loss.item(), scores.size(0))

        losses_rpn_bbox.update(rpn_loss_box_reg.item(), is_useful.float().sum().item())
        losses_rpn_cls.update(rpn_classif_loss.item(), is_useful.float().sum().item())


        """
        pred_cls = scores.detach().max(1)[1]
        accuracy = (labels == pred_cls).float().mean() * 100

        if num_pos > 0:
            fg_accuracy = (labels[:num_pos] == pred_cls[:num_pos]).float().mean() * 100
        else:
            fg_accuracy = 0
        bg_accuracy = (labels[num_pos:] == pred_cls[num_pos:]).float().mean() * 100
        accuracies.update(accuracy, scores.size(0))
        fg_accuracies.update(fg_accuracy, num_pos)
        bg_accuracies.update(bg_accuracy, scores.size(0) - num_pos)
        """
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if False:# i % 5000 == 0 and i > 0:
            logger.info('Checkpointing')
            torch.save({'model': model.state_dict(), 'optimizer':optimizer},
                    '/checkpoint02/fmassa/detectron_v2_faster_rcnn_R50.pth')

        if i % 20 == 0 or i == len(data_loader) - 1:
            logger.info('Epoch [{0}] {1}/{2}\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Acc: {accuracy.val:.4f} ({accuracy.avg:.4f})\t'
                  #'FG Acc: {fg_acc.val:.4f} ({fg_acc.avg:.4f})\t'
                  #'BG Acc: {bg_acc.val:.4f} ({bg_acc.avg:.4f})\t'
                  'loss_cls: {losses_cls.val:.4f} ({losses_cls.avg:.4f})\t'
                  'loss_bbox: {losses_bbox.val:.4f} ({losses_bbox.avg:.4f})\t'
                  'loss_rpn_cls: {losses_rpn_cls.val:.4f} ({losses_rpn_cls.avg:.4f})\t'
                  'loss_rpn_bbox: {losses_rpn_bbox.val:.4f} ({losses_rpn_bbox.avg:.4f})\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      epoch, i, len(data_loader), loss=losses,
                      accuracy=accuracies, fg_acc=fg_accuracies, bg_acc=bg_accuracies,
                      losses_cls=losses_cls, losses_bbox=losses_bbox,
                      losses_rpn_cls=losses_rpn_cls, losses_rpn_bbox=losses_rpn_bbox,
                      batch_time=batch_time, data_time=data_time))
            json_logger.info(json.dumps({
                'loss': losses.val,
                'loss_cls': losses_cls.val,
                'loss_bbox': losses_bbox.val,
                'loss_rpn_cls': losses_rpn_cls.val,
                'loss_rpn_bbox': losses_rpn_bbox.val,
                'epoch': epoch,
                'iter': i
                }))


    epoch_time = time.time() - epoch_time
    epoch_time_hour = int(epoch_time // 3600)
    epoch_time_min = int((epoch_time // 60) % 60)
    epoch_time_sec = int(epoch_time % 60)
    logger.info('Total epoch time: {}h{}m{}s'.format(epoch_time_hour, epoch_time_min, epoch_time_sec))


def adjust_learning_rate(optimizer, it):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = 0.01

    if it < 500:  # cfg.SOLVER.WARM_UP_ITERS:
        alpha = it / 500.0  # cfg.SOLVER.WARM_UP_ITERS

        # warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
        warmup_factor = 1.0 / 3 * (1 - alpha) + alpha
        lr *= warmup_factor
    # print('lr', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def my_normalize(x):
    return (x[[2,1,0]] * 255) - torch.Tensor([102.9801, 115.9465, 122.7717]).view(3,1,1)

if __name__ == '__main__':
    import argparse
    import os, errno
    import sys

    import torch.utils.data
    import torchvision.transforms as T

    from lib.collate_fn import CollatorTraining, Collator
    from lib.data_transforms.transforms import Resize, RandomHorizontalFlip, MyTransf
    from lib.data_transforms.transform_dataset import TransformDataset

    parser = argparse.ArgumentParser(description='PyTorch Faster R-CNN Training')
    parser.add_argument('--save', default='/checkpoint02/fmassa/bottleneck_debug', metavar='DIR')
    parser.add_argument('--dataset', default='coco', choices=['coco', 'pascal'])
    parser.add_argument('--epochs', default=10, type=int, metavar='N')
    parser.add_argument('--num-workers', default=4, type=int)

    parser.add_argument('--checkpoint', default='', metavar='DIR')

    parser.add_argument('--lr', '--learning-rate', default=0.00125, type=float, metavar='LR')
    parser.add_argument('--lr-steps', default=[6, 9], nargs='+', type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)

    parser.add_argument('--imgs-per-batch', default=2, type=int)
    parser.add_argument('--rpn-batch-size-per-img', default=256, type=int)
    parser.add_argument('--batch-size-per-img', default=512, type=int)

    parser.add_argument('--distributed', dest='distributed', action='store_true')
    parser.add_argument('--no-distributed', dest='distributed', action='store_false')
    parser.set_defaults(distributed=True)

    parser.add_argument('--ar-group', dest='ar_group', action='store_true')
    parser.add_argument('--no-ar-group', dest='ar_group', action='store_false')
    parser.set_defaults(ar_group=True)

    parser.add_argument('--rpn-only', dest='rpn_only', action='store_true')
    parser.set_defaults(rpn_only=False)

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
            init_method='env://')

    try:
        os.makedirs(args.save)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if args.local_rank == 0:
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    json_logger = logging.getLogger('json')
    json_logger.setLevel(logging.DEBUG)
    if args.local_rank == 0:
        # TODO this is an approximation: we are discarding the losses from
        # the other GPUs
        jh = logging.FileHandler(os.path.join(args.save, 'log.json'))
        jh.setLevel(logging.DEBUG)
        json_logger.addHandler(jh)

    # TODO this avoids multiple processes overwriting the implementation
    if args.local_rank == 0:
        from lib.layers import _C
    else:
        from time import sleep
        sleep(7)

    logger.info(args)
    if args.distributed:
        from torch.distributed import get_world_size
        logger.info('World size: {}'.format(get_world_size()))

    import random
    torch.manual_seed(3)
    random.seed(3)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    if True:
        normalize = T.Lambda(my_normalize)
    transforms = T.Compose([T.ToTensor(), normalize])

    if args.dataset == 'coco':
        from datasets.coco import COCODataset
        val_ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_val2017_mod.json'
        val_data_dir = '/datasets01/COCO/060817/val2014/'

        valminusminival_ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_valminusminival2017.json'
        from torch.utils.data.dataset import ConcatDataset

        ann_file = '/datasets01/COCO/060817/annotations/instances_train2014.json'
        data_dir = '/datasets01/COCO/060817/train2014/'
        dataset_train = COCODataset(ann_file, data_dir)
        dataset_valminusminival = COCODataset(valminusminival_ann_file, val_data_dir)
        dataset = ConcatDataset([dataset_train, dataset_valminusminival])
        # dataset = COCODataset(val_ann_file, val_data_dir)
        dataset_val = COCODataset(val_ann_file, val_data_dir)
        num_classes = 81
    elif args.dataset == 'pascal':
        from datasets.pascal_voc import PascalVOC
        from lib.data_transforms.transforms import VOCAnnotations
        dataset = PascalVOC("/datasets01/VOC/060817/VOCdevkit/", "train",
                target_transform=VOCAnnotations(keep_difficult=False))
        num_classes = 21
    else:
        raise ValueError('Unsupported dataset {}'.format(args.dataset))

    logger.info(dataset)

    from torch.utils.data.sampler import BatchSampler, RandomSampler
    if args.distributed:
        import torch.utils.data.distributed
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    drop_last = True
    if args.ar_group:
        from lib.samplers import GroupedBatchSampler
        aspect_ratios = []
        # TODO find a better way of handling this with concatdataset
        for i in range(len(dataset)):
            img_data = dataset.get_img_info(i)
            aspect_ratio = float(img_data['height']) / float(img_data['width'])
            aspect_ratios.append(aspect_ratio)
        groups = [ar > 1 for ar in aspect_ratios]
        batch_sampler = GroupedBatchSampler(groups, args.imgs_per_batch, drop_last, sampler)
    else:
        batch_sampler = BatchSampler(sampler, args.imgs_per_batch, drop_last)


    tt = MyTransf(transforms)
    ttt = T.Compose([RandomHorizontalFlip(), Resize(800, 1333), tt])
    # ttt = T.Compose([RandomHorizontalFlip(), Resize(600, 1000), tt])
    dataset = TransformDataset(dataset, ttt)
    dataset_val = TransformDataset(dataset_val, T.Compose([Resize(800, 1333), tt]))

    collate_fn = CollatorTraining()
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1,
            collate_fn=Collator(), num_workers=args.num_workers, pin_memory=True)

    from lib.faster_rcnn import fasterrcnn_resnet18, C2ResNetFasterRCNN
    if False:
        model = fasterrcnn_resnet18(num_classes=num_classes)
    else:
        import pickle
        with open('/private/home/fmassa/R-50.pkl', 'rb') as f:
        # with open('/private/home/fmassa/github/Detectron/detectron_c2_output/train/coco_2014_minival/fast_rcnn/model_final.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
 
        # model = C2ResNetFasterRCNN('/private/home/fmassa/model_final_faster_rcnn.pkl').cuda()
        rpn_batch_size = args.rpn_batch_size_per_img * args.imgs_per_batch
        batch_size = args.batch_size_per_img * args.imgs_per_batch
        model = C2ResNetFasterRCNN(data, num_classes=num_classes, rpn_batch_size=rpn_batch_size, batch_size=batch_size, rpn_only=args.rpn_only).cuda()

    logger.info(model)

    weight_decay = 0.0001
    momentum = 0.9

    model.train()
    if torch.cuda.is_available():
        model.cuda()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
            device_ids=[args.local_rank], output_device=args.local_rank)

    params = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value],'lr': args.lr * (1 + 1), \
                        'weight_decay': 0}]
            else:
                params += [{'params': [value],'lr': args.lr, 'weight_decay': weight_decay}]

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=momentum,
                                weight_decay=weight_decay)


    if isinstance(args.lr_steps, list):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_steps, gamma=args.lr_gamma)


    start_epoch = 1
    if args.checkpoint:
        logger.info('Loading from checkpoint {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        # start_epoch = 11
        # scheduler.last_epoch = 10

    # TODO check if want to start from 1. the scheduler starts at zero, so can be confusing
    for epoch in range(start_epoch, args.epochs + 1):
        scheduler.step()
        model.train()
        logger.info('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        if args.distributed:
            sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, epoch, args)

        if args.local_rank == 0:
            torch.save({
                'model': model.state_dict(), 'optimizer':optimizer.state_dict(),
                'epoch': epoch
                }, os.path.join(args.save, 'model_{}.pth'.format(epoch)))

            if epoch == args.epochs and True:# args.evaluate:
                if args.rpn_only:
                    from testing_rpn import evaluate
                else:
                    from testing_faster_rcnn import evaluate
                logger.info('Evaluating')
                model.eval()
                results = evaluate(model.module, val_data_loader, args)
                logger.info('Result: {}'.format(results))
