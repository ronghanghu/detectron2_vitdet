import time

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from lib.utils.target_assigner import Matcher, ImageBalancedPositiveNegativeSampler, batch_box_iou
from lib.utils.box_coder import BoxCoder
from lib.utils.generate_anchors import generate_anchors

import math


def meshgrid(x, y=None):
    if y is None:
        y = x
    m, n = x.size(0), y.size(0)
    grid_x = x[None].expand(n, m).contiguous()
    grid_y = y[:, None].expand(n, m).contiguous()
    return grid_x, grid_y


class AnchorGridGenerator(object):
    def __init__(self, stride, sizes, aspect_ratios, max_size):
        cell_anchors = generate_anchors(stride, sizes, aspect_ratios).float()
        self.stride = stride
        self.cell_anchors = cell_anchors
        self.max_size = max_size

    def generate(self, coarsest_stride=None):
        coarsest_stride = self.stride  # TODO fix
        fpn_max_size = coarsest_stride * math.ceil(
            self.max_size / float(coarsest_stride)
        )
        # TODO not good because stride is already in cell_anchors
        field_size = int(math.ceil(fpn_max_size / float(self.stride)))
        shifts = torch.arange(0, field_size) * self.stride
        shift_x, shift_y = meshgrid(shifts, shifts)
        shift_x = shift_x.view(-1)
        shift_y = shift_y.view(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        A = self.cell_anchors.size(0)
        K = shifts.size(0)
        field_of_anchors = (
            self.cell_anchors.view(1, A, 4) +
            shifts.view(1, K, 4).permute(1, 0, 2)
        )
        # field_of_anchors = field_of_anchors.view(K * A, 4)
        field_of_anchors = field_of_anchors.view(field_size, field_size, A, 4)

        return field_of_anchors


def train_one_epoch(model, optimizer, data_loader, epoch):
    print('Start epoch')

    losses = MedianMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = MedianMeter()
    fg_accuracies = MedianMeter()
    bg_accuracies = MedianMeter()

    proposal_matcher = Matcher(0.5, 0.0)
    # fg_bg_sampler = BalancedPositiveNegativeSampler(128, 0.25)
    fg_bg_sampler = ImageBalancedPositiveNegativeSampler(128, 0.25)
    # fg_bg_sampler = BalancedPositiveNegativeSampler(4096, 0.25)

    box_coder = BoxCoder(weights=(10., 10., 5., 5.))

    anchor_grid_generator = AnchorGridGenerator(16, (32, 64, 128, 256, 512), (0.5, 1, 2), max_size=1000)
    
    anchors_template = anchor_grid_generator.generate()
    print('anchors template size', anchors_template.shape)

    def _create_hook(n):
        def _hook(module, gi, go):
            if isinstance(gi, tuple):
                gi = gi[0]
            if isinstance(go, tuple):
                go = go[0]
            if gi is None:
                return
            if go is None:
                return
            print('name: ', n, 'norm:',  gi.data.norm(), go.data.norm(), go.data.min(), go.data.max())
        return _hook

    loss_list = []
    labels_list = []
    norm_outputs = {}
    norm_weights = {}
    def _create_forward_hook(name):
        def _hook(module, i, o):
            if name not in norm_outputs:
                norm_outputs[name] = []
            norm_outputs[name].append((o.data.abs().max(), o.data.norm()))
        return _hook

    for n, m in model.named_children():
        m.register_forward_hook(_create_forward_hook(n))

    # for n, m in model.named_modules():
    #     m.register_backward_hook(_create_hook(n))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    end = time.time()
    epoch_time = time.time()
    to_save_data = []
    for i, batch in enumerate(data_loader):
        data_time.update(time.time() - end)

        # scheduler.step()
        # adjust_learning_rate(optimizer, (epoch - 1) * len(data_loader) + i)

        if torch.cuda.is_available():
            batch = tuple(t.cuda(non_blocking=True) for t in batch)

        imgs, proposals, gt_boxes, gt_labels, img_idx_proposal, img_idx_gt, im_shape = batch

        batch_size = imgs.size(0)
        all_anchors = anchors_template.view(1, -1, 4)
        num_anchors_template = all_anchors.size(1)
        all_anchors = all_anchors.expand(batch_size, num_anchors_template, 4).contiguous().view(-1, 4)
        img_idx_proposal_full = torch.arange(batch_size, dtype=torch.int64)[:, None].expand(batch_size, num_anchors_template).contiguous().view(-1)

        if torch.cuda.is_available():
            all_anchors = all_anchors.cuda()
            img_idx_proposal_full = img_idx_proposal_full.cuda()

        # TODO make this line more understandabla
        im_height, im_width = tuple(t.squeeze(1) for t in im_shape[img_idx_proposal_full].split(1, dim=1))

        straddle_thresh = 0
        inds_inside = torch.nonzero(
            (all_anchors[:, 0] >= -straddle_thresh) &
            (all_anchors[:, 1] >= -straddle_thresh) &
            (all_anchors[:, 2] < im_width + straddle_thresh) &
            (all_anchors[:, 3] < im_height + straddle_thresh)
        ).squeeze(1)
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :] 
        proposals = anchors.cuda()
        img_idx_proposal = img_idx_proposal_full[inds_inside].cuda()

        # QUESTION: where to add the proposals?
        # the problem is in the transforms not handling it yet

        # TODO maybe make this a list of matrices if this is too big
        match_quality_matrix = batch_box_iou(gt_boxes, proposals, img_idx_gt, img_idx_proposal)
        matched_idxs = proposal_matcher(match_quality_matrix)
        # sampled_idxs, num_pos = fg_bg_sampler(matched_idxs)
        sampled_idxs, num_pos = fg_bg_sampler(matched_idxs, img_idx_proposal)

        num_proposals = sampled_idxs.size(0)

        sampled_proposals = proposals[sampled_idxs]
        # TODO replace with torch.where once Tensors and Variables are merged? Maybe not
        labels = proposals.new(num_proposals).zero_()
        labels[:num_pos] = gt_labels[matched_idxs[sampled_idxs[:num_pos]]]
        # labels[:num_pos] = 1
        # labels.bernoulli_()
        labels = labels.long()
        # labels = proposals.new(sampled_idxs.size(0)).uniform_().mul_(81).long()
        img_idx = img_idx_proposal[sampled_idxs]
        # to_save_data.append({'imgs': imgs.cpu(), 'labels': labels.cpu(), 'proposals': proposals.cpu(), 'img_idx':img_idx.cpu()})

        # get bbox regression targets
        num_classes = 81  # TODO

        aligned_gt_boxes = gt_boxes[matched_idxs[sampled_idxs]]
        box_targets = box_coder.encode(aligned_gt_boxes, sampled_proposals)

        def expand_bbox_targets(box_targets, labels, num_classes):
            num_proposals = box_targets.size(0)
            expanded_box_targets = torch.zeros(num_proposals * num_classes, 4, dtype=box_targets.dtype)
            map_box_idxs = labels + torch.arange(num_proposals, dtype=labels.dtype) * num_classes
            
            expanded_box_targets[map_box_idxs] = box_targets
            expanded_box_targets = expanded_box_targets.view(num_proposals, num_classes * 4)

            bbox_inside_weights = torch.zeros_like(expanded_box_targets).view(-1, 4)
            bbox_inside_weights[map_box_idxs] = 1
            bbox_inside_weights = bbox_inside_weights.view_as(expanded_box_targets)
            bbox_outside_weights = (bbox_inside_weights > 0).type_as(bbox_inside_weights)

            return expanded_box_targets, bbox_inside_weights, bbox_outside_weights

        expanded_box_targets, bbox_inside_weights, bbox_outside_weights = expand_bbox_targets(box_targets, labels, num_classes)

        # mm = model.cls_score
        # print('bias norm: {:.4f}, {:.4f}, {:.4f} {:.4f}'.format(mm.bias.data.min(), mm.weight.data.norm(), mm.weight.data.min(), mm.weight.data.max()))
        scores, box_deltas = model(imgs, sampled_proposals, img_idx)

        loss_classif = F.cross_entropy(scores, labels, size_average=True)
        loss_box_reg = F.smooth_l1_loss(bbox_inside_weights * (box_deltas - expanded_box_targets), torch.zeros_like(expanded_box_targets), size_average=False, reduce=False) * bbox_outside_weights
        loss_box_reg = loss_box_reg.mean()
        loss = loss_classif  + loss_box_reg

        losses.update(loss.item(), scores.size(0))

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

        optimizer.zero_grad()
        loss.backward()

        for n, v in model.named_parameters():
            if v.requires_grad == False:
                continue
            if n not in norm_weights:
                norm_weights[n] = []
            if v.grad is None:
                norm_weights[n].append((v.detach().abs().max(), v.detach().norm(), 0, 0))
            else:
                norm_weights[n].append((v.detach().abs().max(), v.detach().norm(), v.grad.detach().abs().max(), v.grad.detach().norm()))

        # torch.nn.utils.clip_grad_norm(model.parameters(), 10.)  # TODO check if works

        optimizer.step()

        # print(loss.data[0])
        loss_list.append(loss.item())
        labels_list.append(labels.detach().cpu()[None])
        if loss.item() != loss.item() or loss.item() > 10:
            sys.werwe
            torch.save(to_save_data, 'dump_from_training.pth')
            import pprint
            pprint.pprint({k:v[-10:] for k,v in norm_outputs.items()})
            pprint.pprint({k:v[-5:] for k,v in norm_weights.items()})
            print(loss_list[-30:])
            #print(labels_list[-4:])
            #torch.save({'norm_outputs':norm_outputs, 'loss_list':loss_list, 'labels_list':labels_list}, 'debug_stack.pth')
            print(torch.stack([pred_cls, labels.detach()]))
            # print(F.log_softmax(scores).max(1)[0].view(1, -1))
            print(F.log_softmax(scores, -1))
            # print(scores.max(1)[0].view(1, -1))
            print(loss.item())
            sys.asuhf

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print('Epoch [{0}] {1}/{2}\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc: {accuracy.val:.4f} ({accuracy.avg:.4f})\t'
                  'FG Acc: {fg_acc.val:.4f} ({fg_acc.avg:.4f})\t'
                  'BG Acc: {bg_acc.val:.4f} ({bg_acc.avg:.4f})\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      epoch, i, len(data_loader), loss=losses,
                      accuracy=accuracies, fg_acc=fg_accuracies, bg_acc=bg_accuracies,
                      batch_time=batch_time, data_time=data_time))
    epoch_time = time.time() - epoch_time
    epoch_time_min = int(epoch_time // 60)
    epoch_time_sec = int(epoch_time % 60)
    print('Total epoch time: {}m{}s'.format(epoch_time_min, epoch_time_sec))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from collections import deque
class MedianMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        window_size = 20
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.deque.append(val)
        self.series.append(val)
        self.count += n
        self.total += val * n

    @property
    def val(self):
        return np.median(self.deque)

    @property
    def avg(self):
        return self.total / self.count


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


# https://github.com/jwyang/faster-rcnn.pytorch/blob/df7685772c3ab379ee2ca3dab35809b49e877629/lib/model/utils/net_utils.py
# TODO delete this
import numpy as np
def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    all_norms = {}
    totalnorm = 0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.detach().norm()
            # all_norms[name] = (modulenorm, modulenorm / p.numel())
            # all_norms[name] = modulenorm / p.numel()
            all_norms[name] = modulenorm / (p.detach().norm() + 1e-6)
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    if norm < 1:
        print({k:v for k,v in all_norms.items() if v > 1e-1})
    #     sys.sdfdg
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MyOptimizer, self).__init__(params, defaults)
    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p * group['lr'])
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-1, d_p)

        return loss


if __name__ == '__main__':
    import torch.utils.data
    import torchvision.transforms as T

    from datasets.coco import COCODataset
    from lib.collate_fn import CollatorTraining

    from lib.models.fastrcnn import fastrcnn_resnet18, fastrcnn_resnet50, fastrcnn_vgg16
    from lib.models_caffe2 import load_resnet50_model

    from lib.utils.proposal_dataset import ProposalDataset

    import random
    torch.manual_seed(3)
    random.seed(3)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    transforms = T.Compose([T.ToTensor(), normalize])
    # transforms = T.Compose([T.Resize(800), T.ToTensor(), normalize])

    if True:
        transforms = T.Compose([T.ToTensor(), 
            T.Lambda(lambda x: (x[[2,1,0]] * 255) - torch.Tensor([102.9801, 115.9465, 122.7717]).view(3,1,1))])

    if True:
        ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_val2017.json'
        data_dir = '/datasets01/COCO/060817/val2014/'
        # ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_train2017.json'
        # data_dir = '/datasets01/COCO/060817/train2014/'
        dataset = COCODataset(ann_file, data_dir, transform=transforms)
        # from IPython import embed; embed()
        # dataset = ProposalDataset(dataset, '/private/home/fmassa/rpn_proposals_train.pkl')
        dataset = ProposalDataset(dataset, '/private/home/fmassa/rpn_proposals.pkl')
        num_classes = 81
    else:
        from datasets.pascal_voc import PascalVOC
        from lib.transforms import VOCAnnotations
        dataset = PascalVOC("/datasets01/VOC/060817/VOCdevkit/", "val",
                transform=transforms, target_transform=VOCAnnotations(False))
        dataset = ProposalDatasetVOC(dataset, '/private/home/fmassa/selective_search_data/voc_2012_val.mat')
        num_classes = 21

    collate_fn = CollatorTraining()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2,
            collate_fn=collate_fn, num_workers=4, pin_memory=True, shuffle=True)

    if False:
        # model = fastrcnn_vgg16(num_classes=num_classes)
        model = fastrcnn_resnet18(num_classes=num_classes)
        # model = fastrcnn_resnet50(num_classes=num_classes)
    else:
        # model = load_resnet50_model(file_path='/private/home/fmassa/model_final.pkl')
        import pickle
        with open('/private/home/fmassa/R-50.pkl', 'rb') as f:
        # with open('/private/home/fmassa/github/Detectron/detectron_c2_output/train/coco_2014_minival/fast_rcnn/model_final.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
 
        # model = load_resnet50_model(data['blobs'])
        model = load_resnet50_model(data)
        torch.nn.init.normal(model.cls_score.weight, std=0.01)
        torch.nn.init.constant(model.cls_score.bias, 0)

        torch.nn.init.normal(model.bbox_pred.weight, std=0.001)
        torch.nn.init.constant(model.bbox_pred.bias, 0)

    learning_rate = 0.01 / 2# 10  # 0.01
    weight_decay = 0.0001  # 0.0001
    momentum = 0.9  # TODO check
    print('momentum', momentum)

    model.train()
    # model.eval()
    if torch.cuda.is_available():
        model.cuda()


    params = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            print(key)
            if 'bias' in key:
                params += [{'params':[value],'lr':learning_rate*(1 + 1), \
                        'weight_decay': 0}]
            else:
                params += [{'params':[value],'lr':learning_rate, 'weight_decay': weight_decay}]

    if False:
        # optimizer = torch.optim.SGD([
        optimizer = MyOptimizer(params,
           # [
           # {'params': [m for n, m in model.named_parameters() if '.bias' not in n]},
           # {'params': [m for n, m in model.named_parameters() if '.bias' in n],
           #     # 'lr': 0.02 / 3, 'weight_decay': 0},
           #     'lr': learning_rate * 2, 'weight_decay': 0},
           #  ],
            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        # optimizer = torch.optim.SGD(model.parameters(), learning_rate,
        optimizer = torch.optim.SGD(params, learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    for epoch in range(1, 100):
        train_one_epoch(model, optimizer, data_loader, epoch)
        if False:
            torch.save({'model': model.state_dict(), 'optimizer':optimizer},
                    '/checkpoint02/fmassa/detectron_v0_vgg.pth')
