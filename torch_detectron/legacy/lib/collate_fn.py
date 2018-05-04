import torch
import math


def new_collate(batch):
    flipped_batch = tuple(zip(*batch))
    imgs, targets = flipped_batch

    max_size = tuple(max(s) for s in zip(*[img.shape for img in imgs]))

    # TODO FIX THIS IN GENERAL
    if True: #cfg.FPN.FPN_ON:
        stride = 32# float(cfg.FPN.COARSEST_STRIDE)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)

    batch_shape = (len(imgs),) + max_size
    batched_imgs = imgs[0].new(*batch_shape).zero_()
    for img, pad_img in zip(imgs, batched_imgs):
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

    return batched_imgs, targets


def join_targets(bboxes):
    sizes = [box.bbox.size(0) for box in bboxes]
    idxs = [i for i, s in enumerate(sizes) for _ in range(s)]
    cat_boxes = torch.cat([box.bbox for box in bboxes], dim=0)
    cat_labels = torch.cat([box.get_field('labels') for box in bboxes], dim=0)
    idxs = torch.tensor(idxs)
    return cat_boxes, cat_labels, idxs

class Collator(object):
    def __call__(self, batch):
        flip_batch = tuple(zip(*batch))
        imgs = flip_batch[0]
        max_size = tuple(max(s) for s in zip(*[img.shape for img in imgs]))

        # TODO FIX THIS IN GENERAL
        if True: #cfg.FPN.FPN_ON:
            stride = 32# float(cfg.FPN.COARSEST_STRIDE)
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(imgs),) + max_size
        batched_imgs = imgs[0].new(*batch_shape).zero_()
        for img, pad_img in zip(imgs, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        boxes = flip_batch[1]
        if len(boxes) == 0:
            print(batch)
        n = [b.shape[0] for b in boxes]
        img_idx = []
        count = 0
        for i in n:
            img_idx.append(slice(count, count + i))
            count += i
        cat_boxes = torch.cat(boxes, 0)
        im_shape = [im.shape[1:] for im in imgs]
        # return (batched_imgs,) + flip_batch[1:]
        return batched_imgs, cat_boxes, flip_batch[2], torch.Tensor(im_shape), img_idx


# TODO
# use the trick for namedtuple from https://stackoverflow.com/questions/11351032/namedtuple-and-default-values-for-optional-keyword-arguments
# from collections import namedtuple
# Node = namedtuple('Node', 'val left right')
# Node.__new__.__defaults__ = (None,) * len(Node._fields)
class CollatorTraining(object):
    def __call__(self, batch):
        flip_batch = tuple(zip(*batch))
        imgs = flip_batch[0]
        max_size = tuple(max(s) for s in zip(*[img.shape for img in imgs]))

        # TODO FIX THIS IN GENERAL
        if True: #cfg.FPN.FPN_ON:
            stride = 32# float(cfg.FPN.COARSEST_STRIDE)
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)


        batch_shape = (len(imgs),) + max_size
        batched_imgs = imgs[0].new(*batch_shape).zero_()
        for img, pad_img in zip(imgs, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        im_shape = [im.shape[1:] for im in imgs]

        boxes = flip_batch[1]
        if len(boxes) == 0:
            print(batch)
        n = [b.shape[0] for b in boxes]
        proposal_img_idx = []
        count = 0
        for idx, i in enumerate(n):
            #proposal_img_idx.append(slice(count, count + i))
            proposal_img_idx.extend([idx for _ in range(i)])
            count += i
        proposal_img_idx = torch.LongTensor(proposal_img_idx)

        cat_boxes = torch.cat(boxes, 0)

        gt_cat_labels = torch.cat(flip_batch[2], 0)

        # return (batched_imgs,) + flip_batch[1:]
        # return batched_imgs, cat_boxes, flip_batch[2], im_shape, img_idx
        return batched_imgs, cat_boxes, gt_cat_labels, proposal_img_idx,  torch.Tensor(im_shape)

class CollatorTrainingLegacy(object):
    def __call__(self, batch):
        flip_batch = tuple(zip(*batch))
        imgs = flip_batch[0]
        max_size = tuple(max(s) for s in zip(*[img.shape for img in imgs]))
        batch_shape = (len(imgs),) + max_size
        batched_imgs = imgs[0].new(*batch_shape).zero_()
        for img, pad_img in zip(imgs, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        im_shape = [im.shape[1:] for im in imgs]

        boxes = flip_batch[1]
        if len(boxes) == 0:
            print(batch)
        n = [b.shape[0] for b in boxes]
        proposal_img_idx = []
        count = 0
        for idx, i in enumerate(n):
            #proposal_img_idx.append(slice(count, count + i))
            proposal_img_idx.extend([idx for _ in range(i)])
            count += i
        proposal_img_idx = torch.LongTensor(proposal_img_idx)

        cat_boxes = torch.cat(boxes, 0)

        gt_boxes = flip_batch[2]
        n = [b.shape[0] for b in gt_boxes]
        gt_img_idx = []
        count = 0
        for idx, i in enumerate(n):
            gt_img_idx.extend([idx for _ in range(i)])
            count += i
        gt_img_idx = torch.LongTensor(gt_img_idx)
        
        gt_cat_boxes = torch.cat(gt_boxes, 0)
        gt_cat_labels = torch.cat(flip_batch[3], 0)
        
        # return (batched_imgs,) + flip_batch[1:]
        # return batched_imgs, cat_boxes, flip_batch[2], im_shape, img_idx
        return batched_imgs, cat_boxes, gt_cat_boxes, gt_cat_labels, proposal_img_idx, gt_img_idx, torch.Tensor(im_shape)




if __name__ == '__main__':
    collator = Collator()
    batch = [
            (torch.rand(3, 100, 100), torch.rand(10, 4)),
            (torch.rand(3, 200, 100), torch.rand(20, 4)),
            (torch.rand(3, 100, 200), torch.rand(5, 4))
    ]
    collated_batch = collator(batch)
    assert len(collated_batch) == 2
    assert tuple(collated_batch[0].shape) == (3, 3, 200, 200)
    assert len(collated_batch[1]) == 3
    print(collated_batch)
