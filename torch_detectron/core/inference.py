import torch

import torchvision
from torchvision.structures.bounding_box import BBox

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, ann_file, root, transforms=None):
        super(COCODataset, self).__init__(root, ann_file)

        # filter images without detection annotations
        self.ids = [img_id for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0]

        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        """
        boxes = [obj['bbox'] for obj in anno]
        target = BBox(boxes, img.size, mode='xywh').convert('xyxy')

        classes = [obj['category_id'] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field('labels', classes)
        """
        """
        Idea: maybe don't need to rescale the targets here, but only
        during the loss computation?
        """
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, anno

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    # def __len__(self):
    #     return 16


from tqdm import tqdm
def compute_on_dataset(model, data_loader, device):
    model.eval()
    results = []
    cpu_device = torch.device('cpu')
    for i, batch in tqdm(enumerate(data_loader)):
        images, targets = batch
        images = images.to(device)
        with torch.no_grad():
            output = model.predict(images)
            output = [o.to(cpu_device) for o in output]
        results.extend(output)
    return results


def prepare_for_coco_detection(predictions, dataset):
    assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'bbox': box,
            'score': scores[k]} for k, box in enumerate(boxes)])
    return coco_results


import time
def prepare_for_coco_segmentation(predictions, dataset):
    from core.mask_rcnn import Masker
    import pycocotools.mask as mask_util
    import numpy as np
    masker = Masker(threshold=0.5, padding=1)
    assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field('mask')
        # t = time.time()
        masks = masker(masks, prediction)
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()

        # rles = prediction.get_field('mask')

        rles = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], order='F'))[0] for mask in masks]
        for rle in rles:
            rle['counts'] = rle['counts'].decode('utf-8')

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'segmentation': rle,
            'score': scores[k]} for k, rle in enumerate(rles)])
    return coco_results



def evaluate_predictions_on_coco(coco_gt, coco_results,
        json_result_file, iou_type='bbox'):
    import json
    with open(json_result_file, 'w') as f:
        json.dump(coco_results, f)

    from pycocotools.cocoeval import COCOeval
    coco_dt = coco_gt.loadRes(str(json_result_file))
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    # torch.save(coco_eval, join(args.save, 'computed_results.pth'))
    coco_eval.summarize()



from torchvision.transforms import functional as F
class ImageTransform(object):
    def __call__(self, x):
        x = F.resize(x, 800, max_size=1333)
        x = F.to_tensor(x)

        x = x[[2, 1, 0]] * 255
        x -= torch.tensor([102.9801, 115.9465, 122.7717]).view(3,1,1)

        return x

from core.image_list import to_image_list
class Collator(object):
    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        return images, targets

def get_dataset():
    ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_val2017_mod.json'
    data_dir = '/datasets01/COCO/060817/val2014/'
    
    dataset = COCODataset(ann_file, data_dir, ImageTransform())
    return dataset

def main(model_builder=None,
        pretrained_path=None,
        iou_types=('bbox',),#'segm'),
        batch_size=1):

    dataset = get_dataset()

    device = torch.device('cuda')

    import core.test_script as test_script
    model_builder = 'build_fpn_model'
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/fpn_r50.pth'
    model_builder = 'build_resnet_model'
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/faster_rcnn_resnet50.pth'
    model_builder = 'build_mrcnn_fpn_model'
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/mrcnn_fpn_r50.pth'

    size_divisible = 32 if 'fpn' in model_builder else 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            num_workers=4, collate_fn=Collator(size_divisible))

    model = getattr(test_script, model_builder)(pretrained_path)
    if False: #'mrcnn' in model_builder:
        from core.mask_rcnn import MaskPostProcessorCOCOFormat, Masker
        masker = Masker(threshold=0.5, padding=0)
        mask_postprocessor = MaskPostProcessorCOCOFormat(masker)
        model.heads.heads[1].postprocessor = mask_postprocessor
    model.to(device)

    predictions = compute_on_dataset(model, data_loader, device)
    logger.info('Preparing results for COCO format')
    coco_results = {}
    if 'bbox' in iou_types:
        logger.info('Preparing bbox results')
        coco_results['bbox'] = prepare_for_coco_detection(predictions, dataset)
    if 'segm' in iou_types:
        logger.info('Preparing segm results')
        coco_results['segm'] = prepare_for_coco_segmentation(predictions, dataset)
    logger.info('Evaluating predictions')
    for iou_type in iou_types:
        evaluate_predictions_on_coco(dataset.coco, coco_results[iou_type],
                'tmp_results/{}_{}.json'.format(model_builder, iou_type), iou_type)
    # return predictions


def profile(batch_size=4):
    """
    Test script to be used with nvvp to profile the execution of CUDA kernels
    Run it as follows

    nvprof --profile-from-start off -o trace_name.prof -- python eval.py
    """
    dataset = get_dataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda')

    import core.test_script as test_script
    model_builder = 'build_fpn_model'
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/fpn_r50.pth'
    #model_builder = 'build_resnet_model'
    #pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/faster_rcnn_resnet50.pth'
    model = getattr(test_script, model_builder)(pretrained_path)
    model.to(device)

    iterator = iter(data_loader)
    batch = next(iterator)
    images = batch[0]
    images = images.to(device)

    model.predict(images)
    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():
            for i in range(5):
                model.predict(images)



if __name__ == '__main__':
    main()
    # profile()
