import torchvision
from torchvision.structures.bounding_box import BBox

import torch


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

        boxes = [obj['bbox'] for obj in anno]
        target = BBox(boxes, img.size, mode='xywh').convert('xyxy')

        classes = [obj['category_id'] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        assert classes.numel() > 0
        target.add_field('labels', classes)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


if __name__ == '__main__':
    ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_val2017_mod.json'
    data_dir = '/datasets01/COCO/060817/val2014/'

    ds = COCODataset(ann_file, data_dir)

    print(ds[0])

    from IPython import embed; embed()
