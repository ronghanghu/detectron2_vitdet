import torch.utils.data as data

from datasets import PascalVOC
# todo: need to make this optional?
from torchvision.datasets import coco
from lib.dataset_transforms import pascal_target_transform, \
    coco_target_transform

# TODO: one possible org structure?
class CompositeDetectionDataset(data.Dataset):
    """
    There are multiple datasets for object detection tasks - Pascal VOC, COCO,
    Cityscapes, etc. We want to be able to train our detection models using one
    or more of these datasets. However, the datasets themselves are not
    uniform, for example Pascal VOC's XML annotations vs. COCO's json annotations.
    
    This class serves as an abstraction that provides a uniform interface to
    the underlying datasets. If you want to add a new dataset for training, it
    just needs to be integrated here.
    """

    def __init__(self):
        self.datasets = []

    def register_pascal_voc(self, root, imageset):
        self.datasets.append(PascalVOC(root, imageset, pascal_target_transform))

    def register_coco(self, root, annotation_file):
        self.datasets.append(coco.CocoDetection(root, annotation_file,
            target_transform=coco_target_transform))

    def __getitem__(self, index):
        return self.datasets[1][index]

if __name__ == '__main__':
    composite = CompositeDetectionDataset()
    composite.register_pascal_voc("/datasets01/VOC/060817/VOCdevkit/", "trainval")
    composite.register_coco('/datasets01/COCO/060817/train2014',
        '/datasets01/COCO/060817/annotations/instances_train2014.json')
    print(composite[3][1].detection_annotations()[0].bounding_box())

