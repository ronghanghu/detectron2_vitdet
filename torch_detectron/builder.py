import torch.utils.data as data

from datasets import PascalVOC
from lib.dataset_transforms import pascal_target_transform

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

    def __getitem__(self, index):
        return self.datasets[0][index]

if __name__ == '__main__':
    composite = CompositeDetectionDataset()
    composite.register_pascal_voc("/datasets01/VOC/060817/VOCdevkit/", "trainval")
    print(composite[3][1].detection_annotations()[0].bounding_box())

