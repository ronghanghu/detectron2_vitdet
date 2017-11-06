import numpy as np
import os
import torch.utils.data as data
import xml.etree.ElementTree as ET

class PascalVOC(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    Args:
        root (string): Root directory of VOC devkit
        image_set (string): Which set of images to use, e.g. 'train',
            'trainval', 'test', etc.
        # TODO: year --> we currently only have 2012 on the cluster
    """

    image_sets = ['train', 'val', 'trainval', 'test'] # TODO: test included?

    def _get_image_set_path(self, root, image_set):
        # image set metadata is stored in a text file
        # example: /VOC2012/ImageSets/Main/trainval.txt
        image_set_dir = os.path.join('VOC2012', 'ImageSets', 'Main')
        f = image_set + '.txt'
        return os.path.join(root, image_set_dir, f)

    def _load_image_set(self, image_set_path):
        # The image text file is just a list of numeric identifiers, e.g.
        # 2008_000128
        with open(image_set_path) as isf:
            image_ids = [i.strip() for i in isf.readlines()]
            return image_ids

    def _get_image_dir_path(self, root):
        return os.path.join(root, 'VOC2012', 'JPEGImages')

    def _get_image_path(self, image_dir, img_id):
        # images are .jpg files
        # example: /VOC2012/JPEGImages/2008_000128.jpg
        return os.path.join(image_dir, img_id + '.jpg')

    def _get_annotation_dir_path(self, root):
        return os.path.join(root, 'VOC2012', 'Annotations')

    def _get_annotation_path(self, image_dir, img_id):
        # images are .xml files
        # example: /VOC2012/Annotations/2008_000128.xml
        return os.path.join(image_dir, img_id + '.xml')

    def _parse_annotation_xml(self, annotation_path):
        # TODO: ultimately we will want this image metadata to be in a common
        # format, but for now I will just parse it so I can understand its
        # components

        # Pascal VOC Annotations are in the form of XML files. The key
        # components are subtrees with the <object> tag, these store the ground
        # truth objects in the image, including the class label and the
        # bounding box, and other metadata
        tree = ET.parse(annotation_path)
        objs = tree.findall('object')
        count = len(objs)

        # Use a 2D array for bounding box coordinates (x1, y1, x2, y2)
        boxes = np.ndarray((count, 4), dtype=np.uint16)

        # TODO: ross associates class labels with an index, and only stores the
        # index instead of a string
        classes = list()
        
        # TODO: ross includes more of the metadata, for now we just get classes
        # and bounding boxes
        for idx, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            
            # Pascal VOC dataset has image indices 1-base (#tbt to LuaTorch) so
            # we convert them to be 0-base
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            boxes[idx, :] = [x1, y1, x2, y2]
            
            class_label = obj.find('name').text
            classes.append(class_label)

        return {'boxes': boxes, 'classes': classes}

    def __init__(self, root, image_set):
        self.root = root

        # TODO: should be assertion ?
        if not image_set in self.image_sets:
            raise RuntimeError('Invalid image_set specified for PascalVOC')
        self.image_set = image_set

        # Verify path to Image Set
        self.image_set_path = self._get_image_set_path(root, image_set)
        if not os.path.exists(self.image_set_path):
            raise RuntimeError('Invalid path to Pascal VOC Image Set text file')

        # Load image IDs from Image Set text file
        self.image_ids = self._load_image_set(self.image_set_path)

        # Verify path to Image Directory, Annotations exist
        self.image_dir = self._get_image_dir_path(root)
        if not os.path.exists(self.image_dir):
            raise RuntimeError('Invalid path to Pascal VOC Image directory')
        self.annotation_dir = self._get_annotation_dir_path(root)
        if not os.path.exists(self.annotation_dir):
            raise RuntimeError('Invalid path to Pascal VOC Annotation directory')

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_path = self._get_image_path(self.image_dir, img_id)
        annotation_path = self._get_annotation_path(self.annotation_dir, img_id)

        if not os.path.exists(img_path):
            raise RuntimeError('Invalid path to image')
        if not os.path.exists(annotation_path):
            raise RuntimeError('Invalid path to annotation')

        return img_path, self._parse_annotation_xml(annotation_path)

    def __len__(self):
        return len(self.image_ids)

if __name__ == '__main__':
    ds = PascalVOC("/datasets01/VOC/060817/VOCdevkit/", "trainval")
    print(len(ds))
    print(ds[3])

