import copy
import logging
import os
import torch.utils.data as data


class COCOMeta:
    class_names = None
    contiguous_id_to_json_id = {}  # contiguous id starts from 1
    json_id_to_contiguous_id = {}

    _instance = None

    def __new__(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    @property
    def num_classes(self):
        return len(self.class_names)


class COCODetection(data.Dataset):
    def __init__(self, ann_file, root, remove_images_without_annotations=False):
        """
        Args:
            ann_file (str): the annotation file. Path to a json
            root (str): contains 'train2014', etc
        """
        from pycocotools.coco import COCO

        self.root = root
        self.coco = COCO(ann_file)

        self.meta = COCOMeta()

        # initialize the metadata
        cat_ids = self.coco.getCatIds()
        self.meta.class_names = [c["name"] for c in self.coco.loadCats(cat_ids)]

        id_map = self.meta.json_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(cat_ids)}
        self.meta.contiguous_id_to_json_id = {v: k for k, v in id_map.items()}

        # sort indices for reproducible results
        img_ids = sorted(list(self.coco.imgs.keys()))
        # TODO for inference
        self.imgs = imgs = self.coco.loadImgs(img_ids)
        annotations = [self.coco.imgToAnns[img_id] for img_id in img_ids]

        self.roidbs = list(zip(imgs, annotations))

        logger = logging.getLogger(__name__)
        if remove_images_without_annotations:
            num_before = len(self.roidbs)

            def has_annotation(anns):
                for ann in anns:
                    if ann["iscrowd"] == 0:
                        return True
                return False

            self.roidbs = [x for x in self.roidbs if has_annotation(x[1])]

            num_after = len(self.roidbs)
            logger.info(
                "Remove {} images without no-crowd annotations.".format(num_before - num_after)
            )
        logger.info("Loaded {} images from {}".format(len(self.roidbs), ann_file))

    def __len__(self):
        return len(self.roidbs)

    def __getitem__(self, idx):
        """
        Return a dict for each image. It has the following keys:

        file_name: full path to the image
        id: id of the image
        height, width: int
        annotations: list
        Each annotation is a dict representing an object. The dict has the following keys:
        segmentation: a list of lists, in COCO's original format
            area: float
            iscrowd: 0 or 1
            bbox: x, y, w, h
            category_id: 0~79
        """
        img_json, anno_json = self.roidbs[idx]

        ret_dict = {}
        ret_dict["file_name"] = os.path.join(self.root, img_json["file_name"])
        for field in ["height", "width", "id"]:
            ret_dict[field] = img_json[field]

        objs = []
        for anno in anno_json:
            assert anno.get("ignore", 0) == 0
            obj = {
                field: copy.deepcopy(anno[field])
                for field in ["segmentation", "area", "iscrowd", "bbox"]
            }
            obj["category_id"] = self.meta.json_id_to_contiguous_id[anno["category_id"]]
            objs.append(obj)
        ret_dict["annotations"] = objs
        return ret_dict
