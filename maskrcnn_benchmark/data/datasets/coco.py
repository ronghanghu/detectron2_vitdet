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
        self.coco_api = COCO(ann_file)

        self.meta = COCOMeta()

        # initialize the metadata
        cat_ids = self.coco_api.getCatIds()
        self.meta.class_names = [c["name"] for c in self.coco_api.loadCats(cat_ids)]

        # The values are shifted by + 1 to reserve category 0 for background
        self.meta.json_id_to_contiguous_id = id_map = {v: i + 1 for i, v in enumerate(cat_ids)}
        self.meta.contiguous_id_to_json_id = {v: k for k, v in id_map.items()}

        # sort indices for reproducible results
        img_ids = sorted(list(self.coco_api.imgs.keys()))
        # imgs is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}
        imgs = self.coco_api.loadImgs(img_ids)
        # anns is a list[list[dict]], where each dict is an annotation
        # record for an object. The inner list enumerates the objects in an image
        # and the outer list enumerates over images. Example of anns[0]:
        # [{'segmentation': [[192.81,
        #     247.09,
        #     ...
        #     219.03,
        #     249.06]],
        #   'area': 1035.749,
        #   'iscrowd': 0,
        #   'image_id': 1268,
        #   'bbox': [192.81, 224.8, 74.73, 33.43],
        #   'category_id': 16,
        #   'id': 42986},
        #  ...]
        anns = [self.coco_api.imgToAnns[img_id] for img_id in img_ids]

        imgs_anns = list(zip(imgs, anns))

        logger = logging.getLogger(__name__)
        if remove_images_without_annotations:
            num_before = len(imgs_anns)

            def has_annotation(anns):
                for ann in anns:
                    if ann["iscrowd"] == 0:
                        return True
                return False

            imgs_anns = [x for x in imgs_anns if has_annotation(x[1])]

            num_after = len(imgs_anns)
            logger.info(
                "Remove {} images with no usable annotations.".format(num_before - num_after)
            )
        logger.info("Loaded {} images from {}".format(len(imgs_anns), ann_file))

        self.dataset_dicts = []
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(self.root, img_dict["file_name"])
            record["original_height"] = img_dict["height"]
            record["original_width"] = img_dict["width"]
            record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno.get("ignore", 0) == 0
                obj = {
                    field: copy.deepcopy(anno[field])
                    for field in ["segmentation", "area", "iscrowd", "bbox"]
                }
                obj["category_id"] = self.meta.json_id_to_contiguous_id[anno["category_id"]]
                objs.append(obj)
            record["annotations"] = objs
            self.dataset_dicts.append(record)

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        """
        Return a copy of the dataset dict for each image. It has the following keys:

        file_name: full path to the image
        id: id of the image in the COCO-style json file
        height, width: int shape of the image
        annotations: list of object annotation dicts, each of which has the keys:
            segmentation: a list of lists, in COCO's original format
            area: float segmentation mask area
            iscrowd: 0 or 1
            bbox: x, y, w, h
            category_id: [1, 80] internal contiguous category id
        """
        return copy.deepcopy(self.dataset_dicts[idx])
