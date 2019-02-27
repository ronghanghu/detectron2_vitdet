import copy
import logging
import os


logger = logging.getLogger(__name__)


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


def load_coco_json(json_file, image_root):
    """
    Load a json file in COCO annotation format.

    Args:
        json_file (str): the json file in COCO annotation format.
        image_root (str): the directory where the images in this json file exists.

    Returns:
        list[dict]: a list of per-image annotations. Each dict contains:
            "file_name": the full path to the image file (`image_root` + file name in json)
            "original_height", "original_width":
            "image_id" (str):
            "annotations" (list[dict]): the per-instance annotations of every
                instance in this image. Each annotation dict contains:
                "iscrowd": 0 or 1. Whether this instance is labeled as COCO's "crowd region".
                "bbox" (list[float]): list of 4 numbers (x, y, w, h)
                "category_id" (int): a __positive__ integer in the range [1, num_categories].
                    This is the "continuous id" which may be different from
                    the original id in the annotation file.
                "segmentation" (list[list[float]] or dict, optional):
                    For `list[list[float]]`, it represents the polygons of
                    each object part. Each `list[float]` is one polygon in the
                    format of [x1, y1, ..., xn, yn].
                    For `dict`, it represents the segmentation in COCO's RLE format.
                "keypoints" (list[float]): in the format of [x1, y1, v1,..., xn, yn, vn].
                    v[i] means the visibility of this keypoint.
                    `n` must be equal to the number of keypoint categories.
    """
    from pycocotools.coco import COCO

    coco_api = COCO(json_file)

    meta = COCOMeta()

    # initialize the metadata
    cat_ids = coco_api.getCatIds()
    meta.class_names = [c["name"] for c in coco_api.loadCats(cat_ids)]

    # The values are shifted by + 1 to reserve category 0 for background
    meta.json_id_to_contiguous_id = id_map = {v: i + 1 for i, v in enumerate(cat_ids)}
    meta.contiguous_id_to_json_id = {v: k for k, v in id_map.items()}

    # sort indices for reproducible results
    img_ids = sorted(list(coco_api.imgs.keys()))
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
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
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["original_height"] = img_dict["height"]
        record["original_width"] = img_dict["width"]
        record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno.get("ignore", 0) == 0
            obj = {
                field: copy.deepcopy(anno[field])
                for field in ["segmentation", "iscrowd", "bbox", "keypoints"]
                if field in anno
            }
            obj["category_id"] = meta.json_id_to_contiguous_id[anno["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
