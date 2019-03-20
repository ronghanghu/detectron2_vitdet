import logging
import os

from detectron2.structures import BoxMode

from .metadata import MetadataCatalog

logger = logging.getLogger(__name__)


def load_coco_json(json_file, image_root, dataset_name=None):
    """
    Load a json file with COCO's annotation format.
    Currently only supports instance segmentation annotations.

    Args:
        json_file (str): full path to the json file in COCO annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "coco", "cityscapes").
            If provided, this function will also put "class_names" into
            the metadata associated with this dataset.

    Returns:
        list[dict]: a list of dicts in "Detectron2 Dataset" format. (See DATASETS.md)

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
        2. When `dataset_name=='coco'`,
           this function will translate COCO's
           incontiguous category ids to contiguous ids in [1, 80].
    """
    from pycocotools.coco import COCO

    coco_api = COCO(json_file)

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = coco_api.getCatIds()
        class_names = [c["name"] for c in coco_api.loadCats(cat_ids)]
        meta.class_names = class_names

        # A user can provide a dataset where some category ids have no
        # samples -- that's a valid thing to do.
        # However, in COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        #
        # This is a hack to deal with COCO's id issue and translate
        # the category ids to contiguous ids in [1, 80].
        # We apply this hack for COCO only.
        # If the ids are incontiuguos for datasets other than COCO,
        # we'll just assume that it is intended
        # (you'll just train/test with 0 samples for certain classes).
        #
        # If for some reason this hack is needed for other datasets,
        # we can parse the json and use a different predicate in this if statement.
        if dataset_name == "coco":
            id_map = {v: i + 1 for i, v in enumerate(cat_ids)}
            meta.json_id_to_contiguous_id = id_map
        else:
            if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
                logger.warning("Category ids in annotations are not contiguous!")

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
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno.get("ignore", 0) == 0
            obj = {
                field: anno[field]
                for field in ["segmentation", "iscrowd", "bbox", "keypoints", "category_id"]
                if field in anno
            }
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts