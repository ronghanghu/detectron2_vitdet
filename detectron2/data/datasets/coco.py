import glob
import logging
import os
from PIL import Image

from detectron2.structures import BoxMode

from ..metadata import MetadataCatalog

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
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        class_names = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
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

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0

            obj = {
                field: anno[field]
                for field in ["iscrowd", "bbox", "keypoints", "category_id"]
                if field in anno
            }

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warn(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmenation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]: a list of dicts in "Detectron2 Dataset" format without instance-level
            annotation. (See DATASETS.md)

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their raltive filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively. COCO API works with integer
    # IDs, hence, we try to convert these paths to int if possible.
    def file2id(folder_path, file_path):
        # extract realtive path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        try:
            image_id = int(image_id)
        except ValueError:
            pass
        return image_id

    input_files = sorted(
        glob.iglob(os.path.join(image_root, "**/*.{}".format(image_ext)), recursive=True),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        glob.iglob(os.path.join(gt_root, "**/*.{}".format(gt_ext)), recursive=True),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    logger.info("Loaded {} images from {}".format(len(input_files), image_root))

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        record["image_id"] = file2id(image_root, img_path)
        assert record["image_id"] == file2id(
            gt_root, gt_path
        ), "there is no ground truth for {}".format(img_path)
        img = Image.open(gt_path)
        w, h = img.size
        record["height"] = h
        record["width"] = w
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.vis import draw_coco_dict
    import cv2
    import sys

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    for d in dicts:
        vis = draw_coco_dict(d, ["0"] + meta.class_names)
        fpath = os.path.join("coco-data-vis", os.path.basename(d["file_name"]))
        cv2.imwrite(fpath, vis)
