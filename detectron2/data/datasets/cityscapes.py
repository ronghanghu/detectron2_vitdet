import functools
import glob
import logging
import multiprocessing as mp
import numpy as np
import os
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode

from .metadata import MetadataCatalog

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


def load_cityscapes_instances(data_dir, gt_dir, use_polygons=False):
    """
    Args:
        data_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        use_polygons (bool): whether to represent the segmentation as masks
            (cityscapes's format) or polygons (COCO's format).

    Returns:
        list[dict]: a list of dicts in "Detectron2 Dataset" format. (See DATASETS.md)
    """
    files = []
    for image_file in glob.glob(os.path.join(data_dir, "**/*.png")):
        suffix = "leftImg8bit.png"
        assert image_file.endswith(suffix)
        prefix = data_dir
        instance_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "gtFine_instanceIds.png"
        assert os.path.isfile(instance_file), instance_file

        label_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "gtFine_labelIds.png"
        assert os.path.isfile(label_file), label_file
        files.append((image_file, instance_file, label_file))

    logger = logging.getLogger(__name__)
    logger.info("Preprocessing cityscapes annotations ...")
    pool = mp.Pool(processes=min(8, mp.cpu_count()))

    ret = pool.map(functools.partial(cityscapes_files_to_dict, use_polygons=use_polygons), files)
    logger.info("Loaded {} images from {}".format(len(ret), data_dir))
    return ret


def cityscapes_files_to_dict(files, use_polygons=False):
    """
    Parse cityscapes annotation files to a dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file)
        use_polygons (bool): whether to represent the segmentation as masks
            (cityscapes's format) or polygons (COCO's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    from cityscapesscripts.helpers.labels import id2label

    meta = MetadataCatalog.get("cityscapes")
    CATEGORY_TO_ID = {c: i + 1 for i, c in enumerate(meta.class_names)}

    # See also the official annotation parsing scripts at
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py
    inst_image = np.asarray(Image.open(files[1]), order="F")
    # ids < 24 are stuff labels (filtering them first is about 5% faster)
    flattened_ids = np.unique(inst_image[inst_image >= 24])

    ret = {
        "file_name": files[0],
        "image_id": os.path.basename(files[0]),
        "height": inst_image.shape[0],
        "width": inst_image.shape[1],
    }
    annos = []
    for instance_id in flattened_ids:
        # For non-crowd annotations, instance_id // 1000 is the label_id
        # Crowd annotations have <1000 instance ids
        label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
        label = id2label[label_id]
        if not label.hasInstances or label.ignoreInEval:
            continue

        anno = {}
        anno["iscrowd"] = instance_id < 1000
        anno["category_id"] = CATEGORY_TO_ID[label.name]

        mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

        inds = np.nonzero(mask)
        ymin, ymax = inds[0].min(), inds[0].max()
        xmin, xmax = inds[1].min(), inds[1].max()
        # TODO Maybe need an offset (xmax + 1 ?). Revisit this when we train the models.
        anno["bbox"] = (xmin, ymin, xmax, ymax)
        anno["bbox_mode"] = BoxMode.XYXY_ABS
        if use_polygons:
            # This conversion comes from D4809743 and D5171122, when Mask-RCNN was first developed.
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            polygons = [c.reshape(-1).tolist() for c in contours]
            anno["segmentation"] = polygons
        else:
            anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
        annos.append(anno)
    ret["annotations"] = annos
    return ret


if __name__ == "__main__":
    """
    Test the cityscapes dataset loader.

    Usage:
        python -m detectron2.data.datasets.cityscapes \
            cityscapes/leftImg8bit/train cityscapes/gtFine/train
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.vis import draw_coco_dict
    import sys

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get("cityscapes")

    dicts = load_cityscapes_instances(sys.argv[1], sys.argv[2], use_polygons=False)
    logger.info("Done loading {} samples.".format(len(dicts)))

    for d in dicts:
        vis = draw_coco_dict(d, ["0"] + meta.class_names)
        cv2.imshow("", vis)
        cv2.waitKey()
