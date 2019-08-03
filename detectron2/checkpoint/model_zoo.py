import logging
import os
import shutil
import tempfile

from detectron2.utils.comm import is_main_process, synchronize
from detectron2.utils.file_io import PathHandler, PathManager, get_cache_dir


def cache_file(file_name, cache_dir=None, distributed=True):
    """
    Caches a (presumably remote) file under cache_dir.

    When `distributed==True`,
    this function contains a barrier call. Therefore all processes must all
    call this method to avoid deadlock. Only the main process will actually
    perform the caching.

    When `distributed==False`, this function may cause race condition if called
    from multiple processes.
    """
    cache_dir = get_cache_dir(cache_dir)
    if file_name.startswith(cache_dir):
        return file_name
    src_dir, base_name = os.path.split(file_name)
    if src_dir[0].startswith(os.path.sep):
        src_dir = src_dir[len(os.path.sep) :]
    dst_dir = os.path.join(cache_dir, src_dir)
    dst_file_name = os.path.join(dst_dir, base_name)
    assert dst_file_name != file_name
    if os.path.exists(dst_file_name):
        return dst_file_name

    if (not distributed) or is_main_process():

        os.makedirs(dst_dir, exist_ok=True)
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            logger = logging.getLogger(__name__)
            logger.info("Caching {} locally...".format(file_name))
            shutil.copy(file_name, f.name)
            shutil.move(f.name, dst_file_name)
        finally:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)
    if distributed:
        synchronize()
    return dst_file_name


class ModelCatalog(object):
    """
    Store mappings from names to third-party models.
    """

    S3_C2_DETECTRON_PREFIX = "https://dl.fbaipublicfiles.com/detectron"

    # MSRA models have STRIDE_IN_1X1=True. False otherwise.
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "FAIR/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/X-101-64x4d": "ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl",
        "FAIR/X-152-32x8d-IN5k": "ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl",
    }

    C2_DETECTRON_PATH_FORMAT = (
        "{prefix}/{url}/output/train/{dataset}/{type}/model_final.pkl"
    )  # noqa B950

    C2_DATASET_COCO = "coco_2014_train%3Acoco_2014_valminusminival"
    C2_DATASET_COCO_KEYPOINTS = "keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival"

    # format: {model_name} -> part of the url
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "35857197/12_2017_baselines/e2e_faster_rcnn_R-50-C4_1x.yaml.01_33_49.iAX0mXvW",  # noqa B950
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "35857345/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml.01_36_30.cUF7QR7I",  # noqa B950
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "35857890/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_1x.yaml.01_38_50.sNxI7sX7",  # noqa B950
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "36761737/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml.06_31_39.5MIHi1fZ",  # noqa B950
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "35858791/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml.01_45_57.ZgkA7hPB",  # noqa B950
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "35858933/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC",  # noqa B950
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT",  # noqa B950
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "36761843/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI",  # noqa B950
        "48616381/e2e_mask_rcnn_R-50-FPN_2x_gn": "GN/48616381/04_2018_gn_baselines/e2e_mask_rcnn_R-50-FPN_2x_gn_0416.13_23_38.bTlTI97Q",  # noqa B950
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "37697547/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml.08_42_54.kdzV35ao",  # noqa B950
        "35998355/rpn_R-50-C4_1x": "35998355/12_2017_baselines/rpn_R-50-C4_1x.yaml.08_00_43.njH5oD9L",  # noqa B950
        "35998814/rpn_R-50-FPN_1x": "35998814/12_2017_baselines/rpn_R-50-FPN_1x.yaml.08_06_03.Axg0r179",  # noqa B950
        "36225147/fast_R-50-FPN_1x": "36225147/12_2017_baselines/fast_rcnn_R-50-FPN_1x.yaml.08_39_09.L3obSdQ2",  # noqa B950
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog._get_c2_detectron_baseline(name)
        if name.startswith("ImageNetPretrained/"):
            return ModelCatalog._get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog: {}".format(name))

    @staticmethod
    def _get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_PREFIX
        name = name[len("ImageNetPretrained/") :]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def _get_c2_detectron_baseline(name):
        name = name[len("Caffe2Detectron/COCO/") :]
        url = ModelCatalog.C2_DETECTRON_MODELS[name]
        if "keypoint_rcnn" in name:
            dataset = ModelCatalog.C2_DATASET_COCO_KEYPOINTS
        else:
            dataset = ModelCatalog.C2_DATASET_COCO

        if "35998355/rpn_R-50-C4_1x" in name:
            # this one model is somehow different from others ..
            type = "rpn"
        else:
            type = "generalized_rcnn"

        # Detectron C2 models are stored in the structure defined in `C2_DETECTRON_PATH_FORMAT`.
        url = ModelCatalog.C2_DETECTRON_PATH_FORMAT.format(
            prefix=ModelCatalog.S3_C2_DETECTRON_PREFIX, url=url, type=type, dataset=dataset
        )
        return url


class ModelCatalogHandler(PathHandler):
    """
    Resolve URL like catalog:// by the model zoo.
    """

    def _support(self, path):
        return self._has_protocol(path, "catalog")

    def _get_file_name(self, path):
        # TODO keep D2 model zoo handler for BC. Remove when release.
        d2_prefix = "catalog://Detectron2/"
        if path.startswith(d2_prefix):
            return PathManager.get_file_name("detectron2://" + path[len(d2_prefix) :])

        logger = logging.getLogger(__name__)
        catalog_path = ModelCatalog.get(path[len("catalog://") :])
        logger.info("Catalog entry {} points to {}".format(path, catalog_path))
        return PathManager.get_file_name(catalog_path)


class Detectron2Handler(PathHandler):
    """
    Resolve anything that's in Detectron2 model zoo.
    """

    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    def _support(self, path):
        return self._has_protocol(path, "detectron2")

    def _get_file_name(self, path):
        name = path[len("detectron2://") :]
        return PathManager.get_file_name(self.S3_DETECTRON2_PREFIX + name)


PathManager.register_handler(ModelCatalogHandler())
PathManager.register_handler(Detectron2Handler())
