import logging
import os
import shutil
import tempfile

from detectron2.utils.comm import is_main_process, synchronize

try:
    # TODO avoid these torch internals
    from torch.hub import HASH_REGEX, _download_url_to_file, urlparse
except ImportError:
    from torch.utils.model_zoo import HASH_REGEX, _download_url_to_file, urlparse


# very similar to https://github.com/pytorch/pytorch/blob/master/torch/utils/model_zoo.py
# but with a few improvements and modifications
def cache_url(url, model_dir=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object.
            Defaults to :func:`get_cache_dir()`
        progress (bool, optional): whether or not to display a progress bar to stderr
    Example:
        >>> cached_file = detectron2.utils.model_zoo.cache_url('https://dl.fbaipublicfiles.com/detectron/pytorch/models/resnet18-5c106cde.pth')  # noqa
    """
    model_dir = get_cache_dir(model_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if filename == "model_final.pkl":
        # workaround as pre-trained Caffe2 models from Detectron have all the same filename
        # so make the full path the filename by replacing / with _
        filename = parts.path.replace("/", "_")
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) and is_main_process():
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        logger = logging.getLogger(__name__)
        logger.info("Downloading: {} to {}".format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename)
        if hash_prefix is not None:
            hash_prefix = hash_prefix.group(1)
            # workaround: Caffe2 models don't have a hash, but follow the R-50 convention,
            # which matches the hash PyTorch uses. So we skip the hash matching
            # if the hash_prefix is less than 6 characters
            if len(hash_prefix) < 6:
                hash_prefix = None
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    synchronize()
    return cached_file


def cache_file(file_name, model_dir=None):
    """
    Caches a (presumably remote) file under model_dir.

    This function contains a barrier call. Therefore all processes must all
    call this method to avoid deadlock. Only the main process will actually
    perform the caching.
    """
    model_dir = get_cache_dir(model_dir)
    if file_name.startswith(model_dir):
        return file_name
    src_dir, base_name = os.path.split(file_name)
    if src_dir[0].startswith(os.path.sep):
        src_dir = src_dir[len(os.path.sep) :]
    dst_dir = os.path.join(model_dir, src_dir)
    dst_file_name = os.path.join(dst_dir, base_name)
    assert dst_file_name != file_name

    if is_main_process():
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            if not os.path.exists(dst_file_name):
                logger = logging.getLogger(__name__)
                logger.info("Caching {} locally...".format(file_name))
                shutil.copy(file_name, f.name)
                shutil.move(f.name, dst_file_name)
        finally:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)
    synchronize()
    return dst_file_name


def get_cache_dir(model_dir=None):
    """
    Args:
        model_dir (None or str): if None, returns the default cache directory as:

        1) $TORCH_MODEL_ZOO env variable, if set
        2) otherwise $TORCH_HOME, if set
        3) otherwise ~/.torch
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
    return model_dir


class ModelCatalog(object):
    """
    Store mappings from names to third-party models.
    """

    S3_C2_DETECTRON_PREFIX = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
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
