import logging
import os
import shutil
import tempfile
from torch.hub import HASH_REGEX, _download_url_to_file, urlparse

from detectron2.utils.comm import is_main_process, synchronize


# very similar to https://github.com/pytorch/pytorch/blob/master/torch/utils/model_zoo.py
# but with a few improvements and modifications
def cache_url(url, model_dir=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    The default value of `model_dir` is ``$TORCH_HOME/models`` where
    ``$TORCH_HOME`` defaults to ``~/.torch``. The default directory can be
    overridden with the ``$TORCH_MODEL_ZOO`` environment variable.
    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr
    Example:
        >>> cached_file = detectron2.utils.model_zoo.cache_url('https://dl.fbaipublicfiles.com/detectron/pytorch/models/resnet18-5c106cde.pth')  # noqa
    """
    model_dir = _get_model_dir(model_dir)
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
    """Caches a (presumably remote) file under model_dir."""
    model_dir = _get_model_dir(model_dir)
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


def _get_model_dir(model_dir):
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
    return model_dir
