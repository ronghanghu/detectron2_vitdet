# -*- coding: utf-8 -*-

import logging
import os
from abc import ABCMeta, abstractmethod
from urllib.parse import urlparse
from borc.common.download import download

__all__ = ["PathManager"]


def get_cache_dir(cache_dir=None):
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $TORCH_HOME/detectron2_cache, if set
        2) otherwise ~/.torch/detectron2_cache
    """
    if cache_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        cache_dir = os.path.join(torch_home, "detectron2_cache")
    return cache_dir


class PathHandler(metaclass=ABCMeta):
    """
    Base class for path handler. Path handler takes a
    generic path (which may look like "protocol://*") which identifies a file,
    and returns either a file name or a file-like object.
    """

    @abstractmethod
    def _support(self, path):
        """
        Returns:
            bool: whether this handler can handle this path
        """
        pass

    @abstractmethod
    def _get_file_name(self, path):
        """
        Returns:
            str: a file name which exists on the file system.
        """
        pass

    def _open(self, path, mode="r"):
        """
        Returns:
            file: a file-like object.
        """
        return open(self._get_file_name(path), mode)

    @staticmethod
    def _has_protocol(url, protocol):
        return url.startswith(protocol + "://")


# Implement the handlers we need:


class NativePathHandler(PathHandler):
    def _support(self, path):
        return os.path.isfile(path)

    def _get_file_name(self, path):
        return path


class HTTPURLHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    def _support(self, path):
        return any(self._has_protocol(path, k) for k in ["http", "https", "ftp"])

    def _get_file_name(self, path):
        logger = logging.getLogger(__name__)

        url = urlparse(path)
        dirname = os.path.join(get_cache_dir(), os.path.dirname(url.path.lstrip("/")))
        cached = download(path, dirname)
        logger.info("URL {} cached in {}".format(path, cached))
        return cached


class PathManager:
    """
    A class for users to open generic paths or translate generic paths to file names.
    """

    _PATH_HANDLERS = [HTTPURLHandler(), NativePathHandler()]

    @staticmethod
    def open(path, mode="r"):
        for h in PathManager._PATH_HANDLERS:
            if h._support(path):
                return h._open(path, mode)
        raise OSError("Unable to open {}".format(path))

    @staticmethod
    def get_file_name(path):
        for h in PathManager._PATH_HANDLERS:
            if h._support(path):
                return h._get_file_name(path)
        raise OSError("Unable to lookup file name for {}".format(path))

    @staticmethod
    def register_handler(handler):
        """
        Add a handler to the end of the available handlers.

        Args:
            handler (PathHandler):
        """
        assert isinstance(handler, PathHandler), handler
        PathManager._PATH_HANDLERS.append(handler)
