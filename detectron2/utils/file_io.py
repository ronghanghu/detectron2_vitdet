# -*- coding: utf-8 -*-

import errno
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

    def _get_local_path(self, path):
        """
        Returns:
            str: a file path which exists on the local file system. This function
                 can download/cache if the resource is located remotely.
        """
        return path

    def _open(self, path, mode="r"):
        """
        Returns:
            file: a file-like object.
        """
        raise NotImplementedError()

    def _exists(self, path):
        """
        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def _isfile(self, path):
        """
        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()

    def _ls(self, path):
        """
        Returns:
            List[str]: list of contents in given path
        """
        raise NotImplementedError()

    def _mkdirs(self, path):
        """
        Creates path recursively
        """
        raise NotImplementedError()

    @staticmethod
    def _has_protocol(url, protocol):
        return url.startswith(protocol + "://")


class NativePathHandler(PathHandler):
    def _support(self, path):
        return bool(path)

    def _get_local_path(self, path):
        return path

    def _open(self, path, mode="r"):
        return open(path, mode)

    def _exists(self, path):
        return os.path.exists(path)

    def _isfile(self, path):
        return os.path.isfile(path)

    def _ls(self, path):
        return os.listdir(path)

    def _mkdirs(self, path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # EEXIST it can still happen if multiple processes are creating the dir
            if e.errno != errno.EEXIST:
                raise


class HTTPURLHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    def __init__(self):
        self.cache_map = {}

    def _support(self, path):
        return any(self._has_protocol(path, k) for k in ["http", "https", "ftp"])

    def _get_local_path(self, path):
        if path not in self.cache_map:
            logger = logging.getLogger(__name__)
            url = urlparse(path)
            dirname = os.path.join(get_cache_dir(), os.path.dirname(url.path.lstrip("/")))
            cached = download(path, dirname)
            logger.info("URL {} cached in {}".format(path, cached))
            self.cache_map[path] = cached
        return self.cache_map[path]

    def _open(self, path, mode="r"):
        assert mode in ("r", "rb"), "{} does not support open with {} mode".format(
            self.__class__.__name__, mode
        )
        local_path = self._get_local_path(path)
        return open(local_path, mode)


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
    def get_local_path(path):
        for h in PathManager._PATH_HANDLERS:
            if h._support(path):
                return h._get_local_path(path)
        raise OSError("Unable to lookup file name for {}".format(path))

    @staticmethod
    def exists(path):
        for h in PathManager._PATH_HANDLERS:
            if h._support(path):
                return h._exists(path)
        raise OSError("Unable to lookup file name for {}".format(path))

    @staticmethod
    def isfile(path):
        for h in PathManager._PATH_HANDLERS:
            if h._support(path):
                return h._isfile(path)
        raise OSError("Unable to lookup file name for {}".format(path))

    @staticmethod
    def ls(path):
        for h in PathManager._PATH_HANDLERS:
            if h._support(path):
                return h._ls(path)
        raise OSError("Unable to lookup file name for {}".format(path))

    @staticmethod
    def mkdirs(path):
        for h in PathManager._PATH_HANDLERS:
            if h._support(path):
                return h._mkdirs(path)
        raise OSError("Unable to lookup file name for {}".format(path))

    @staticmethod
    def register_handler(handler):
        """
        Add a handler to the end of the available handlers.

        Args:
            handler (PathHandler):
        """
        assert isinstance(handler, PathHandler), handler
        PathManager._PATH_HANDLERS.insert(0, handler)
