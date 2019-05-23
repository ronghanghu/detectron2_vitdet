# -*- coding: utf-8 -*-

import logging
import os
from abc import ABCMeta, abstractmethod

from .model_zoo import ModelCatalog, cache_url

__all__ = ["PathManager"]


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
        cached = cache_url(path)
        logger.info("URL {} cached in {}".format(path, cached))
        return cached


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


class PathManager:
    """
    A class for users to open generic paths or translate generic paths to file names.
    """

    _PATH_HANDLERS = [
        HTTPURLHandler(),
        NativePathHandler(),
        ModelCatalogHandler(),
        Detectron2Handler(),
    ]

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
