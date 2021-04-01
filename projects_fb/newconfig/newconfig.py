import ast
import builtins
import importlib
import inspect
import logging
import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from typing import List, Tuple, Union
import cloudpickle
import yaml
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, ListConfig, OmegaConf

from detectron2.utils.file_io import PathManager
from detectron2.utils.registry import _convert_target_to_string

logger = logging.getLogger("detectron2.config.instantiate")


_CFG_PACKAGE_NAME = "detectron2._cfg_loader"


def _visit_dict_config(cfg, func):
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _visit_dict_config(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _visit_dict_config(v, func)


def _random_config_package(filename):
    # generate a random package name when loading config files
    return _CFG_PACKAGE_NAME + str(uuid.uuid4())[:4] + "." + os.path.basename(filename)


def apply_overrides(cfg, overrides: List[str]):
    parser = OverridesParser.create()
    overrides = parser.parse_overrides(overrides)
    for o in overrides:
        key = o.key_or_group
        value = o.value()
        assert not o.is_delete()  # TODO
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. support other storage system through PathManager
    4. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip(".")
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)
        # NOTE: directory import is not handled
        if not cur_file.endswith(".py"):
            cur_file += ".py"
        if not PathManager.isfile(cur_file):
            raise ImportError(
                f"Cannot import name {relative_import_path} from "
                f"{original_file}: {cur_file} does not exist."
            )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and globals.get("__package__", "").startswith(_CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            ConfigFile._validate_py_syntax(cur_file)
            spec = importlib.machinery.ModuleSpec(
                _random_config_package(cur_file), None, origin=cur_file
            )
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with PathManager.open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            for name in fromlist:  # turn imported dict into DictConfig automatically
                val = module.__dict__[name]
                if isinstance(val, dict):
                    val = DictConfig(val, flags={"allow_objects": True})
                    module.__dict__[name] = val
            return module
        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


class ConfigFile:
    @staticmethod
    def load_rel(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Similar to :meth:`load()`, but load path relative to the caller's
        source file.

        This has the same functionality as a relative import, except that it
        accepts filename as a string, so more characters are allowed in the name.
        """
        caller_frame = inspect.stack()[1]
        caller_fname = caller_frame[0].f_code.co_filename
        assert caller_fname != "<string>", "load_rel Unable to find caller"
        caller_dir = os.path.dirname(caller_fname)
        filename = os.path.join(caller_dir, filename)
        return ConfigFile.load(filename, keys)

    @staticmethod
    def load(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Load a config file.

        Args:
            filename: absolute path or relative path w.r.t. the current working directory
            keys: keys to load and return. If not given, return all keys
                (whose values are config objects) in a dict.
        """
        filename = filename.replace("/./", "/")  # redundant
        if filename.endswith(".py"):
            ConfigFile._validate_py_syntax(filename)

            with _patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": _random_config_package(filename),
                }
                with PathManager.open(filename) as f:
                    content = f.read()
                # Compile first with filename to:
                # 1. make filename appears in stacktrace
                # 2. make load_rel able to find its parent's (possibly remote) location
                exec(compile(content, filename, "exec"), module_namespace)

            top_level = DictConfig(
                {
                    name: value
                    for name, value in module_namespace.items()
                    if isinstance(value, (DictConfig, ListConfig, dict))
                    and not name.startswith("_")
                },
                flags={"allow_objects": True},
            )  # this does a deepcopy under the hood
            ret = top_level
        else:
            assert filename.endswith(".yaml"), filename
            with PathManager.open(filename) as f:
                obj = yaml.unsafe_load(f)
            ret = OmegaConf.create(obj, flags={"allow_objects": True})
        if keys is not None:
            if isinstance(keys, str):
                keys = (keys,)
            ret = tuple(getattr(ret, a) for a in keys)
            return ret[0] if len(ret) == 1 else ret
        else:
            return ret

    @staticmethod
    def save(cfg, filename):
        cfg = deepcopy(cfg)

        def _replace_type_by_name(x):
            if "_target_" in x and not isinstance(x._target_, str):
                try:
                    x._target_ = _convert_target_to_string(x._target_)
                except AttributeError:
                    pass

        # not necessary, but makes yaml looks nicer
        _visit_dict_config(cfg, _replace_type_by_name)

        try:
            OmegaConf.save(cfg, filename)
        except Exception:
            logger.exception("Unable to serialize the config to yaml. Error:")
            new_filename = filename + ".pkl"
            try:
                # retry by pickle
                with PathManager.open(new_filename, "wb") as f:
                    cloudpickle.dump(cfg, f)
                logger.warning(f"Config saved using cloudpickle at {new_filename} ...")
            except Exception:
                pass

    @staticmethod
    def _validate_py_syntax(filename):
        # see also https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
        with PathManager.open(filename, "r") as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(f"Config file {filename} has syntax error!") from e


if __name__ == "__main__":
    pass
