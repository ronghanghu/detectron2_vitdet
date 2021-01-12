import ast
import importlib
import inspect
import os
from copy import deepcopy
from typing import List
import yaml
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, ListConfig, OmegaConf

from detectron2.utils.registry import _convert_target_to_string, locate


"""
Highly experimental.

Reference:
https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
"""


def _visit_dict_config(cfg, func):
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _visit_dict_config(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _visit_dict_config(v, func)


def apply_overrides(cfg, overrides: List[str]):
    parser = OverridesParser.create()
    overrides = parser.parse_overrides(overrides)
    for o in overrides:
        key = o.key_or_group
        value = o.value()
        assert not o.is_delete()  # TODO
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


def omegaconf_resolve(conf):
    """
    Resolve interpolation but still return DictConfig/ListConfig
    """
    return DictConfig(
        content=OmegaConf.to_container(conf, resolve=True), flags={"allow_objects": True}
    )


# TODO: better names ..
def Lazy(_target_, *args, **kwargs):
    assert len(args) == 0  # TODO?
    kwargs["_target_"] = _target_
    return DictConfig(content=kwargs, flags={"allow_objects": True})


Config = Lazy


class ConfigFile:
    def __init__(self, dict, filename=None):
        super().__setattr__("_dict", dict)
        super().__setattr__("_filename", filename)

    def __getattr__(self, name):
        return self._dict[name]

    def __setattr__(self, name, val):
        self._dict[name] = val

    def __delattr__(self, name):
        del self._dict[name]

    @staticmethod
    def load_rel(filename):
        """
        Load path relative to the caller's file.
        """
        caller_frame = inspect.stack()[1]
        # TODO: use filename directly if caller is not a file
        caller_fname = caller_frame[0].f_code.co_filename
        caller_dir = os.path.dirname(caller_fname)
        filename = os.path.join(caller_dir, filename)
        return ConfigFile.load(filename)

    @staticmethod
    def load(filename):
        if filename.endswith(".py"):
            ConfigFile._validate_py_syntax(filename)
            spec = importlib.util.spec_from_file_location("detectron2_tmp_module", filename)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            top_level = DictConfig(
                {
                    name: value
                    for name, value in mod.__dict__.items()
                    if isinstance(value, (DictConfig, ListConfig)) and not name.startswith("_")
                }
            )  # this does a deepcopy under the hood
            return top_level
        else:
            assert filename.endswith(".yaml"), filename
            with open(filename) as f:
                obj = yaml.load(f)
            return OmegaConf.create(obj, flags={"allow_objects": True})

    @staticmethod
    def save(cfg, filename):
        cfg = deepcopy(cfg)

        def _replace_type_by_name(x):
            if "_target_" in x:
                x._target_ = _convert_target_to_string(x._target_)

        # not necessary, but makes yaml looks nicer
        _visit_dict_config(cfg, _replace_type_by_name)
        OmegaConf.save(cfg, filename)

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, "r") as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(f"Config file {filename} has syntax error") from e


def instantiate(cfg):
    cfg = omegaconf_resolve(cfg)

    # def _add_convert_flag(x):
    # if "_target_" in x:
    # x["_convert_"] = "all"
    # _visit_dict_config(cfg, _add_convert_flag)
    # return hydra.utils.instantiate(cfg)
    # slow. https://github.com/facebookresearch/hydra/issues/1200
    return _instantiate_after_resolve(cfg)


def _instantiate_after_resolve(cfg):
    if isinstance(cfg, DictConfig):
        newcfg = {}  # use python dict to be efficient.
        for k in list(cfg.keys()):
            newcfg[k] = _instantiate_after_resolve(cfg[k])
        cfg = newcfg
    elif isinstance(cfg, ListConfig):
        # TODO: specialize for list, because many classes take
        # list[constructible objects], such as ResNet, DatasetMapper
        # alternative: wrap the list under a Lazy([...]) call in config
        cfg = [_instantiate_after_resolve(cfg[k]) for k in range(len(cfg))]

    if isinstance(cfg, dict) and "_target_" in cfg:
        cls = cfg.pop("_target_")
        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        assert callable(cls), cls
        return cls(**cfg)
    return cfg


if __name__ == "__main__":
    pass
