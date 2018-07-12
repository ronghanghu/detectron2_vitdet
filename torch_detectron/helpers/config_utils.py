"""
Utilities for building configuration files
"""

import importlib
import importlib.util


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def import_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_config(config_path):
    config_module = import_file("torch_detectron.config", config_path)
    return config_module.config


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class ConfigClass(object):
    """
    This is the base class that should be used when creating configuration
    classes.
    All it does is to give a nicer error message if an attribute is not
    set, and also has a nice repr that allows for json-like printing
    """

    def __init__(self, config=None):
        self._config = config

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            raise AttributeError(
                "{} is not specified. There is either "
                "a typo or you forgot to set a value".format(attr)
            )
        return object.__getattr__(self, attr)

    def __repr__(self):
        r = object.__repr__(self)
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        if attrs:
            r += ": \n"
            s = []
            for k, v in attrs.items():
                attr_str = "{}: {}".format(repr(k), repr(v))
                attr_str = _addindent(attr_str, 2)
                s.append(attr_str)
            r += "  " + "\n  ".join(s)
        return r

    @property
    def config(self):
        return self._config


class AttrDict(dict):
    """
    A simple attribute dictionary used for representing configuration options.
    """

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

    def __repr__(self):
        r = object.__repr__(self)
        s = []
        for k, v in self.items():
            attr_str = "{}: {}".format(repr(k), repr(v))
            attr_str = _addindent(attr_str, 2)
            s.append(attr_str)

        return r + "\n  " + "\n  ".join(s)


def _walk_to(obj, path):
    """
    Given a nested object (that can be accessed via getattr),
    iterates over a nested path defined by a string with dot
    as a delimiter and returns the last object that should
    be accessed, and the attribute that access this object.

    For example, if path is 'MODEL.ROI_HEADS.NMS', then
    the returned position will be the object represented by
    MODEL.ROI_HEADS, and attr will be NMS.

    Arguments:
        obj: an arbitrarily-nested object with attributes
        path (string)
    """
    path = path.split(".")
    path, attr = path[:-1], path[-1]
    position = obj
    for step in path:
        position = getattr(position, step)
    return position, attr


def update_config_with_args(config, args):
    assert len(args) % 2 == 0
    keys = args[::2]
    values = args[1::2]
    for k, v in zip(keys, values):
        parent, attr = _walk_to(config, k)
        attr_type = type(getattr(parent, attr))
        # overwrite with new attribute
        setattr(parent, attr, attr_type(v))
