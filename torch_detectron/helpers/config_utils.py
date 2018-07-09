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
    return import_file("torch_detectron.config", config_path)


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

    For example, if path is 'MODEL.HEADS.NMS', then
    the returned position will be the object represented by
    MODEL.HEADS, and attr will be NMS.

    Arguments:
        obj: an arbitrarily-nested object with attributes
        path (string)
    """
    path = path.split('.')
    path, attr = path[:-1], path[-1]
    position = obj
    for step in path:
        position = getattr(position, step)
    return position, attr


class LinkedNode(object):
    """
    Helper class that for creating linked nodes
    """
    def __init__(self, base_obj, name):
        self.base_obj = base_obj
        self.name = name

    def __call__(self, obj):
        attrs = self.name.split('.')
        base_obj = self.base_obj
        for attr in attrs:
            base_obj = getattr(base_obj, attr)
        return base_obj

    def __repr__(self):
        s = object.__repr__(self)
        name = "{} (link to {}: {})".format(s, self.name, self.__call__(self))
        return name


class LinkedAttrProperty(object):
    """
    Helper class that for creating linked nodes
    """
    def __init__(self, obj, path_to_attribute_to_link):
        self.obj = obj
        self.path_to_attribute_to_link = path_to_attribute_to_link

    def __call__(self, obj):
        attr = self.path_to_attribute_to_link.split('.')[-1]
        if attr in obj.__dict__:
            return obj.__dict__[attr](obj)
        raise RuntimeError("Internal error. Write a bug report")


# inspired by
# https://stackoverflow.com/questions/7117892/how-do-i-assign-a-property-to-an-instance-in-python
def link_nodes(obj, attr_to_set, attr_to_link):
    """
    Function that links two arbitrary nodes in the graph, so that
    once one of the nodes is queryied the returned object will be
    the the other object.

    This is useful to create graphs that do not represent a tree.

    Note: this behavior is obtained via Python properties.
    But Python properties need to be added to the `class`, and
    not to instances of the class. Thus, in order to open the
    possibility of having instance-level attributes, the attribute
    in the base class will query for a specific element in the
    instance object.
    This enables both having a pretty-print that is compatible
    with what we had before, but also to be able to have
    different instances having different behaviors.

    Arguments:
        obj (AttrDict): the base config object
        attr_to_set (string): the path to the attribute to be
            set, using dot (.) as the separator
        attr_to_link (string): the path to the attribute to be
            linked against, using dot (.) as the separator
    """
    linked_attr_property = LinkedAttrProperty(obj, attr_to_set)
    # iterate over obj until attr_to_set
    target_obj, attr = _walk_to(obj, attr_to_set)
    # set the property of the whole class
    setattr(target_obj.__class__, attr, property(linked_attr_property))
    # specialize the result value for the specific instance
    target_obj.__dict__[attr] = LinkedNode(obj, attr_to_link)


def get_attributes_of(self):
    """
    This function is equivalent to vars(self), but
    replaces the instances of LinkedNode with the
    linked values
    """
    attrs = vars(self)
    for k, v in attrs.items():
        if isinstance(v, LinkedNode):
            dummy_value = 0
            attrs[k] = v(dummy_value)
    return attrs
