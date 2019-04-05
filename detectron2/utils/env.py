import importlib
import os

from detectron2.utils.imports import import_file


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $DETECTRON2_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    custom_module_path = os.environ.get("DETECTRON2_ENV_MODULE")

    # TODO remove this after a while
    if custom_module_path == "infra.fb.env":
        custom_module_path = "detectron2.fb.env"
        print("---------------------- NOTE ---------------------------------")
        print("Please use `DETECTRON2_ENV_MODULE=detectron2.fb.env` instead!")
        print("-------------------------------------------------------------")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """
    if custom_module.endswith(".py"):
        module = import_file("detectron2.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(custom_module)
    module.setup_environment()


# Force environment setup when this module is imported
setup_environment()
