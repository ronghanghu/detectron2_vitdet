import os

from libfb.py import parutil


def setup_environment():
    # When running from a PAR, we need to define the MATPLOTLIBDATA location
    os.environ["MATPLOTLIBDATA"] = parutil.get_dir_path("matplotlib/mpl-data")
