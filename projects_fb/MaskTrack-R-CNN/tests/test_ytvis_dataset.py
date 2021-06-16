# Copyright (c) Facebook, Inc. and its affiliates.
import os
import pathlib
import sys

import detectron2.projects_fb.masktrack_rcnn.lib.datasets.builtin_ytvis  # noqa
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.projects_fb.masktrack_rcnn.lib.datasets.ytvis import load_ytvis_json
from detectron2.projects_fb.masktrack_rcnn.lib.utils.ytvis_visualizer import YTVISVisualizer
from detectron2.utils.logger import setup_logger

"""
This file tests that the YTVIS dataset is registered properly and can be visualized.
"""

if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.

    Usage:
        python -m detectron2.data.datasets.ytvis \
            path/to/json path/to/image_root dataset_name
    """

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_ytvis_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info(f"Done loading {len(dicts)} samples.")

    dirname = "ytvis-data-vis"
    os.makedirs(dirname, exist_ok=True)
    instance_colors = {}

    for vid_dict in dicts:
        logger.info(vid_dict.keys())
        visualizer = YTVISVisualizer(metadata=meta)
        for frame_dict in vid_dict["frames"]:
            vis = visualizer.draw_dataset_dict(frame_dict, instance_colors)
            file_name = pathlib.Path(frame_dict["file_name"]).resolve()
            file_name = file_name.relative_to(file_name.parent.parent)
            file_name = dirname / file_name
            file_name.parent.mkdir(parents=True, exist_ok=True)
            vis.save(str(file_name))
