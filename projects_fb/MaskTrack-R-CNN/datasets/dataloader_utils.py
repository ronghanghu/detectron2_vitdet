import copy
import itertools
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.build import print_instances_class_histogram
from detectron2.data.detection_utils import check_metadata_consistency


def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out video frames with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).

    Args:
        dataset_dicts (list[dict]): annotations in YTVIS Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = sum([len(x["frames"]) for x in dataset_dicts])

    def valid(anns):
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    # Filters out invalid frames from videos
    # Ensures both initial and reference frames have non-empty annotations
    new_dataset_dicts = []
    for vid_dict in dataset_dicts:
        n_vid_dict = copy.deepcopy(vid_dict)
        n_vid_dict["frames"] = [x for x in vid_dict["frames"] if valid(x["annotations"])]
        new_dataset_dicts.append(n_vid_dict)

    num_after = sum([len(x["frames"]) for x in new_dataset_dicts])
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return new_dataset_dicts


def get_VIS_dataset_dicts(names, filter_empty=True, min_keypoints=0, proposal_files=None):
    """
    Load and prepare dataset dicts for video instance segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]["frames"][0]
    print(f"Instances check: {has_instances}, Filter: {filter_empty}")
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)

    if has_instances:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            check_metadata_consistency("thing_classes", names)
            f_dataset_dicts = [x for y in dataset_dicts for x in y["frames"]]
            print_instances_class_histogram(f_dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts
