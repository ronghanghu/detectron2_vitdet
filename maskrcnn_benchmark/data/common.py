import bisect

import torch.utils.data as data
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList


class ConcatDataset(data.dataset.ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)


class MapDataset(data.Dataset):
    def __init__(self, ds, map_func):
        self.ds = ds
        self.map_func = map_func

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.map_func(self.ds[idx])


class NaiveListDataset(object):
    """
    Simple dataset class that wraps a list of path names
    """

    def __init__(self, image_lists, transforms=None):
        self.image_lists = image_lists
        self.transforms = transforms

    def __getitem__(self, item):
        img = Image.open(self.image_lists[item]).convert("RGB")

        # dummy target
        w, h = img.size
        target = BoxList([[0, 0, w, h]], img.size, mode="xyxy")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_lists)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        pass
