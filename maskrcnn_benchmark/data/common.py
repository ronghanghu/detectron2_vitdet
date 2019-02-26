import copy
import torch.utils.data as data


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._map_func(self._dataset[idx])


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst: list, copy: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
        """
        self._lst = lst
        self._copy = copy

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        if self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]
