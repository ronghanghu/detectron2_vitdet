import torch.utils.data as data


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    """

    def __init__(self, dataset, map_func):
        self.dataset = dataset
        self.map_func = map_func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.map_func(self.dataset[idx])
