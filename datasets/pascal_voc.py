import torch.utils.data as data


class PascalVOC(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    Args:
        root (string): Root directory of VOC devkit
    """

    def __init__(self, root):
        self.root = root

    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0

if __name__ == '__main__':
    ds = PascalVOC("/")
