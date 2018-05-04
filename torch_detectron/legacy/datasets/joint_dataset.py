import torch.utils.data as data

class JointDataset(data.Dataset):
    """Abstraction to augment a single dataset with multiple joint transforms.

    Motivated by the following issue. In training for our detection platform,
    we may want to augment our dataset by flipping the images to generate 2x
    the training data. We can think of our new dataset being the same original
    set of data, where the images/targets are transformed by an "identity"
    transformation in the first case, and a "flip" transformation in the
    second.

    This class allows us to register a dataset and a series of joint_transforms
    that operate jointly on the example and target and provide a single
    interface to the permutations of the base dataset.

    Note that in theory, we could also imagine creating two Datasets (say two
    instances of the PascalVOC dataset, one with the flip transform and one
    without) and simply concatenate them. However, for datasets like COCO,
    which explicitly create an underlying index, this would have the effect of
    creating duplicates indices in memory.
    """

    def __init__(self, dataset, joint_transforms):
        self.dataset = dataset
        self.joint_transforms = joint_transforms

    def __len__(self):
        return len(self.dataset) * len(self.joint_transforms)

    def __getitem__(self, index):
        transform = int(index / len(self.dataset))
        img, target = self.dataset[index % len(self.dataset)]

        return self.joint_transforms[transform](img, target)
