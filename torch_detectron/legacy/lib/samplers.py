import itertools
import math
import numpy as np

from torch.utils.data.sampler import BatchSampler


class GroupedBatchSampler(BatchSampler):
    """Samples elements randomly, while enforcing that elements from the same group
    should appear in groups of group_size
    """
    def __init__(self, group_ids, group_size, drop_uneven=True, sampler=None):
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
        self.groups = np.unique(self.group_ids)
        self.group_size = group_size
        self.drop_uneven = drop_uneven
        self.sampler = sampler

    def __iter__(self):
        if self.sampler is not None:
            sampled_ids = list(self.sampler)
            mask = np.zeros((len(self.group_ids),), dtype=np.bool)
            mask[sampled_ids] = True
        else:
            mask = np.ones((len(self.group_ids),), dtype=np.bool)
        # TODO make it take into account the sampler that is passed for the order
        # cluster and shuffle indices that belong to the same group
        clusters = [np.random.permutation(np.where((self.group_ids == i) & mask)[0])
                for i in self.groups]

        # numpy split don't specify the size of each split easily
        def _split(a, n):
            return [a[i:i + n] for i in range(0, len(a), n)]
        splits = [_split(c, self.group_size) for c in clusters]

        if self.drop_uneven:
            for c in splits:
                if len(c[-1]) != self.group_size:
                    c.pop()

        # concatenate them all together
        merged = tuple(itertools.chain.from_iterable(splits))
        perm = np.random.permutation(len(merged))
        # shuffle each batch
        merged = tuple(merged[i] for i in perm)

        return iter(merged)

    def __len__(self):
        if self.sampler is not None:
            sampled_ids = list(self.sampler)
            mask = np.zeros((len(self.group_ids),), dtype=np.bool)
            mask[sampled_ids] = True
        else:
            mask = np.ones((len(self.group_ids),), dtype=np.bool)
        nums = tuple(((self.group_ids == i) & mask).sum() for i in self.groups)
        if self.drop_uneven:
            return sum(n // self.group_size for n in nums)
        else:
            return sum((n + self.group_size - 1) // self.group_size for n in nums)



if __name__ == '__main__':
    def test_grouped_sampler(group_ids, group_size, drop_uneven):
        sampler = GroupedBatchSampler(group_ids, group_size, drop_uneven)
        numel = 0
        for group in sampler:
            numel += 1
            elements = set()
            for sample in group:
                elements.add(group_ids[sample])
            # check they all belong to the same group
            assert len(elements) == 1
        assert numel == len(sampler)

    for _ in range(5):
        test_grouped_sampler(np.random.randint(2, size=30), 6, True)
        test_grouped_sampler(np.random.randint(2, size=30), 6, False)
