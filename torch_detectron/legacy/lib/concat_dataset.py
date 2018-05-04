import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

# maybe use https://stackoverflow.com/questions/6307761/how-can-i-decorate-all-functions-of-a-class-without-typing-it-over-and-over-for?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

class ConcatDataset(_ConcatDataset):
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
