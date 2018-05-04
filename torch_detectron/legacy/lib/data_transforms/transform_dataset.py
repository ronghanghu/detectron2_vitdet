class TransformDataset(object):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item)
    
    def __len__(self):
        return len(self.dataset)
    # TODO doesn't work in forkserver mode, infinite recursion

    def __getattr__(self, name):
        if name in self.dataset.__dict__:
            return getattr(self.dataset, name)
        return object.__getattr__(self, name)

