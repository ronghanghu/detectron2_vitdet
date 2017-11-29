import torch_detectron as td
import torch.utils.data as data
import unittest

class DummyDataset(data.Dataset):

    def __getitem__(self, index):
        return 1, 2

    def __len__(self):
        return 1

class Tester(unittest.TestCase):

    def test_joint_dataset(self):
        ds = DummyDataset()
        
        def identity(x, y):
            return x, y

        def double(x, y):
            return x * 2, y * 2

        jds = td.datasets.JointDataset(ds, [identity, double])
        assert len(jds) == 2
        
        a, b = jds[0]
        c, d = jds[1]

        assert (a, b) == (1, 2)
        assert (c, d) == (2, 4)


if __name__ == '__main__':
    unittest.main()

