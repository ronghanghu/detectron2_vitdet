import math
import torch
import lib.utils.box as box_utils


class ProposalDataset(object):
    def __init__(self, dataset, proposal_file):
        import pickle
        self.dataset = dataset
        with open(proposal_file, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

        proposals_map_to_id = {v: k for k, v in enumerate(self.data['ids'])}
        self.img_id_map_to_proposal_id = {k: proposals_map_to_id[v]
                for k, v in enumerate(dataset.img_ids)}

    def __getitem__(self, idx):
        item = self.dataset[idx]
        proposal_id = self.img_id_map_to_proposal_id[idx]
        boxes = torch.from_numpy(self.data['boxes'][proposal_id])

        boxes = boxes.clone()

        img = item[0]

        """
        # preproceesing the boxes
        height = img.size(1)
        width= img.size(2)
        boxes = box_utils.clip_boxes_to_image(
            boxes, height, width
        )
        keep = box_utils.unique_boxes(boxes)
        boxes = boxes[keep, :]
        min_proposal_size = 2
        keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
        boxes = boxes[keep, :]


        min_size = min(img.shape[1:])
        max_size = max(img.shape[1:])
        MIN_SIZE = 600.  # 800.
        MAX_SIZE = 1000.  # 1333.
        ratio = MIN_SIZE / min_size
        if max_size * ratio > MAX_SIZE:
            ratio = MAX_SIZE / max_size
        sizes = [int(math.ceil(img.shape[1] * ratio)), int(math.ceil(img.shape[2] * ratio))]
        img = torch.nn.functional.upsample(Variable(img[None]), size=sizes, mode='bilinear').data[0]
        boxes = boxes * ratio
        gt_boxes = item[1] * ratio
        """

        new_item = (item[0], boxes) + item[1:]
        # new_item = (img, boxes, gt_boxes) + item[2:]

        return new_item

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name in self.dataset.__dict__:
            return getattr(self.dataset, name)
        return object.__getattr__(self, name)

class ProposalDatasetVOC(ProposalDataset):
    def __init__(self, dataset, proposal_file):
        import scipy.io as sio
        raw_data = sio.loadmat(proposal_file)['boxes'].ravel()
        self.data = [b[:, (1, 0, 3, 2)].astype(np.float32) - 1 for b in raw_data]
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        boxes = torch.from_numpy(self.data[idx])
        boxes = torch.cat([item[1][0], boxes], 0)
        new_item = (item[0], boxes) + item[1]
        return new_item


