import torch


class Collator(object):
    def __call__(self, batch):
        flip_batch = tuple(zip(*batch))
        imgs = flip_batch[0]
        max_size = tuple(max(s) for s in zip(*[img.shape for img in imgs]))
        batch_shape = (len(imgs),) + max_size
        batched_imgs = imgs[0].new(*batch_shape).zero_()
        for img, pad_img in zip(imgs, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        return (batched_imgs,) + flip_batch[1:]


if __name__ == '__main__':
    collator = Collator()
    batch = [
            (torch.rand(3, 100, 100), torch.rand(10, 4)),
            (torch.rand(3, 200, 100), torch.rand(20, 4)),
            (torch.rand(3, 100, 200), torch.rand(5, 4))
    ]
    collated_batch = collator(batch)
    assert len(collated_batch) == 2
    assert tuple(collated_batch[0].shape) == (3, 3, 200, 200)
    assert len(collated_batch[1]) == 3
    print(collated_batch)
