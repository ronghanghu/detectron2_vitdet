import datasets as ds
import torch.utils.data as data

def _collate_fn(batch):
    imgs = list()
    annotations = list()
    for img, annotation in batch:
        imgs.append(img)
        annotations.append(annotation)

    return (imgs, annotations)

if __name__ == '__main__':
    dataset = ds.PascalVOC("/datasets01/VOC/060817/VOCdevkit/", "trainval")
    dataloader = data.DataLoader(dataset, batch_size=8, collate_fn=_collate_fn)

    i = 0
    for item in enumerate(dataloader):
        if i < 1:
            print(item)
            i += 1
        else:
            raise RuntimeError('poop')
