from faster_rcnn import FasterRCNN, RPN
from resnet import ResNet50COCO
from detection_model import DetectionModel


class ResNet50FasterRCNN(FasterRCNN, ResNet50COCO, DetectionModel):
    def __init__(self, rpn, **kwargs):
        super(ResNet50FasterRCNN, self).__init__(rpn=rpn, **kwargs)
 

if __name__ == '__main__':
    import torch
    from torchvision.transforms import functional as F

    from PIL import Image

    from os.path import join


    classes = ['__background__', 'person', 'bicycle', 'car', 'motorcycle',
       'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
       'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
       'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
       'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
       'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
       'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
       'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
       'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
       'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
       'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
       'scissors', 'teddy bear', 'hair drier', 'toothbrush'] 

   
    rpn = RPN(inplanes=256 * 4)
    model = ResNet50FasterRCNN(rpn, min_size=800, max_size=1333, class_names=classes)
    state_dict = torch.load(
            '/private/home/fmassa/github/detectron.pytorch/torch_detectron/lib/clean_inference/faster_rcnn_resnet50.pth')
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    data_dir = '/datasets01/COCO/060817/val2014/'
    img_paths = [
        'COCO_val2014_000000000139.jpg',
        'COCO_val2014_000000000285.jpg',
        'COCO_val2014_000000000632.jpg',
        'COCO_val2014_000000000724.jpg',
        'COCO_val2014_000000000776.jpg']

    images = [F.to_tensor(Image.open(join(data_dir, p))) for p in img_paths]

    images = [img.cuda() for img in images]
    with torch.no_grad():
        # can pass either a single 3xHxW tensor, or a list of 3xHxW tensors, which
        # can potentially have different sizes
        o = model.predict(images[0])

    """
    import pickle
    with open('results.plk', 'wb') as f:
        pickle.dump({'prob': o[0][0].numpy(), 'boxes': o[1][0].numpy(), 'names':o[2][0]}, f)
    """
