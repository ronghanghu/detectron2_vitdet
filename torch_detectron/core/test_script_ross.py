import torch

from core.resnet_builder import resnet50_conv4_body, resnet50_conv5_head
from core.anchor_generator import AnchorGenerator, FPNAnchorGenerator
from core.box_selector import RPNBoxSelector, FPNRPNBoxSelector, ROI2FPNLevelsMapper
from core import detection_model_ross as detection_model
from core.faster_rcnn import RPNHeads, Pooler
from core.post_processor import PostProcessor, FPNPostProcessor
from core.region_proposal import RandomRegionProposal, FPNRandomRegionProposal

# copied from test_script.py with small modificatons

def _get_rpn_state_dict(pretrained_path):
    from collections import OrderedDict
    pretrained_state_dict = torch.load(pretrained_path)
    rpn_state_dict = OrderedDict()
    for k in ['conv.weight', 'conv.bias',
            'cls_logits.weight', 'cls_logits.bias',
            'bbox_pred.weight', 'bbox_pred.bias']:
        tensor = pretrained_state_dict['rpn.' + k]
        rpn_state_dict[k] = tensor
    return rpn_state_dict



def build_resnet_without_rpn_model(pretrained_path):
    # create FeatureProvider
    backbone = resnet50_conv4_body(pretrained_path)

    region_proposal = RandomRegionProposal(2000)
    # create heads
    pooler = Pooler()
    classifier_layers = resnet50_conv5_head(num_classes=81, pretrained=pretrained_path)
    postprocessor = PostProcessor()
    head = detection_model.Head(classifier_layers, postprocessor)
    
    # build model
    model = detection_model.GeneralizedRCNN(backbone, region_proposal, pooler, head)
    return model


def build_fpn_without_rpn_model(pretrained_path=None):
    from core.fpn import fpn_resnet50_conv5_body, FPNPooler, fpn_classification_head

    backbone = fpn_resnet50_conv5_body(pretrained_path)


    roi_to_fpn_level_mapper = ROI2FPNLevelsMapper(2, 5)
    region_proposal = FPNRandomRegionProposal(2000, roi_to_fpn_level_mapper)

    # rpn_provider = detection_model.RPNProvider(backbone, rpn, box_sampler=None)

    pooler = FPNPooler(
            output_size=(7, 7), scales=[2 ** (-i) for i in range(2, 6)], sampling_ratio=2, drop_last=True)
    classifier_layers = fpn_classification_head(num_classes=81, pretrained=pretrained_path)
    postprocessor = FPNPostProcessor()
    head = detection_model.Head(classifier_layers, postprocessor)

    model = detection_model.GeneralizedRCNN(backbone, region_proposal, pooler, head)
    return model


def build_resnet_model(pretrained_path):
    """
    Example function of how to build a detection model for inference.
    This function builds a Faster R-CNN network based on resnet50, without FPN nor mask
    prediction.
    """
    # create FeatureProvider
    backbone = resnet50_conv4_body(pretrained_path)

    anchor_generator = AnchorGenerator(scales=(0.125, 0.25, 0.5, 1., 2.), anchor_offset=(8.5, 8.5))
    rpn_heads = RPNHeads(256 * 4, anchor_generator.num_anchors_per_location()[0])
    rpn_heads.load_state_dict(_get_rpn_state_dict(pretrained_path))
    box_selector = RPNBoxSelector(6000, 1000, 0.7, 0)
    
    rpn = detection_model.RPN(rpn_heads, anchor_generator, box_selector, box_sampler=None)

    # create heads
    pooler = Pooler()
    classifier_layers = resnet50_conv5_head(num_classes=81, pretrained=pretrained_path)
    postprocessor = PostProcessor()
    head = detection_model.Head(classifier_layers, postprocessor)
    
    # build model
    model = detection_model.GeneralizedRCNN(backbone, rpn, pooler, head)
    return model


def build_fpn_model(pretrained_path=None):
    from core.fpn import fpn_resnet50_conv5_body, FPNPooler, fpn_classification_head

    backbone = fpn_resnet50_conv5_body(pretrained_path)

    anchor_generator = FPNAnchorGenerator(
            scales=(0.125, 0.25, 0.5, 1., 2.), anchor_strides=(4, 8, 16, 32, 64))
    rpn_heads = RPNHeads(256, anchor_generator.num_anchors_per_location()[0])
    rpn_heads.load_state_dict(_get_rpn_state_dict(pretrained_path))

    roi_to_fpn_level_mapper = ROI2FPNLevelsMapper(2, 5)
    box_selector = FPNRPNBoxSelector(roi_to_fpn_level_mapper=roi_to_fpn_level_mapper,
            fpn_post_nms_top_n=2000,
            pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.7, min_size=0)

    rpn = detection_model.RPN(rpn_heads, anchor_generator, box_selector, box_sampler=None)

    # rpn_provider = detection_model.RPNProvider(backbone, rpn, box_sampler=None)

    pooler = FPNPooler(
            output_size=(7, 7), scales=[2 ** (-i) for i in range(2, 6)], sampling_ratio=2, drop_last=True)
    classifier_layers = fpn_classification_head(num_classes=81, pretrained=pretrained_path)
    postprocessor = FPNPostProcessor()
    head = detection_model.Head(classifier_layers, postprocessor)

    model = detection_model.GeneralizedRCNN(backbone, rpn, pooler, head)
    return model


"""
Can't do in a nice way the inference part  -- drawback for this approach
"""
def build_mrcnn_fpn_model(pretrained_path=None):
    from core.fpn import fpn_resnet50_conv5_body, FPNPooler, fpn_classification_head

    backbone = fpn_resnet50_conv5_body(pretrained_path)

    anchor_generator = FPNAnchorGenerator(
            scales=(0.125, 0.25, 0.5, 1., 2.), anchor_strides=(4, 8, 16, 32, 64))
    rpn_heads = RPNHeads(256, anchor_generator.num_anchors_per_location()[0])
    rpn_heads.load_state_dict(_get_rpn_state_dict(pretrained_path))

    roi_to_fpn_level_mapper = ROI2FPNLevelsMapper(2, 5)
    box_selector = FPNRPNBoxSelector(roi_to_fpn_level_mapper=roi_to_fpn_level_mapper,
            fpn_post_nms_top_n=2000,
            pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.7, min_size=0)

    rpn = detection_model.RPN(rpn_heads, anchor_generator, box_selector, box_sampler=None)

    classifier_pooler = FPNPooler(
            output_size=(7, 7), scales=[2 ** (-i) for i in range(2, 6)], sampling_ratio=2, drop_last=True)
    classifier_layers = fpn_classification_head(num_classes=81, pretrained=pretrained_path)
    postprocessor = FPNPostProcessor()
    classifier_head = detection_model.Head(classifier_layers, postprocessor)

    from core.mask_rcnn import maskrcnn_head, MaskFPNPooler, MaskPostProcessor
    mask_pooler = MaskFPNPooler(roi_to_fpn_level_mapper=roi_to_fpn_level_mapper,
            output_size=(14, 14), scales=[2 ** (-i) for i in range(2, 6)], sampling_ratio=2, drop_last=True)
    mask_layers = maskrcnn_head(num_classes=81, pretrained=pretrained_path)
    mask_postprocessor = MaskPostProcessor()


    class CombinedMaskPoolerAndClassifier(torch.nn.Module):
        def __init__(self, pooler, layers):
            super(CombinedMaskPoolerAndClassifier, self).__init__()
            self.pooler = pooler
            self.layers = layers

        def forward(self, x):
            """
            x is a tuple of features and boxes
            """
            result = self.pooler(x[0], x[1])
            result = self.layers(result)
            return result

    mask_head = detection_model.Head(CombinedMaskPoolerAndClassifier(mask_pooler, mask_layers), mask_postprocessor)

    class SelectFeatures(torch.nn.Module):
        def forward(self, features, proposals):
            return features

    pooler = detection_model.MultiPoolers([classifier_pooler, SelectFeatures()])
    # raise RuntimeError("This doesn't work. Need to hack around by
    #         adding a new pooler inside the mask_head to be able to perform testing
    #         as done in Detectron. Training works fine though")
    head = detection_model.MaskOnTopOfClassifierHead([classifier_head, mask_head])

    model = detection_model.GeneralizedRCNN(backbone, rpn, pooler, head)
    return model


def get_image():
    from PIL import Image
    import numpy as np
    img = Image.open('/datasets01/COCO/060817/val2014/COCO_val2014_000000000139.jpg').convert('RGB').resize((1216, 800), Image.BILINEAR)

    x = torch.from_numpy(np.array(img)).float()[:, :, [2,1,0]]
    x = x - torch.tensor([102.9801, 115.9465, 122.7717])
    x = x.permute(2, 0, 1)

    return x

def run_model(model, bs):
    x = get_image()
    device = torch.device('cuda')
    x = x.to(device)
    model.to(device)

    x = [x for _ in range(bs)]

    with torch.no_grad():
        o = model.predict(x)

    print(o)

def run_model_without_rpn(model, bs, is_fpn=False):
    run_model(model, bs)

    print('Passing pre-computed proposals')
    device = torch.device('cuda')
    model.to(device)

    x = get_image().to(device)
    height, width = x.shape[-2:]

    bbox = get_random_bboxes(2000, height, width, device)

    x = [x for _ in range(bs)]
    if is_fpn:
        bbox = [[bbox for _ in range(bs)] for _ in range(2, 6)]
    else:
        bbox = [[bbox for _ in range(bs)]]

    with torch.no_grad():
        o = model.predict(x, proposals=bbox)
    print(o)


def get_random_bboxes(number_of_proposals, height, width, device):
    from torchvision.structures.bounding_box import BBox
    boxes = torch.rand(number_of_proposals, 4, device=device)
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    bbox = BBox(boxes, (width, height), mode='xywh').convert('xyxy')
    bbox.bbox[:, 0].clamp_(min=0, max=width)
    bbox.bbox[:, 1].clamp_(min=0, max=height)
    bbox.bbox[:, 2].clamp_(min=0, max=width)
    bbox.bbox[:, 3].clamp_(min=0, max=height)
    return bbox


def test_resnet_without_rpn(bs=2):
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/faster_rcnn_resnet50.pth'
    model = build_resnet_without_rpn_model(pretrained_path)
    run_model_without_rpn(model, bs, is_fpn=False)

def test_fpn_without_rpn(bs=2):
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/fpn_r50.pth'
    model = build_fpn_without_rpn_model(pretrained_path)
    run_model_without_rpn(model, bs, is_fpn=True)

def test_resnet(bs=2):
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/faster_rcnn_resnet50.pth'
    model = build_resnet_model(pretrained_path)
    run_model(model, bs)

def test_fpn(bs=2):
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/fpn_r50.pth'
    model = build_fpn_model(pretrained_path)
    run_model(model, bs)

def test_mrcnn_fpn(bs=2):
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/models/mrcnn_fpn_r50.pth'
    model = build_mrcnn_fpn_model(pretrained_path)
    run_model(model, bs)

def test_all():
    print('Fast R-CNN ResNet without RPN')
    test_resnet_without_rpn()
    print('Faster R-CNN ResNet')
    test_resnet()
    print('Faster R-CNN ResNet FPN without RPN')
    test_fpn_without_rpn()
    print('Faster R-CNN ResNet FPN')
    test_fpn()
