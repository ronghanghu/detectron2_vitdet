import torch

from core.resnet_builder import resnet50_conv4_body, resnet50_conv5_head
from core.anchor_generator import AnchorGenerator, FPNAnchorGenerator
from core.box_selector import RPNBoxSelector, FPNRPNBoxSelector, ROI2FPNLevelsMapper
from core import detection_model
from core.faster_rcnn import RPNHeads, Pooler
from core.post_processor import PostProcessor, FPNPostProcessor


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


def build_resnet_model():
    """
    Example function of how to build a detection model for inference.
    This function builds a Faster R-CNN network based on resnet50, without FPN nor mask
    prediction.
    """
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/lib/clean_inference/faster_rcnn_resnet50.pth'
    # create FeatureProvider
    backbone = resnet50_conv4_body(pretrained_path)

    anchor_generator = AnchorGenerator(scales=(0.125, 0.25, 0.5, 1., 2.), anchor_offset=(8.5, 8.5))
    rpn_heads = RPNHeads(256 * 4, anchor_generator.num_anchors_per_location()[0])
    rpn_heads.load_state_dict(_get_rpn_state_dict(pretrained_path))
    box_selector = RPNBoxSelector(6000, 1000, 0.7, 0)
    
    rpn = detection_model.RPN(rpn_heads, anchor_generator, box_selector)

    rpn_provider = detection_model.RPNProvider(backbone, rpn, box_sampler=None)
    
    # create heads
    pooler = Pooler()
    classifier_layers = resnet50_conv5_head(num_classes=81, pretrained=pretrained_path)
    postprocessor = PostProcessor()
    classifier_head = detection_model.ClassificationHead(pooler, classifier_layers, postprocessor)
    
    # build model
    model = detection_model.GeneralizedRCNN(rpn_provider, classifier_head)
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

    rpn = detection_model.RPN(rpn_heads, anchor_generator, box_selector)

    rpn_provider = detection_model.RPNProvider(backbone, rpn, box_sampler=None)

    pooler = FPNPooler(
            output_size=(7, 7), scales=[2 ** (-i) for i in range(2, 6)], sampling_ratio=2, drop_last=True)
    classifier_layers = fpn_classification_head(num_classes=81, pretrained=pretrained_path)
    postprocessor = FPNPostProcessor()
    classifier_head = detection_model.ClassificationHead(pooler, classifier_layers, postprocessor)

    model = detection_model.GeneralizedRCNN(rpn_provider, classifier_head)
    return model


def test_fpn():
    pretrained_path = '/private/home/fmassa/github/detectron.pytorch/torch_detectron/core/fpn_r50.pth'
    model = build_fpn_model(pretrained_path)
    # x = torch.rand(1, 3, 800, 800)

    from PIL import Image
    import numpy as np
    img = Image.open('/datasets01/COCO/060817/val2014/COCO_val2014_000000000139.jpg').convert('RGB').resize((1216, 800), Image.BILINEAR)

    x = torch.from_numpy(np.array(img)).float()[:, :, [2,1,0]]
    x = x - torch.tensor([102.9801, 115.9465, 122.7717])
    x = x.permute(2, 0, 1).unsqueeze(0)


    device = torch.device('cuda')
    x = x.to(device)
    model.to(device)

    with torch.no_grad():
        o = model.predict([x[0], x[0]])

    print(o)

