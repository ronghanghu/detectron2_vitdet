from torch import nn
from torch.nn import functional as F

from torch_detectron.modeling.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from torch_detectron.modeling.mask_rcnn import MaskFPNPooler

from torch_detectron.modeling.box_selector import ROI2FPNLevelsMapper

class MaskRCNNSharedFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNSharedFeatureExtractor, self).__init__()
        self.feature_extractor = feature_extractor
    
    def forward(self, x, proposals):
        if self.training:
            return x
        # proposals have been flattened by this point, so aren't
        # in the format of list[list[BBox]] anymore. Add one more level to it
        proposals = [proposals]
        return self.feature_extractor(x, proposals)


class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        roi_to_fpn_level_mapper = ROI2FPNLevelsMapper(2, 5)  # TODO expose these options
        pooler = MaskFPNPooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            drop_last=True,
            roi_to_fpn_level_mapper=roi_to_fpn_level_mapper,
        )
        input_size = cfg.BACKBONE.OUTPUT_DIM #  * resolution ** 2
        self.pooler = pooler

        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = nn.Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

        #self.conv5_mask = nn.ConvTranspose2d(next_feature, next_feature, 2, stride=2)
        #nn.init.kaiming_normal_(
        #    self.conv5_mask.weight, mode="fan_out", nonlinearity="relu"
        #)
        #nn.init.constant_(self.conv5_mask.bias, 0)


    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        # TODO need to handle case of no boxes -> empty x
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x



_ROI_MASK_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "MaskRCNNFPNFeatureExtractor": MaskRCNNFPNFeatureExtractor,
}


def make_roi_mask_feature_extractor(cfg):
    func = _ROI_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
