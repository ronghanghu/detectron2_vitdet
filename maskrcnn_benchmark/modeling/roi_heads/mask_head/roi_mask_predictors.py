from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


# Temporarily keep backward compatibility
def MaskRCNNC4Predictor(cfg):
    num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

    if cfg.MODEL.ROI_HEADS.USE_FPN:
        num_inputs = dim_reduced
    else:
        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor
    return MaskRCNNConvUpsampleHead(0, num_classes, num_inputs, dim_reduced)


class MaskRCNNConvUpsampleHead(nn.Module):
    def __init__(self, num_convs, num_classes, input_channels, feature_channels):
        super(MaskRCNNConvUpsampleHead, self).__init__()
        self.blocks = []

        for k in range(num_convs):
            layer = Conv2d(input_channels if k == 0 else feature_channels,
                           feature_channels, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            # nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.constant_(layer.bias, 0)
            self.add_module('mask_fcn{}'.format(k), layer)
            self.blocks.append(layer)

        self.deconv = ConvTranspose2d(feature_channels if num_convs > 0 else input_channels,
                                      feature_channels, 2, 2, 0)
        self.predictor = Conv2d(feature_channels, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            # print("INIT", name)
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for layer in self.blocks:
            x = F.relu(layer(x))
        x = F.relu(self.deconv(x))
        return self.predictor(x)


_ROI_MASK_PREDICTOR = {"MaskRCNNC4Predictor": MaskRCNNC4Predictor}


def make_roi_mask_predictor(cfg):
    func = _ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg)
