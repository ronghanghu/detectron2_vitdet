from torch import nn


class FastRCNNPredictor(nn.Module):
    """
    TODO bad name. It's just a C4-specific predictor.
    """
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class BoxHeadPredictor(nn.Module):
    """
    2 FC layers that does bbox regression and classification, respectively.
    """
    def __init__(self, cfg, input_size=None):
        """
        TODO: takes num_classes instead of cfg
        Args:
            input_size (int): Defaults to be ROI_BOX_HEAD.MLP_HEAD_DIM, for compatibility
        """
        super(BoxHeadPredictor, self).__init__()

        if input_size is None:
            # TODO remove this in the future and compute it from the outside
            input_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.cls_score = nn.Linear(input_size, num_classes)
        self.bbox_pred = nn.Linear(input_size, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


_ROI_BOX_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "BoxHeadPredictor": BoxHeadPredictor,
    "FPNPredictor": BoxHeadPredictor,  # TODO remove it
}


def make_roi_box_predictor(cfg):
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
