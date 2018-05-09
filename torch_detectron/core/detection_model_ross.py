import torch
from torch import nn

from core.image_list import to_image_list


class GeneralizedRCNN(nn.Module):
    def __init__(self, backbone, region_proposal, roi_feature, head):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.region_proposal = region_proposal
        self.roi_feature = roi_feature
        self.head = head

    def predict(self, images, proposals=None):
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        if proposals is None:
            # TODO instead of bypassing here (which makes the
            # FG/BG sampling be elsewhere), we could unconditionally
            # pass the proposals, and return the proposals
            # this is only relevant for training though
            proposals = self.region_proposal.predict(images, features)
        x = self.roi_feature(features, proposals)
        x = self.head.predict(x, proposals)
        return x

    def forward(self, images, proposals=None, targets=None):
        pass

    def loss(self, images, proposals=None, targets=None):
        # TODO targets is necessary here, but would imply
        # changing the order of which the arguments
        # are passed
        pass


class RPN(nn.Module):
    """
    From an image and a set of image features, returns a set of
    candidate bounding boxes.

    The only difference of this class compared to the other one
    I proposed is that the box sampler now comes here.
    Also, we are passing the images directly, instead of only the
    image sizes
    """
    def __init__(self, layers, anchor_generator, box_selector, box_sampler):
        super(RPN, self).__init__()
        self.layers = layers
        self.anchor_generator = anchor_generator
        self.box_selector = box_selector
        self.box_sampler = box_sampler

    def predict(self, images, features):
        objectness, box_regression = self.layers(features)

        # anchors is a BBox with image size information
        anchors = self.anchor_generator(images.image_sizes, features)
        boxes = self.box_selector(anchors, objectness, box_regression)
        return boxes

    def loss(self, images, features, targets):
        objectness, box_regression = self.layers(features)
        
        anchors = self.anchor_generator(images.image_sizes, features)
        # TODO returning the scores from the box selector might be unecessary
        # it's here originally because of FPN, but the selector will take
        # care of the selection / reshuffle of the boxes to have things in the
        # correct order
        boxes = self.box_selector(anchors, objectness, box_regression)

        loss = self.compute_loss(anchors, objectness, box_regression, targets)
        boxes = self.box_sampler(boxes, targets)
        return boxes, loss

    def compute_loss(self, anchors, objectness_score, box_regression, targets):
        """
        This can subsample the total predictions.
        """
        # TODO use loss_subsampler here, should be passed as a constructor arg?
        pass


class Head(nn.Module):
    """
    Given a set of features and bounding boxes, computes the predictions
    """
    def __init__(self, layers, postprocessor):
        super(Head, self).__init__()
        self.layers = layers
        self.postprocessor = postprocessor

    def forward(self, features):
        # x can be a tuple here
        x = self.layers(features)
        return x

    def predict(self, features, boxes):
        x = self(features)
        x = self.postprocessor(x, boxes)
        return x

    def loss(self, features, boxes, targets):
        predictions = self(features)
        loss = self.compute_loss(predictions, boxes, targets)
        return predictions, loss

    def compute_loss(self, predictions, boxes, targets):
        pass


