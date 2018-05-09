import torch
from torch import nn

from core.image_list import to_image_list


class FeatureProvider(nn.Module):
    """
    This class generates the bounding boxes that will be
    used to feed the RoI Layer. It can be learned,
    or use pre-computed boxes.
    It also computes the initial conv features of the network,
    if necessary
    """
    def __init__(self, *args, **kwargs):
        super(FeatureProvider, self).__init__()

    def predict(self, x):
        """
        Returns the features and bounding boxes.
        x can be either an image, or eventually
        pre-computed features.
        In the case we have pre-computed boxes already,
        it might be more efficient to pass them
        from the dataset in the dataLoader. In this case,
        this class should simply return the boxes
        """
        pass

    def forward(self, x, targets=None):
        """
        Should return the anchors, the objectness probabilities
        and the box regression
        """
        pass

    def loss(self, x, targets):
        """
        Should return x, the boxes and the losses
        """
        pass

"""
Idea: BoxProvider should return the input and the boxes.
This way, we can make it compute the features internally,
and make backbone a dummy class?

Another idea: Fuse the BoxProvider and the Backbone
maybe call it FeatureProvider?

I'm fusing the backbone into the box_provider
"""


class HeadProvider(nn.Module):
    def predict(self, x):
        pass
    def forward(self, x, targets=None):
        pass
    def loss(self, x, targets):
        pass


class GeneralizedRCNN(nn.Module):
    """
    This class holds general detection models based on the Generalized R-CNN
    extension.
    It contains two parts: a FeatureProvider (which can be seen as
    image-centric operations) and the Heads (which perform operations
    on top of the Features and proposed resions).

    In Faster R-CNN style network, FeatureProvider will compute both the
    features from the backbone architecture, as well as the RPN boxes.
    The heads will use the features and the proposals to compute
    the class probabilities for each box, as well as the regression
    targets.

    Note that several Heads can be combined to perform different tasks.
    """
    def __init__(self, feature_provider, heads):
        super(GeneralizedRCNN, self).__init__()
        
        assert isinstance(feature_provider, FeatureProvider)

        self.feature_provider = feature_provider
        self.heads = heads

    def predict(self, x, proposals=None):
        """
        Idea: instead of passing everywhere the proposals argument,
        why not also allow ImageList to hold the proposals?
        This would make it very specific to object detection though
        """
        x = to_image_list(x)
        x, boxes = self.feature_provider.predict(x, proposals)
        x = self.heads.predict(x, boxes)
        return x  # can be a tuple, or dict
    
    def forward(self, x, targets=None, proposals=None):
        pass

    def loss(self, x, targets, proposals=None):
        x = to_image_list(x)
        x, boxes, box_provider_loss = self.feature_provider.loss(x, targets, proposals)
        x, heads_loss = self.heads.loss(x, boxes, targets)
        return x, heads_loss, box_provider_loss


# TODO this ImaeList being passed in some places but tensors in others is not good
class RPNProvider(FeatureProvider):
    """
    Takes care of taking the input images, computing the
    image features and applying the RPN
    """
    def __init__(self, backbone, rpn_net, box_sampler):
        super(RPNProvider, self).__init__()
        self.backbone = backbone
        self.rpn_net = rpn_net
        self.box_sampler = box_sampler

    def predict(self, x, proposals=None):
        """
        x should be an ImageList object
        """
        x = to_image_list(x)
        features = self.backbone(x.tensors)
        if proposals is None:
            proposals = self.rpn_net.predict(x, features)
        return features, proposals

    def loss(self, x, targets, proposals=None):
        # TODO the order targets, proposals is not good
        x = to_image_list(x)
        features = self.backbone(x.tensors)
        if proposals is None:
            boxes, loss = self.rpn_net.loss(x, features, targets)
        else:
            boxes = proposals
            loss = 0
        # subsample FG / BG
        boxes = self.box_sampler(boxes, targets)
        return features, boxes, loss


class RPN(nn.Module):
    """
    From an image and a set of image features, returns a set of
    candidate bounding boxes.
    """
    def __init__(self, layers, anchor_generator, box_selector):
        super(RPN, self).__init__()
        self.layers = layers
        self.anchor_generator = anchor_generator
        self.box_selector = box_selector

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
        return boxes, loss

    def compute_loss(self, anchors, objectness_score, box_regression, targets):
        """
        This can subsample the total predictions.
        """
        # TODO use loss_subsampler here, should be passed as a constructor arg?
        pass


class ClassificationHead(HeadProvider):
    """
    Given a set of features and bounding boxes, computes the predictions
    """
    def __init__(self, pooler, layers, postprocessor):
        super(ClassificationHead, self).__init__()
        self.pooler = pooler
        self.layers = layers
        self.postprocessor = postprocessor

    def forward(self, features, boxes):
        x = self.pooler(features, boxes)
        # x can be a tuple here
        x = self.layers(x)
        return x

    def predict(self, features, boxes):
        x = self(features, boxes)
        x = self.postprocessor(x, boxes)
        return x

    def loss(self, features, boxes, targets):
        predictions = self(features, boxes)
        loss = self.compute_loss(predictions, boxes, targets)
        return predictions, loss

    def compute_loss(self, predictions, boxes, targets):
        pass


if __name__ == '__main__':
    backbone = ResNet50Backbone(...)
    # after each line add getter from config
    backbone = config.get('backbone', backbone)

    anchor_generator = AnchorGenerator(...)
    box_selector = TopKBoxSelector(...)
    rpn_net = RPN(backbone, anchor_generator, box_selector)
    box_sampler = FGBGBoxSampler(...)
    feature_provider = RPNProvider(backbone, rpn_net, box_sampler)

    pooler = ROIAlign(...)
    classifiers = ResNet50Heads(...)
    heads = ClassificationHeads(pooler, classifiers)

    detection_model = GeneralizedRCNN(feature_provider=box_provider, heads=heads)
