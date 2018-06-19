import torch

from torch_detectron.core.image_list import to_image_list
from torch_detectron.core.utils import cat_bbox


class GeneralizedRCNN(torch.nn.Module):
    def __init__(self, backbone, region_proposal, heads, rpn_only=False):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.region_proposal = region_proposal
        self.heads = heads
        self.rpn_only = rpn_only

    def forward(self, images, targets=None):
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.region_proposal(images, features, targets)
        if self.rpn_only:
            result = proposals
            detector_losses = {}
        else:
            result, detector_losses = self.heads(features, proposals, targets)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
        
class RPNModule(torch.nn.Module):
    def __init__(self, anchor_generator, heads, box_selector_train, box_selector_test, loss_evaluator, rpn_only):
        super(RPNModule, self).__init__()
        self.anchor_generator = anchor_generator
        self.heads = heads
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.rpn_only = rpn_only

    def forward(self, images, features, targets=None):
        objectness, rpn_box_regression = self.heads(features)
        anchors = self.anchor_generator(images.image_sizes, features)

        if not self.training:
            boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
            if self.rpn_only:
                # concatenate all boxes from different levels if in inference and rpn_only
                boxes = list(zip(*boxes))
                boxes = [cat_bbox(box) for box in boxes]
                # sort scores in decreasing order
                inds = [box.get_field('objectness').sort(descending=True)[1] for box in boxes]
                boxes = [box[ind] for box, ind in zip(boxes, inds)]
            return boxes, {}

        boxes = anchors
        if not self.rpn_only:
            with torch.no_grad():
                boxes = self.box_selector_train(anchors, objectness, rpn_box_regression)
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
                anchors, objectness, rpn_box_regression, targets)
        return boxes, dict(loss_objectness=loss_objectness, loss_rpn_box_reg=loss_rpn_box_reg)


class DetectionHead(torch.nn.Module):
    def __init__(self, pooler, heads, post_processor, loss_evaluator):
        super(DetectionHead, self).__init__()
        self.pooler = pooler
        self.heads = heads
        self.post_processor = post_processor
        self.loss_evaluator = loss_evaluator

    def forward(self, features, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        pooled_features = self.pooler(features, proposals)
        class_logits, box_regression = self.heads(pooled_features)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return result, ()

        loss_classifier, loss_box_reg = self.loss_evaluator(
                [class_logits], [box_regression])

        return None, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)


class DetectionAndMaskHead(torch.nn.Module):
    def __init__(self, pooler, heads, post_processor, loss_evaluator, classifier, heads_mask, loss_evaluator_mask, mask_post_processor):
        super(DetectionAndMaskHead, self).__init__()
        self.pooler = pooler
        self.heads = heads
        self.post_processor = post_processor
        self.loss_evaluator = loss_evaluator

        self.classifier = classifier
        self.mask_heads = heads_mask
        self.loss_evaluator_mask = loss_evaluator_mask
        self.mask_post_processor = mask_post_processor

    def forward(self, features, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        pooled_features = self.pooler(features, proposals)
        x = self.heads(pooled_features)
        class_logits, box_regression = self.classifier(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)

            # the network can't handle the case of no selected proposals,
            # so need to shortcut before
            if sum(r.bbox.shape[0] for r in result) == 0:
                for r in result:
                    r.add_field('mask', class_logits.new())
                return result, ()

            pooled_features = self.pooler(features, [result])
            x = self.heads(pooled_features)
            mask_logits = self.mask_heads(x)
            result = self.mask_post_processor(mask_logits, result)
            return result, ()

        mask_logits = self.mask_heads(x)

        loss_classifier, loss_box_reg = self.loss_evaluator(
                [class_logits], [box_regression])

        loss_mask = self.loss_evaluator_mask(
                proposals, mask_logits, targets)

        return None, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_mask=loss_mask)
