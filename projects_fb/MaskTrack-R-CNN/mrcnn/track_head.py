# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from .utils import accuracy, weighted_cross_entropy

__all__ = ["build_track_head", "ROI_TRACK_HEAD_REGISTRY"]

ROI_TRACK_HEAD_REGISTRY = Registry("ROI_TRACK_HEAD")
ROI_TRACK_HEAD_REGISTRY.__doc__ = """
Registry for tracking head.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_TRACK_HEAD_REGISTRY.register()
class TrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
    Use dynamic option to deal with different number of objects in different
    images. A non-match entry is added to the reference objects with all-zero
    features. Object matched with the non-match entry is considered as a new
    object.
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_fcs: int,
        in_channels: int,
        fc_out_channels: int,
        roi_feat_size: int,
        match_coeff: List[int],
        with_avg_pool=False,
        bbox_dummy_iou=0,
        dynamic=True,
    ):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_fcs (int): number of fc layers in track head
            in_channels (int): number of input channels for first fc layer
            fc_out_channels (int): number of output channels for fc layers
            roi_feat_size (int): used to scale number of input channels
                if not using average pool
            match_coeff (list[int]): scalar coefficients used when
                computing comprehensive matching scores
            with_avg_pool (bool): option to use average pool
            bbox_dummy_iou (int): scales iou scores of dummy bboxes
            dynamic (bool): option to handle different number
                of objects in different images
        """
        super().__init__()
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = roi_feat_size
        self.match_coeff = match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= self.roi_feat_size * self.roi_feat_size
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = in_channels if i == 0 else fc_out_channels
            fc = nn.Linear(in_channels, fc_out_channels)
            self.fcs.append(fc)

        self.relu = nn.ReLU(inplace=True)
        self.dynamic = dynamic

        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_fcs = cfg.MODEL.ROI_TRACK_HEAD.NUM_FCS
        in_channels = cfg.MODEL.ROI_TRACK_HEAD.IN_CHANNELS
        fc_out_channels = cfg.MODEL.ROI_TRACK_HEAD.FC_OUT_CHANNELS
        roi_feat_size = cfg.MODEL.ROI_TRACK_HEAD.ROI_FEAT_SIZE
        match_coeff = cfg.MODEL.ROI_TRACK_HEAD.MATCH_COEFF

        return {
            "input_shape": input_shape,
            "num_fcs": num_fcs,
            "in_channels": in_channels,
            "fc_out_channels": fc_out_channels,
            "roi_feat_size": roi_feat_size,
            "match_coeff": match_coeff,
        }

    def compute_comp_scores(
        self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False
    ):
        # compute comprehensive matching score based on matching likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy = (
                torch.ones(bbox_ious.size(0), 1, device=match_ll.device) * self.bbox_dummy_iou
            )
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1, device=match_ll.device)
            label_delta = torch.cat((label_dummy, label_delta), dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert len(self.match_coeff) == 3
            return (
                match_ll
                + self.match_coeff[0] * torch.log(bbox_scores)
                + self.match_coeff[1] * bbox_ious
                + self.match_coeff[2] * label_delta
            )

    def forward(self, x, ref_x, x_n, ref_x_n):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch,
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add an all 0 column which denotes no matching
        assert len(x_n) == len(ref_x_n)

        if self.with_avg_pool:
            x = self.avg_pool(x)
            ref_x = self.avg_pool(ref_x)
        x = x.view(x.size(0), -1)
        ref_x = ref_x.view(ref_x.size(0), -1)
        for idx, fc in enumerate(self.fcs):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        for i in range(n):
            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            prods.append(prod)
        if self.dynamic:
            match_score = []
            for prod in prods:
                m = prod.size(0)
                dummy = torch.zeros(m, 1, device=x.device)
                prod_ext = torch.cat([dummy, prod], dim=1)
                match_score.append(prod_ext)
        else:
            dummy = torch.zeros(n, m, device=x.device)
            prods_all = torch.cat(prods, dim=0)
            match_score = torch.cat([dummy, prods_all], dim=2)
        return match_score

    def loss(self, match_score, ids, id_weights, reduce=True):
        losses = dict()
        if self.dynamic:
            n = len(match_score)
            x_n = [s.size(0) for s in match_score]
            ids = torch.split(ids, x_n, dim=0)
            loss_match = torch.zeros(1, device=match_score[0].device)
            match_acc = 0.0
            n_total = 0
            for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
                valid_idx = torch.nonzero(cur_weights).squeeze()
                if len(valid_idx.size()) == 0:
                    continue
                n_valid = valid_idx.size(0)
                n_total += n_valid
                loss_match += weighted_cross_entropy(score, cur_ids, cur_weights, reduce=reduce)
                match_acc += (
                    accuracy(
                        torch.index_select(score, 0, valid_idx),
                        torch.index_select(cur_ids, 0, valid_idx),
                    )
                    * n_valid
                )
            losses["loss_match"] = loss_match / n
            storage = get_event_storage()
            storage.put_scalar("match_acc", match_acc / n_total if n_total > 0 else 1.0)
        else:
            if match_score is not None:
                valid_idx = torch.nonzero(cur_weights).squeeze()
                losses["loss_match"] = weighted_cross_entropy(
                    match_score, ids, id_weights, reduce=reduce
                )
                losses["match_acc"] = accuracy(
                    torch.index_select(match_score, 0, valid_idx),
                    torch.index_select(ids, 0, valid_idx),
                )
        return losses


def build_track_head(cfg, input_shape):
    """
    Build a tracking head defined by `cfg.MODEL.ROI_TRACK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_TRACK_HEAD.NAME
    return ROI_TRACK_HEAD_REGISTRY.get(name)(cfg, input_shape)
