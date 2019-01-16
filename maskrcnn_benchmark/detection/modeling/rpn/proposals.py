import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, cat_boxlist, remove_small_boxes


def generate_rpn_proposals(
    boxes, objectness, image_sizes, nms_thresh, pre_nms_topk, post_nms_topk, min_size=0
):
    """
    Generate RPN proposals from RPN boxes at all locations.
    It does so by NMS and keeping the topk.

    Args:
        boxes: tensor of size (B, N, 4), decoded boxes, where N is the total number of anchors on the feature map.
        objectness: tensor of size (B, N), before sigmoid
        image_sizes: B (w, h) tuples.
        pre_nms_topk, post_nms_topk: int
        min_size: int

    Returns:
        list[BoxList], with B BoxList. Each BoxList is the proposals on the corresponding image.
    """
    B, N = objectness.shape
    device = objectness.device
    assert boxes.shape[:2] == (B, N)

    pre_nms_topk = min(pre_nms_topk, N)
    objectness, topk_idx = objectness.topk(pre_nms_topk, dim=1)

    batch_idx = torch.arange(B, device=device)[:, None]
    boxes = boxes[batch_idx, topk_idx]  # B, topk, 4

    result = []
    for proposals, scores, im_shape in zip(boxes, objectness, image_sizes):
        """
        proposals: tensor of shape (topk, 4)
        scores: top-k scores of shape (topk, )
        """

        boxlist = BoxList(proposals, im_shape, mode="xyxy")
        # TODO why using "field" at all? Avoid storing arbitrary states
        boxlist.add_field("objectness", scores)
        boxlist = boxlist.clip_to_image(remove_empty=False)
        boxlist = remove_small_boxes(boxlist, min_size)
        boxlist = boxlist_nms(
            boxlist, nms_thresh, max_proposals=post_nms_topk, score_field="objectness"
        )
        result.append(boxlist)
    return result


def generate_fpn_proposals(
    multilevel_boxes,
    multilevel_objectness,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_size=0,
    training=False,
):
    """
    Generate RPN proposals from multilevel features in FPN.
    Currently it does so by merging the outputs of `generate_rpn_proposals`.

    Args:
        multilevel_boxes(list[Tenosr]): #lvl decoded boxes. Each tensor has size (B, N, 4).
        multilevel_objectness(list[Tenosr]): #lvl objectness scores. Each tensor has size (B, N).
        image_sizes: B (w, h) tuples.

    Returns:
        list[BoxList], with B BoxList. Each BoxList is the proposals on the corresponding image.
    """
    # TODO an alternative is to merge them first, then do topk and NMS
    multilevel_proposals = [
        generate_rpn_proposals(
            boxes, objectness, image_sizes, nms_thresh, pre_nms_topk, post_nms_topk, min_size
        )
        for boxes, objectness in zip(multilevel_boxes, multilevel_objectness)
    ]  # #lvl x #img BoxList.
    multilevel_proposals = list(zip(*multilevel_proposals))  # #img x #lvl BoxList
    # concat proposals across feature levels
    proposals_per_img = [cat_boxlist(boxlist) for boxlist in multilevel_proposals]  # #img BoxList
    num_img = len(proposals_per_img)

    # different behavior during training and during testing:
    # during training, post_nms_top_n is over *all* the proposals combined, while
    # during testing, it is over the proposals for each image
    # TODO resolve this difference and make it consistent. It should be per image,
    # and not per batch
    if training:
        objectness = torch.cat(
            [boxlist.get_field("objectness") for boxlist in proposals_per_img], dim=0
        )
        box_sizes = [len(boxlist) for boxlist in proposals_per_img]
        post_nms_top_n = min(post_nms_topk, len(objectness))
        _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
        inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
        inds_mask[inds_sorted] = 1
        inds_mask = inds_mask.split(box_sizes)
        for i in range(num_img):
            proposals_per_img[i] = proposals_per_img[i][inds_mask[i]]
    else:
        for i in range(num_img):
            objectness = proposals_per_img[i].get_field("objectness")
            post_nms_top_n = min(post_nms_topk, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            proposals_per_img[i] = proposals_per_img[i][inds_sorted]
    return proposals_per_img
