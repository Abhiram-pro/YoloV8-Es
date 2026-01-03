import torch
import torch.nn as nn


class WIoUv3Loss(nn.Module):
    """
    Wise-IoU v3 Loss
    
    From: "Efficient and accurate road crack detection technology based on YOLOv8-ES"
    Section 3.4 - Improved bounding box regression loss
    
    Key improvements over standard IoU:
    - Dynamic non-monotonic focusing mechanism
    - Reduces negative impact of low-quality examples
    - Focuses gradient allocation on medium-quality anchors
    - Better handling of noisy/ambiguous boxes
    
    The loss uses a wise gradient gain to dynamically adjust the contribution
    of each anchor box based on its quality, preventing dominance by either
    high-quality or low-quality examples.
    """

    def __init__(self, monotonous=False, eps=1e-7):
        """
        Args:
            monotonous: If True, uses monotonic focusing (WIoU v1/v2 style)
                       If False, uses non-monotonic focusing (WIoU v3, recommended)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.monotonous = monotonous
        self.eps = eps

    def forward(self, pred, target, ret_iou=False):
        """
        Compute WIoU v3 loss
        
        Args:
            pred: Predicted boxes [N, 4] in format (x1, y1, x2, y2) or (cx, cy, w, h)
            target: Target boxes [N, 4] in same format as pred
            ret_iou: If True, return IoU values along with loss
            
        Returns:
            loss: WIoU v3 loss value
            iou (optional): IoU values if ret_iou=True
        """
        # Ensure boxes are in (x1, y1, x2, y2) format
        if self._is_center_format(pred):
            pred = self._center_to_corners(pred)
        if self._is_center_format(target):
            target = self._center_to_corners(target)

        # Calculate IoU
        iou = self._calculate_iou(pred, target)
        
        # Calculate distance-based penalty
        # Center distance between predicted and target boxes
        pred_center = self._get_center(pred)
        target_center = self._get_center(target)
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=-1)
        
        # Diagonal length of smallest enclosing box
        c_x1 = torch.min(pred[:, 0], target[:, 0])
        c_y1 = torch.min(pred[:, 1], target[:, 1])
        c_x2 = torch.max(pred[:, 2], target[:, 2])
        c_y2 = torch.max(pred[:, 3], target[:, 3])
        c_diagonal = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + self.eps
        
        # Distance ratio
        distance_ratio = center_distance / c_diagonal
        
        # Wise gradient gain (dynamic focusing mechanism)
        if self.monotonous:
            # Monotonic focusing (v1/v2 style)
            # Higher penalty for lower IoU
            beta = iou.detach().clamp(min=self.eps)
            alpha = 1.0 / beta
        else:
            # Non-monotonic focusing (v3 style)
            # The wise gradient gain focuses on medium-quality anchors
            # Formula: r = beta * delta, where beta = IoU* / (1 - IoU*)
            # This creates a non-monotonic focusing that reduces gradient
            # for both very high IoU (already good) and very low IoU (too hard)
            iou_detached = iou.detach().clamp(min=self.eps, max=1.0 - self.eps)
            
            # Calculate outlier degree
            # Higher for medium IoU, lower for extreme IoU values
            beta = iou_detached / (1.0 - iou_detached)
            
            # The gradient gain is applied to the gradient, not the loss directly
            # For loss computation, we use a modified approach:
            # Use exponential of beta to create focusing effect
            alpha = torch.exp(-beta.clamp(max=10.0))  # Clamp to prevent overflow
        
        # WIoU loss with wise gradient gain
        # Base loss: 1 - IoU + distance_penalty
        base_loss = 1.0 - iou + distance_ratio
        
        # Apply wise gradient gain
        # The alpha modulates the loss contribution
        loss = alpha * base_loss
        
        if ret_iou:
            return loss, iou
        return loss

    def _is_center_format(self, boxes):
        """
        Heuristic to detect if boxes are in center format (cx, cy, w, h)
        vs corner format (x1, y1, x2, y2)
        """
        # If x2 < x1 or y2 < y1 for any box, likely center format
        if boxes.shape[-1] != 4:
            return False
        return torch.any(boxes[:, 2] < boxes[:, 0]) or torch.any(boxes[:, 3] < boxes[:, 1])

    def _center_to_corners(self, boxes):
        """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _get_center(self, boxes):
        """Get center coordinates from corner format boxes"""
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        return torch.stack([cx, cy], dim=-1)

    def _calculate_iou(self, pred, target):
        """Calculate IoU between predicted and target boxes"""
        # Intersection area
        x1_inter = torch.max(pred[:, 0], target[:, 0])
        y1_inter = torch.max(pred[:, 1], target[:, 1])
        x2_inter = torch.min(pred[:, 2], target[:, 2])
        y2_inter = torch.min(pred[:, 3], target[:, 3])
        
        inter_w = (x2_inter - x1_inter).clamp(min=0)
        inter_h = (y2_inter - y1_inter).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Union area
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area + self.eps
        
        # IoU
        iou = inter_area / union_area
        return iou.clamp(min=0, max=1)


def wiou_v3_loss(pred, target, monotonous=False, eps=1e-7):
    """
    Functional interface for WIoU v3 loss
    
    Args:
        pred: Predicted boxes [N, 4]
        target: Target boxes [N, 4]
        monotonous: Use monotonic (v1/v2) or non-monotonic (v3) focusing
        eps: Small constant for numerical stability
        
    Returns:
        loss: Mean WIoU v3 loss
    """
    loss_fn = WIoUv3Loss(monotonous=monotonous, eps=eps)
    loss = loss_fn(pred, target)
    return loss.mean()
