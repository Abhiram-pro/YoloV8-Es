import torch
import torch.nn as nn

class WIoUv3Loss(nn.Module):
    """
    Wise-IoU v3 Loss with Dynamic Focusing Mechanism
    Includes Momentum-based Moving Average for outlier definition.
    """
    def __init__(self, momentum=0.9, alpha=1.9, delta=3):
        super().__init__()
        self.momentum = momentum
        self.alpha = alpha
        self.delta = delta
        # Register buffer for moving average IoU loss to persist across batches
        self.register_buffer('iou_mean', torch.tensor(1.0))

    def forward(self, pred, target, ret_iou=False):
        """
        pred: [N, 4] (x1, y1, x2, y2)
        target: [N, 4] (x1, y1, x2, y2)
        """
        iou = self._bbox_iou(pred, target)
        dist_penalty = self._distance_penalty(pred, target)
        
        # 1. Base IoU Loss
        l_iou = 1.0 - iou
        
        # 2. Distance Penalty (R_WIoU denominator is detached)
        l_wiou_v1 = l_iou + dist_penalty
        
        # 3. Dynamic Focusing (WIoU v3)
        # Calculate outlier degree beta = L_IoU / L_IoU_Mean
        # Detach gradients for beta calculation to avoid instability
        loss_curr = l_iou.detach().mean()
        
        if self.training:
            # Update moving average
            self.iou_mean = self.iou_mean * self.momentum + \
                            loss_curr * (1 - self.momentum)
        
        # Beta (Outlier degree)
        # Adding epsilon to avoid div by zero
        beta = l_iou.detach() / (self.iou_mean.clamp(min=1e-6))
        
        # Focusing coefficient r
        # r = beta / (delta * alpha^(beta - delta))
        r = beta / (self.delta * torch.pow(self.alpha, beta - self.delta))
        
        # Final Loss
        loss = r * l_wiou_v1
        
        if ret_iou:
            return loss, iou
        return loss

    def _bbox_iou(self, box1, box2, eps=1e-7):
        # ... (Standard IoU calculation: Intersection/Union) ...
        # Assume box1/box2 are x1y1x2y2
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = b1_area + b2_area - inter_area + eps
        
        return inter_area / union

    def _distance_penalty(self, box1, box2, eps=1e-7):
        # Center Distance / Diagonal^2
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        
        # The paper specifies decoupling Wg, Hg from computational graph (*)
        c2 = (cw ** 2 + ch ** 2).detach() + eps 
        
        b1_cx, b1_cy = (b1_x1 + b1_x2)/2, (b1_y1 + b1_y2)/2
        b2_cx, b2_cy = (b2_x1 + b2_x2)/2, (b2_y1 + b2_y2)/2
        
        rho2 = (b1_cx - b2_cx)**2 + (b1_cy - b2_cy)**2
        
        return torch.exp(rho2 / c2)