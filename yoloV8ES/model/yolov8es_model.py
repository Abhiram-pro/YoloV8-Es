"""
YOLOv8-ES Model Integration
Integrates EDCM, SGAM, and WIoU-v3 into Ultralytics YOLOv8
"""
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import yaml_load
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

# Import our custom modules
from .edcm import EDCM
from .sgam import SGAM
from .loss_wiou import WIoUv3Loss


class YOLOv8ES(YOLO):
    """
    YOLOv8-ES: Enhanced YOLOv8 for road crack detection
    
    Modifications from YOLOv8:
    1. EDCM in backbone (replaces some Conv layers)
    2. SGAM in neck (feature fusion enhancement)
    3. WIoU-v3 loss (replaces standard IoU loss)
    """
    
    def __init__(self, model='yolov8es.yaml', task='detect'):
        """
        Initialize YOLOv8-ES model
        
        Args:
            model: Path to model YAML or pretrained weights
            task: Task type (default: 'detect')
        """
        # If model is a YAML file, use it; otherwise use default
        if isinstance(model, str) and model.endswith('.yaml'):
            model_path = model
        else:
            # Use our custom YAML
            model_path = Path(__file__).parent / 'yolov8es.yaml'
        
        super().__init__(model_path, task=task)
        
        # Replace loss function with WIoU-v3
        self._setup_wiou_loss()
    
    def _setup_wiou_loss(self):
        """Setup WIoU-v3 loss for the model"""
        # This will be called during training setup
        # The actual loss replacement happens in the trainer
        pass


class YOLOv8ESDetectionModel(DetectionModel):
    """
    Custom Detection Model for YOLOv8-ES
    Extends Ultralytics DetectionModel to use custom modules
    """
    
    def __init__(self, cfg='yolov8es.yaml', ch=3, nc=None, verbose=True):
        """
        Initialize YOLOv8-ES Detection Model
        
        Args:
            cfg: Model configuration (YAML file or dict)
            ch: Number of input channels
            nc: Number of classes
            verbose: Print model info
        """
        # Register custom modules before initialization
        self._register_custom_modules()
        
        super().__init__(cfg, ch, nc, verbose)
    
    def _register_custom_modules(self):
        """Register EDCM and SGAM as custom modules"""
        # These will be available in the YAML parser
        import ultralytics.nn.modules as modules
        
        # Add our custom modules to Ultralytics registry
        if not hasattr(modules, 'EDCM'):
            modules.EDCM = EDCM
        if not hasattr(modules, 'SGAM'):
            modules.SGAM = SGAM


class WIoUv3DetectionLoss:
    """
    Detection Loss with WIoU-v3
    Replaces standard IoU loss in YOLOv8 with WIoU-v3
    """
    
    def __init__(self, model, use_dfl=True):
        """
        Initialize WIoU-v3 detection loss
        
        Args:
            model: Detection model
            use_dfl: Use Distribution Focal Loss for bbox regression
        """
        self.device = next(model.parameters()).device
        self.use_dfl = use_dfl
        
        # Get model parameters
        h = model.args
        m = model.model[-1]  # Detect() module
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.wiou = WIoUv3Loss(monotonous=False)  # WIoU-v3
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        
        # Task aligned assigner
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0
        )
        
        # Loss weights
        self.hyp_box = h.get('box', 7.5)
        self.hyp_cls = h.get('cls', 0.5)
        self.hyp_dfl = h.get('dfl', 1.5)
        
        # Projection for DFL
        if self.use_dfl:
            self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)
    
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets for loss computation"""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = out[..., 1:5].mul_(scale_tensor)
        return out
    
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted bbox distribution to actual bbox"""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
            pred_dist = pred_dist.matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def __call__(self, preds, batch):
        """
        Compute loss
        
        Args:
            preds: Model predictions
            batch: Batch data with targets
            
        Returns:
            loss: Total loss
            loss_items: Individual loss components
        """
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        # Predictions: [box, cls]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        
        # Anchors and strides
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # Targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        # Decode predictions
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        # Assign targets
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # Classification loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        
        # Bbox loss (WIoU-v3)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            
            # Convert to corner format for WIoU
            pred_bboxes_pos = pred_bboxes[fg_mask]
            target_bboxes_pos = target_bboxes[fg_mask]
            
            # WIoU-v3 loss
            bbox_loss = self.wiou(pred_bboxes_pos, target_bboxes_pos)
            loss[0] = (bbox_loss * target_scores[fg_mask].sum(-1)).sum() / target_scores_sum
            
            # DFL loss (optional)
            if self.use_dfl:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                loss[2] = self._df_loss(pred_distri[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) / target_scores_sum
        
        # Apply loss weights
        loss[0] *= self.hyp_box
        loss[1] *= self.hyp_cls
        loss[2] *= self.hyp_dfl
        
        return loss.sum() * batch_size, loss.detach()
    
    def _df_loss(self, pred_dist, target):
        """Distribution Focal Loss"""
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            torch.nn.functional.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl
            + torch.nn.functional.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr
        ).mean(-1, keepdim=True)


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox to distance format"""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def create_yolov8es_model(cfg='yolov8es.yaml', pretrained=None, nc=4):
    """
    Create YOLOv8-ES model
    
    Args:
        cfg: Model configuration YAML
        pretrained: Path to pretrained weights (optional)
        nc: Number of classes
        
    Returns:
        model: YOLOv8-ES model ready for training
    """
    # Register custom modules
    import ultralytics.nn.modules as modules
    modules.EDCM = EDCM
    modules.SGAM = SGAM
    
    # Load configuration
    if isinstance(cfg, str):
        cfg_path = Path(__file__).parent / cfg
        cfg_dict = yaml_load(cfg_path)
    else:
        cfg_dict = cfg
    
    # Override number of classes
    if nc is not None:
        cfg_dict['nc'] = nc
    
    # Create model
    model = DetectionModel(cfg_dict, ch=3, nc=nc)
    
    # Load pretrained weights if provided
    if pretrained:
        model.load(pretrained)
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Creating YOLOv8-ES model...")
    model = create_yolov8es_model(nc=4)
    print(f"Model created successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shapes: {[yi.shape for yi in y]}")
