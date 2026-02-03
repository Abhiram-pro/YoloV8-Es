import sys, os
from pathlib import Path
import torch

# Prevent W&B noise
os.environ["WANDB_MODE"] = "disabled"
os.environ["RAY_AIR_NEW_CALLBACK_API"] = "0"

from ultralytics import YOLO
from ultralytics.utils import SETTINGS
SETTINGS["ray"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 1. Import Custom Modules
from model.edcm import EDCM
from model.sgam import SGAM
from model.loss_wiou import wiou_v3_loss

# 2. Register modules into Ultralytics
import ultralytics.nn.tasks as tasks
# Safety check to prevent double registration issues
if not hasattr(tasks, 'EDCM'):
    tasks.EDCM = EDCM
if not hasattr(tasks, 'SGAM'):
    tasks.SGAM = SGAM

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

# --- CUSTOM LOSS CLASS ---
class WIoUDetectionLoss(v8DetectionLoss):
    """Custom Loss Class that uses WIoUv3 instead of CIoU"""
    def __init__(self, model):
        super().__init__(model)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # Box loss

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            
            # --- WIoU INJECTION ---
            loss[0] = wiou_v3_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], monotonous=False)
            loss[0] = (loss[0] * target_scores[fg_mask].sum(-1)).sum() / target_scores_sum

            if self.use_dfl:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                loss[2] = self._df_loss(pred_distri[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) / target_scores_sum

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

def bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)

class WiouTrainer(DetectionTrainer):
    def get_loss(self, dataloader=None):
        loss = WIoUDetectionLoss(self.model)
        loss.hyp = self.args
        return loss

def main():
    model_file = ROOT / "yolov8es.yaml"
    data_file  = ROOT / "rdd2022.yaml"
    weights_file = "yolov8n.pt"

    print(f"\n📄 Using model file: {model_file}")
    
    # 1. Build Model (Random Weights)
    model = YOLO(str(model_file))

    # 2. Load Weights (Transfer Learning)
    # This will throw a warning about size mismatch for EDCM/SGAM - THIS IS NORMAL.
    print("⬇️ Loading pre-trained COCO weights...")
    model.load(weights_file)

    # 3. Train
    model.train(
        data=str(data_file),
        imgsz=512,
        epochs=100,
        batch=16,
        device=0,
        optimizer="SGD",
        lr0=0.01,
        freeze=0,  # CRITICAL: Must be 0 so EDCM/SGAM can learn
        project="runs",
        name="yolov8es_reproduction",
        exist_ok=True,
        verbose=True,
        trainer=WiouTrainer, 
    )

if __name__ == "__main__":
    main()