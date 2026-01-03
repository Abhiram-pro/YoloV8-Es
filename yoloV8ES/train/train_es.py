import sys, os
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"   # 👈 prevents W&B prompts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

# 1️⃣ import YOUR custom modules
from model.edcm import EDCM
from model.sgam import SGAM

# 2️⃣ register them into Ultralytics' model builder namespace
import ultralytics.nn.tasks as tasks
tasks.EDCM = EDCM
tasks.SGAM = SGAM

from model.loss_wiou import WIoUv3Loss
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect import DetectionTrainer


# 3️⃣ Custom trainer to inject WIoU-v3 loss
class WiouTrainer(DetectionTrainer):
    def build_loss(self):
        loss = super().build_loss()
        print("\n🔁 Injecting WIoU-v3 into loss...\n")
        loss.iou_loss = WIoUv3Loss()
        return loss


def main():

    model_file = ROOT / "yolov8es.yaml"
    data_file  = ROOT / "rdd2022.yaml"

    print(f"\n📄 Using model file: {model_file}")
    print(f"📄 Using data file:  {data_file}\n")

    model = YOLO(str(model_file))

    print("\n================ MODEL SUMMARY (TABLE) ================\n")
    model.model.info(verbose=True)

    print("\n================ FULL ARCHITECTURE (PyTorch) ================\n")
    print(model.model)

    # ---------------- TRAIN ----------------
    model.train(
        data=str(data_file),
        imgsz=512,
        epochs=100,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        auto_augment=False,
        freeze=22,
        warmup_epochs=5,
        project="runs",
        name="yolov8es_wiouv3_layers",
        exist_ok=True,
        verbose=True,
        trainer=WiouTrainer,   # 👈 important
    )


if __name__ == "__main__":
    main()
