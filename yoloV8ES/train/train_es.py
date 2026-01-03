import sys, os
from pathlib import Path

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



def main():

    # ---------------- LOAD MODEL ----------------
    model_file = ROOT / "yolov8es.yaml"         # <-- architecture file
    data_file  = ROOT / "rdd2022.yaml"          # <-- dataset config

    print(f"\n📄 Using model file: {model_file}")
    print(f"📄 Using data file:  {data_file}\n")

    model = YOLO(str(model_file))

    # ---------------- PRINT ARCHITECTURE ----------------
    print("\n================ MODEL SUMMARY (TABLE) ================\n")
    model.model.info(verbose=True)

    print("\n================ FULL ARCHITECTURE (PyTorch) ================\n")
    print(model.model)

    # build graph so loss exists
    model.model.build()

    # ---------------- REPLACE LOSS WITH WIoU-V3 ----------------
    print("\n🔁 Replacing default IoU loss with WIoU-v3...\n")
    model.model.loss.iou_loss = WIoUv3Loss()

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
    )


if __name__ == "__main__":
    main()
