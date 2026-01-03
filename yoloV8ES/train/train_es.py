import sys, os
from pathlib import Path

# ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]   # <-- yoloV8ES/
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from model.loss_wiou import WIoUv3Loss
def main():

    # 2️⃣ load your ES model
    model = YOLO("yolov8es.yaml")

    # ---- PRINT THE ENTIRE MODEL ----
    print("\n================ MODEL SUMMARY (TABLE) ================\n")
    model.model.info(verbose=True)

    print("\n================ FULL ARCHITECTURE (PyTorch) ================\n")
    print(model.model)        # <-- expands every submodule and layer

    # 3️⃣ build graph so loss object exists
    model.model.build()

    # 4️⃣ enforce WIoU-v3 loss
    print("\n🔁 Replacing default IoU loss with WIoU-v3...\n")
    model.model.loss.iou_loss = WIoUv3Loss()

    # 5️⃣ train (paper-style config)
    model.train(
        data="rdd2022.yaml",
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
