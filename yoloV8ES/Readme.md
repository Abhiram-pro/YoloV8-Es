YOLOv8-ES: Efficient Road Crack Detection (Reproduction)This repository contains a PyTorch reproduction of the YOLOv8-ES architecture described in the paper "Efficient and accurate road crack detection technology based on YOLOv8-ES" (Zeng et al., 2025).The model enhances the standard YOLOv8n baseline with three key innovations:EDCM (Enhanced Dynamic Convolution Module): Embedded in the backbone for multi-dimensional attention.SGAM (Selective Global Attention Mechanism): Added to the neck for better feature fusion.WIoUv3 (Wise-IoU v3): A dynamic focusing loss function to handle low-quality crack samples.📂 Project StructureEnsure your project directory is organized exactly as follows for the scripts to work:Plaintextyolov8-es/
├── model/
│   ├── __init__.py       # (Empty file to make it a package)
│   ├── edcm.py           # ODConv + PSA implementation
│   ├── sgam.py           # SE + GAM + CA implementation
│   └── loss_wiou.py      # Wise-IoU v3 loss function
├── yolov8es.yaml         # Model architecture config
├── es_baseline.yaml      # Hyperparameters (optional, can be set in train script)
├── train_es.py           # Main training script with Loss Injection
├── rdd2022.yaml          # Dataset config
└── README.md
🛠️ PrerequisitesPython 3.8+CUDA-enabled GPU (Recommended: 8GB+ VRAM for batch size 16)Dependencies:Bashpip install ultralytics torch torchvision
📊 Dataset PreparationThe paper uses the RDD2022 (Road Damage Dataset).Download the dataset (specifically the China_MotorBike subset as per the paper).Format it for YOLO (images in /images, labels in /labels).Create a rdd2022.yaml file in your root directory:YAML# rdd2022.yaml
path: /path/to/your/RDD2022  # Update this absolute path!
train: images/train
val: images/val
test: images/test

# Classes (Longitudinal, Transverse, Alligator, Pothole)
names:
  0: D00
  1: D10
  2: D20
  3: D40
🚀 TrainingTo train the model on your local GPU:Download Base Weights:Ensure yolov8n.pt is in the root directory (the script will try to download it, but having it ready helps).Run the Training Script:Bashpython train_es.py
Critical Configuration NoteThe train_es.py script is pre-configured with the paper's exact hyperparameters:Epochs: 100Batch Size: 16Image Size: 512x512Optimizer: SGD (lr=0.01)Freeze: 0 (Crucial! Do not change this to 22. Custom layers must remain unfrozen).🔍 Architecture Details1. EDCM (Backbone)Replaces standard convolution at Layer 2. It uses Omni-Dimensional Dynamic Convolution (ODConv) to learn attention across four dimensions:Spatial kernel size ($k \times k$)Input channels ($c_{in}$)Output filters ($c_{out}$)Number of kernels ($n$)2. SGAM (Neck)A sequential attention module placed in the upsampling path:Input $\rightarrow$ SE (Channel) $\rightarrow$ GAM (Global/Permutation) $\rightarrow$ CA (Coordinate) $\rightarrow$ Output.3. WIoU v3 LossReplaces the standard CIoU loss. It calculates a dynamic non-monotonic focusing coefficient $r$ to reduce the gradient penalty from low-quality examples (blurry cracks) while preventing overfitting to high-quality ones.📈 Monitoring TrainingSince WANDB is disabled in the script to keep things simple, you can monitor progress via:Console Output: Real-time loss and mAP logging.Results Directory: Check runs/detect/yolov8es_reproduction/ for:results.csv (Metrics over time)train_batch*.jpg (Visual validation of training batches)weights/best.pt (Your final trained model)⚠️ Common Issues & FixesQ: I get a "size mismatch" warning when loading weights.A: This is normal.Transferred 319/355 items...The standard YOLOv8n weights cannot map to your custom EDCM and SGAM layers because those layers don't exist in the original model. They will initialize with random weights and learn during training (which is why freeze=0 is required).Q: CUDA Out of Memory?A: Reduce batch=16 to batch=8 in train_es.py.Q: The loss isn't decreasing?A: Check your dataset labels. Ensure your rdd2022.yaml paths are correct and that labels are in standard YOLO format (normalized class x_center y_center width height).