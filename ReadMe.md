# Deep Learning–Driven Semantic Segmentation of Off-Road Environments

## Overview

This project presents a semantic segmentation model designed for off-road desert environments using synthetic data generated from Duality AI's Falcon digital twin platform. The objective is to train a robust segmentation model on synthetic training data and evaluate its generalization performance on unseen desert scenes.

The model performs pixel-level classification for terrain and object categories relevant to unmanned ground vehicle (UGV) navigation.

## Problem Statement

Off-road autonomous navigation requires fine-grained scene understanding in unstructured environments. Traditional data collection and annotation methods are expensive and limited in coverage. Synthetic digital twins provide a scalable alternative for generating diverse, labeled datasets.

This project focuses on:

- Training a semantic segmentation model using synthetic desert data
- Evaluating performance on unseen test environments
- Analyzing robustness and failure cases

## Dataset

The Offroad Segmentation Training Dataset consists of synthetic RGB images with pixel-level semantic annotations. The dataset includes 10 terrain and object classes relevant to desert navigation:

### Class Encoding

The segmentation masks use the following value mapping:

- **0**: Background
- **100**: Grass/Vegetation
- **200**: Dirt/Soil
- **300**: Sand
- **500**: Rock/Stone
- **550**: Water
- **700**: Asphalt/Road
- **800**: Path/Trail
- **7100**: Building/Structure
- **10000**: Sky/Air

## Model Architecture

The model uses a hybrid Transformer–Convolution architecture:

- **Encoder:** Pretrained DINOv2 Vision Transformer (ViT-B/14)
- **Decoder:** Lightweight ConvNeXt-style segmentation head
- **Final Layer:** 1×1 convolution for pixel-level classification

The backbone was frozen during training, and only the segmentation head was optimized.

An ensemble of three models trained with different random seeds (42, 123, 999) was used during evaluation to reduce prediction variance and improve stability.

### Architecture Details

- **Input Resolution:** 896×504 pixels
- **Patch Size:** 14×14 (resulting in 64×36 patch grid)
- **Feature Dimension:** 768-dimensional embeddings from DINOv2
- **Segmentation Head:**
  - Stem: 7×7 convolution with batch normalization and GELU activation
  - Block: Depthwise separable convolutions with residual connections
  - Classifier: 1×1 convolution outputting 10 class logits
- **Upsampling:** Bilinear interpolation to input resolution

## Training Details

- **Framework:** PyTorch
- **Optimizer:** AdamW
- **Learning Rate:** 5e-4
- **Scheduler:** OneCycle Learning Rate Policy
- **Loss Function:** Cross-Entropy + Dice Loss (0.5 weighting each)
- **Batch Size:** 12
- **Epochs:** 10
- **Precision:** Automatic Mixed Precision (AMP)
- **Weight Decay:** 1e-4

### Handling Class Imbalance

Class imbalance was addressed using a combined loss function that incorporates both Cross-Entropy and Dice loss. Geometric and photometric augmentations were also applied to improve minority class representation.

## Data Augmentation

The following augmentations were applied during training:

- Random horizontal flipping (50% probability)
- Random 90-degree rotations (50% probability)
- Shift, scale, and rotation transformations (max 15-degree rotation)
- Color jitter (brightness, contrast, saturation)
- Brightness and contrast adjustments
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Validation data was only resized and normalized.

## Test-Time Augmentation

Eight augmentation variants were applied during inference for improved robustness:

1. Original image
2. 90° rotation
3. 180° rotation
4. 270° rotation
5. Transpose
6. Transpose + 90° rotation
7. Transpose + 180° rotation
8. Transpose + 270° rotation

Final predictions are obtained by averaging the ensemble outputs across all augmentation variants.

## Evaluation Metric

Performance was measured using mean Intersection over Union (mIoU):

```
IoU = (Prediction ∩ GroundTruth) / (Prediction ∪ GroundTruth)
```

Mean IoU is computed across all 10 classes, excluding classes with zero union in the ground truth.

## Results

- **Validation mIoU:** 0.519
- **Test mIoU:** 0.324

The running mean IoU stabilizes around 0.324 across test images, indicating consistent segmentation performance on unseen scenes. The gap between validation and test performance suggests domain shift between training and test environments.

## Failure Cases

Observed failure modes include:

- Confusion between visually similar vegetation classes
- Reduced accuracy for partially occluded objects
- Sensitivity to extreme shadow regions
- Misclassification at object boundaries

These issues primarily arise due to high intra-class similarity and limited representation of certain visual conditions in training data.

## How to Run

### 1. Environment Setup

Create a Python environment using Conda:

```bash
conda create -n offroad_seg
conda activate offroad_seg
```

Install required dependencies:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics segmentation-models-pytorch albumentations xformers==0.0.27.post2
pip install opencv-python pillow tqdm matplotlib
```

### 2. Train the Model

Open `main.ipynb` and execute the training cells:

- Configure dataset paths in the notebook
- Adjust hyperparameters if needed
- Run training for 10 epochs
- Three model checkpoints will be saved (one per random seed)

Training logs and checkpoints are saved automatically. Models are named:
- `offroad_model_seed_42.pth`
- `offroad_model_seed_123.pth`
- `offroad_model_seed_999.pth`

### 3. Run Inference

Execute the testing cells in `main.ipynb`:

- Load trained ensemble weights
- Run inference with 8-way test-time augmentation
- Compute IoU metrics on test set
- Generate colored segmentation masks
- Save results to `test_results/`

Output includes:
- Color-coded segmentation masks
- IoU distribution histogram
- Running mean IoU plot
- Complete results archive (`test_results.zip`)

## Reproducibility

To reproduce the reported results:

- Use the provided hyperparameters
- Keep the DINOv2 backbone frozen
- Train for exactly 10 epochs
- Use ensemble averaging across three random seeds (42, 123, 999)
- Apply 8-way test-time augmentation during evaluation

## Repository Structure

```
OffRoad_Semantic_Scene_Segmentation/
├── models/
│   ├── offroad_model_seed_42.pth
│   ├── offroad_model_seed_123.pth
│   └── offroad_model_seed_999.pth
├── main.ipynb
├── test_results.zip
└── ReadMe.md
```

## Future Improvements

- **Class-balanced focal loss** to better handle rare classes
- **Fine-tuning the backbone** for improved feature adaptation
- **Multi-scale feature fusion** using FPN or similar architectures
- **Integration of depth information** for enhanced 3D scene understanding
- **Domain adaptation techniques** to reduce train-test performance gap
- **Post-processing with CRF** for boundary refinement

## Key Implementation Details

- Mixed precision training reduces memory usage and accelerates computation
- DataLoaders with 4 workers enable parallel data loading
- Frame-by-frame processing during inference for memory efficiency
- Per-sample IoU computation enables distribution analysis
- Conditional IoU calculation excludes absent classes from evaluation

## References

- **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
- **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch
- **Albumentations**: Fast and flexible image augmentation library
