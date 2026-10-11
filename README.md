# Weakly Supervised Pneumonia Localization from Chest X-Rays

This repository implements an educational computer-vision pipeline that classifies chest X-ray images as **Pneumonia** or **No Pneumonia**, then uses Class Activation Mapping (CAM) and DFS clustering to propose regions that contributed to a Pneumonia prediction. A supervised Faster R-CNN detector is included as a localization benchmark.

> **Academic use only.** This project is not clinically validated and must not be used for medical diagnosis or treatment decisions.

## Dataset

The supplied dataset is a 200-image COCO-format export located in `train/`:

```text
train/
├── <image files>.jpg
└── _annotations.coco.json
```

The JSON is the source of truth; a CSV annotation file is not required. It contains image metadata, category names, and COCO bounding boxes `[x, y, width, height]`.

Image-level labels are created using this rule:

- An image with at least one `pneumonia` annotation is labeled **Pneumonia (1)**.
- Otherwise, an image with `nopneumonia` annotations is labeled **No Pneumonia (0)**.
- For mixed-category images, the image-level label is Pneumonia, and only `pneumonia` boxes are used as localization ground truth.


## What this project runs

```text
COCO annotations + X-ray images
             ↓
    data validation and split
             ↓
   ┌─────────┴───────────┐
   ↓                     ↓
LR / SVM / RF          CNN classifier
                         ↓
                        CAM
                         ↓
                 threshold + DFS
                         ↓
                 predicted boxes
                         ↓
                        IoU

Parallel localization benchmark: Faster R-CNN trained with pneumonia boxes
```

The data is split at the **image level** into 70% train, 20% validation, and 10% test (expected counts: 140/40/20). A fixed seed makes the split reproducible. The validation set is used for model/threshold selection; the test set is reserved for the final evaluation.

The source reference project used U-Net lung segmentation, but this supplied dataset does not contain lung masks or a verified lung-segmentation checkpoint. U-Net code is retained as an optional extension; the default pipeline does not require it.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch cannot be installed for your particular OS/GPU with the generic requirement, use the official PyTorch installation selector to choose a compatible `torch`/`torchvision` pair, then install the remaining requirements.

## Run from a clean state

This package intentionally excludes **previous model checkpoints, generated splits, old metrics, and generated figures**. The commands below will create fresh outputs so you can inspect them yourself.

First validate the raw dataset and create the split files:

```bash
python scripts/run_project.py --prepare-data
```

Run the automated tests and syntax check:

```bash
python -m pytest -q
python -m compileall -q src scripts demo tests
```

Run the complete training and evaluation pipeline:

```bash
python scripts/run_project.py --all
```

Or run it one stage at a time:

```bash
python scripts/run_project.py --prepare-data
python scripts/run_project.py --train-baselines
python scripts/run_project.py --train-cnn
python scripts/run_project.py --select-cam-threshold
python scripts/run_project.py --train-rcnn
python scripts/run_project.py --evaluate
python scripts/run_project.py --generate-results
```

Training the CNN and Faster R-CNN can take a while, especially on CPU. `--all` trains the project models from scratch, except that the Faster R-CNN backbone starts from ImageNet-pretrained weights, which are downloaded automatically on the first run (internet required).

## Where fresh outputs will be written

The first run creates these folders automatically:

```text
data/splits/       # parsed metadata and reproducible train/val/test CSVs
models/             # freshly trained baselines, CNN and Faster R-CNN checkpoints
results/metrics/    # metrics, histories, selected thresholds and per-image records
results/tables/     # model comparison and per-image CSVs
results/figures/    # training curves, confusion matrix, IoU charts and examples
```

These are intentionally absent from this clean package. `.gitignore` excludes generated models, split files, results, caches and the local dataset from normal Git commits.

## Master notebook

Open `notebooks/pneumonia_localization_master.ipynb` in Jupyter or VS Code and run its cells from top to bottom. It calls the same command-line stages described above. The notebook is shipped without saved execution outputs so its results come from your own run.

## Run inference after training

After `--all` has successfully trained the CNN, run:

```bash
python demo/inference.py "train/<an-image-filename>.jpg" --output results/figures/demo_prediction.png
```

Replace the example path with an actual JPG filename from `train/`. The demo reports the predicted class and the model-derived Pneumonia probability; for positive predictions, it also produces CAM-based boxes. The confidence is the model's output, not a ground-truth annotation.

## Notes about evaluation

- Ground-truth class labels and pneumonia boxes come from `_annotations.coco.json`.
- The dataset's labels do not provide ground-truth confidence scores; probabilities are produced by the trained classifier.
- Classification metrics and localization IoU are written only after running the pipeline.
- The dataset is small. A 20-image test set is suitable for demonstrating an educational workflow, not for strong statistical or clinical conclusions.
- CAM is a weakly supervised localization heuristic; a heatmap box is not guaranteed to correspond to a true disease boundary.

## Main files

```text
src/                    Data, preprocessing, model, CAM, box, metric and visualization code
scripts/run_project.py  Reproducible pipeline CLI
scripts/generate_final_artifacts.py  Generates fresh result tables and figures
notebooks/              Master execution notebook
demo/inference.py       Single-image inference demo
tests/                  Automated correctness tests
train/                  Local input dataset
```
