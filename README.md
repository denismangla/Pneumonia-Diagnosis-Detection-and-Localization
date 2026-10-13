# Pneumonia Detection & Localization from Chest X-Rays

**Machine Learning Mini-Project**  
**Team:** Denis Mangla (SRN: PES2UG24CS151) · Farheen Akhtar (SRN: PES2UG24CS164)

This project compares machine-learning methods for classifying chest X-ray images as **Pneumonia** or **No Pneumonia** and explores approximate localization of pneumonia regions. It includes classical classification baselines, a CNN with Class Activation Mapping (CAM), and a Faster R-CNN object detector.

> **Educational use only.** This project is not clinically validated and must not be used for medical diagnosis or treatment decisions.

## 1. Dataset

The dataset is a 200-image COCO-format export. The dataset itself is not included in this repository ZIP and must be obtained separately. Place the extracted files in `train/` so the structure looks like this:

```text
train/
├── <image files>.jpg
└── _annotations.coco.json
```

The COCO JSON must contain `images`, `annotations`, and `categories` arrays. The expected category names are `pneumonia` and `nopneumonia` (case-insensitive). Bounding boxes use COCO format `[x, y, width, height]`.

Image-level labels are derived from the supplied annotations:

- An image with at least one pneumonia annotation is labelled **Pneumonia (1)**.
- Otherwise, an image with a `nopneumonia` annotation is labelled **No Pneumonia (0)**.
- For images containing both categories, the image-level label is Pneumonia.
- Only pneumonia boxes are used as localization ground truth.

The dataset snapshot used for the saved results contains 200 images: 120 pneumonia-positive and 80 no-pneumonia images. The standard image-level split is 70% training, 20% validation, and 10% test (140/40/20), using seed 42 by default.

## 2. Methods

The pipeline includes:

- **Logistic Regression, Linear SVM, and Random Forest:** baseline classifiers trained on features extracted from resized images using PCA.
- **CNN classifier:** predicts the image-level class from image pixels.
- **CNN + CAM:** creates a class activation map and groups activated pixels using thresholding and depth-first search (DFS) to produce approximate bounding boxes.
- **Faster R-CNN:** a supervised localization benchmark trained with pneumonia bounding boxes. The detector uses an ImageNet-pretrained MobileNetV3-Large backbone; the detector itself is trained on this project dataset.

Data is split at the image level. Validation data is used for model selection and CAM threshold selection, while the test set is reserved for final evaluation. The project reports classification metrics and localization Intersection over Union (IoU). The current localization score uses greedy one-to-one box matching per image; an image with an empty ground-truth set or no predicted boxes receives IoU 0. It is a project-specific localization metric, not object-detection mAP.

## 3. Installation

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include PyTorch and torchvision. If installation fails or you need GPU support, install a compatible `torch` and `torchvision` pair for your operating system using the official PyTorch installation instructions, then install the remaining packages from `requirements.txt`. The first Faster R-CNN training run may download ImageNet-pretrained MobileNetV3 backbone weights, so internet access may be required.

## 4. Prepare the dataset

After extracting the dataset into `train/`, run:

```bash
python scripts/run_project.py --prepare-data
```

The command validates the COCO annotations, prepares image-level labels, checks image paths and bounding boxes, and creates reproducible train/validation/test split files under `data/splits/`.

If your dataset is stored elsewhere, specify its path explicitly. For example:

```bash
python scripts/run_project.py --prepare-data --data-dir "/path/to/train"
```

The specified directory should contain both the image files and `_annotations.coco.json`, unless you also pass `--annotation-file` and/or `--image-dir`.

## 5. Run the project

To train the models and generate the evaluation artifacts in one command:

```bash
python scripts/run_project.py --all
```

The full pipeline runs data preparation, baseline training, CNN training, CAM threshold selection, Faster R-CNN training, test evaluation, and result generation. Training can take a while on CPU.

You can also run the stages separately:

```bash
python scripts/run_project.py --prepare-data
python scripts/run_project.py --train-baselines
python scripts/run_project.py --train-cnn
python scripts/run_project.py --select-cam-threshold
python scripts/run_project.py --train-rcnn
python scripts/run_project.py --evaluate
python scripts/run_project.py --generate-results
```

`--evaluate` evaluates the baselines, CNN + CAM, and Faster R-CNN on the held-out test split. `--evaluate-cam` is also available when you want to evaluate the CAM model separately.

## 6. Run a single-image demo

Run the full pipeline first so the CNN checkpoint exists. Then choose an image from `train/` and run:

```bash
python demo/inference.py "train/<image-filename>.jpg" --output results/figures/demo_prediction.png
```

Replace `<image-filename>.jpg` with the exact filename of a real image. The demo prints the predicted class, the model's pneumonia probability, and any predicted CAM boxes. It also saves a visualization to the requested output path. PNG, JPEG, and DICOM inputs are supported by the inference loader.

This command demonstrates **CNN + CAM inference**; it does not run Faster R-CNN inference.

## 7. Tests and code checks

With the required dataset in place, run:

```bash
python -m pytest -q
python -m compileall -q src scripts demo tests
```

The COCO data tests need the dataset and annotation JSON. If those files are missing, the data-dependent tests or data-preparation step will fail.

## 8. Results and output files

The repository includes a saved set of metrics, tables, and figures from a previous experiment under `results/`. They are provided as a result snapshot; they are not proof that training has been freshly reproduced on the machine where the ZIP is extracted. Running the pipeline again updates the generated artifacts.

Important output locations:

```text
data/splits/                       # generated metadata and train/val/test CSVs
models/                             # generated model checkpoints and baseline artifacts
results/metrics/                    # metric summaries and training histories
results/tables/                     # model comparison and per-image CSV files
results/figures/                    # charts and qualitative localization examples
```

### Saved result snapshot

The saved evaluation uses a 20-image held-out test set. Its reported classification accuracies are:

| Model | Test accuracy | Mean IoU across all test images |
|---|---:|---:|
| Logistic Regression | 65% | — |
| SVM | 70% | — |
| Random Forest | 65% | — |
| CNN + CAM | 60% | 0.022 |
| Faster R-CNN | 65% | 0.252 |

For Faster R-CNN, the saved positive-image-only mean IoU is approximately 0.420. This is a different subset from the all-image mean IoU, so the two values should not be compared as if they used the same evaluation set. The test set is small: each image changes accuracy by 5 percentage points. The results are educational and should be interpreted cautiously.

The saved Faster R-CNN test results also show high sensitivity (approximately 91.7%) and low specificity (25%) at the current prediction threshold. This means it detects most positive images but incorrectly flags many no-pneumonia images. Localization quality and image-level classification are different measures.

## 9. Notebook

The notebook walkthrough is available at:

```text
notebooks/pneumonia_localization_master.ipynb
```

Open it in Jupyter or VS Code. Run the cells in order after installing dependencies and placing the dataset in the expected location.

## 10. Main files

```text
src/                                # data, preprocessing, models, CAM, boxes and metrics
scripts/run_project.py              # command-line training/evaluation pipeline
scripts/generate_final_artifacts.py  # generates result tables and figures
notebooks/                          # notebook walkthrough
demo/inference.py                   # CNN + CAM single-image demo
tests/                              # automated tests
train/                              # local dataset
results/                            # saved experiment metrics, tables and figures
```

## 11. References

1. Zhou, B. et al. “Learning Deep Features for Discriminative Localization.” CVPR, 2016. (Class Activation Mapping.)
2. Ren, S. et al. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” NeurIPS, 2015.
3. Supplied COCO-format chest X-ray dataset and annotation JSON.
