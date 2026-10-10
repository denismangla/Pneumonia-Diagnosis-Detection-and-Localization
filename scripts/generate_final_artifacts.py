from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
TABLES = RESULTS / 'tables'
METRICS = RESULTS / 'metrics'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)


def load_json(name: str):
    return json.loads((METRICS / name).read_text(encoding='utf-8'))


def parse_boxes(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return []


def draw_boxes(ax, boxes, edgecolor, linewidth=2):
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=edgecolor, linewidth=linewidth)
        ax.add_patch(rect)


def make_comparison_table():
    baselines = load_json('baseline_test_metrics.json')
    cam = load_json('test_cam.json')['metrics']
    rcnn = load_json('test_rcnn.json')['metrics']
    rows = []
    for name, m in baselines.items():
        rows.append({
            'model': name,
            'accuracy': m['accuracy'],
            'balanced_accuracy': m.get('balanced_accuracy', np.nan),
            'sensitivity': m['recall_sensitivity'],
            'specificity': m['specificity'],
            'f1': m['f1'],
            'mean_iou': np.nan,
        })
    rows.extend([
        {'model': 'CNN + CAM', 'accuracy': cam['accuracy'], 'balanced_accuracy': cam.get('balanced_accuracy', np.nan),
         'sensitivity': cam['recall_sensitivity'], 'specificity': cam['specificity'], 'f1': cam['f1'], 'mean_iou': cam['mean_iou']},
        {'model': 'Faster R-CNN', 'accuracy': rcnn['accuracy'], 'balanced_accuracy': rcnn.get('balanced_accuracy', np.nan),
         'sensitivity': rcnn['recall_sensitivity'], 'specificity': rcnn['specificity'], 'f1': rcnn['f1'], 'mean_iou': rcnn['mean_iou']},
    ])
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / 'final_model_comparison.csv', index=False)
    return df


def make_charts(df):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(df['model'], df['accuracy'] * 100)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Classification test accuracy')
    ax.tick_params(axis='x', rotation=25)
    for i, v in enumerate(df['accuracy'] * 100):
        ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / 'final_classification_accuracy.png', dpi=180)
    plt.close(fig)

    loc = df.dropna(subset=['mean_iou'])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(loc['model'], loc['mean_iou'])
    ax.set_ylabel('Mean IoU')
    ax.set_ylim(0, max(0.12, float(loc['mean_iou'].max()) * 1.4))
    ax.set_title('Localization mean IoU on held-out test set')
    for i, v in enumerate(loc['mean_iou']):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / 'final_localization_iou.png', dpi=180)
    plt.close(fig)


def make_qualitative():
    split = pd.read_csv(ROOT / 'data' / 'splits' / 'test.csv')
    split['image_id'] = split['image_id'].astype(str)
    cam = load_json('test_cam.json')['per_image']
    rcnn = load_json('test_rcnn.json')['per_image']
    cam_map = {str(x['image_id']): x for x in cam}
    rcnn_map = {str(x['image_id']): x for x in rcnn}
    merged = []
    for _, row in split.iterrows():
        rec = cam_map[str(row['image_id'])]
        merged.append((float(rec['IoU']), row, rec, rcnn_map[str(row['image_id'])]))
    # one good CAM case, one poor positive, one false positive, one true negative
    positives = [x for x in merged if int(x[2]['true_label']) == 1]
    negatives = [x for x in merged if int(x[2]['true_label']) == 0]
    good = max(positives, key=lambda x: x[0])
    poor = min(positives, key=lambda x: x[0])
    fp = next((x for x in negatives if int(x[2]['predicted_label']) == 1), negatives[0])
    tn = next((x for x in negatives if int(x[2]['predicted_label']) == 0), negatives[-1])
    picks = [good, poor, fp, tn]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (_, row, crecord, rrecord) in zip(axes.flat, picks):
        img = np.asarray(Image.open(row['image_path']).convert('L'))
        ax.imshow(img, cmap='gray')
        draw_boxes(ax, parse_boxes(row['boxes_xywh']), 'lime', 2)
        draw_boxes(ax, crecord['predicted_boxes'], 'red', 2)
        ax.set_title(
            f"ID {row['image_id']} | GT={'P' if int(row['target']) else 'N'} | "
            f"Pred={'P' if int(crecord['predicted_label']) else 'N'} | "
            f"P(Pneu)={crecord['pneumonia_probability']:.2f} | CAM IoU={crecord['IoU']:.3f}\n"
            f"GT boxes: {len(parse_boxes(row['boxes_xywh']))}, CAM boxes: {len(crecord['predicted_boxes'])}, R-CNN boxes: {len(rrecord['predicted_boxes'])}",
            fontsize=9,
        )
        ax.axis('off')
    fig.suptitle('Representative held-out test predictions (green = ground truth, red = CAM prediction)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / 'qualitative_test_predictions.png', dpi=160)
    plt.close(fig)


def write_summary(df):
    cam = load_json('test_cam.json')['metrics']
    rcnn = load_json('test_rcnn.json')['metrics']
    summary = {
        'dataset': {
            'images': 200,
            'train': 140,
            'validation': 40,
            'test': 20,
            'pneumonia_images': 120,
            'no_pneumonia_images': 80,
        },
        'classification_and_localization': df.to_dict(orient='records'),
        'cam_positive_only_iou': {
            'mean_iou': float(np.mean([r['IoU'] for r in load_json('test_cam.json')['per_image'] if r['true_label'] == 1])),
            'median_iou': float(np.median([r['IoU'] for r in load_json('test_cam.json')['per_image'] if r['true_label'] == 1])),
        },
        'rcnn_positive_only_iou': {
            'mean_iou': float(np.mean([r['IoU'] for r in load_json('test_rcnn.json')['per_image'] if r['true_label'] == 1])),
            'median_iou': float(np.median([r['IoU'] for r in load_json('test_rcnn.json')['per_image'] if r['true_label'] == 1])),
        },
        'cam_threshold': cam['cam_threshold'],
        'notes': [
            'Results are measured on a 20-image held-out test split.',
            'The dataset is a small educational COCO export, not the full RSNA dataset.',
            'The weakly supervised CAM model uses image-level labels only for training; boxes are used for evaluation.',
        ],
    }
    (METRICS / 'final_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main():
    df = make_comparison_table()
    make_charts(df)
    make_qualitative()
    write_summary(df)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
