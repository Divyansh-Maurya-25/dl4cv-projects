# Deep Learning for Computer Vision — Projects 1, 2 & 4

> Three end-to-end deep learning projects covering image classification, semantic segmentation, and multi-modal visual grounding. Built for CIS 6930 at USF.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org)

---

## Project 1 — Car Classifier with EfficientNet-B1

**Task:** Multi-class image classification of car models from photos.

**Architecture:** `CarNet` wraps EfficientNet-B1 with a custom `NormalizeInModel` layer so normalization is baked into the model graph (no preprocessing step needed at inference). Test-Time Augmentation (TTA) with horizontal flip is applied inside `forward()`.

**Training strategy:**
- 2-stage: head warmup (5 epochs, backbone frozen) → full fine-tune (35 epochs)
- Exponential Moving Average (EMA) with decay=0.999
- MixUp augmentation (α=0.2)
- RandAugment
- Stratified 85/15 train/val split
- Label smoothing = 0.08

**Results:** 93.88% validation accuracy. Exported as TorchScript `model.pt` (26MB).

**Core concepts:** Transfer learning, EMA, MixUp, TTA, TorchScript export.

---

## Project 2 — Binary Semantic Segmentation with LRASPP MobileNetV3

**Task:** Pixel-wise binary segmentation — classify each pixel as foreground or background.

**Architecture:** `FastSegModel` wraps LRASPP MobileNet V3 Large with a custom 1-channel output head replacing the default multi-class head.

**Loss function:** Combined BCE + Dice loss.
- BCE handles per-pixel binary cross entropy
- Dice loss directly optimizes the IoU metric, preventing the model from ignoring small foreground regions

**Evaluation:** Custom `compute_iou()` function computes intersection-over-union on the validation set.

**Results:** Best validation IoU: **0.9736** over 10 epochs.

**Core concepts:** Semantic segmentation, Dice loss, IoU evaluation, lightweight mobile architecture.

---

## Project 4 — Visual Grounding with Molmo2-4B

**Task:** Given an image, use a vision-language model to locate a described object and return its pixel coordinates.

**Approach:** Rather than fine-tuning, leveraged Molmo2-4B (4B parameter multimodal model via HuggingFace) with a multi-prompt consensus voting strategy.

**Key implementation:**
```python
def build_prompt_variants():
    # Returns 6 prompts with weights 1.0–3.0
    # Higher weight = more confident/specific phrasing

def choose_consensus_point(predictions, weights):
    # Clusters predictions by radius = 12% of max image dimension
    # Applies weighted voting within clusters
    # Applies edge penalty to suppress predictions near image boundaries
```

**Inference:** Greedy decoding (temperature=None, top_p=None) for deterministic output.

**Core concepts:** Vision-language models, prompt engineering, weighted consensus voting, multi-modal inference.

---

## Setup

```bash
git clone https://github.com/Divyansh-Maurya-25/dl4cv-projects.git
cd dl4cv-projects
pip install torch torchvision transformers numpy matplotlib
```

Each project is self-contained in its subdirectory. Open the `.ipynb` file in Jupyter or Google Colab.

> ⚠️ Project 4 requires a GPU with ≥16GB VRAM to load Molmo2-4B. Use Google Colab A100 or equivalent.

---

## File Structure

```
dl4cv-projects/
├── Project1/
│   └── Project1.ipynb       # EfficientNet-B1 car classifier
├── Project2/
│   └── Project2.ipynb       # LRASPP segmentation (IoU 0.9736)
├── Project4/
│   └── project4.ipynb       # Molmo2-4B visual grounding
└── README.md
```

---

## Course Context

Projects for **CIS 4930 — Deep Learning for Computer Vision** at the University of South Florida.
