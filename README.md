# Pediatric Brain Tumor Segmentation (BraTS-PEDs 2024)

In this project, we built and evaluated deep learning models that **segment pediatric brain tumors** (binary tumor vs background)
from **multi-modal MRI** (**T1c, T1n, T2w, T2f**) using **PyTorch + MONAI**. Our V1 pipeline trains a **2D U-Net** on tumor-biased
axial slices, evaluates on a strict patient-level TEST split, and saves slide-ready figures + qualitative overlays.

> ⚠️ This model is a **research/educational demo only**. It is not a clinical diagnostic tool and should not be used for real medical decisions.

---

## Problem Statement
Pediatric brain tumors can affect critical brain functions and may cause symptoms like headaches, nausea/vomiting, seizures, balance issues,
and behavioral changes. MRI is commonly preferred over CT in children because MRI avoids ionizing radiation and provides detailed soft-tissue
contrast for brain anatomy and tumor-related abnormalities.

**Goal:** build a segmentation model that outlines tumor regions on pediatric brain MRI scans to support education and prototyping.

---

## Key Results (V1)
**Task:** Binary segmentation (**tumor vs background**) using `seg > 0`.

**Models trained**
- **Baseline:** 2D U-Net (4-channel input) trained on tumor-biased slices with **Dice + BCE** loss
- **Improved (final V1):** same 2D U-Net with a better recipe: **augmentations** (flips + intensity jitter + Gaussian noise) and **Dice + Focal** loss

**Saved outputs (see `results/`)**
- Per-patient metrics CSV: `results/test_metrics_long.csv` (Dice/IoU/precision/recall + empty prediction rates)
- Plots + overlays (PNG): `results/figures/`

> Note: Earlier validation Dice (~0.55) was from random-slice evaluation and can be noisy. Notebook 3 uses a deterministic per-patient protocol.

---

## Dataset
- **Source:** Kaggle — `srutorshibasuray/brats-ped-2024`
- **Location (Colab/Drive):** `brats_peds_2024_data/BraTS-PEDs2024_Training`

### Dataset quirks we handled
- Extracted **261** subject folders
- Found **2 invalid subjects** (empty modality file or missing modality path)
- Filtered valid subjects by requiring all modality paths exist and file size ≥ 1KB
- Final usable subjects: **259**
- Modalities are stored in folders named like `*-t1c.nii/` containing the actual NIfTI inside

---

## Methodology

### 1) Dataset loading & inspection
- Verified per-subject structure: 4 modalities + 1 segmentation mask
- Built robust file resolution to handle modality-folder structure

### 2) Label scheme (V1)
- Converted multi-class BraTS labels to binary: **tumor = `seg > 0`**

### 3) Split strategy (leakage prevention)
- **STRICT patient-level split** (never slice-level split)
- Saved artifacts to `splits/` so results are reproducible

### 4) Model training
- Architecture: **2D U-Net**, input channels = 4, output channels = 1
- Slice sampling: tumor-biased sampling to address class imbalance
- Baseline loss: Dice + BCE
- Improved loss: Dice + Focal
- Improved training adds augmentations for robustness

### 5) Deterministic test evaluation
To make evaluation stable and patient-level:
- Select **K fixed slices per patient** deterministically
  - If tumor exists: choose slices from tumor-containing slices evenly
  - Else: choose evenly across the volume
- Report per-patient Dice + IoU + precision/recall
- Track **empty prediction rate** (how often the model predicts no tumor)

### 6) Qualitative visualization
- Saved overlay grids showing MRI slice + **GT mask (green)** + **Prediction (red)**
- Exported best / worst / random examples for slides

---

## How to Run (Colab)
1. `notebooks/01_data_sanity.ipynb`
   - Validate dataset structure + handle quirks
   - Produce valid subjects list + splits

2. `notebooks/02_train_unet2d.ipynb`
   - Train baseline + improved model
   - Save best checkpoints to Drive

3. `notebooks/03_test_eval_figures.ipynb`
   - Load splits + checkpoints
   - Run deterministic TEST evaluation
   - Save CSV metrics + figures + overlay grids

---

## Streamlit Demo (planned)
A lightweight demo app will allow:
- selecting a patient + slice slider
- viewing MRI slice + predicted mask overlay
- optional GT overlay toggle (if available)

> This will remain an educational demo with a clear disclaimer.

---

## Technologies Used
- **Language:** Python
- **Core libraries:** NumPy, Pandas, PyTorch, MONAI, nibabel, scikit-learn, Matplotlib
- **Tools / Platforms:** Google Colab, Kaggle, Git/GitHub, Streamlit

---

## Authors / Credits
- **ML Engineering / Implementation:** **Penuel Stanley-Zebulon** (GitHub: `@iampenuel`, Email: `pcs5301@psu.edu`)
- **Research / Clinical context + write-up support:** **Chioma Kalu** (Email: `cfk5451@psu.edu`) — Research + clinical context + write-up support

---

## Repository Structure
```text
pediatric-brain-tumor-segmentation/
  notebooks/
    01_data_sanity.ipynb
    02_train_unet2d.ipynb
    03_test_eval_figures.ipynb
  splits/
    splits.json
    valid_subjects.txt
  results/
    figures/
      *.png
  src/
    (helpers: loading, preprocessing, metrics, inference)
  app/
    (streamlit demo)
  README.md
  requirements.txt
  .gitignore
```

---

## Disclaimer
This project is for **educational and research demonstration only**.
It is **not** a medical device and must **not** be used for clinical diagnosis or medical decision-making.
