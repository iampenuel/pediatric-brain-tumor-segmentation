# Pediatric Brain Tumor Segmentation (BraTS-PEDs 2024)

Binary tumor segmentation (tumor vs background) using a 2D U-Net trained on multi-modal MRI (T1c, T1n, T2w, T2f).
Built with PyTorch + MONAI. Includes deterministic test evaluation and slide-ready figures.

## Disclaimer
For research/education only. Not for clinical use.

## Dataset
BraTS-PEDs 2024 (Kaggle: srutorshibasuray/brats-ped-2024).
Dataset files are **not included** in this repo.

## Highlights
- Patient-level train/val/test split (prevents leakage)
- Baseline: 2D U-Net + Dice+BCE
- Improved: augmentations + Dice+Focal
- Deterministic test evaluation using fixed slice selection per patient
- Figures + overlays saved in `results/figures/`

## Repo layout
- `notebooks/` : Colab notebooks
- `splits/` : committed patient splits
- `results/figures/` : PNG figures
- `app/` : Streamlit demo (to be added/expanded)
- `src/` : reusable code (to be added/expanded)

## Setup
```bash
pip install -r requirements.txt
```

## Streamlit (later)
```bash
streamlit run app/streamlit_app.py
```
