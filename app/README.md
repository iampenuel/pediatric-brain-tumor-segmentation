# Streamlit Demo (Local)

> ⚠️ Educational / research demo only. Not for clinical use.

## What it does
- Select a patient ID (subject) and an axial slice (z)
- Run inference using either baseline or improved model
- View MRI slice + overlays:
  - **Green** = Ground Truth (optional)
  - **Red** = Prediction

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Set paths (recommended via environment variables)
```bash
export BRATS_DATA_ROOT="/path/to/BraTS-PEDs2024_Training"
export BRATS_BASELINE_CKPT="/path/to/best_unet2d_binary.pt"
export BRATS_IMPROVED_CKPT="/path/to/best_unet2d_binary_aug_focal.pt"
export BRATS_VALID_LIST="/path/to/valid_subjects.txt"   # optional
```

## Run
From the repo root:
```bash
streamlit run app/streamlit_app.py
```

If you don’t set env vars, you can paste the paths directly into the Streamlit sidebar.
