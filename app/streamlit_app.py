
import os, glob
import numpy as np
import streamlit as st
import nibabel as nib
import torch

from monai.networks.nets import UNet

# =========================
# App Config
# =========================
st.set_page_config(page_title="Pediatric Brain Tumor Segmentation Demo", layout="wide")

st.title("Pediatric Brain Tumor Segmentation (BraTS-PEDs 2024) — Demo")
st.markdown(
    """
> ⚠️ **Educational / research demo only.**
> This tool is **not** a medical device and must **not** be used for clinical diagnosis or medical decisions.
"""
)

# =========================
# Helpers (dataset quirks)
# =========================
MODS = ["t1c", "t1n", "t2w", "t2f"]

def _find_single_nii_inside(path_or_folder):
    if os.path.isfile(path_or_folder) and (path_or_folder.endswith(".nii") or path_or_folder.endswith(".nii.gz")):
        return path_or_folder

    if os.path.isdir(path_or_folder):
        candidates = []
        candidates += glob.glob(os.path.join(path_or_folder, "*.nii"))
        candidates += glob.glob(os.path.join(path_or_folder, "*.nii.gz"))
        if len(candidates) == 0:
            candidates += glob.glob(os.path.join(path_or_folder, "*", "*.nii"))
            candidates += glob.glob(os.path.join(path_or_folder, "*", "*.nii.gz"))
        if len(candidates) == 0:
            return None
        return sorted(candidates)[0]
    return None

def get_subject_files(subject_id, data_root):
    subj_dir = os.path.join(data_root, subject_id)
    if not os.path.isdir(subj_dir):
        raise FileNotFoundError(f"Missing subject folder: {subj_dir}")

    seg_candidates = glob.glob(os.path.join(subj_dir, f"{subject_id}-seg.nii")) + \
                     glob.glob(os.path.join(subj_dir, f"{subject_id}-seg.nii.gz"))
    if len(seg_candidates) == 0:
        seg_candidates = glob.glob(os.path.join(subj_dir, "*-seg.nii")) + \
                         glob.glob(os.path.join(subj_dir, "*-seg.nii.gz"))
    if len(seg_candidates) == 0:
        raise FileNotFoundError(f"Could not find seg file for {subject_id}")
    seg_path = sorted(seg_candidates)[0]

    mod_paths = []
    for m in MODS:
        folder_like = os.path.join(subj_dir, f"{subject_id}-{m}.nii")
        folder_like_gz = os.path.join(subj_dir, f"{subject_id}-{m}.nii.gz")
        direct_file = os.path.join(subj_dir, f"{subject_id}-{m}.nii")
        direct_file_gz = os.path.join(subj_dir, f"{subject_id}-{m}.nii.gz")

        candidate = None
        for p in [folder_like, folder_like_gz, direct_file, direct_file_gz]:
            candidate = _find_single_nii_inside(p)
            if candidate is not None:
                break

        if candidate is None:
            g = glob.glob(os.path.join(subj_dir, f"*-{m}.nii")) + glob.glob(os.path.join(subj_dir, f"*-{m}.nii.gz"))
            g += glob.glob(os.path.join(subj_dir, f"*-{m}.nii", "*.nii")) + glob.glob(os.path.join(subj_dir, f"*-{m}.nii", "*.nii.gz"))
            candidate = sorted(g)[0] if len(g) else None

        mod_paths.append(candidate)

    if any(p is None for p in mod_paths):
        raise FileNotFoundError(f"Missing modality for {subject_id}. Got: {mod_paths}")

    return {"mods": mod_paths, "seg": seg_path}

def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)

def load_seg(path):
    return nib.load(path).get_fdata().astype(np.int16)

def zscore_normalize(volume, eps=1e-8):
    mask = (volume != 0)
    if mask.sum() > 100:
        mean = volume[mask].mean()
        std = volume[mask].std()
    else:
        mean = volume.mean()
        std = volume.std()
    std = std if std > eps else 1.0
    return (volume - mean) / std

def normalize_01(img):
    x = img.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def dice_2d(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2*inter) / denom) if denom > 0 else 1.0

def make_overlay_rgb(base_gray_01, gt=None, pred=None, alpha=0.35):
    """
    base_gray_01: (H,W) float in [0,1]
    gt: (H,W) 0/1 -> green overlay
    pred: (H,W) 0/1 -> red overlay
    """
    rgb = np.stack([base_gray_01, base_gray_01, base_gray_01], axis=-1)  # (H,W,3)
    if gt is not None:
        g = gt.astype(bool)
        rgb[g, 1] = np.clip((1 - alpha) * rgb[g, 1] + alpha * 1.0, 0, 1)
    if pred is not None:
        r = pred.astype(bool)
        rgb[r, 0] = np.clip((1 - alpha) * rgb[r, 0] + alpha * 1.0, 0, 1)
    return (rgb * 255).astype(np.uint8)

# =========================
# Model helpers
# =========================
def build_unet2d(in_channels=4, out_channels=1):
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )

@st.cache_resource
def load_model_cached(ckpt_path, device_str):
    device = torch.device(device_str)
    model = build_unet2d()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

@st.cache_data
def load_subject_cached(subject_id, data_root):
    files = get_subject_files(subject_id, data_root)
    mods = [zscore_normalize(load_nifti(p)) for p in files["mods"]]  # each (H,W,D)
    seg = load_seg(files["seg"])                                     # (H,W,D)
    seg_bin = (seg > 0).astype(np.uint8)
    return mods, seg_bin

@torch.no_grad()
def infer_slice_prob(model, mods_4, z, device_str):
    device = torch.device(device_str)
    x = np.stack([m[:, :, z] for m in mods_4], axis=0)  # (4,H,W)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)    # (1,4,H,W)
    logits = model(xt)
    prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
    return prob

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("Settings")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: **{device_str}**")

    data_root = st.text_input(
        "DATA_ROOT (folder containing patient subject folders)",
        value=os.environ.get("BRATS_DATA_ROOT", "./data/BraTS-PEDs2024_Training")
    )

    # Optional: load list of valid subjects from splits/valid_subjects.txt if present
    default_valid_path = os.path.join(os.path.dirname(__file__), "..", "splits", "valid_subjects.txt")
    default_valid_path = os.path.abspath(default_valid_path)

    valid_list_path = st.text_input(
        "valid_subjects.txt (optional)",
        value=os.environ.get("BRATS_VALID_LIST", default_valid_path)
    )

    # checkpoint paths
    baseline_ckpt = st.text_input(
        "Baseline checkpoint (.pt)",
        value=os.environ.get("BRATS_BASELINE_CKPT", "./models/best_unet2d_binary.pt")
    )
    improved_ckpt = st.text_input(
        "Improved checkpoint (.pt)",
        value=os.environ.get("BRATS_IMPROVED_CKPT", "./models/best_unet2d_binary_aug_focal.pt")
    )

    model_choice = st.selectbox("Choose model", ["improved", "baseline"])

    threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
    modality_bg = st.selectbox("Background modality", MODS, index=0)
    show_gt = st.checkbox("Show GT overlay (if available)", value=True)
    alpha = st.slider("Overlay alpha", 0.1, 0.8, 0.35, 0.05)

# =========================
# Load subject IDs
# =========================
def list_subjects(data_root, valid_list_path):
    subjects = []
    if os.path.isfile(valid_list_path):
        with open(valid_list_path, "r") as f:
            subjects = [line.strip() for line in f if line.strip()]
        return subjects

    # fallback: list directories
    if os.path.isdir(data_root):
        subjects = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    return subjects

subject_ids = list_subjects(data_root, valid_list_path)

if not subject_ids:
    st.error("No subjects found. Check DATA_ROOT and/or valid_subjects.txt path.")
    st.stop()

subject_id = st.selectbox("Select patient (subject_id)", subject_ids)

# =========================
# Load model
# =========================
ckpt_path = improved_ckpt if model_choice == "improved" else baseline_ckpt

if not os.path.isfile(ckpt_path):
    st.error(f"Checkpoint not found: {ckpt_path}\n\nTip: set BRATS_* env vars or edit the sidebar path.")
    st.stop()

try:
    model = load_model_cached(ckpt_path, device_str)
except Exception as e:
    st.error(f"Failed to load model from checkpoint:\n{ckpt_path}\n\nError: {e}")
    st.stop()

# =========================
# Load subject volumes
# =========================
try:
    mods, seg_bin = load_subject_cached(subject_id, data_root)
except Exception as e:
    st.error(f"Failed to load subject: {subject_id}\n\nError: {e}")
    st.stop()

D = seg_bin.shape[-1]
z = st.slider("Axial slice index (z)", 0, D - 1, D // 2, 1)

# =========================
# Inference
# =========================
prob = infer_slice_prob(model, mods, z, device_str)
pred = (prob >= threshold).astype(np.uint8)

bg_idx = MODS.index(modality_bg)
bg = mods[bg_idx][:, :, z]
bg_01 = normalize_01(bg)

gt = seg_bin[:, :, z] if show_gt else None
overlay = make_overlay_rgb(bg_01, gt=gt, pred=pred, alpha=alpha)

# =========================
# Display
# =========================
col1, col2 = st.columns([1.2, 1.0])

with col1:
    st.subheader("Overlay view")
    st.image(overlay, caption=f"{subject_id} | z={z} | model={model_choice} | thr={threshold:.2f}", use_container_width=True)

with col2:
    st.subheader("Slice summary")
    gt_area = int(seg_bin[:, :, z].sum())
    pred_area = int(pred.sum())
    st.write(f"**GT tumor pixels:** {gt_area}" if show_gt else "**GT tumor pixels:** (hidden)")
    st.write(f"**Pred tumor pixels:** {pred_area}")

    if show_gt:
        st.write(f"**Slice Dice:** {dice_2d(pred, seg_bin[:, :, z]):.4f}")

    st.caption("Green = Ground Truth (if enabled). Red = Prediction.")

# Optional: show raw probability map
with st.expander("Show probability map (debug)"):
    prob_01 = normalize_01(prob)
    st.image((prob_01 * 255).astype(np.uint8), caption="Predicted probability (sigmoid) rescaled to 0–255", use_container_width=True)
