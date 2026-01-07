
import os, glob, tempfile, urllib.request
import numpy as np
import streamlit as st
import nibabel as nib
import torch
from monai.networks.nets import UNet

st.set_page_config(page_title="Pediatric Brain Tumor Segmentation Demo", layout="wide")

st.title("Pediatric Brain Tumor Segmentation (BraTS-PEDs 2024) — Demo")
st.markdown(
    """
> ⚠️ **Educational / research demo only.**  
> This tool is **not** a medical device and must **not** be used for clinical diagnosis or medical decisions.
"""
)

MODS = ["t1c", "t1n", "t2w", "t2f"]

# -------------------------
# Utils
# -------------------------
def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)

def load_seg(path):
    return nib.load(path).get_fdata().astype(np.int16)

def zscore_normalize(volume, eps=1e-8):
    mask = (volume != 0)
    if mask.sum() > 100:
        mean = float(volume[mask].mean())
        std = float(volume[mask].std())
    else:
        mean = float(volume.mean())
        std = float(volume.std())
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
    return float((2 * inter) / denom) if denom > 0 else 1.0

def make_overlay_rgb(base_gray_01, gt=None, pred=None, alpha=0.35):
    rgb = np.stack([base_gray_01, base_gray_01, base_gray_01], axis=-1)
    if gt is not None:
        g = gt.astype(bool)
        rgb[g, 1] = np.clip((1 - alpha) * rgb[g, 1] + alpha * 1.0, 0, 1)
    if pred is not None:
        r = pred.astype(bool)
        rgb[r, 0] = np.clip((1 - alpha) * rgb[r, 0] + alpha * 1.0, 0, 1)
    return (rgb * 255).astype(np.uint8)

def save_upload_to_temp(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path

# -------------------------
# Dataset quirks support (local mode)
# -------------------------
def _find_single_nii_inside(path_or_folder):
    if os.path.isfile(path_or_folder) and (path_or_folder.endswith(".nii") or path_or_folder.endswith(".nii.gz")):
        return path_or_folder
    if os.path.isdir(path_or_folder):
        candidates = glob.glob(os.path.join(path_or_folder, "*.nii")) + glob.glob(os.path.join(path_or_folder, "*.nii.gz"))
        if not candidates:
            candidates = glob.glob(os.path.join(path_or_folder, "*", "*.nii")) + glob.glob(os.path.join(path_or_folder, "*", "*.nii.gz"))
        return sorted(candidates)[0] if candidates else None
    return None

def get_subject_files(subject_id, data_root):
    subj_dir = os.path.join(data_root, subject_id)
    if not os.path.isdir(subj_dir):
        raise FileNotFoundError(f"Missing subject folder: {subj_dir}")

    seg_candidates = glob.glob(os.path.join(subj_dir, f"{subject_id}-seg.nii")) + \
                     glob.glob(os.path.join(subj_dir, f"{subject_id}-seg.nii.gz")) + \
                     glob.glob(os.path.join(subj_dir, "*-seg.nii")) + \
                     glob.glob(os.path.join(subj_dir, "*-seg.nii.gz"))
    if not seg_candidates:
        raise FileNotFoundError(f"Could not find seg file for {subject_id}")
    seg_path = sorted(seg_candidates)[0]

    mod_paths = []
    for m in MODS:
        # handle folder-named-like *.nii/ containing real NIfTI
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
            candidate = sorted(g)[0] if g else None

        mod_paths.append(candidate)

    if any(p is None for p in mod_paths):
        raise FileNotFoundError(f"Missing modality for {subject_id}. Got: {mod_paths}")

    return {"mods": mod_paths, "seg": seg_path}

# -------------------------
# Model
# -------------------------
def build_unet2d():
    return UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )

def ensure_file(local_path, url):
    """Download once if missing."""
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 1024:
        return local_path
    if not url:
        return None
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    st.info(f"Downloading model weights… ({os.path.basename(local_path)})")
    urllib.request.urlretrieve(url, local_path)
    return local_path

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

@torch.no_grad()
def infer_slice_prob(model, mods_4, z, device_str):
    device = torch.device(device_str)
    x = np.stack([m[:, :, z] for m in mods_4], axis=0)  # (4,H,W)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)    # (1,4,H,W)
    logits = model(xt)
    prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
    return prob

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Settings")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: **{device_str}**")

    model_choice = st.selectbox("Choose model", ["improved", "baseline"])
    threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
    modality_bg = st.selectbox("Background modality", MODS, index=0)
    show_gt = st.checkbox("Show GT overlay (if available)", value=True)
    alpha = st.slider("Overlay alpha", 0.1, 0.8, 0.35, 0.05)

st.markdown("### Choose input mode")
tab_upload, tab_local = st.tabs(["🌐 Upload Mode (recommended for Cloud)", "💻 Local Dataset Mode (your machine)"])

# -------------------------
# Get model checkpoint (Cloud-friendly)
# -------------------------
# Prefer secrets for public URLs:
# BASELINE_URL="https://github.com/.../releases/download/v1/best_unet2d_binary.pt"
# IMPROVED_URL="https://github.com/.../releases/download/v1/best_unet2d_binary_aug_focal.pt"
baseline_url = st.secrets.get("BASELINE_URL", "")
improved_url = st.secrets.get("IMPROVED_URL", "")

ckpt_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "brats_peds_models")
baseline_local = os.path.join(ckpt_cache_dir, "best_unet2d_binary.pt")
improved_local = os.path.join(ckpt_cache_dir, "best_unet2d_binary_aug_focal.pt")

if model_choice == "baseline":
    ckpt_path = ensure_file(baseline_local, baseline_url)
else:
    ckpt_path = ensure_file(improved_local, improved_url)

if ckpt_path is None:
    st.error("Model checkpoint not available. Add BASELINE_URL and IMPROVED_URL in Streamlit Cloud Secrets.")
    st.stop()

# Load model
model = load_model_cached(ckpt_path, device_str)

# -------------------------
# Upload Mode
# -------------------------
with tab_upload:
    st.info("Upload 4 modalities (T1c, T1n, T2w, T2f). Optionally upload a GT seg for Dice display.")

    c1, c2, c3, c4 = st.columns(4)
    up_t1c = c1.file_uploader("T1c (.nii/.nii.gz)", type=["nii","gz"], key="t1c")
    up_t1n = c2.file_uploader("T1n (.nii/.nii.gz)", type=["nii","gz"], key="t1n")
    up_t2w = c3.file_uploader("T2w (.nii/.nii.gz)", type=["nii","gz"], key="t2w")
    up_t2f = c4.file_uploader("T2f (.nii/.nii.gz)", type=["nii","gz"], key="t2f")

    up_seg = st.file_uploader("Optional GT Seg (.nii/.nii.gz)", type=["nii","gz"], key="seg")

    if not (up_t1c and up_t1n and up_t2w and up_t2f):
        st.warning("Upload all 4 modalities to run inference.")
        st.stop()

    # save uploads to temp
    p_t1c = save_upload_to_temp(up_t1c)
    p_t1n = save_upload_to_temp(up_t1n)
    p_t2w = save_upload_to_temp(up_t2w)
    p_t2f = save_upload_to_temp(up_t2f)
    seg_path = save_upload_to_temp(up_seg) if (up_seg and show_gt) else None

    mods = [zscore_normalize(load_nifti(p)) for p in [p_t1c, p_t1n, p_t2w, p_t2f]]
    seg_bin = None
    if seg_path is not None:
        seg = load_seg(seg_path)
        seg_bin = (seg > 0).astype(np.uint8)

    D = mods[0].shape[-1]
    z = st.slider("Axial slice index (z)", 0, D - 1, D // 2, 1, key="upload_z")

    prob = infer_slice_prob(model, mods, z, device_str)
    pred = (prob >= threshold).astype(np.uint8)

    bg_idx = MODS.index(modality_bg)
    bg = mods[bg_idx][:, :, z]
    bg_01 = normalize_01(bg)

    gt = seg_bin[:, :, z] if (seg_bin is not None and show_gt) else None
    overlay = make_overlay_rgb(bg_01, gt=gt, pred=pred, alpha=alpha)

    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("Overlay view")
        st.image(overlay, use_container_width=True)

    with right:
        st.subheader("Slice summary")
        st.write(f"**Pred tumor pixels:** {int(pred.sum())}")
        if seg_bin is not None and show_gt:
            st.write(f"**GT tumor pixels:** {int(seg_bin[:, :, z].sum())}")
            st.write(f"**Slice Dice:** {dice_2d(pred, seg_bin[:, :, z]):.4f}")
        st.caption("Green = Ground Truth (if provided). Red = Prediction.")

    with st.expander("Show probability map (debug)"):
        prob_01 = normalize_01(prob)
        st.image((prob_01 * 255).astype(np.uint8), caption="Sigmoid prob map (rescaled 0–255)", use_container_width=True)

# -------------------------
# Local Dataset Mode (your machine)
# -------------------------
with tab_local:
    st.caption("This mode is for running locally with your downloaded BraTS-PEDs dataset folder.")

    data_root = st.text_input("DATA_ROOT (BraTS-PEDs2024_Training)", value=os.environ.get("BRATS_DATA_ROOT", ""))

    valid_list_path = st.text_input(
        "valid_subjects.txt (optional)",
        value=os.environ.get("BRATS_VALID_LIST", "")
    )

    def list_subjects(data_root, valid_list_path):
        if os.path.isfile(valid_list_path):
            with open(valid_list_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        if os.path.isdir(data_root):
            return sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        return []

    subject_ids = list_subjects(data_root, valid_list_path)
    if not subject_ids:
        st.warning("No subjects found. Set DATA_ROOT or valid_subjects.txt.")
        st.stop()

    subject_id = st.selectbox("Select patient (subject_id)", subject_ids)

    try:
        files = get_subject_files(subject_id, data_root)
        mods = [zscore_normalize(load_nifti(p)) for p in files["mods"]]
        seg = load_seg(files["seg"])
        seg_bin = (seg > 0).astype(np.uint8)
    except Exception as e:
        st.error(f"Failed to load subject: {e}")
        st.stop()

    D = seg_bin.shape[-1]
    z = st.slider("Axial slice index (z)", 0, D - 1, D // 2, 1, key="local_z")

    prob = infer_slice_prob(model, mods, z, device_str)
    pred = (prob >= threshold).astype(np.uint8)

    bg_idx = MODS.index(modality_bg)
    bg = mods[bg_idx][:, :, z]
    bg_01 = normalize_01(bg)

    gt = seg_bin[:, :, z] if show_gt else None
    overlay = make_overlay_rgb(bg_01, gt=gt, pred=pred, alpha=alpha)

    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("Overlay view")
        st.image(overlay, caption=f"{subject_id} | z={z} | model={model_choice} | thr={threshold:.2f}", use_container_width=True)

    with right:
        st.subheader("Slice summary")
        st.write(f"**GT tumor pixels:** {int(seg_bin[:, :, z].sum())}" if show_gt else "**GT tumor pixels:** (hidden)")
        st.write(f"**Pred tumor pixels:** {int(pred.sum())}")
        if show_gt:
            st.write(f"**Slice Dice:** {dice_2d(pred, seg_bin[:, :, z]):.4f}")
        st.caption("Green = Ground Truth (optional). Red = Prediction.")
