# Structural MRI Processing Pipeline

A lightweight, end-to-end pipeline for **brain extraction**, **registration to MNI**, **post-processing (N4 + masked Z-score + cropping)**, **QC visualization**, and **CSV metadata** generation.  
Built around simple, explicit Python scripts with a shared config and a unified folder schema.

> This README is a complete, copy‑pasteable guide for releasing the repo publicly. It mirrors the style/structure of a typical research code README (overview → install → usage → outputs → troubleshooting → citations) **and adds a step‑by‑step explanation of each stage**.

---

## Features

- **Brain extraction (SynthStrip)** with automatic Docker/Singularity/FS binary fallbacks.
- **Two-stage registration** (Rigid → Affine) using SimpleITK to **MNI152NLin2009cAsym (1 mm)**.
- **Post-processing**: N4 bias correction (optional), **masked Z-score normalization**, and **auto-crop**.
- **QC**:
  - Per-subject PNG panels (raw, brain, mask overlays).
  - CSV summary with brain volume + status.
  - Multi-slice post-process visualization.
- **Unified structure** resolver (subject/session aware).
- **CSV metadata** exporter (paths + demographics) for downstream ML.

---

## Repo Layout

```
.
├── config.py                               # Central configuration (paths, options, patterns)
├── skullstrip.py                            # Brain extraction + QC
├── reg.py                                   # T1/T2 → MNI (Rigid + Affine), saves transforms
├── postprocess_mni_mask_zscore_crop.py      # Warp masks, N4, masked Z-score, crop, viz
├── structural_qc.py                          # extra QC helpers
├── generate_csv_metadata.py                  # Build CSVs with file paths + demographics
├── structure_resolver.py                     # Unified subject/session path discovery
├── dicom_to_nifti.py                         # (If needed) DICOM → NIfTI helper
└── README.md                                 # You are here
```

---

## Data Layout

The pipeline uses a **unified pattern**:
```
{root}/{subject}/{session}/anat/
```
- `subject` looks like `sub-ON12345`.
- `session` looks like `ses-01` (or may be absent if your dataset is single-session).
- Input typically contains NIfTI files named like:
  - `*acq-MPRAGE_T1w.nii.gz`
  - `*acq-CUBE_T2w.nii.gz`

Configure the **stage roots** in `config.py`:
- `STAGE_ROOTS["skullstrip"]` → extracted brains & masks (+ QC)
- `STAGE_ROOTS["registration"]` → MNI outputs & transforms
- `STAGE_ROOTS["qc"]` → global QC CSV/images (optional)

---

## Requirements

- **Python** ≥ 3.8
- Python packages:
  - `numpy`, `nibabel`, `SimpleITK`, `matplotlib`, `tqdm`, `templateflow`, `pandas`
- **FreeSurfer** (for SynthStrip binary) **or** Docker **or** Singularity
  - If using FreeSurfer binary: `FREESURFER_HOME` set in `config.py`
  - If using containers: access to `freesurfer/synthstrip:latest`
- (Optional) TemplateFlow cache access (auto-downloads MNI template)

Create and activate an environment (example with `conda`):
```bash
conda create -y -n smri python=3.10
conda activate smri
pip install numpy nibabel SimpleITK matplotlib tqdm templateflow pandas
```

---

## Configuration

All configuration is centralized in **`config.py`**.

### `config.example.py` (safe to commit)
```python
import os

# --- Paths (replace locally) ---
INPUT_DIR  = "/path/to/raw_or_nifti_root"
OUTPUT_DIR = "/path/to/derivatives/Structural"

# Stage roots
STAGE_ROOTS = {
    "skullstrip":   os.path.join(OUTPUT_DIR, "skullstrip"),
    "registration": os.path.join(OUTPUT_DIR, "registration"),
    "qc":           os.path.join(OUTPUT_DIR, "QC"),
}

# Unified structure
STRUCTURE = "{root}/{subject}/{session}/anat"

# Processing
MODALITIES = ["T1w", "T2w"]
SUBJECTS = None        # e.g. ["sub-ON12345"]
SESSIONS = None        # e.g. ["ses-01"]
FORCE_REPROCESSING = False

# Logging/parallel
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CPU_CORES = 16

# SynthStrip / FreeSurfer
FREESURFER_HOME = "/opt/freesurfer"  # if using FS binary
SYNTHSTRIP_BIN = os.path.join(FREESURFER_HOME, "bin", "mri_synthstrip")
SYNTHSTRIP_SIF_PATH = "/path/to/synthstrip.sif"  # if using Singularity

# File patterns
SKULLSTRIP_T1_FILE_PATTERN = "*acq-MPRAGE_T1w.nii.gz"
SKULLSTRIP_T2_FILE_PATTERN = "*acq-CUBE_T2w.nii.gz"

# QC
ENABLE_QC = True
QC_GENERATE_IMAGES = True
QC_SUMMARY_FILE = "qc_summary.csv"
QC_BRAIN_VOLUME_MIN_ML = 800
QC_BRAIN_VOLUME_MAX_ML = 2000

# Registration
MNI_TEMPLATE_VERSION = "MNI152NLin2009cAsym"
MNI_TEMPLATE_RESOLUTION = 1

# Postprocess
POSTPROCESS_APPLY_N4 = True
POSTPROCESS_CROP_MARGIN = 1
POSTPROCESS_MODALITIES = ["T1", "T2"]
POSTPROCESS_SUBJECTS = SUBJECTS

# CSV metadata
GENERATE_CSV_METADATA = True
CSV_METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
T1_CSV_FILENAME = "T1_metadata.csv"
T2_CSV_FILENAME = "T2_metadata.csv"
PARTICIPANTS_TSV_PATH = os.path.join(CSV_METADATA_DIR, "participants.tsv")

CSV_COLUMNS = [
    "subjectId", "MNI_Warped", "MNI_ZSCORE", "MNI_Z_Cropped",
    "age", "sex", "handedness"
]

T1_PROCESSED_PATTERNS = {
    "MNI_Warped":   "*_T1w_brain_mni_warped.nii.gz",
    "MNI_ZSCORE":   "*_T1w_brain_mni_zscore.nii.gz",
    "MNI_Z_Cropped":"*_T1w_brain_mni_zscore_cropped.nii.gz",
}
T2_PROCESSED_PATTERNS = {
    "MNI_Warped":   "*_T2w_brain_mni_warped.nii.gz",
    "MNI_ZSCORE":   "*_T2w_brain_mni_zscore.nii.gz",
    "MNI_Z_Cropped":"*_T2w_brain_mni_zscore_cropped.nii.gz",
}
```
Then, on your machine/cluster:
```bash
cp config.example.py config.py
# edit config.py with your private paths/settings (do NOT commit)
```

---

## Quick Start

1) **Brain Extraction + QC**
```bash
python skullstrip.py \
  --input_dir  "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --modalities T1w T2w \
  --workers 8 \
  --enable_qc
```

2) **Registration (T1/T2 → MNI)**
```bash
python reg.py --subject sub-ON12345   # optional filter
# or simply
python reg.py
```

3) **Post-processing (mask warp, N4, masked Z-score, crop, visualization)**
```bash
python postprocess_mni_mask_zscore_crop.py
# optional: --subject sub-ON12345
```

4) **CSV Metadata**
```bash
python generate_csv_metadata.py
```

> The pipeline is modular: you can run each stage independently or all in sequence.

---

## Stage-by-Stage Details (English)

### Stage 0 — (Optional) DICOM → NIfTI
- **Script**: `dicom_to_nifti.py`
- **What**: Converts DICOM series into BIDS-ish NIfTI files if your dataset is not already in NIfTI.
- **Inputs**: DICOM directories, output root.
- **Outputs**: NIfTI files under `INPUT_DIR` in `{subject}/{session}/anat` layout.
- **Notes**: Skip if your data is already NIfTI.

---

### Stage 1 — Brain Extraction (SynthStrip) + QC
- **Script**: `skullstrip.py`
- **Algorithm**: FreeSurfer **SynthStrip** (deep-learning skull-stripping). The script auto-chooses:
  1. **Singularity** (preferred if available) using `SYNTHSTRIP_SIF_PATH`,
  2. else **Docker** image `freesurfer/synthstrip:latest`,
  3. else local **FreeSurfer binary** `mri_synthstrip`.
- **Inputs**: `*acq-MPRAGE_T1w.nii.gz`, `*acq-CUBE_T2w.nii.gz` discovered via `STRUCTURE`.
- **Outputs** (per subject/session in `STAGE_ROOTS["skullstrip"]`):
  - `*_brain.nii.gz` — brain-extracted volume.
  - `*_brain_mask.nii.gz` — binary mask.
  - `*_desc-qc.png` — 3-view overlays (raw/brain/mask).
  - `QC/qc_summary.csv` — rows with subject, session, modality, volume(ml), PASS/FAIL.
- **QC**: Brain volume computed from mask and voxel size; `QC_BRAIN_VOLUME_MIN_ML`–`MAX_ML` thresholds.
- **CLI flags**: `--input_dir`, `--output_dir`, `--modalities`, `--subjects`, `--sessions`, `--workers`, `--force`, `--enable_qc`.
- **Tips**:
  - Ensure `FREESURFER_HOME` (if using FS binary) or container runtime (Docker/Singularity).
  - If re-running, use `--force` to overwrite.

---

### Stage 2 — Registration to MNI (Rigid → Affine)
- **Script**: `reg.py`
- **Algorithm**:
  - **Rigid**: Euler3D (rotation + translation) with Mattes Mutual Information, multi-resolution (4–2–1), linear interp.
  - **Affine**: 12-DOF affine starting from identity, multi-resolution (2–1), linear interp.
  - **Transforms**: Saved as SimpleITK `.mat` (rigid + affine). Final resampling uses **composite** transform for single interpolation.
  - **T2 handling**: Applies T1-derived transforms to T2 (within the same subject/session).
- **Template**: `MNI152NLin2009cAsym` (1 mm) via TemplateFlow.
- **Inputs**: `*_brain.nii.gz` from Stage 1.
- **Outputs** (per subject/session in `STAGE_ROOTS["registration"]`):
  - `*_mni_rigid_warped.nii.gz` — rigid-only (QC/ref).
  - `*_mni_warped.nii.gz` — final (rigid+affine) warped volume.
  - `other/*_rigid.mat`, `other/*_affine.mat` — transforms.
- **Metrics**: Processing times per subject are logged; optional fail lists saved in JSON report.
- **Tips**:
  - TemplateFlow must be able to write cache (set `TEMPLATEFLOW_HOME` if needed).
  - Ensure T1/T2 **session** matching for T2 warping.

---

### Stage 3 — Post-processing (Mask Warp → N4 → Masked Z-score → Crop → Viz)
- **Script**: `postprocess_mni_mask_zscore_crop.py`
- **Steps**:
  1. **Warp native masks to MNI** using the **composite** (rigid+affine) transform in a **single nearest-neighbor resample**.
  2. **N4 bias correction** (optional; `POSTPROCESS_APPLY_N4`) using the MNI-space mask.
  3. **Masked Z-score**: μ/σ computed **inside brain mask**; background kept at 0 by default.
  4. **Crop**: Tight bounding box around mask (+ margin `POSTPROCESS_CROP_MARGIN`), affine adjusted.
  5. **Visualization**: Saves a multi-slice PNG montage per subject/session.
- **Inputs**: Stage 2 warped images + Stage 1 masks.
- **Outputs** (per subject/session in registration `anat/`):
  - `T1w_mni_mask.nii.gz` / `T2w_mni_mask.nii.gz`
  - `T1w_mni_warped_n4.nii.gz` (if enabled)
  - `T1w_mni_zscore_fixed.nii.gz` → `T1w_mni_zscore_fixed_cropped.nii.gz`
  - `multislice_visualization_<subj>.png`
  - (analogous T2 files when available)
- **Tips**:
  - If T2 native mask missing, script falls back to T1’s MNI mask for T2.
  - Cropping writes qform/sform to preserve spatial orientation.

---

### Stage 4 — Quality Control & Reports
- **Script(s)**: `skullstrip.py` (volume QC, masks overlay), `structural_qc.py` (optional extras).
- **What**:
  - Image panels (`*_desc-qc.png`) for skull-stripping.
  - Brain volume check (min/max ml) → PASS/FAIL.
  - Optional Dice-based thresholds (`DICE_PASS_THRESHOLD`) if you compute Dice against a reference mask (custom).
- **Output**: `QC/qc_summary.csv` with columns: subject, session, modality, qc_status, brain_volume_ml, qc_image_path, error_message.

---

### Stage 5 — CSV Metadata for ML
- **Script**: `generate_csv_metadata.py`
- **What**: Scans registration outputs, collects paths to **MNI warped**, **Z-score**, **cropped** files; merges demographics from `participants.tsv`; encodes `sex` / `handedness` (mapping in `config.py`).
- **Outputs**: `metadata/T1_metadata.csv`, `metadata/T2_metadata.csv` with columns from `CSV_COLUMNS`.
- **Notes**: Set `REQUIRE_BOTH_MODALITIES`/`VALIDATE_FILE_EXISTENCE` to control inclusion.

---

## What Gets Produced? (Summary)

### Skullstrip (`STAGE_ROOTS["skullstrip"]`)
- `*_brain.nii.gz`, `*_brain_mask.nii.gz`, `*_desc-qc.png`, and global `QC/qc_summary.csv`.

### Registration (`STAGE_ROOTS["registration"]`)
- `*_mni_rigid_warped.nii.gz`, `*_mni_warped.nii.gz`, and transforms in `other/`.

### Post-process (under the same `anat/`)
- `T1w_mni_mask.nii.gz`, `T1w_mni_zscore_fixed.nii.gz`, `T1w_mni_zscore_fixed_cropped.nii.gz`, `multislice_visualization_<subj>.png` (and T2 analogs).

### CSVs (`CSV_METADATA_DIR`)
- `T1_metadata.csv`, `T2_metadata.csv` → paths + demographics for ML-ready tables.

---

## Unified Structure Resolver

- **File**: `structure_resolver.py`
- **Purpose**: Discover `{subject}/{session}/anat` directories and build output paths consistently across stages.
- **APIs**:
  - `discover(STRUCTURE, root, subjects=None, sessions=None)` → yields `(subject, session, path)`.
  - `make_path(STRUCTURE, root, "sub-XXXXX", "ses-YY")` → returns path string.
- **Why**: Portable across datasets (BIDS-like or custom) by changing a single pattern in `config.py`.

---

## Troubleshooting

- **SynthStrip not found**:
  - Ensure one of: `singularity` (and `SYNTHSTRIP_SIF_PATH`), `docker`, or `mri_synthstrip` in `$FREESURFER_HOME/bin`.
- **TemplateFlow cache errors**:
  - Set `TEMPLATEFLOW_HOME` to a writable path, re-run `reg.py`.
- **No sessions dataset**:
  - Change `STRUCTURE` to `"{root}/{subject}/anat"` and re-run. Scripts do not require BIDS per se.
- **Weird cropping / orientation**:
  - Postprocess writes qform/sform—if viewers complain, re-check header with `nib-ls` / `fslhd`.
- **Memory/threads**:
  - Control parallelism via `--workers` (skullstrip) and `CPU_CORES` in `config.py`. SimpleITK threads set from config in `reg.py`.

---

## Privacy & Release Checklist

- ✅ Put real cluster/paths only in **`config.py`** and keep it **untracked**.
- ✅ Commit **`config.example.py`** with placeholders.
- ✅ Review this README for accidental path leaks before publishing.
- ✅ (Optional) Add `.gitignore` entries:
  ```
  config.py
  *.sif
  */QC/*
  */logs/*
  ```

---

## Reproducing a Full Run (Example)

```bash
# 0) Environment
conda create -y -n smri python=3.10
conda activate smri
pip install numpy nibabel SimpleITK matplotlib tqdm templateflow pandas

# 1) Configure
cp config.example.py config.py
# edit config.py (private paths)

# 2) Brain extraction
python skullstrip.py --workers 8 --enable_qc

# 3) Registration
python reg.py

# 4) Post-process
python postprocess_mni_mask_zscore_crop.py

# 5) CSVs
python generate_csv_metadata.py
```

---

## Citation

If you use this pipeline in your work, please consider citing the tools it builds upon:

- **SynthStrip** (FreeSurfer)
- **TemplateFlow**
- **SimpleITK**
- **N4 Bias Field Correction** (Tustison et al.)
- (Any dataset you process, e.g., OpenNeuro accession)

*(Add your preferred BibTeX entries here.)*

---

## License

Add your license (e.g., MIT) as `LICENSE` in the repo root.

---

## Acknowledgements

Thanks to the open-source neuroimaging community (FreeSurfer, TemplateFlow, SimpleITK, nibabel) and dataset providers. Special thanks to lab colleagues for testing/feedback.

---
