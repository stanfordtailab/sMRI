# Structural MRI Processing Pipeline

A lightweight, end-to-end pipeline for **brain extraction**, **registration to MNI**, **post-processing (N4 + masked Z‑score + cropping)**, **QC visualization**, and **CSV metadata** generation.  
Built around simple Python scripts with a shared config and a unified folder schema.

> This README is a complete, copy‑pasteable guide for public release. It explains each stage step‑by‑step and matches the current CLI of the scripts you shared.

---

## Features

- **Brain extraction (SynthStrip)** with automatic Singularity/Docker/FreeSurfer‑binary fallback.
- **Two‑stage registration** (Rigid → Affine) using SimpleITK to **MNI152NLin2009cAsym (1 mm)**.
- **Post‑processing**: **N4 bias correction**, **masked Z‑score normalization**, and **auto‑crop** with affine update.
- **QC**:
  - Per‑subject PNG panels (raw, brain, mask overlays).
  - CSV summary with brain volume and PASS/FAIL.
  - Multi‑slice post‑process visualization.
- **Unified structure** resolver (subject/session aware).
- **CSV metadata** exporter (paths + demographics) for downstream ML.

---

## Repository Layout

```
.
├── config.py                               # Central configuration (paths, options, patterns)
├── skullstrip.py                            # Brain extraction + QC
├── reg.py                                   # T1/T2 → MNI (Rigid + Affine), saves transforms
├── postprocess_mni_mask_zscore_crop.py      # Warp masks, N4, masked Z-score, crop, viz
├── structural_qc.py                          # Optional/extra QC helpers
├── generate_csv_metadata.py                  # Build CSVs with file paths + demographics
├── structure_resolver.py                     # Unified subject/session path discovery
├── dicom_to_nifti.py                         # (Optional) DICOM → NIfTI helper
└── README.md                                 # You are here
```

---

## Data Layout

The pipeline uses a **unified pattern**:
```
{root}/{subject}/{session}/anat/
```
- `subject` like `sub-ON12345`
- `session` like `ses-01` (may be absent in single‑session datasets)
- Input NIfTI filenames (examples):
  - `*acq-MPRAGE_T1w.nii.gz`
  - `*acq-CUBE_T2w.nii.gz`

Configure **stage roots** in `config.py`:
- `STAGE_ROOTS["skullstrip"]` → extracted brains & masks (+ QC)
- `STAGE_ROOTS["registration"]` → MNI outputs & transforms
- `STAGE_ROOTS["qc"]` → global QC CSV/images (optional)

---

## Requirements

- **Python** ≥ 3.8
- Python packages: `numpy`, `nibabel`, `SimpleITK`, `matplotlib`, `tqdm`, `templateflow`, `pandas`
- **FreeSurfer** (for SynthStrip binary) **or** Docker **or** Singularity  
  If using FreeSurfer binary: set `FREESURFER_HOME` in `config.py`  
  If using containers: access to `freesurfer/synthstrip:latest` (Docker or SIF)
- (Optional) TemplateFlow cache (auto‑downloads MNI template)

Create an environment (example with `conda`):
```bash
conda create -y -n smri python=3.10
conda activate smri
pip install numpy nibabel SimpleITK matplotlib tqdm templateflow pandas
```

---

## Configuration

All configuration is centralized in **`config.py`**.

### Minimal `config.py` (example)
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

# Unified structure (change once for your dataset)
STRUCTURE = "{root}/{subject}/{session}/anat"

# Processing
MODALITIES = ["T1w", "T2w"]
SUBJECTS = None        # e.g. ["sub-ON12345"]
SESSIONS = None        # e.g. ["ses-01"]
FORCE_REPROCESSING = False

# Logging / parallel
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CPU_CORES = 16

# SynthStrip / FreeSurfer
FREESURFER_HOME   = "/opt/freesurfer"  # if using FS binary
SYNTHSTRIP_BIN    = os.path.join(FREESURFER_HOME, "bin", "mri_synthstrip")
SYNTHSTRIP_SIF_PATH = "/path/to/synthstrip.sif"  # if using Singularity

# File patterns for inputs
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
> **Note:** Keep real paths only in your local `config.py`. Publish `config.example.py` with placeholders.

---

## Quick Start

1. **Brain Extraction + QC**
   ```bash
   python skullstrip.py           # processes all subjects
   # optional: python skullstrip.py --subject sub-ON12345
   ```

2. **Registration (T1/T2 → MNI)**
   ```bash
   python reg.py                  # processes all subjects with skull-stripped brains
   # optional: python reg.py --subject sub-ON12345
   ```

3. **Post‑processing (mask warp, N4, masked Z‑score, crop, visualization)**
   ```bash
   python postprocess_mni_mask_zscore_crop.py
   # optional: --subject sub-ON12345
   ```

4. **CSV Metadata**
   ```bash
   python generate_csv_metadata.py
   ```

> The pipeline is modular—run each stage independently or all in sequence.

---

## Stage‑by‑Stage Explanation

### Stage 0 — (Optional) DICOM → NIfTI
- **Script**: `dicom_to_nifti.py`
- **Goal**: Convert DICOM series into BIDS‑like NIfTI if needed.
- **Input/Output**: Reads DICOM folders, writes NIfTI under `INPUT_DIR` using `{subject}/{session}/anat` layout.
- **Skip if** your dataset is already NIfTI.

### Stage 1 — Brain Extraction (SynthStrip) + QC
- **Script**: `skullstrip.py`
- **Logic**: Tries **Singularity** (SIF), then **Docker**, then local **FreeSurfer `mri_synthstrip`**.
- **Inputs**: Files matching `SKULLSTRIP_T1_FILE_PATTERN` and `SKULLSTRIP_T2_FILE_PATTERN` discovered via `STRUCTURE`.
- **Outputs** (per subject/session in `STAGE_ROOTS["skullstrip"]`):
  - `*_brain.nii.gz` and `*_brain_mask.nii.gz`
  - `*_desc-qc.png` montage (raw/brain/mask overlays)
  - Global `QC/qc_summary.csv` with brain volume and PASS/FAIL
- **QC rule**: Brain volume (ml) must be within `[QC_BRAIN_VOLUME_MIN_ML, QC_BRAIN_VOLUME_MAX_ML]`.
- **CLI**: Supports `--subject` (optional filter). Other parameters come from `config.py`.

### Stage 2 — Registration to MNI (Rigid → Affine)
- **Script**: `reg.py`
- **Method**:
  - **Rigid**: Euler3D with Mattes Mutual Information, multi‑resolution (4–2–1), linear interp.
  - **Affine**: 12‑DOF, multi‑resolution (2–1), linear interp.
  - **Transforms**: Saved as `.mat` (SimpleITK). Final resampling uses a **composite** (rigid+affine) for single interpolation.
  - **T2 handling**: Applies T1 transforms to T2 from the same subject/session.
- **Template**: `MNI152NLin2009cAsym` at 1 mm via TemplateFlow.
- **Outputs** (per subject/session in `STAGE_ROOTS["registration"]`):
  - `*_mni_rigid_warped.nii.gz` (QC/ref)
  - `*_mni_warped.nii.gz` (final)
  - `other/*_rigid.mat`, `other/*_affine.mat`
- **Threads**: SimpleITK threads set from `CPU_CORES` in `config.py`.
- **CLI**: Supports `--subject` (optional filter).

### Stage 3 — Post‑processing (Mask Warp → N4 → Masked Z‑score → Crop → Viz)
- **Script**: `postprocess_mni_mask_zscore_crop.py`
- **Steps**:
  1. **Warp native masks to MNI** (nearest‑neighbor) using composite transform (rigid+affine).
  2. **N4 bias correction** (if `POSTPROCESS_APPLY_N4`) with the MNI‑space mask.
  3. **Masked Z‑score**: mean/std computed **inside the mask**; background kept at 0 by default.
  4. **Crop**: Tight bounding box around mask (+ `POSTPROCESS_CROP_MARGIN`), affine updated (qform/sform set).
  5. **Visualization**: Multi‑slice PNG montage per subject/session.
- **Outputs** (under each registration `anat/`):
  - `T1w_mni_mask.nii.gz`, `T1w_mni_zscore_fixed.nii.gz`, `T1w_mni_zscore_fixed_cropped.nii.gz`, `multislice_visualization_<subj>.png`
  - T2 equivalents when available
- **Fallbacks**: If T2 native mask is missing, T1’s MNI mask is reused for T2.

### Stage 4 — QC & Reports
- **Scripts**: `skullstrip.py` (QC images + volume check), `structural_qc.py`.
- **CSV**: `QC/qc_summary.csv` (columns: input_file, output_file, modality, qc_status, brain_volume_ml, qc_image_path, error_message).

### Stage 5 — CSV Metadata for ML
- **Script**: `generate_csv_metadata.py`
- **What**: Scans registration outputs, collects **MNI warped**, **Z‑score**, **cropped** paths; merges demographics from `participants.tsv`; encodes `sex` and `handedness` (mappings in `config.py`)(These demographics columns depend on the Dataset, any dataset may have its own demographics information).
- **Outputs**: `metadata/T1_metadata.csv`, `metadata/T2_metadata.csv` using `CSV_COLUMNS`.

---

## What Gets Produced? (Summary)

**Skullstrip (`STAGE_ROOTS["skullstrip"]`)**
- `*_brain.nii.gz`, `*_brain_mask.nii.gz`, `*_desc-qc.png`, global `QC/qc_summary.csv`

**Registration (`STAGE_ROOTS["registration"]`)**
- `*_mni_rigid_warped.nii.gz`, `*_mni_warped.nii.gz`, `other/*_rigid.mat`, `other/*_affine.mat`(+ T2)

**Post‑process (under same `anat/`)**
- `T1w_mni_mask.nii.gz`, `T1w_mni_zscore_fixed.nii.gz`, `T1w_mni_zscore_fixed_cropped.nii.gz`(+ T2), `multislice_visualization_<subj>.png` (+ T2)

**CSVs (`CSV_METADATA_DIR`)**
- `T1_metadata.csv`, `T2_metadata.csv`

---

## Unified Structure Resolver

- **File**: `structure_resolver.py`
- **APIs**:
  - `discover(STRUCTURE, root, subjects=None, sessions=None)` → yields `(subject, session, path)`
  - `make_path(STRUCTURE, root, "sub-XXXXX", "ses-YY")` → returns path string
- **Why**: Keep the same scripts portable across datasets by changing a single pattern in `config.py`.

---

## Troubleshooting

- **SynthStrip not found**: Ensure one of Singularity (`SYNTHSTRIP_SIF_PATH`), Docker, or FreeSurfer `mri_synthstrip` is available.
- **TemplateFlow cache errors**: set `TEMPLATEFLOW_HOME` to a writable directory and re‑run `reg.py`.
- **No sessions dataset**: change `STRUCTURE` to `"{root}/{subject}/anat"` and re‑run.
- **Orientation after crop**: post‑process writes qform/sform; check with `nib-ls` or `fslhd` if needed.
- **Threads/memory**: set `CPU_CORES` in `config.py`. SimpleITK threads respect this in `reg.py`.

---

## Privacy & Release Checklist

- ✅ Keep real paths only in **`config.py`** and **do not commit** it.
- ✅ Publish **`config.py`** with placeholders.
- ✅ Add a `.gitignore` (recommended):
  ```gitignore
  config.py
  *.sif
  */QC/*
  */logs/*
  ```

---

## Reproduce a Full Run (Example)

```bash
# 0) Environment
conda create -y -n smri python=3.10
conda activate smri
pip install numpy nibabel SimpleITK matplotlib tqdm templateflow pandas

# 1) Configure
cp config.example.py config.py
# edit config.py (private paths)

# 2) Brain extraction
python skullstrip.py

# 3) Registration
python reg.py

# 4) Post-process
python postprocess_mni_mask_zscore_crop.py

# 5) CSVs
python generate_csv_metadata.py
```

---

## Citation

Please consider citing the tools this pipeline builds upon:

- **SynthStrip** (FreeSurfer)
- **TemplateFlow**
- **SimpleITK**
- **N4 Bias Field Correction** (Tustison et al.)
- (And the dataset(s) you process, e.g., OpenNeuro accession.)

---

## Authors

Mohammad H Abbasi (mabbasi [at] stanford.edu)

---

## Acknowledgements

Thanks to the open‑source neuroimaging community (FreeSurfer, TemplateFlow, SimpleITK, nibabel) and dataset providers.
