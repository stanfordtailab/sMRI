# Structural MRI Processing Pipeline

A lightweight, end‑to‑end pipeline for **brain extraction**, **registration to MNI**, **post‑processing (N4 + masked Z‑score + cropping)**, **QC visualization**, and **CSV metadata** generation.

The structure of this README mirrors the DTI project style: **Prerequisites → Configuration → Processing Steps (numbered) → Quick Start → Outputs/QC → Authors/Version**.

---

## Prerequisites

- **Python** ≥ 3.8
- Python packages: `numpy`, `nibabel`, `SimpleITK`, `matplotlib`, `tqdm`, `templateflow`, `pandas`
- **SynthStrip** via one of:
  - **Singularity** image: `freesurfer/synthstrip:latest` (recommended)
  - **Docker** image: `freesurfer/synthstrip:latest`
  - **FreeSurfer** installation with `mri_synthstrip` in `$FREESURFER_HOME/bin`
- (Optional) **TemplateFlow** cache (auto‑downloads `MNI152NLin2009cAsym` template)
- (Optional) **conda** for environment management

> Example env setup
```bash
conda create -y -n smri python=3.10
conda activate smri
pip install numpy nibabel SimpleITK matplotlib tqdm templateflow pandas
```

---

## Configuration

Paths and behavior are controlled by **`config.py`** (shared by all scripts). The scripts are primarily **config‑driven**; minimal CLI flags are provided for convenience.

> **Privacy‑safe release**: publish `config.example.py` with placeholders; keep your real `config.py` untracked.

**Key items in `config.py`:**
- Paths: `INPUT_DIR`, `OUTPUT_DIR`, `STAGE_ROOTS`
- Unified structure pattern: `STRUCTURE = "{root}/{subject}/{session}/anat"`
- Patterns: `SKULLSTRIP_T1_FILE_PATTERN`, `SKULLSTRIP_T2_FILE_PATTERN`
- Processing: `MODALITIES`, `SUBJECTS`, `SESSIONS`, `FORCE_REPROCESSING`
- QC thresholds, logging, CPU threads
- Registration: `MNI_TEMPLATE_VERSION`, `MNI_TEMPLATE_RESOLUTION`
- Post‑process: `POSTPROCESS_APPLY_N4`, `POSTPROCESS_CROP_MARGIN`
- CSV metadata: output dir, columns, demographics encoders

---

## Processing Steps

### 1) Brain Extraction — `skullstrip.py`
Removes non‑brain tissue using **SynthStrip**. The script tries **Singularity → Docker → FreeSurfer binary** (in this order). Generates brain, mask, and QC images.

**Inputs**
- NIfTI files discovered via `STRUCTURE` and patterns:
  - `*acq-MPRAGE_T1w.nii.gz`
  - `*acq-CUBE_T2w.nii.gz`

**Outputs (per subject/session in `STAGE_ROOTS["skullstrip"]`)**
- `*_brain.nii.gz`, `*_brain_mask.nii.gz`
- `*_desc-qc.png` overlay montage
- Global CSV: `QC/qc_summary.csv` with brain volume (ml) and PASS/FAIL

**CLI**
```bash
python skullstrip.py                  # process all
python skullstrip.py --subject sub-ON12345
```

---

### 2) Registration to MNI (Rigid → Affine) — `reg.py`
Registers **T1w** to **MNI152NLin2009cAsym (1mm)** using **SimpleITK** (Mattes MI; multi‑resolution). Applies the **same transforms** to **T2w** of the same session.

**Method**
- Rigid: Euler3D, shrink [4,2,1], linear
- Affine: 12‑DOF, shrink [2,1], linear
- Final resampling: **composite transform** (rigid+affine) for single interpolation
- Saves `*.mat` transforms

**Outputs (per subject/session in `STAGE_ROOTS["registration"]`)**
- `*_mni_rigid_warped.nii.gz` (QC/ref)
- `*_mni_warped.nii.gz` (final)
- `other/*_rigid.mat`, `other/*_affine.mat`

**CLI**
```bash
python reg.py
python reg.py --subject sub-ON12345
```

---

### 3) Post‑processing — `postprocess_mni_mask_zscore_crop.py`
Warp masks to MNI, run **N4 bias correction** (optional), compute **masked Z‑score**, **crop** around the brain, and save a **multi‑slice visualization**.

**Steps**
1. Warp native masks → MNI with **nearest‑neighbor** using composite transform (rigid+affine)
2. N4 bias correction (if `POSTPROCESS_APPLY_N4 = True`)
3. Masked Z‑score: μ/σ computed **within mask**; background kept at 0
4. Crop: tight bounding box (+ `POSTPROCESS_CROP_MARGIN`), affine updated (qform/sform set)
5. Save multi‑slice PNG montage

**Outputs (per subject/session; under the same registration `anat/`)**
- `T1w_mni_mask.nii.gz`
- `T1w_mni_warped_n4.nii.gz` (if N4 enabled)
- `T1w_mni_zscore_fixed.nii.gz`
- `T1w_mni_zscore_fixed_cropped.nii.gz`
- `multislice_visualization_<subj>.png`
- (T2 equivalents when available; falls back to T1 mask if T2 mask missing)

**CLI**
```bash
python postprocess_mni_mask_zscore_crop.py
python postprocess_mni_mask_zscore_crop.py --subject sub-ON12345
```

---

### 4) Quality Control & Reports — `skullstrip.py`, `structural_qc.py`
- `skullstrip.py` writes **per‑subject QC montages** and a global **QC CSV** with brain volume.
- `structural_qc.py` (optional helpers) can be used for extended validations (e.g., Dice vs reference).

**Outputs**
- `QC/qc_summary.csv` (columns: input_file, output_file, modality, qc_status, brain_volume_ml, qc_image_path, error_message)

---

### 5) CSV Metadata — `generate_csv_metadata.py`
Scans registration outputs and builds CSVs listing **MNI warped**, **Z‑score**, and **cropped** paths, with optional demographics (`participants.tsv`) and encodings (`sex`, `handedness`).

**Outputs**
- `metadata/T1_metadata.csv`
- `metadata/T2_metadata.csv`

**CLI**
```bash
python generate_csv_metadata.py
```

---

## Quick Start

```bash
# 0) (optional) create environment
conda create -y -n smri python=3.10
conda activate smri
pip install numpy nibabel SimpleITK matplotlib tqdm templateflow pandas

# 1) configure
cp config.example.py config.py
# edit config.py (private paths, thresholds)

# 2) brain extraction
python skullstrip.py

# 3) registration
python reg.py

# 4) post-process
python postprocess_mni_mask_zscore_crop.py

# 5) metadata tables
python generate_csv_metadata.py
```

---

## Outputs (Where to Find Things)

- **Skullstrip** → `STAGE_ROOTS["skullstrip"]`  
  `*_brain.nii.gz`, `*_brain_mask.nii.gz`, `*_desc-qc.png`, global `QC/qc_summary.csv`

- **Registration** → `STAGE_ROOTS["registration"]`  
  `*_mni_rigid_warped.nii.gz`, `*_mni_warped.nii.gz`, `other/*_rigid.mat`, `other/*_affine.mat`

- **Post‑process** → same `anat/` under `registration`  
  `T1w_mni_mask.nii.gz`, `T1w_mni_zscore_fixed.nii.gz`, `T1w_mni_zscore_fixed_cropped.nii.gz`, `multislice_visualization_<subj>.png` (and T2 analogs)

- **CSV metadata** → `CSV_METADATA_DIR`  
  `T1_metadata.csv`, `T2_metadata.csv`

---

## Authors / Version

- **Author**: Mohammad H. Abbasi (mabbasi [at] stanford.edu)  
- **Lab**: Stanford University, STAI Lab (https://stai.stanford.edu)  
- **Created**: 2025 &nbsp;|&nbsp; **Version**: 1.0.0

---

## Acknowledgements

Thanks to the open‑source neuroimaging community (FreeSurfer, TemplateFlow, SimpleITK, nibabel) and dataset providers.

---
