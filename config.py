#!/path/to/REDACTED python
# -*- coding: utf-8 -*-
"""
Structural MRI Pipeline Configuration
Contains all settings for the structural MRI processing pipeline

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""

import os

# =============================================================================
# 1. GENERAL CONFIGURATION
# =============================================================================

# Dataset Information
DATASET_NAME = "OpenNeuro ds004215"

# Path Configuration
INPUT_DIR = "/path/to/REDACTED"
OUTPUT_DIR = "/path/to/REDACTED"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Hardware Configuration
CPU_CORES = 64

# Processing Configuration
MODALITIES = ["T1w", "T2w"]
SUBJECTS = None
SESSIONS = None
FORCE_REPROCESSING = True

# Parallel Processing
ENABLE_PARALLEL = True
MAX_WORKERS = 8

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# FreeSurfer Configuration
FREESURFER_HOME = "/path/to/REDACTED"
SYNTHSTRIP_BIN = os.path.join(FREESURFER_HOME, "bin", "mri_synthstrip")
SYNTHSTRIP_SIF_PATH = "/path/to/REDACTED"

# File Patterns
SKULLSTRIP_T1_FILE_PATTERN = "*acq-MPRAGE_T1w.nii.gz"
SKULLSTRIP_T2_FILE_PATTERN = "*acq-CUBE_T2w.nii.gz"

# =============================================================================
# 2. SKULLSTRIP CONFIGURATION
# =============================================================================

# Quality Control
ENABLE_QC = True
QC_GENERATE_IMAGES = True
QC_SUMMARY_FILE = "qc_summary.csv"
QC_BRAIN_VOLUME_MIN_ML = 800
QC_BRAIN_VOLUME_MAX_ML = 2000

# =============================================================================
# 3. REGISTRATION CONFIGURATION
# =============================================================================

# MNI Template
MNI_TEMPLATE_VERSION = "MNI152NLin2009cAsym"
MNI_TEMPLATE_RESOLUTION = 1

# =============================================================================
# 4. POSTPROCESS CONFIGURATION
# =============================================================================

# Processing Options
POSTPROCESS_APPLY_N4 = True
POSTPROCESS_CROP_MARGIN = 1

# =============================================================================
# 5. METADATA CONFIGURATION
# =============================================================================

# CSV Metadata Generation
GENERATE_CSV_METADATA = True
CSV_METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
T1_CSV_FILENAME = "T1_metadata.csv"
T2_CSV_FILENAME = "T2_metadata.csv"
PARTICIPANTS_TSV_PATH = "/path/to/REDACTED"

# CSV Columns
CSV_COLUMNS = [
    "subjectId",
    "session",
    "MNI_Warped",
    "MNI_ZSCORE", 
    "MNI_Z_Cropped",
    "age",
    "sex",
    "handedness"
]

# File Patterns for Processed Data
T1_PROCESSED_PATTERNS = {
    "MNI_Warped": "*T1w*_mni_warped.nii.gz",
    "MNI_ZSCORE": "T1w_mni_zscore_fixed.nii.gz",
    "MNI_Z_Cropped": "T1w_mni_zscore_fixed_cropped.nii.gz"
}

T2_PROCESSED_PATTERNS = {
    "MNI_Warped": "*T2w*_mni_warped.nii.gz",
    "MNI_ZSCORE": "T2w_mni_zscore_fixed.nii.gz",
    "MNI_Z_Cropped": "T2w_mni_zscore_fixed_cropped.nii.gz"
}

# Encoding Dictionaries
SEX_ENCODING = {
    "male": 1,
    "female": 2,
    "n/a": 0
}

HANDEDNESS_ENCODING = {
    "right": 1,
    "left": 2,
    "ambidextrous": 3,
    "n/a": 0
}

# Missing Data
MISSING_DATA_VALUE = "n/a"
REQUIRE_BOTH_MODALITIES = False
VALIDATE_FILE_EXISTENCE = True

# =============================================================================
# 6. QUALITY CONTROL CONFIGURATION
# =============================================================================

# Dice Coefficient Threshold
DICE_PASS_THRESHOLD = 0.8

# =============================================================================
# UNIFIED STRUCTURE CONFIGURATION
# =============================================================================

# Stage-specific root directories (only these differ between pipeline stages)
STAGE_ROOTS = {
    "skullstrip":   os.path.join(OUTPUT_DIR, "skullstrip"),
    "registration": os.path.join(OUTPUT_DIR, "registration"), 
    "qc":          os.path.join(OUTPUT_DIR, "QC"),
}

# Unified structure pattern - adapt this for different datasets:
# Examples:
#   OpenNeuro:     "{root}/path/to/REDACTED"
#   ADNI:          "{root}/path/to/REDACTED" 
#   ABCD:          "{root}/path/to/REDACTED"
#   No sessions:   "{root}/path/to/REDACTED"
STRUCTURE = "{root}/{modality}/{subject}/{session}/anat"

# Legacy aliases for backward compatibility
FORCE_REPROCESS = FORCE_REPROCESSING
QC_DIR = STAGE_ROOTS["qc"]
SKULLSTRIP_DIR = STAGE_ROOTS["skullstrip"] 
REGISTRATION_DIR = STAGE_ROOTS["registration"]
POSTPROCESS_MODALITIES = ["T1", "T2"]  # Different naming convention for postprocess
POSTPROCESS_SUBJECTS = SUBJECTS
