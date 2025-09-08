#!/usr/bin/env python3
"""
DICOM → NIfTI conversion utility
Uses dcm2niix (https://github.com/rordenlab/dcm2niix)

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""

import os
import argparse
import subprocess
from datetime import datetime

def convert_dicom_to_nifti(dicom_dir, output_dir, subject=None):
    """
    Convert a DICOM folder into NIfTI using dcm2niix.
    
    Args:
        dicom_dir (str): Path to input DICOM directory.
        output_dir (str): Path to save the NIfTI output.
        subject (str, optional): Subject ID (for naming).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # dcm2niix command
    cmd = [
        "dcm2niix",
        "-z", "y",               # compress to .nii.gz
        "-o", output_dir,        # output directory
        "-f", "%p_%s",           # filename: protocol_series
        dicom_dir                # input directory
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Conversion finished.")
    if subject:
        print(f"Subject {subject} NIfTI saved in: {output_dir}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert DICOM folder to NIfTI using dcm2niix")
    parser.add_argument("--dicom_dir", type=str, required=True, help="Input DICOM directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for NIfTI")
    parser.add_argument("--subject", type=str, default=None, help="Optional subject ID for logging")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    convert_dicom_to_nifti(args.dicom_dir, args.output_dir, args.subject)
