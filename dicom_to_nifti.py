#!/usr/bin/env python3
"""
DICOM → NIfTI conversion utility
Wrapper around dcm2niix (https://github.com/rordenlab/dcm2niix)

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""

import os
import argparse
import subprocess
from datetime import datetime
import shutil
import sys


def check_dcm2niix():
    """Ensure dcm2niix is available in PATH."""
    if not shutil.which("dcm2niix"):
        sys.stderr.write(
            "ERROR: dcm2niix not found in PATH.\n"
            "Please install from https://github.com/rordenlab/dcm2niix "
            "and ensure it is accessible in your shell.\n"
        )
        sys.exit(1)


def convert_dicom_to_nifti(dicom_dir, output_dir, subject=None, opts=None):
    """
    Convert a DICOM folder into NIfTI using dcm2niix.

    Args:
        dicom_dir (str): Path to input DICOM directory.
        output_dir (str): Path to save the NIfTI output.
        subject (str, optional): Subject ID (for logging).
        opts (dict, optional): Extra options controlling dcm2niix flags.
    """
    os.makedirs(output_dir, exist_ok=True)
    if opts is None:
        opts = {}

    # Build dcm2niix command
    cmd = ["dcm2niix"]

    # Compression
    if opts.get("no_compress", False):
        cmd += ["-z", "n"]  # no gzip
    else:
        cmd += ["-z", "y"]  # gzip .nii.gz

    # BIDS sidecar JSON
    if opts.get("no_bids", False):
        cmd += ["-b", "n"]
    else:
        cmd += ["-b", "y"]

    # Recursive
    if opts.get("no_recurse", False):
        cmd += ["-r", "n"]
    else:
        cmd += ["-r", "y"]

    # Crop
    if opts.get("crop", False):
        cmd += ["-x", "y"]

    # Merge
    if opts.get("no_merge", False):
        cmd += ["-m", "n"]
    else:
        cmd += ["-m", "y"]

    # Philips precise scaling
    if opts.get("no_philips_precise", False):
        cmd += ["-p", "n"]
    else:
        cmd += ["-p", "y"]

    # Overwrite
    if opts.get("overwrite", False):
        cmd += ["-w", "y"]
    else:
        cmd += ["-w", "n"]

    # Filename pattern
    fname_pattern = opts.get("fname", "%p_%s")
    cmd += ["-f", fname_pattern]

    # Protocol filter
    if opts.get("protocol_filter"):
        cmd += ["-d", opts["protocol_filter"]]

    # Output directory
    cmd += ["-o", output_dir]

    # Input dicom folder
    cmd += [dicom_dir]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running:", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"ERROR during dcm2niix execution: {e}\n")
        sys.stderr.write(e.stdout + "\n" + e.stderr)
        sys.exit(1)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Conversion finished.")
    if subject:
        print(f"Subject {subject} NIfTI saved in: {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert DICOM folder to NIfTI using dcm2niix")

    parser.add_argument("--dicom_dir", type=str, required=True, help="Input DICOM directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for NIfTI")
    parser.add_argument("--subject", type=str, default=None, help="Optional subject ID for logging")

    parser.add_argument("--no_bids", action="store_true", help="Disable BIDS sidecar JSON output")
    parser.add_argument("--no_recurse", action="store_true", help="Disable recursive scan of subfolders")
    parser.add_argument("--no_compress", action="store_true", help="Do not gzip output NIfTI")
    parser.add_argument("--crop", action="store_true", help="Enable crop of black borders")
    parser.add_argument("--no_merge", action="store_true", help="Disable merging of identical series")
    parser.add_argument("--no_philips_precise", action="store_true", help="Disable Philips precise scaling")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--fname", type=str, default="%p_%s", help="Filename pattern (default: %%p_%%s)")
    parser.add_argument("--protocol_filter", type=str, help="Only convert series matching this string")

    return parser.parse_args()


if __name__ == "__main__":
    check_dcm2niix()
    args = parse_arguments()
    opts = {
        "no_bids": args.no_bids,
        "no_recurse": args.no_recurse,
        "no_compress": args.no_compress,
        "crop": args.crop,
        "no_merge": args.no_merge,
        "no_philips_precise": args.no_philips_precise,
        "overwrite": args.overwrite,
        "fname": args.fname,
        "protocol_filter": args.protocol_filter,
    }
    convert_dicom_to_nifti(args.dicom_dir, args.output_dir, args.subject, opts)
