#!/usr/bin/env python3

"""
Structural MRI Post-processing Pipeline

This script performs post-processing of registered MRI images including:
- Warping brain masks from native space to MNI space
- N4 bias field correction for improved image quality
- Z-score normalization with masked brain regions
- Automatic cropping to remove non-brain voxels
- Multi-slice visualization for quality control

Input: Registered T1w/T2w images and brain masks from previous pipeline stages
Output: Preprocessed, normalized, and cropped images ready for analysis

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""
import os
import glob
import argparse
import logging
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Read configuration and structure resolver
from config import (
    REGISTRATION_DIR, SKULLSTRIP_DIR, POSTPROCESS_MODALITIES, 
    POSTPROCESS_SUBJECTS, POSTPROCESS_APPLY_N4, POSTPROCESS_CROP_MARGIN, 
    LOG_DIR, LOG_LEVEL, LOG_FORMAT,
    # Unified structure
    STAGE_ROOTS, STRUCTURE
)
from structure_resolver import discover, make_path

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, f'postprocess_{timestamp}.log'))
    ]
)
logger = logging.getLogger('postprocess')


def discover_session_dirs(root_dir: str, subj: str) -> List[str]:
    """Discover session directories for a subject. Returns list of session dirs or [subj_root] if no sessions."""
    subj_root = os.path.join(root_dir, subj)
    if not os.path.exists(subj_root):
        return []
    
    ses_dirs = sorted(glob.glob(os.path.join(subj_root, "ses-*")))
    return ses_dirs if ses_dirs else [subj_root]  # Return subject root if no sessions


def extract_session_from_dir(session_dir: str) -> Optional[str]:
    """Extract session ID from session directory path."""
    session_name = os.path.basename(session_dir)
    if session_name.startswith('ses-'):
        return session_name
    return None  # No session (single-session dataset)


def warp_mask_to_mni(mask_native_path: str,
                     rigid_transform_path: str,
                     affine_transform_path: str,
                     reference_image_path: str,
                     out_mask_path: str) -> Optional[str]:
    try:
        native_mask = sitk.ReadImage(mask_native_path, sitk.sitkUInt8)
        reference_image = sitk.ReadImage(reference_image_path, sitk.sitkFloat32)

        rigid_transform = sitk.ReadTransform(rigid_transform_path)
        affine_transform = sitk.ReadTransform(affine_transform_path)

        # Combine transforms into single composite (better quality)
        combo_transform = sitk.CompositeTransform(3)
        combo_transform.AddTransform(rigid_transform)
        combo_transform.AddTransform(affine_transform)

        # Single resample with combined transform
        final_resampler = sitk.ResampleImageFilter()
        final_resampler.SetReferenceImage(reference_image)
        final_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        final_resampler.SetTransform(combo_transform)
        final_mask_mni = final_resampler.Execute(native_mask)

        final_mask_mni = sitk.Cast(final_mask_mni > 0, sitk.sitkUInt8)
        sitk.WriteImage(final_mask_mni, out_mask_path)
        print(f"Saved MNI-space mask to {out_mask_path}")
        return out_mask_path
    except Exception as e:
        print(f"ERROR warping mask: {e}")
        return None


def zscore_normalization_masked(in_file: str, mask_file: str, out_file: str, keep_background_zero: bool = True) -> Optional[str]:
    try:
        img = nib.load(in_file)
        data = img.get_fdata(dtype=np.float32)
        mask = nib.load(mask_file).get_fdata(dtype=np.float32) > 0

        if np.sum(mask) == 0:
            raise ValueError("Mask is empty")

        # Make Z-score robust to NaN/Inf values
        vals = data[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or np.std(vals) == 0:
            print("WARNING: No finite values or std=0 inside mask; setting masked region to zero")
            z = np.zeros_like(data)
            z[mask] = 0.0  # Keep background and masked region both zero
        else:
            mu = float(vals.mean())
            sd = float(vals.std())
            z = np.zeros_like(data)
            z[mask] = (data[mask] - mu) / sd
            if not keep_background_zero:
                z = (data - mu) / sd

        nii = nib.Nifti1Image(z.astype(np.float32), img.affine, img.header)
        if hasattr(nii, "set_qform"):
            nii.set_qform(img.affine, code=1)
            nii.set_sform(img.affine, code=1)
        nib.save(nii, out_file)
        print(f"Saved masked Z-score image to {out_file}")
        return out_file
    except Exception as e:
        print(f"ERROR masked z-score: {e}")
        return None


def n4_bias_correction(in_file: str, mask_file: Optional[str], out_file: str) -> Optional[str]:
    try:
        im = sitk.ReadImage(in_file, sitk.sitkFloat32)
        if mask_file and os.path.exists(mask_file):
            m = sitk.ReadImage(mask_file, sitk.sitkUInt8)
        else:
            m = sitk.Cast(im > 0, sitk.sitkUInt8)
        corrected = sitk.N4BiasFieldCorrection(im, m)
        sitk.WriteImage(corrected, out_file)
        print(f"Saved N4 corrected image to {out_file}")
        return out_file
    except Exception as e:
        print(f"ERROR N4: {e}")
        return None


def crop_with_mask(in_file: str, mask_file: str, out_file: str, margin: int = 1) -> Tuple[bool, Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]]:
    try:
        img = nib.load(in_file)
        data = img.get_fdata(dtype=np.float32)
        mask = nib.load(mask_file).get_fdata(dtype=np.float32) > 0
        if np.sum(mask) == 0:
            print("ERROR: Empty mask; cannot crop")
            return False, None

        indices = np.where(mask)
        min_x, max_x = max(0, int(np.min(indices[0]) - margin)), min(mask.shape[0], int(np.max(indices[0]) + margin + 1))
        min_y, max_y = max(0, int(np.min(indices[1]) - margin)), min(mask.shape[1], int(np.max(indices[1]) + margin + 1))
        min_z, max_z = max(0, int(np.min(indices[2]) - margin)), min(mask.shape[2], int(np.max(indices[2]) + margin + 1))

        cropped = data[min_x:max_x, min_y:max_y, min_z:max_z]

        new_affine = img.affine.copy()
        shift = np.array([min_x, min_y, min_z], dtype=float)
        new_affine[:3, 3] = new_affine[:3, 3] + new_affine[:3, :3].dot(shift)

        nii = nib.Nifti1Image(cropped.astype(np.float32), new_affine, img.header)
        if hasattr(nii, "set_qform"):
            nii.set_qform(new_affine, code=1)
            nii.set_sform(new_affine, code=1)
        nib.save(nii, out_file)
        print(f"Saved cropped image to {out_file} | shape {cropped.shape}")
        bbox = ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        return True, bbox
    except Exception as e:
        print(f"ERROR crop: {e}")
        return False, None


def find_native_masks(mask_root: str, subj: str, session_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Find native-space T1/T2 brain masks for a subject in skullstrip directory."""
    # session_dir is already the full path to skullstrip anat directory
    anat_mask_dir = session_dir if session_dir else None
    
    if not anat_mask_dir or not os.path.exists(anat_mask_dir):
        return None, None
    
    t1_mask = None
    t2_mask = None
    
    if os.path.exists(anat_mask_dir):
        # Find T1 mask
        t1_pattern = os.path.join(anat_mask_dir, "*T1w*_brain_mask.nii.gz")
        t1_files = glob.glob(t1_pattern)
        if t1_files:
            t1_mask = t1_files[0]
        
        # Find T2 mask  
        t2_pattern = os.path.join(anat_mask_dir, "*T2w*_brain_mask.nii.gz")
        t2_files = glob.glob(t2_pattern)
        if t2_files:
            t2_mask = t2_files[0]
    
    return t1_mask, t2_mask


def create_visualization(subj: str, anat_dir: str):
    """Create comprehensive multi-slice visualization of postprocess results."""
    try:
        # File paths - use T1 cropped file as main reference
        t1_cropped_file = os.path.join(anat_dir, "T1w_mni_zscore_fixed_cropped.nii.gz")
        if not os.path.exists(t1_cropped_file):
            print("WARNING: T1 cropped file not found for visualization")
            return
            
        # Load the processed data
        img = nib.load(t1_cropped_file)
        data = img.get_fdata(dtype=np.float32)
        
        # Define vmin/vmax for consistent visualization
        vmin = -3
        vmax = 3
        
        # Define slices to show based on actual image dimensions
        # Create 5 evenly spaced slices across each dimension
        slice_indices = []
        
        # For Z dimension (axial)
        z_max = data.shape[2]
        if z_max > 5:
            z_slices = [int(z_max * i / 6) for i in range(1, 6)]  # Skip first (0) and last
        else:
            z_slices = list(range(z_max))
        
        # For X dimension (sagittal) 
        x_max = data.shape[0]
        if x_max > 5:
            x_slices = [int(x_max * i / 6) for i in range(1, 6)]
        else:
            x_slices = list(range(x_max))
            
        # For Y dimension (coronal)
        y_max = data.shape[1] 
        if y_max > 5:
            y_slices = [int(y_max * i / 6) for i in range(1, 6)]
        else:
            y_slices = list(range(y_max))
        
        # Use the same slice indices for all views (based on smallest dimension)
        num_slices = min(5, min(len(z_slices), len(x_slices), len(y_slices)))
        slice_indices = z_slices[:num_slices]
        
        # Create the plot
        fig, axes = plt.subplots(3, len(slice_indices), figsize=(15, 8))
        fig.suptitle(f'Multi-slice Analysis - {subj} (T1w_mni_zscore_fixed_cropped)', fontsize=14)
        
        # Axial (XY plane, vary z)
        for i, s in enumerate(z_slices[:len(slice_indices)]):
            slice_data = data[:, :, s]
            # Use masked array for proper black background
            ma_slice = np.ma.masked_equal(slice_data, 0)
            cmap = plt.cm.gray.copy()
            cmap.set_bad('black')
            axes[0, i].imshow(np.rot90(ma_slice), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f'Axial z={s}')
            axes[0, i].axis('off')
        
        # Sagittal (YZ plane, vary x)  
        for i, s in enumerate(x_slices[:len(slice_indices)]):
            slice_data = data[s, :, :]
            # Use masked array for proper black background
            ma_slice = np.ma.masked_equal(slice_data, 0)
            cmap = plt.cm.gray.copy()
            cmap.set_bad('black')
            axes[1, i].imshow(np.rot90(ma_slice), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f'Sagittal x={s}')
            axes[1, i].axis('off')
        
        # Coronal (XZ plane, vary y)
        for i, s in enumerate(y_slices[:len(slice_indices)]):
            slice_data = data[:, s, :]
            # Use masked array for proper black background
            ma_slice = np.ma.masked_equal(slice_data, 0)
            cmap = plt.cm.gray.copy()
            cmap.set_bad('black')
            axes[2, i].imshow(np.rot90(ma_slice), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[2, i].set_title(f'Coronal y={s}')
            axes[2, i].axis('off')
        
        # Add statistics using MNI mask for more accurate brain-only stats
        t1_mask_file = os.path.join(anat_dir, "T1w_mni_mask.nii.gz")
        if os.path.exists(t1_mask_file):
            try:
                mask_img = nib.load(t1_mask_file)
                # Resample mask to match cropped data if needed
                if mask_img.shape != data.shape:
                    brain_mask = data > 0  # Fallback to intensity-based mask
                else:
                    brain_mask = mask_img.get_fdata(dtype=np.float32) > 0
            except:
                brain_mask = data > 0  # Fallback
        else:
            brain_mask = data > 0  # Fallback to intensity-based mask
            
        brain_values = data[brain_mask]
        if len(brain_values) > 0:
            stats_text = f'Shape: {data.shape} | Brain voxels: {len(brain_values)} | Z-score: μ={brain_values.mean():.3f}, σ={brain_values.std():.3f} | Range: [{brain_values.min():.2f}, {brain_values.max():.2f}]'
        else:
            stats_text = f'Shape: {data.shape} | No brain data found'
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(anat_dir, f"multislice_visualization_{subj}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Multi-slice visualization saved: {viz_path}")
        
    except Exception as e:
        print(f"WARNING: Could not create multi-slice visualization: {e}")


def main():
    # Parse command line for optional subject filter
    parser = argparse.ArgumentParser(description="Post-process registration outputs: warp masks to MNI, masked Z-score, and crop.")
    parser.add_argument("--subject", type=str, help="Process only this specific subject (e.g., sub-ON12345)")
    args = parser.parse_args()

    # Load all paths and settings from config
    apply_n4 = POSTPROCESS_APPLY_N4
    crop_margin = int(POSTPROCESS_CROP_MARGIN)
    modalities = POSTPROCESS_MODALITIES
    
    # Subject filter: CLI arg overrides config
    subjects_filter = None
    if args.subject:
        subjects_filter = [args.subject]
        print(f"Processing only subject: {args.subject}")
    elif POSTPROCESS_SUBJECTS:
        subjects_filter = POSTPROCESS_SUBJECTS
        print(f"Processing subjects from config: {POSTPROCESS_SUBJECTS}")
    else:
        print("Processing all subjects")

    # Use unified structure to discover registration outputs
    triples = list(discover(STRUCTURE, STAGE_ROOTS["registration"], subjects=subjects_filter, sessions=None))
    
    if not triples:
        print(f"No registration outputs found matching STRUCTURE in {STAGE_ROOTS['registration']}")
        return

    print(f"Found {len(triples)} subject-session pairs to process")

    for subj, session, reg_anat_dir in triples:
        print(f"\n=== Subject {subj}, Session {session} ===")
        
        # reg_anat_dir is the registration output directory for this subject/session
        anat_dir = reg_anat_dir
        other_dir = os.path.join(anat_dir, "other")
        
        # Transform files are in other/ subdirectory
        rigid_mat = next(iter(glob.glob(os.path.join(other_dir, "*T1w*rigid.mat"))), None)
        affine_mat = next(iter(glob.glob(os.path.join(other_dir, "*T1w*affine.mat"))), None)
        
        if not (rigid_mat and affine_mat and os.path.exists(rigid_mat) and os.path.exists(affine_mat)):
            print("Missing transforms; skipping session")
            continue

        # Find warped files
        t1_warped = next(iter(glob.glob(os.path.join(anat_dir, "*T1w*mni_warped.nii.gz"))), None)
        t2_warped = next(iter(glob.glob(os.path.join(anat_dir, "*T2w*mni_warped.nii.gz"))), None)

        # Find native masks using unified structure
        skullstrip_anat_dir = make_path(STRUCTURE, STAGE_ROOTS["skullstrip"], subj, session)
        t1_native_mask, t2_native_mask = find_native_masks(STAGE_ROOTS["skullstrip"], subj, session_dir=skullstrip_anat_dir)

        # Warp masks to MNI
        t1_mni_mask = None
        t2_mni_mask = None
        if t1_native_mask and t1_warped:
            t1_mni_mask = os.path.join(anat_dir, "T1w_mni_mask.nii.gz")
            warp_mask_to_mni(t1_native_mask, rigid_mat, affine_mat, t1_warped, t1_mni_mask)
        else:
            print("WARNING: No native T1 mask or warped T1 found")

        if t2_native_mask and t2_warped:
            t2_mni_mask = os.path.join(anat_dir, "T2w_mni_mask.nii.gz")
            warp_mask_to_mni(t2_native_mask, rigid_mat, affine_mat, t2_warped, t2_mni_mask)
        else:
            if t1_mni_mask:
                t2_mni_mask = t1_mni_mask
                print("INFO: Using T1 MNI mask for T2")
            else:
                print("WARNING: No mask available for T2")

        # T1 post-processing
        if "T1" in modalities and t1_warped and t1_mni_mask and os.path.exists(t1_mni_mask):
            z_input = t1_warped
            if apply_n4:
                z_input = os.path.join(anat_dir, "T1w_mni_warped_n4.nii.gz")
                n4_bias_correction(t1_warped, t1_mni_mask, z_input)

            t1_z_fixed = os.path.join(anat_dir, "T1w_mni_zscore_fixed.nii.gz")
            zscore_normalization_masked(z_input, t1_mni_mask, t1_z_fixed, keep_background_zero=True)

            t1_z_fixed_cropped = os.path.join(anat_dir, "T1w_mni_zscore_fixed_cropped.nii.gz")
            crop_with_mask(t1_z_fixed, t1_mni_mask, t1_z_fixed_cropped, margin=crop_margin)
        elif "T1" in modalities:
            print("WARNING: Missing T1 warped image or mask; skipping T1")

        # T2 post-processing
        if "T2" in modalities and t2_warped and t2_mni_mask and os.path.exists(t2_mni_mask):
            z_input = t2_warped
            if apply_n4:
                z_input = os.path.join(anat_dir, "T2w_mni_warped_n4.nii.gz")
                n4_bias_correction(t2_warped, t2_mni_mask, z_input)

            t2_z_fixed = os.path.join(anat_dir, "T2w_mni_zscore_fixed.nii.gz")
            zscore_normalization_masked(z_input, t2_mni_mask, t2_z_fixed, keep_background_zero=True)

            t2_z_fixed_cropped = os.path.join(anat_dir, "T2w_mni_zscore_fixed_cropped.nii.gz")
            crop_with_mask(t2_z_fixed, t2_mni_mask, t2_z_fixed_cropped, margin=crop_margin)
        elif "T2" in modalities:
            print("WARNING: Missing T2 warped image or mask; skipping T2")
        
        # Create visualization
        create_visualization(subj, anat_dir)


if __name__ == "__main__":
    main()


