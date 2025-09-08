#!/usr/bin/env python3

"""
Structural MRI Registration Pipeline

This script performs registration of T1w and T2w images to MNI space using SimpleITK.
It performs a two-stage registration (rigid + affine) and saves the warped images 
and transformation matrices.

Note: Z-score normalization and cropping are handled by postprocess_mni_mask_zscore_crop.py

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""

import os
import glob
import nibabel as nib
import numpy as np
import json
from datetime import datetime
import time
import logging
import SimpleITK as sitk
import argparse

from templateflow import api as tflow

# Import configuration and structure resolver
from config import *
from structure_resolver import discover, make_path

# Configure SimpleITK threading for better performance on HPC
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(min(CPU_CORES, os.cpu_count()))

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, f'registration_{timestamp}.log'))
    ]
)
logger = logging.getLogger('registration')

# ----------------------------
# 1. Settings and Directories
# ----------------------------

def parse_arguments():
    """Parse command line arguments for the registration script."""
    parser = argparse.ArgumentParser(
        description='Register T1w and T2w brain images to MNI space using two-stage registration (rigid + affine).'
    )
    parser.add_argument('--subject', type=str, default=None,
                        help='Process only a specific subject (e.g., ON93426 or sub-ON93426)')
    return parser.parse_args()

# Note: Subject and session extraction functions removed - using discover() instead

# Parse command line arguments
args = parse_arguments()

# Use unified structure to discover brain-extracted images from skullstrip directory
logger.info(f"Using brain-extracted images from skullstrip")
logger.info(f"Discovering files using STRUCTURE: {STRUCTURE}")

# Log package versions for reproducibility
logger.info(f"SimpleITK version: {sitk.Version.VersionString()}")
logger.info(f"SimpleITK thread count: {sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()}")

# Discover all (subject, session, base_path) combinations from skullstrip output
triples = list(discover(STRUCTURE, STAGE_ROOTS["skullstrip"], subjects=SUBJECTS, sessions=SESSIONS))
logger.info(f"Found {len(triples)} (subject, session, base_path) combinations")

# Find T1 and T2 brain-extracted files using discovered paths
t1_files, t2_files = [], []
for (subj, sess, base) in triples:
    # Find T1 brain files in this base directory
    t1_pattern = os.path.join(base, SKULLSTRIP_T1_FILE_PATTERN.replace(".nii.gz", "_brain.nii.gz"))
    t1_files.extend(glob.glob(t1_pattern))
    
    # Find T2 brain files in this base directory  
    t2_pattern = os.path.join(base, SKULLSTRIP_T2_FILE_PATTERN.replace(".nii.gz", "_brain.nii.gz"))
    t2_files.extend(glob.glob(t2_pattern))
    
# Note: cropping is now handled by postprocess_mni_mask_zscore_crop.py

# Set output directory from config
output_base_dir = STAGE_ROOTS["registration"]
os.makedirs(output_base_dir, exist_ok=True)
logger.info(f"Output directory set to: {output_base_dir}")

logger.info(f"Number of T1 files found: {len(t1_files)}")
for f in t1_files:
    logger.debug(f"T1 file: {f}")

logger.info(f"Number of T2 files found: {len(t2_files)}")
for f in t2_files:
    logger.debug(f"T2 file: {f}")

# Use the same triples we already discovered above
expected_pairs = [(subj, sess) for (subj, sess, base) in triples]
logger.info(f"Found {len(expected_pairs)} expected (subject, session) pairs")

# Create mapping of files to (subject, session) pairs based on discovered triples
file_to_pair = {}
for (subj, sess, base) in triples:
    t1_pattern = os.path.join(base, SKULLSTRIP_T1_FILE_PATTERN.replace(".nii.gz", "_brain.nii.gz"))
    for t1_file in glob.glob(t1_pattern):
        file_to_pair[t1_file] = (subj, sess)

# Find (subject, session) pairs with available T1 files
found_t1_pairs = set(file_to_pair.values())
missing_pairs = [(subj, sess) for (subj, sess) in expected_pairs if (subj, sess) not in found_t1_pairs]
logger.info(f"{len(missing_pairs)} (subject, session) pairs are missing T1 files")
for subj, sess in missing_pairs[:10]:  # Show only first 10 to avoid spam
    logger.debug(f"Missing T1: subject {subj}, session {sess}")
if len(missing_pairs) > 10:
    logger.debug(f"... and {len(missing_pairs) - 10} more missing pairs")

# For backward compatibility, keep subject-level list
expected_subjects = sorted({s for (s, sess) in expected_pairs})
missing_subjects = [s for s in expected_subjects if s not in {s for (s, sess) in found_t1_pairs}]

# ----------------------------
# 2. Download Template
# ----------------------------

# Download the skull-stripped MNI T1 template (using version from config)
try:
    template_T1_path = tflow.get(MNI_TEMPLATE_VERSION, resolution=MNI_TEMPLATE_RESOLUTION, suffix="T1w", desc="brain")
except Exception:
    # Fallback if desc="brain" not available
    template_T1_path = tflow.get(MNI_TEMPLATE_VERSION, resolution=MNI_TEMPLATE_RESOLUTION, suffix="T1w")
template_T1 = str(template_T1_path)  # Convert PosixPath to string for SimpleITK
print("DEBUG: MNI T1 template downloaded to:", template_T1)
if os.path.exists(template_T1):
    print("DEBUG: Verified that the template file exists.")

# ----------------------------
# 3. Utility Functions
# ----------------------------


# Add a results tracking dictionary
registration_results = {
    'successful_subjects': [],
    'failed_subjects': [],
    'missing_subjects': [],  # New category for subjects with missing files
    'errors': {},
    'processing_times': {},  # per-subject processing time
    'cropping_results': {},  # cropping results
    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'end_time': None,
    'total_subjects': 0,
    'success_rate': 0,
    'average_processing_time': 0  # average processing time
}

def save_registration_report(results, output_dir):
    """Save registration results to a JSON file."""
    report_path = os.path.join(output_dir, 'registration_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Detailed registration report saved to: {report_path}")

def format_time(seconds):
    """Convert seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def print_registration_summary(results):
    """Print a summary of registration results."""
    print("\n=== Registration Summary ===")
    print(f"Total expected subjects: {results['total_subjects']}")
    print(f"Successfully processed: {len(results['successful_subjects'])}")
    print(f"Failed during processing: {len(results['failed_subjects'])}")
    print(f"Missing input files: {len(results['missing_subjects'])}")
    total_issues = len(results['failed_subjects']) + len(results['missing_subjects'])
    if results['total_subjects'] > 0:
        print(f"Success rate: {(len(results['successful_subjects']) / results['total_subjects']) * 100:.2f}%")
        print(f"Total fail rate: {(total_issues / results['total_subjects']) * 100:.2f}%")
    else:
        print("Success rate: N/A (no subjects processed)")
        print("Total fail rate: N/A (no subjects processed)")
    
    if results['processing_times']:
        avg_time = results['average_processing_time']
        print(f"\nTiming Information:")
        print(f"Average processing time per subject: {format_time(avg_time)}")
        print("\nProcessing times per subject:")
        for subj, t in results['processing_times'].items():
            print(f"- Subject {subj}: {format_time(t)}")
    
    if results['failed_subjects']:
        print("\nFailed subjects (processing errors):")
        for subject, error in results['errors'].items():
            print(f"- {subject}: {error}")
    
    if results['missing_subjects'] and len(results['missing_subjects']) < 20:
        print("\nMissing subjects (input files not found):")
        for subject in results['missing_subjects']:
            print(f"- {subject}")
    elif results['missing_subjects']:
        print(f"\n{len(results['missing_subjects'])} subjects missing input files (details in log)")

def register_image(fixed, moving, output_prefix, subject_id=None, base_name=None, other_dir=None):
    """
    Run a two-stage registration with SimpleITK:
    1. Rigid registration (rotation + translation)
    2. Affine registration (using rigid result as initial position)
    """
    try:
        output_dir = os.path.dirname(output_prefix)
        if not os.path.exists(output_dir):
            logger.debug(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Set default values if not provided
        if base_name is None:
            base_name = os.path.basename(moving).replace("_brain.nii.gz", "")
        if other_dir is None:
            other_dir = os.path.join(output_dir, "other")
        os.makedirs(other_dir, exist_ok=True)
        
        logger.debug(f"Loading images for registration")
        # Load images using SimpleITK
        fixed_image = sitk.ReadImage(fixed, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving, sitk.sitkFloat32)
        
        logger.debug(f"Fixed image size: {fixed_image.GetSize()}, dimension: {fixed_image.GetDimension()}")
        logger.debug(f"Moving image size: {moving_image.GetSize()}, dimension: {moving_image.GetDimension()}")
        
        # Ensure both images are 3D
        if fixed_image.GetDimension() != 3 or moving_image.GetDimension() != 3:
            raise ValueError("Both images must be 3D")
            
        # STAGE 1: Rigid Registration
        logger.debug("Starting Rigid registration (Stage 1)")
        
        # Initialize registration method for Rigid
        rigid_registration = sitk.ImageRegistrationMethod()
        
        # Set up similarity metric for Rigid
        rigid_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        rigid_registration.SetMetricSamplingStrategy(rigid_registration.RANDOM)
        rigid_registration.SetMetricSamplingPercentage(0.25)
        # Note: SetMetricSamplingPercentageRandomSeed not available in SimpleITK 2.1.0
        
        # Set up interpolator for Rigid
        rigid_registration.SetInterpolator(sitk.sitkLinear)
        
        # Set up optimizer for Rigid
        rigid_registration.SetOptimizerAsGradientDescent(learningRate=0.1,
                                                       numberOfIterations=1000,
                                                       convergenceMinimumValue=1e-6,
                                                       convergenceWindowSize=10)
        rigid_registration.SetOptimizerScalesFromPhysicalShift()
        
        # Set up Rigid transform
        rigid_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
        rigid_registration.SetInitialTransform(rigid_transform)
        
        # Multi-resolution framework for Rigid
        rigid_registration.SetShrinkFactorsPerLevel([4, 2, 1])
        rigid_registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        
        try:
            # Execute Rigid registration
            final_rigid_transform = rigid_registration.Execute(fixed_image, moving_image)
            logger.debug("Rigid registration completed successfully")
            
            # Save Rigid transform for reference
            rigid_transform_path = os.path.join(other_dir, f"{base_name}_rigid.mat")
            sitk.WriteTransform(final_rigid_transform, rigid_transform_path)
            
            # Apply Rigid transform to get intermediate result
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(final_rigid_transform)
            
            rigid_moved_image = resampler.Execute(moving_image)
            
            # Save intermediate Rigid-registered image for reference
            rigid_warped_path = os.path.join(os.path.dirname(output_prefix), f"{base_name}_mni_rigid_warped.nii.gz")
            sitk.WriteImage(rigid_moved_image, rigid_warped_path)
            logger.debug(f"Saved Rigid-registered intermediate image to {rigid_warped_path}")
            
            # STAGE 2: Affine Registration (starting from Rigid result)
            logger.debug("Starting Affine registration (Stage 2) using Rigid result")
            
            # Initialize registration method for Affine
            affine_registration = sitk.ImageRegistrationMethod()
            
            # Set up similarity metric for Affine
            affine_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            affine_registration.SetMetricSamplingStrategy(affine_registration.RANDOM)
            affine_registration.SetMetricSamplingPercentage(0.25)
            # Note: SetMetricSamplingPercentageRandomSeed not available in SimpleITK 2.1.0
            
            # Set up interpolator for Affine
            affine_registration.SetInterpolator(sitk.sitkLinear)
            
            # Set up optimizer for Affine
            affine_registration.SetOptimizerAsGradientDescent(learningRate=0.05,  # Lower learning rate for fine-tuning
                                                            numberOfIterations=1000,
                                                            convergenceMinimumValue=1e-6,
                                                            convergenceWindowSize=10)
            affine_registration.SetOptimizerScalesFromPhysicalShift()
            
            # Initialize Affine transform from the Rigid result
            # Create a new Affine transform instead of trying to convert from Rigid
            affine_transform = sitk.AffineTransform(3)
            
            # Use identity matrix for initialization (standard starting point)
            # We'll use the Rigid-registered image as input, so we don't need 
            # to copy the rigid transformation parameters
            affine_registration.SetInitialTransform(affine_transform)
            
            # Multi-resolution framework for Affine (finer levels)
            affine_registration.SetShrinkFactorsPerLevel([2, 1])
            affine_registration.SetSmoothingSigmasPerLevel([1, 0])
            
            # Execute Affine registration
            final_affine_transform = affine_registration.Execute(fixed_image, rigid_moved_image)
            logger.debug("Affine registration completed successfully")
            
            # Create CompositeTransform for final result with single interpolation
            logger.debug("Creating CompositeTransform for final T1 output")
            combo = sitk.CompositeTransform(3)
            combo.AddTransform(final_rigid_transform)
            combo.AddTransform(final_affine_transform)
            
            # Single resample from original moving image for better quality
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(combo)
            
            final_moved_image = resampler.Execute(moving_image)
            
            # Save the final results (output of two-stage registration)
            # The final warped image is the result of both transforms
            final_warped_path = os.path.join(os.path.dirname(output_prefix), f"{base_name}_mni_warped.nii.gz")
            affine_transform_path = os.path.join(other_dir, f"{base_name}_affine.mat")
            
            sitk.WriteImage(final_moved_image, final_warped_path)
            sitk.WriteTransform(final_affine_transform, affine_transform_path)
            logger.debug(f"Saved final warped image to {final_warped_path}")
            
            # Note: Success tracking moved to main loop after complete processing
            
            # Create a mock result object to maintain compatibility
            class MockResult:
                class Outputs:
                    def __init__(self, prefix, base_name, other_dir):
                        # Final results (two-stage pipeline)
                        self.warped_image = os.path.join(os.path.dirname(prefix), f"{base_name}_mni_warped.nii.gz")
                        # Both transforms needed for complete transformation
                        self.forward_transforms = [
                            os.path.join(other_dir, f"{base_name}_rigid.mat"),
                            os.path.join(other_dir, f"{base_name}_affine.mat")
                        ]
                        
                        # Individual stage results for reference
                        self.rigid_warped_image = os.path.join(os.path.dirname(prefix), f"{base_name}_mni_rigid_warped.nii.gz")
                        self.rigid_transform = os.path.join(other_dir, f"{base_name}_rigid.mat")
                        self.affine_warped_image = os.path.join(os.path.dirname(prefix), f"{base_name}_mni_warped.nii.gz")
                        self.affine_transform = os.path.join(other_dir, f"{base_name}_affine.mat")
                
                def __init__(self, prefix, base_name, other_dir):
                    self.outputs = self.Outputs(prefix, base_name, other_dir)
            
            return MockResult(output_prefix, base_name, other_dir)
            
        except Exception as e:
            logger.error(f"Registration failed with error: {str(e)}")
            raise
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Registration failed for subject {subject_id}: {error_msg}")
        if subject_id:
            registration_results['failed_subjects'].append(subject_id)
            registration_results['errors'][subject_id] = error_msg
        raise

# ----------------------------
# 4. T1 Registration (T1 -> MNI)
# ----------------------------

# Dictionary to store T1->MNI transforms for each subject (for later use in T2 transformation)
t1_transforms = {}

# Main processing
try:
    # Add missing subjects to results
    registration_results['missing_subjects'] = missing_subjects
    for subj in missing_subjects:
        registration_results['errors'][subj] = "Input files not found"
    
    # Update total subjects count based on discovered triples
    registration_results['total_subjects'] = len(set((s, se) for (s, se, _) in triples))
    
    # Process each (subject, session, base_path) combination
    for (subj, sess, base) in triples:
        # Filter for specific subject if requested
        if args.subject:
            target_subject = args.subject.replace('sub-', '')  # Remove 'sub-' prefix if present
            current_subject = subj.replace('sub-', '')  # Remove 'sub-' prefix from discovered subject
            if current_subject != target_subject:
                continue
        
        try:
            subj_start_time = time.time()
            logger.info(f"Starting processing for subject {subj}, session {sess} at {datetime.now().strftime('%H:%M:%S')}")
            
            # Find T1 and T2 files for this specific (subject, session) combination
            t1_pattern = os.path.join(base, SKULLSTRIP_T1_FILE_PATTERN.replace(".nii.gz", "_brain.nii.gz"))
            t2_pattern = os.path.join(base, SKULLSTRIP_T2_FILE_PATTERN.replace(".nii.gz", "_brain.nii.gz"))
            
            current_t1_files = glob.glob(t1_pattern)
            current_t2_files = glob.glob(t2_pattern)
            
            if not current_t1_files:
                logger.warning(f"No T1 files found for subject {subj}, session {sess} in {base}")
                continue
                
            # Use unified structure to build output directory
            subj_dir = make_path(STRUCTURE, STAGE_ROOTS["registration"], subj, sess, mkdirs=True)
            other_dir = os.path.join(subj_dir, "other")
            os.makedirs(other_dir, exist_ok=True)
            logger.debug(f"Created output directory: {subj_dir}")
            logger.debug(f"Created other directory: {other_dir}")
            
            # Process the first (usually only) T1 file for this subject/session
            t1 = current_t1_files[0]
            base_name = os.path.basename(t1).replace("_brain.nii.gz", "")
            out_prefix = os.path.join(subj_dir, f"{base_name}_")
            
            # T1 Registration
            logger.info(f"Processing T1 for subject {subj}")
            reg_result = register_image(fixed=template_T1, moving=t1, output_prefix=out_prefix, subject_id=subj, base_name=base_name, other_dir=other_dir)
            
            # Store transforms using (subject, session) as key for multi-session support
            t1_transforms[(subj, sess)] = reg_result.outputs.forward_transforms
            
            # Process T2 if available for this (subject, session)
            if current_t2_files:
                t2_file = current_t2_files[0]  # Use the first T2 file
                logger.info(f"Processing T2 for subject {subj}")
                
                # Create T2 base name for transform file naming
                t2_base_name = os.path.basename(t2_file).replace("_brain.nii.gz", "")
                
                # STEP 1: T2→T1 rigid registration for better inter-modality alignment
                logger.debug(f"Starting T2→T1 rigid registration for subject {subj}")
                
                # Read images
                t1_image = sitk.ReadImage(t1, sitk.sitkFloat32)  # T1 as fixed
                t2_image = sitk.ReadImage(t2_file, sitk.sitkFloat32)  # T2 as moving
                
                # T2→T1 rigid registration
                t2_to_t1_registration = sitk.ImageRegistrationMethod()
                
                # Set up similarity metric for T2→T1
                t2_to_t1_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                t2_to_t1_registration.SetMetricSamplingStrategy(t2_to_t1_registration.RANDOM)
                t2_to_t1_registration.SetMetricSamplingPercentage(0.25)
                # Note: SetMetricSamplingPercentageRandomSeed not available in SimpleITK 2.1.0
                
                # Set up interpolator and optimizer for T2→T1
                t2_to_t1_registration.SetInterpolator(sitk.sitkLinear)
                t2_to_t1_registration.SetOptimizerAsGradientDescent(learningRate=0.1,
                                                                  numberOfIterations=1000,
                                                                  convergenceMinimumValue=1e-6,
                                                                  convergenceWindowSize=10)
                t2_to_t1_registration.SetOptimizerScalesFromPhysicalShift()
                
                # Initialize T2→T1 rigid transform
                t2_to_t1_transform = sitk.CenteredTransformInitializer(t1_image, 
                                                                      t2_image,
                                                                      sitk.Euler3DTransform(),
                                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
                t2_to_t1_registration.SetInitialTransform(t2_to_t1_transform)
                
                # Multi-resolution for T2→T1
                t2_to_t1_registration.SetShrinkFactorsPerLevel([4, 2, 1])
                t2_to_t1_registration.SetSmoothingSigmasPerLevel([2, 1, 0])
                
                # Execute T2→T1 rigid registration
                final_t2_to_t1_transform = t2_to_t1_registration.Execute(t1_image, t2_image)
                logger.debug(f"T2→T1 rigid registration completed for subject {subj}")
                
                # Save T2→T1 transform
                t2_to_t1_transform_path = os.path.join(other_dir, f"{t2_base_name}_t2_to_t1_rigid.mat")
                sitk.WriteTransform(final_t2_to_t1_transform, t2_to_t1_transform_path)
                
                # STEP 2: Create CompositeTransform: (T1→MNI affine) ∘ (T1→MNI rigid) ∘ (T2→T1 rigid)
                logger.debug(f"Creating CompositeTransform for T2→MNI via T1 for subject {subj}")
                
                # Get T1→MNI transform paths using (subject, session) key
                t1_rigid_transform_path, t1_affine_transform_path = t1_transforms[(subj, sess)]
                
                # Create composite transform: T2→T1→MNI
                combo_transform = sitk.CompositeTransform(3)
                combo_transform.AddTransform(final_t2_to_t1_transform)  # First: T2→T1
                combo_transform.AddTransform(sitk.ReadTransform(t1_rigid_transform_path))   # Second: T1→MNI rigid
                combo_transform.AddTransform(sitk.ReadTransform(t1_affine_transform_path))  # Third: T1→MNI affine
                
                # STEP 3: Single resample T2→MNI with composite transform
                template_image = sitk.ReadImage(template_T1, sitk.sitkFloat32)
                
                final_resampler = sitk.ResampleImageFilter()
                final_resampler.SetReferenceImage(template_image)
                final_resampler.SetInterpolator(sitk.sitkLinear)
                final_resampler.SetTransform(combo_transform)
                
                # Single resample for final T2 result (best quality)
                final_warped_t2 = final_resampler.Execute(t2_image)
                
                # Save final transformed T2
                t2_warped_path = os.path.join(subj_dir, f"{t2_base_name}_mni_warped.nii.gz")
                sitk.WriteImage(final_warped_t2, t2_warped_path)
                logger.debug(f"Saved final T2 warped image (T2→T1→MNI CompositeTransform) to {t2_warped_path}")
                
                # Note: zscore normalization and cropping will be handled by postprocess_mni_mask_zscore_crop.py
            
            # Record successful completion (only if no exceptions occurred)
            registration_results['successful_subjects'].append(subj)
            
            # Record processing time
            processing_time = time.time() - subj_start_time
            registration_results['processing_times'][subj] = processing_time
            logger.info(f"Completed subject {subj} in {format_time(processing_time)}")
            
        except Exception as e:
            processing_time = time.time() - subj_start_time
            registration_results['processing_times'][subj] = processing_time
            print(f"Failed processing subject {subj} after {format_time(processing_time)}: {str(e)}")
            registration_results['failed_subjects'].append(subj)
            registration_results['errors'][subj] = str(e)
            continue

except Exception as e:
    print(f"Critical error in main processing: {str(e)}")
finally:
    # Calculate final statistics
    registration_results['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if registration_results['total_subjects'] > 0:
        registration_results['success_rate'] = (len(registration_results['successful_subjects']) / 
                                              registration_results['total_subjects'] * 100)
        if registration_results['processing_times']:
            total_time = sum(registration_results['processing_times'].values())
            registration_results['average_processing_time'] = total_time / len(registration_results['processing_times'])
    
    # Save report in the base derivatives directory
    save_registration_report(registration_results, output_base_dir)
    print_registration_summary(registration_results)
    
    print("\nRegistration completed. Zscore normalization and cropping will be handled by postprocess_mni_mask_zscore_crop.py")


