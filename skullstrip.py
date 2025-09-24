#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brain Extraction Script using FreeSurfer SynthStrip
Supports T1w and T2w MRI modalities
Reads settings from config.py

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""

import os
import glob
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import logging
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import csv
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
 
# Import configuration and structure resolver
try:
    from config import (
        INPUT_DIR, OUTPUT_DIR, QC_DIR, SKULLSTRIP_DIR,
        MODALITIES,
        SUBJECTS, FORCE_REPROCESS,
        FREESURFER_HOME, SYNTHSTRIP_BIN,
        ENABLE_QC, QC_BRAIN_VOLUME_MIN_ML, QC_BRAIN_VOLUME_MAX_ML,
        QC_GENERATE_IMAGES, QC_SUMMARY_FILE,
        SKULLSTRIP_T1_FILE_PATTERN, SKULLSTRIP_T2_FILE_PATTERN,
        SESSIONS, LOG_LEVEL, LOG_FORMAT, LOG_DIR,
        SYNTHSTRIP_SIF_PATH,
        # Unified structure
        STAGE_ROOTS, STRUCTURE
    )
    from structure_resolver import discover, make_path
except ImportError as e:
    print(f"ERROR: Could not import required configuration from config.py or structure_resolver.py")
    print(f"Import error: {e}")
    print("Please ensure config.py and structure_resolver.py are properly configured.")
    exit(1)

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, f'skullstrip_{timestamp}.log'))
    ]
)
logger = logging.getLogger('synthstrip')

def perform_quality_check(output_brain_file, input_raw_file, qc_dir, qc_root, modality, subject, session):
    """Perform quality check on extracted brain."""
    os.makedirs(qc_dir, exist_ok=True)
    
    def _normalize(p):
        nz = p[p>0]
        denom = np.percentile(nz, 99) if nz.size else 1.0
        denom = denom if np.isfinite(denom) and denom>0 else 1.0
        x = p/denom
        return np.clip(x, 0, 1)
    
    qc_result = {
        'subject': subject,
        'session': session,
        'modality': modality,
        'passed_qc': True,
        'brain_volume_ml': 0,
        'qc_image_path': '',
        'error': None
    }
    
    try:
        brain_img = nib.load(output_brain_file)
        brain_data = brain_img.get_fdata()
        
        raw_img = nib.load(input_raw_file)
        raw_data = raw_img.get_fdata()
        
        mask_file = output_brain_file.replace('.nii.gz', '_mask.nii.gz')
        if not os.path.exists(mask_file):
            logger.warning("Mask missing for %s; skipping volume QC.", output_brain_file)
            qc_result['passed_qc'] = False
            qc_result['error'] = "Mask file missing"
            return qc_result
        
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()
        
        voxel_volume_mm3 = np.prod(brain_img.header.get_zooms())
        brain_volume_mm3 = np.sum(mask_data > 0) * voxel_volume_mm3
        brain_volume_ml = brain_volume_mm3 / 1000
        
        qc_result['brain_volume_ml'] = brain_volume_ml
        
        if ENABLE_QC and (brain_volume_ml < QC_BRAIN_VOLUME_MIN_ML or brain_volume_ml > QC_BRAIN_VOLUME_MAX_ML):
            qc_result['passed_qc'] = False
            qc_result['error'] = f"Brain volume ({brain_volume_ml:.2f} ml) outside acceptable range ({QC_BRAIN_VOLUME_MIN_ML}-{QC_BRAIN_VOLUME_MAX_ML} ml)"
            logger.warning(f"QC failed for sub-{subject} ses-{session} {modality}: {qc_result['error']}")
        
        if QC_GENERATE_IMAGES:
            input_basename = os.path.basename(input_raw_file)
            qc_image_file = os.path.join(qc_dir, input_basename.replace('.nii.gz', '_desc-qc.png'))
            qc_result['qc_image_path'] = qc_image_file
            
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            
            x_mid = brain_data.shape[0] // 2
            y_mid = brain_data.shape[1] // 2
            z_mid = brain_data.shape[2] // 2
            
            raw_data_norm = _normalize(raw_data)
            brain_data_norm = _normalize(brain_data)
            
            def add_colorbar(im, ax):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            
            def _slice3d(arr, axis, index):
                if axis == 0: return arr[index, :, :]
                if axis == 1: return arr[:, index, :]
                return arr[:, :, index]
            
            views = [
                (0, x_mid, 'Sagittal'),
                (1, y_mid, 'Coronal'),
                (2, z_mid, 'Axial')
            ]
            
            for row, (axis, index, view_name) in enumerate(views):
                raw_slice = np.rot90(_slice3d(raw_data_norm, axis, index))
                brain_slice = np.rot90(_slice3d(brain_data_norm, axis, index))
                mask_slice = np.rot90(_slice3d(mask_data, axis, index))
                
                im0 = axes[row, 0].imshow(raw_slice, cmap='gray')
                axes[row, 0].set_title(f'{view_name}\nRaw Image')
                add_colorbar(im0, axes[row, 0])
                
                im1 = axes[row, 1].imshow(brain_slice, cmap='gray')
                axes[row, 1].set_title(f'{view_name}\nBrain Extracted')
                add_colorbar(im1, axes[row, 1])
                
                axes[row, 2].imshow(raw_slice, cmap='gray')
                im2 = axes[row, 2].imshow(mask_slice, cmap='Reds', alpha=0.3)
                axes[row, 2].set_title(f'{view_name}\nRaw + Mask')
                add_colorbar(im2, axes[row, 2])
                
                for ax in axes[row]:
                    ax.axis('off')
            
            for col in range(3):
                if col == 0:
                    im = axes[3, col].imshow(mask_slice, cmap='Reds')
                    axes[3, col].set_title('Brain Mask')
                elif col == 1:
                    diff = raw_slice - brain_slice
                    im = axes[3, col].imshow(diff, cmap='RdBu_r')
                    axes[3, col].set_title('Raw - Brain (Difference)')
                else:
                    axes[3, col].text(0.5, 0.5, 
                                    f"Brain Volume: {brain_volume_ml:.2f} ml\n" +
                                    f"Status: {'PASS' if qc_result['passed_qc'] else 'FAIL'}\n" +
                                    f"Valid Range: {QC_BRAIN_VOLUME_MIN_ML}-{QC_BRAIN_VOLUME_MAX_ML} ml",
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    transform=axes[3, col].transAxes,
                                    fontsize=12)
                    axes[3, col].axis('off')
                    continue
                
                add_colorbar(im, axes[3, col])
                axes[3, col].axis('off')
            
            plt.suptitle(f"Brain Extraction QC Report\n" +
                        f"{subject} {session} {modality}",
                        fontsize=16, y=0.95)
            
            plt.tight_layout()
            plt.savefig(qc_image_file, dpi=150, bbox_inches='tight', pad_inches=0.2)
            plt.close()
            
            logger.info(f"Generated QC image at: {qc_image_file}")
        
        if QC_SUMMARY_FILE:
            qc_summary_path = os.path.join(STAGE_ROOTS["skullstrip"], QC_SUMMARY_FILE)
            file_exists = os.path.isfile(qc_summary_path)
            
            with open(qc_summary_path, mode='a') as csvfile:
                fieldnames = ['input_file', 'output_file', 'modality', 'qc_status', 'brain_volume_ml', 'qc_image_path', 'error_message']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'input_file': input_raw_file,
                    'output_file': output_brain_file,
                    'modality': modality,
                    'qc_status': 'PASS' if qc_result['passed_qc'] else 'FAIL',
                    'brain_volume_ml': f"{qc_result['brain_volume_ml']:.2f}",
                    'qc_image_path': qc_result['qc_image_path'],
                    'error_message': qc_result['error'] if 'error' in qc_result and qc_result['error'] else 'None'
                })
    
    except Exception as e:
        error_msg = f"Error in quality check: {str(e)}"
        logging.error(error_msg)
        qc_result['passed_qc'] = False
        qc_result['error'] = error_msg
    
    return qc_result

def extract_brain_with_synthstrip(input_file, output_file, modality, qc_dir=None, qc_root=None, subject="", session=""):
    """Extract brain using FreeSurfer's SynthStrip or Docker."""
    if os.path.exists(output_file) and not FORCE_REPROCESS:
        logger.info(f"Output file already exists: {output_file}")
        # Even if the brain file exists, ensure QC artifacts are generated/updated
        if ENABLE_QC and qc_dir:
            try:
                qc_result = perform_quality_check(output_file, input_file, qc_dir, qc_root, modality, subject, session)
                logger.info(f"QC result: {qc_result['error'] if 'error' in qc_result and qc_result['error'] else 'passed'}")
            except Exception as e:
                logger.warning(f"QC generation failed for existing output {output_file}: {e}")
        return True

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    input_abs = os.path.abspath(input_file)
    output_abs = os.path.abspath(output_file)
    mask_file = f"{output_abs.replace('.nii.gz', '')}_mask.nii.gz"

    # Try Singularity first
    try:
        # Check if Singularity is available
        subprocess.run(['singularity', '--version'], check=True, capture_output=True)

        # Use Singularity
        logger.info("Using Singularity SynthStrip...")
        sif_file = SYNTHSTRIP_SIF_PATH
        if not os.path.exists(sif_file):
            logger.info("Pulling SynthStrip Singularity image...")
            subprocess.run(['singularity', 'pull', f"{sif_file}", 'docker://freesurfer/synthstrip:latest'],
                         check=True, capture_output=True)

        singularity_cmd = [
            'singularity', 'exec', '--cleanenv',
            '-B', f"{os.path.dirname(input_abs)}:/input",
            '-B', f"{os.path.dirname(output_abs)}:/output",
            sif_file,
            'mri_synthstrip',
            '-i', f"/input/{os.path.basename(input_abs)}",
            '-o', f"/output/{os.path.basename(output_abs)}",
            '-m', f"/output/{os.path.basename(mask_file)}"
        ]

        logger.info(f"Running Singularity command: {' '.join(singularity_cmd)}")
        result = subprocess.run(singularity_cmd, check=True, capture_output=True, text=True)
        logger.info("Singularity SynthStrip completed successfully")
        # After successful container run, perform QC and return
        if os.path.exists(output_file):
            if ENABLE_QC and qc_dir:
                try:
                    qc_result = perform_quality_check(output_file, input_file, qc_dir, qc_root, modality, subject, session)
                    logger.info(f"QC result: {qc_result['error'] if 'error' in qc_result and qc_result['error'] else 'passed'}")
                except Exception as e:
                    logger.warning(f"QC generation failed (Singularity) for {output_file}: {e}")
            return True
        else:
            logger.error(f"Output file was not created: {output_file}")
            return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try Docker as fallback
        try:
            # Check if Docker is available
            subprocess.run(['docker', '--version'], check=True, capture_output=True)

            # Use Docker
            logger.info("Using Docker SynthStrip...")
            docker_cmd = [
                'docker', 'run', '--rm',
                '-v', f"{os.path.dirname(input_abs)}:/input",
                '-v', f"{os.path.dirname(output_abs)}:/output",
                'freesurfer/synthstrip:latest',
                '-i', f"/input/{os.path.basename(input_abs)}",
                '-o', f"/output/{os.path.basename(output_abs)}",
                '-m', f"/output/{os.path.basename(mask_file)}"
            ]

            logger.info(f"Running Docker command: {' '.join(docker_cmd)}")
            result = subprocess.run(docker_cmd, check=True, capture_output=True, text=True)
            logger.info("Docker SynthStrip completed successfully")
            # After successful container run, perform QC and return
            if os.path.exists(output_file):
                if ENABLE_QC and qc_dir:
                    try:
                        qc_result = perform_quality_check(output_file, input_file, qc_dir, qc_root, modality, subject, session)
                        logger.info(f"QC result: {qc_result['error'] if 'error' in qc_result and qc_result['error'] else 'passed'}")
                    except Exception as e:
                        logger.warning(f"QC generation failed (Docker) for {output_file}: {e}")
                return True
            else:
                logger.error(f"Output file was not created: {output_file}")
                return False

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to FreeSurfer binary
            logger.info("Docker/Singularity not available, trying FreeSurfer binary...")

        # subject/session already parsed above for QC

        env = os.environ.copy()
        env["FREESURFER_HOME"] = FREESURFER_HOME
        env["PATH"] = f"{os.path.join(FREESURFER_HOME, 'bin')}:{env.get('PATH', '')}"

        cmd = [
            SYNTHSTRIP_BIN,
            "-i", input_abs,
            "-o", output_abs,
            "-m", mask_file
        ]
    
    try:
        logger.info(f"Running SynthStrip on {input_file}")
        
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        if os.path.exists(output_file):
            logger.info(f"Successfully extracted brain: {output_file}")
            
            if ENABLE_QC and qc_dir:
                qc_result = perform_quality_check(output_file, input_file, qc_dir, qc_root, modality, subject, session)
                logger.info(f"QC result: {qc_result['error'] if 'error' in qc_result and qc_result['error'] else 'passed'}")
                return qc_result['passed_qc']
            
            return True
        else:
            logger.error(f"Output file was not created: {output_file}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {input_file}: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        return False


def process_subject(subject_id, session_id, in_base, output_root, qc_root):
    """Process a single subject's images."""
    out_path = make_path(STRUCTURE, output_root, subject_id, session_id)
    os.makedirs(out_path, exist_ok=True)
    qc_path = out_path

    any_ok = False
    for modality in MODALITIES:
        pattern = SKULLSTRIP_T1_FILE_PATTERN if modality == "T1w" else SKULLSTRIP_T2_FILE_PATTERN
        input_candidates = glob.glob(os.path.join(in_base, pattern))
        if not input_candidates:
            logger.warning(f"No {modality} for {subject_id} {session_id} under {in_base}")
            continue

        input_candidates.sort(key=os.path.getmtime, reverse=True)
        input_file = input_candidates[0]
        stem = os.path.splitext(os.path.splitext(os.path.basename(input_file))[0])[0]
        output_file = os.path.join(out_path, f"{stem}_brain.nii.gz")

        try:
            success = extract_brain_with_synthstrip(input_file, output_file, modality,
                                                    qc_dir=qc_path, qc_root=qc_root, subject=subject_id, session=session_id)
            any_ok = any_ok or bool(success)
            if success:
                logger.info(f"Successfully processed {modality} for subject {subject_id}, session {session_id}")
            else:
                logger.error(f"Failed to process {modality} for subject {subject_id}, session {session_id}")
            
        except Exception as e:
            logger.error(f"Error processing {modality} for subject {subject_id}, session {session_id}: {str(e)}")
            continue
    
    return any_ok

def batch_process(triples, qc_root, max_workers=None):
    """Process subjects in parallel batches."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STAGE_ROOTS["skullstrip"], exist_ok=True)
    if ENABLE_QC:
        os.makedirs(qc_root, exist_ok=True)
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count() - 2
    
    total_subjects = len(triples)
    logger.info(f"Processing {total_subjects} subjects with {max_workers} workers")
    
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futs = {executor.submit(process_subject, subj, sess, in_base, STAGE_ROOTS["skullstrip"], qc_root): (subj, sess)
                    for (subj, sess, in_base) in triples}

            for future in tqdm(as_completed(futs), total=len(futs), desc="Processing subjects"):
                subj, sess = futs[future]
                try:
                    result = future.result()
                    if result:
                        logger.info(f"Successfully processed subject: {subj}, session: {sess}")
                    else:
                        logger.warning(f"Issues processing subject: {subj}, session: {sess}")
                except Exception as e:
                    logger.error(f"Error processing subject {subj}, session {sess}: {e}")
    else:
        for subj, sess, in_base in tqdm(triples, desc="Processing subjects"):
            try:
                result = process_subject(subj, sess, in_base, STAGE_ROOTS["skullstrip"], qc_root)
                if result:
                    logger.info(f"Successfully processed subject: {subj}, session: {sess}")
                else:
                    logger.warning(f"Issues processing subject: {subj}, session: {sess}")
            except Exception as e:
                logger.error(f"Error processing subject {subj}, session {sess}: {e}")

def main():
    """Main function to run the brain extraction pipeline."""
    parser = argparse.ArgumentParser(description="Brain extraction using FreeSurfer SynthStrip")
    parser.add_argument("--subject", type=str, default=None,
                        help="Process only a specific subject (e.g., ON93426 or sub-ON93426)")
    
    args = parser.parse_args()
    
    # Override SUBJECTS if specific subject requested
    global SUBJECTS
    if args.subject:
        # Handle both formats: ON93426 or sub-ON93426
        subject_id = args.subject.replace('sub-', '')
        SUBJECTS = [subject_id]
    
    logger.info("===== Configuration =====")
    logger.info(f"Input Directory: {INPUT_DIR}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Modalities: {MODALITIES}")
    logger.info(f"Subjects: {SUBJECTS if SUBJECTS else 'All'}")
    logger.info(f"Sessions: {SESSIONS if SESSIONS else 'All'}")
    logger.info(f"Force Reprocessing: {FORCE_REPROCESS}")
    logger.info(f"Quality Control Enabled: {ENABLE_QC}")
    logger.info(f"FreeSurfer Home: {FREESURFER_HOME}")
    logger.info(f"SynthStrip Binary: {SYNTHSTRIP_BIN}")
    logger.info("=========================")
    
    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input directory does not exist: {INPUT_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    qc_root = STAGE_ROOTS.get("qc", STAGE_ROOTS["skullstrip"])
    
    if ENABLE_QC and QC_GENERATE_IMAGES:
        os.makedirs(qc_root, exist_ok=True)
        logger.info(f"Quality control images will be saved to: {qc_root}")
    
    triples = list(discover(STRUCTURE, INPUT_DIR, subjects=SUBJECTS, sessions=SESSIONS))
    
    if not triples:
        logger.warning(f"No subjects found matching STRUCTURE in {INPUT_DIR}")
        return
    
    logger.info(f"Found {len(triples)} subjects to process")
    
    batch_process(triples, qc_root, max_workers=max_workers if 'max_workers' in locals() else None)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main() 
