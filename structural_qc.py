#!/usr/bin/env python3
"""
Structural MRI Quality Control Script
Author: Mohammad Abbasi (mabbasi@stanford.edu)
Created: 2025

This script generates QC reports for structural MRI preprocessing including:
- Skull stripping quality assessment for T1w and T2w
- Registration quality metrics (Dice scores for rigid and affine)
- HTML reports with detailed visualizations
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime
from pathlib import Path
import re
from urllib.parse import quote 
import nibabel as nib
import config
import numbers

# Configure logging (console only, file handler added in __init__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def calculate_dice_coefficient(image1_path, image2_path, mask1_path=None, mask2_path=None, bin_thr=0.5):
    """
    Calculate Dice coefficient strictly between two binary masks in the same space.
    Requires both mask paths; returns NaN if any mask is missing.
    Resamples masks with nearest-neighbor (order=0) when grids differ.
    """
    try:
        from nibabel.processing import resample_from_to
        img1 = nib.load(image1_path)
        ref = (img1.shape, img1.affine)

        # Require masks
        if not (mask1_path and os.path.exists(mask1_path)) or not (mask2_path and os.path.exists(mask2_path)):
            logger.error("Dice requires binary masks but at least one mask file is missing.")
            return float('nan')

        m1_img = nib.load(mask1_path)
        m2_img = nib.load(mask2_path)

        if (m1_img.shape != img1.shape) or (not np.allclose(m1_img.affine, img1.affine, atol=1e-3)):
            m1_img = resample_from_to(m1_img, ref, order=0)
        if (m2_img.shape != img1.shape) or (not np.allclose(m2_img.affine, img1.affine, atol=1e-3)):
            m2_img = resample_from_to(m2_img, ref, order=0)

        d1 = m1_img.get_fdata()
        d2 = m2_img.get_fdata()
        m1 = d1.astype(bool) if np.issubdtype(d1.dtype, np.bool_) else (d1 > bin_thr)
        m2 = d2.astype(bool) if np.issubdtype(d2.dtype, np.bool_) else (d2 > bin_thr)

        inter = np.count_nonzero(m1 & m2)
        denom = np.count_nonzero(m1) + np.count_nonzero(m2)
        if denom == 0:
            return float('nan')
        return float(2.0 * inter / denom)
    except Exception as e:
        logger.error(f"Error calculating Dice: {os.path.basename(image1_path)} vs {os.path.basename(image2_path)}: {e}")
        return float('nan')

class StructuralQC:
    """Structural MRI Quality Control Class"""
    
    def __init__(self, dataset_root=None):
        # Use config values or fallback to provided dataset_root
        if dataset_root is None:
            # Extract dataset root from config OUTPUT_DIR
            self.processed_dir = Path(config.OUTPUT_DIR)
            self.dataset_root = self.processed_dir.parent
        else:
            self.dataset_root = Path(dataset_root)
            self.processed_dir = self.dataset_root / "processed" / "Structural"

        # Dataset configuration from config
        self.dataset_name = getattr(config, 'DATASET_NAME', 'Unknown Dataset')
        self.has_sessions = getattr(config, 'HAS_SESSIONS', True)
        self.folder_structure = getattr(config, 'FOLDER_STRUCTURE', {
            "skullstrip": "{subject}/{session}/anat/",
            "registration": "{subject}/{session}/anat/",
            "qc_images": "{subject}/{session}/anat/",
        })

        # Use paths from config
        self.skullstrip_dir = Path(config.SKULLSTRIP_DIR)
        self.registration_dir = Path(config.REGISTRATION_DIR)
        self.qc_dir = Path(config.QC_DIR)

        # Create QC directory if it doesn't exist
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for logging after directory creation
        fh = logging.FileHandler(self.qc_dir / 'structural_qc.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        logger.propagate = False

        # QC thresholds from config
        self.brain_volume_min = config.QC_BRAIN_VOLUME_MIN_ML
        self.brain_volume_max = config.QC_BRAIN_VOLUME_MAX_ML

        # Cache for MNI template mask path
        self._mni_mask_path = None

        logger.info(f"Structural QC initialized")
        logger.info(f"  Dataset: {self.dataset_name}")
        logger.info(f"  Has sessions: {self.has_sessions}")
        logger.info(f"  Dataset root: {self.dataset_root}")
        logger.info(f"  Skullstrip dir: {self.skullstrip_dir}")
        logger.info(f"  Registration dir: {self.registration_dir}")
        logger.info(f"  QC output dir: {self.qc_dir}")
    
    def load_skullstrip_data(self):
        """Load skull stripping QC data from CSV file"""
        qc_csv_path = self.skullstrip_dir / "qc_summary.csv"
        
        if not qc_csv_path.exists():
            logger.error(f"Skullstrip QC file not found: {qc_csv_path}")
            return None
        
        try:
            df = pd.read_csv(qc_csv_path)
            
            # Check required columns
            required_cols = {'input_file', 'modality', 'brain_volume_ml', 'qc_image_path'}
            missing = required_cols - set(df.columns)
            if missing:
                logger.error(f"Skullstrip QC file missing columns: {missing}")
                return None
                
            logger.info(f"Loaded {len(df)} skullstrip QC records")
            return df
        except Exception as e:
            logger.error(f"Error loading skullstrip data: {e}")
            return None
    
    def extract_subject_session(self, filepath):
        """Extract subject and session from file path (Windows/Unix compatible)"""
        # Normalize path separators for cross-platform compatibility
        p = str(filepath).replace('\\', '/')

        # If dataset doesn't have sessions, only extract subject
        if not self.has_sessions:
            # Pattern for datasets without sessions: sub-ON12345/...
            match = re.search(r'sub-([^/]+)', p)
            if match:
                return f"sub-{match.group(1)}", None
            return None, None

        # For datasets with sessions
        # Pattern: sub-ON12345/ses-01/...
        match = re.search(r'sub-([^/]+)/ses-([^/]+)', p)
        if match:
            return f"sub-{match.group(1)}", f"ses-{match.group(2)}"

        # Fallback pattern: sub-ON12345_ses-01
        match = re.search(r'sub-([^_]+)_ses-([^_]+)', p)
        if match:
            return f"sub-{match.group(1)}", f"ses-{match.group(2)}"

        return None, None
    
    def assess_brain_volume(self, volume_ml):
        """Assess if brain volume is within acceptable range"""
        if pd.isna(volume_ml):
            return "UNKNOWN"
        
        volume_ml = float(volume_ml)
        if self.brain_volume_min <= volume_ml <= self.brain_volume_max:
            return "PASS"
        else:
            return "FAIL"
    


    def calculate_registration_dice(self, subject, session):
        """Calculate Dice coefficients for registration quality using binary masks"""
        # Build path using configurable folder structure
        if self.has_sessions and session:
            folder_path = self.folder_structure["registration"].format(subject=subject, session=session)
        else:
            # For datasets without sessions, use subject only
            folder_path = self.folder_structure["registration"].format(subject=subject, session="")

        registration_base = self.registration_dir / folder_path
        
        dice_results = {
            'T1_Dice_MNI': float('nan'),     # T1 mask vs MNI template
            'T2_Dice_MNI': float('nan')      # T2 mask vs MNI template
        }
        
        try:
            # Get MNI reference masks
            mni_template_mask = self.get_mni_template_mask()
            
            # Subject's warped masks (binary masks in MNI space)
            t1_mask = registration_base / "T1w_mni_mask.nii.gz"
            t2_mask = registration_base / "T2w_mni_mask.nii.gz"
            
            # Calculate T1 Dice coefficient: Subject T1 mask vs MNI template mask
            if t1_mask.exists() and mni_template_mask:
                logger.debug(f"Calculating T1 Dice: {t1_mask} vs MNI template")
                dice_results['T1_Dice_MNI'] = calculate_dice_coefficient(
                    str(t1_mask), mni_template_mask,
                    mask1_path=str(t1_mask), mask2_path=mni_template_mask
                )
            
            # Calculate T2 Dice coefficient: Subject T2 mask vs MNI template mask
            if t2_mask.exists() and mni_template_mask:
                logger.debug(f"Calculating T2 Dice: {t2_mask} vs MNI template")
                dice_results['T2_Dice_MNI'] = calculate_dice_coefficient(
                    str(t2_mask), mni_template_mask,
                    mask1_path=str(t2_mask), mask2_path=mni_template_mask
                )
                
        except Exception as e:
            logger.warning(f"Error calculating Dice coefficients for {subject} {session}: {e}")
            
        return dice_results
    
    def get_mni_template_mask(self):
        """Get MNI template brain mask using templateflow (cached)"""
        if self._mni_mask_path:
            return self._mni_mask_path
        try:
            from templateflow import api as tflow
            self._mni_mask_path = str(tflow.get(
                config.MNI_TEMPLATE_VERSION,
                resolution=config.MNI_TEMPLATE_RESOLUTION,
                suffix="mask", desc="brain"
            ))
            logger.info(f"Using MNI template mask: {self._mni_mask_path}")
            return self._mni_mask_path
        except Exception as e:
            logger.warning(f"Could not get MNI template mask: {e}")
            logger.warning("Try: pip install templateflow")
            logger.warning("Or set custom template path in config.py")
            return None
    
    def create_summary_dataframe(self, subject_filter=None):
        """Create summary dataframe with all QC metrics"""
        
        # Load skullstrip data
        skullstrip_df = self.load_skullstrip_data()
        if skullstrip_df is None:
            return None
        
        # Note: Registration metrics loaded per subject in calculate_registration_dice()
        
        # Process skullstrip data
        summary_data = []
        
        for _, row in skullstrip_df.iterrows():
            subject, session = self.extract_subject_session(row['input_file'])

            if subject is None:
                logger.warning(f"Could not extract subject from: {row['input_file']}")
                continue

            # For datasets without sessions, session will be None
            if self.has_sessions and session is None:
                logger.warning(f"Could not extract session from: {row['input_file']}")
                continue
            
            # Filter by subject if specified
            if subject_filter and subject != subject_filter:
                continue
            
            # Find existing record for this subject/session
            existing_record = None
            session_to_compare = session if session else ''  # Use empty string for comparison
            for record in summary_data:
                record_session = record['Session'] if record['Session'] else ''
                if record['Subject'] == subject and record_session == session_to_compare:
                    existing_record = record
                    break
            
            # Create new record if doesn't exist
            if existing_record is None:
                record = {
                    'Subject': subject,
                    'Session': session if session else '',  # Empty string for datasets without sessions
                    'T1_SkullStrip_QC': 'N/A',
                    'T1_Brain_Volume_ml': 'N/A',
                    'T2_SkullStrip_QC': 'N/A',
                    'T2_Brain_Volume_ml': 'N/A',
                    'T1_Dice_MNI': 'N/A',        # Simplified: just one Dice per modality
                    'T2_Dice_MNI': 'N/A',
                    'Overall_Status': 'UNKNOWN',
                    'HTML_Report': 'N/A'
                }
                summary_data.append(record)
                existing_record = record
            
            # Update modality-specific data
            modality = row['modality']
            qc_status = row['qc_status']  # Use the status from CSV directly
            brain_volume = row['brain_volume_ml']
            
            if modality == 'T1w':
                existing_record['T1_SkullStrip_QC'] = qc_status
                existing_record['T1_Brain_Volume_ml'] = brain_volume
            elif modality == 'T2w':
                existing_record['T2_SkullStrip_QC'] = qc_status
                existing_record['T2_Brain_Volume_ml'] = brain_volume
        
        # Add registration metrics (real Dice coefficients using MNI template)
        for record in summary_data:
            subject = record['Subject']
            session = record['Session']
            
            # Calculate actual Dice coefficients
            dice_results = self.calculate_registration_dice(subject, session)
            record.update(dice_results)
        
        # Filter out records without T1 (since T2 registration depends on T1)
        summary_data = [record for record in summary_data 
                       if record.get('T1_SkullStrip_QC') != 'N/A']
        
        # Calculate overall status
        for record in summary_data:
            status_flags = []
            
            # Check T1 skull stripping
            if record['T1_SkullStrip_QC'] == 'PASS':
                status_flags.append(True)
            elif record['T1_SkullStrip_QC'] == 'FAIL':
                status_flags.append(False)
            
            # Check T2 skull stripping  
            if record['T2_SkullStrip_QC'] == 'PASS':
                status_flags.append(True)
            elif record['T2_SkullStrip_QC'] == 'FAIL':
                status_flags.append(False)
            
            # Check T1 Dice - REQUIRED (if missing = FAIL)
            t1_dice = record['T1_Dice_MNI']
            if (not isinstance(t1_dice, numbers.Real)) or (isinstance(t1_dice, numbers.Real) and np.isnan(t1_dice)):
                status_flags.append(False)
            else:
                status_flags.append(t1_dice >= config.DICE_PASS_THRESHOLD)
            
            # Check T2 Dice - OPTIONAL (if missing/NaN = ignore)
            t2_dice = record['T2_Dice_MNI']
            if isinstance(t2_dice, numbers.Real) and not np.isnan(t2_dice):
                status_flags.append(t2_dice >= config.DICE_PASS_THRESHOLD)
            
            # Overall status
            if len(status_flags) == 0:
                record['Overall_Status'] = 'UNKNOWN'
            elif all(status_flags):
                record['Overall_Status'] = 'PASS'
            elif any(status_flags):
                record['Overall_Status'] = 'WARNING'
            else:
                record['Overall_Status'] = 'FAIL'
        
        # Create HTML report links
        for record in summary_data:
            subject = record['Subject']
            session = record['Session']
            if self.has_sessions and session:
                html_filename = f"{subject}_{session}_report.html"
            else:
                html_filename = f"{subject}_report.html"
            record['HTML_Report'] = html_filename
        
        # Create DataFrame and sort by Subject then Session for proper grouping
        df = pd.DataFrame(summary_data)
        if not df.empty:
            if self.has_sessions:
                df = df.sort_values(['Subject', 'Session']).reset_index(drop=True)
            else:
                df = df.sort_values(['Subject']).reset_index(drop=True)
        return df
    
    def generate_subject_html_report(self, subject, session, output_dir):
        """Generate detailed HTML report for a specific subject/session"""
        
        # Helper function for safe CSS class names
        def status_class(s):
            s = s or 'UNKNOWN'
            s_norm = str(s).strip().upper()
            return {'PASS':'pass','FAIL':'fail','WARNING':'warning','UNKNOWN':'unknown'}.get(s_norm,'unknown')
        
        subject_short = subject.replace('sub-', '')
        session_short = session.replace('ses-', '') if session else 'no-session'
        
        # Paths to relevant files using configurable folder structure
        if self.has_sessions and session:
            skullstrip_folder = self.folder_structure["skullstrip"].format(subject=subject, session=session)
            registration_folder = self.folder_structure["registration"].format(subject=subject, session=session)
        else:
            # For datasets without sessions
            skullstrip_folder = self.folder_structure["skullstrip"].format(subject=subject, session="")
            registration_folder = self.folder_structure["registration"].format(subject=subject, session="")

        skullstrip_base = self.skullstrip_dir / skullstrip_folder
        registration_base = self.registration_dir / registration_folder
        
        # T1w/T2w QC PNGs (flexible patterns across datasets)
        t1_matches = list(skullstrip_base.glob("*T1*_desc-qc.png"))
        t1_qc_img = t1_matches[0] if t1_matches else (skullstrip_base / "__missing__")
        
        t2_matches = list(skullstrip_base.glob("*T2*_desc-qc.png"))
        t2_qc_img = t2_matches[0] if t2_matches else (skullstrip_base / "__missing__")
        
        # Registration visualizations
        multislice_viz = registration_base / f"multislice_visualization_{subject}.png"
        
        # Registration output files
        reg_files = {
            'T1_warped': registration_base / "T1w_mni_warped_n4.nii.gz",
            'T1_zscore': registration_base / "T1w_mni_zscore_fixed.nii.gz", 
            'T1_zscore_cropped': registration_base / "T1w_mni_zscore_fixed_cropped.nii.gz",
            'T1_mask': registration_base / "T1w_mni_mask.nii.gz",
            'T2_warped': registration_base / "T2w_mni_warped_n4.nii.gz",
            'T2_zscore': registration_base / "T2w_mni_zscore_fixed.nii.gz",
            'T2_zscore_cropped': registration_base / "T2w_mni_zscore_fixed_cropped.nii.gz", 
            'T2_mask': registration_base / "T2w_mni_mask.nii.gz"
        }
        
        # Get actual QC data for this subject
        skullstrip_df = self.load_skullstrip_data()
        subject_data = {'T1': {}, 'T2': {}}
        
        if skullstrip_df is not None:
            for _, row in skullstrip_df.iterrows():
                subj, sess = self.extract_subject_session(row['input_file'])
                if subj == subject and sess == session:
                    modality = 'T1' if row['modality'] == 'T1w' else 'T2'
                    subject_data[modality] = {
                        'volume': row['brain_volume_ml'],
                        'status': self.assess_brain_volume(row['brain_volume_ml']),
                        'qc_image': row['qc_image_path']
                    }
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Structural QC Report - {subject} {session}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 30px; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .info-card {{ background: #ecf0f1; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .info-card h3 {{ margin-top: 0; color: #2c3e50; }}
        .status-pass {{ color: #27ae60; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        .status-warning {{ color: #f39c12; font-weight: bold; }}
        .status-unknown {{ color: #95a5a6; font-weight: bold; }}
        .image-container {{ text-align: center; margin: 20px 0; }}
        .image-container img {{ max-width: 100%; height: auto; border: 2px solid #bdc3c7; border-radius: 8px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .navigation {{ text-align: center; margin: 30px 0; }}
        .nav-button {{ background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 0 10px; }}
        .nav-button:hover {{ background-color: #2980b9; }}

        /* Modal Styles */
        .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); }}
        .modal-content {{ margin: auto; display: block; max-width: 90%; max-height: 90%; }}
        .modal-content img {{ width: 100%; height: auto; }}
        .close {{ position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }}
        .close:hover, .close:focus {{ color: #bbb; text-decoration: none; cursor: pointer; }}
        .image-container img {{ cursor: pointer; transition: transform 0.3s; }}
        .image-container img:hover {{ transform: scale(1.02); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.dataset_name} Structural MRI QC Report</h1>
        <h2>Subject: {subject}{" | Session: " + session if session else ""}</h2>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>📊 Processing Summary</h3>
                <p><strong>Subject ID:</strong> {subject}</p>
                <p><strong>Session:</strong> {session}</p>
                <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="info-card">
                <h3>Skull Stripping QC</h3>
                <p><strong>T1w Status:</strong> <span class="status-{status_class(subject_data['T1'].get('status'))}">{subject_data['T1'].get('status', 'N/A')}</span></p>
                <p><strong>T1w Volume:</strong> {subject_data['T1'].get('volume', 'N/A')} ml</p>
                <p><strong>T2w Status:</strong> <span class="status-{status_class(subject_data['T2'].get('status'))}">{subject_data['T2'].get('status', 'N/A')}</span></p>
                <p><strong>T2w Volume:</strong> {subject_data['T2'].get('volume', 'N/A')} ml</p>
            </div>
            
            <div class="info-card">
                <h3>Registration QC</h3>
                <p><strong>T1 → MNI:</strong> <span class="status-pass">AVAILABLE</span></p>
                <p><strong>T2 → MNI:</strong> <span class="status-pass">AVAILABLE</span></p>
                <p><strong><a href="{os.path.relpath(registration_base, output_dir)}" style="color: #3498db;">Browse Files</a></strong></p>
            </div>
        </div>
        
        <h2>Skull Stripping Visualizations</h2>
        <div class="image-grid">
"""
        
        # Add T1w skull stripping visualization
        if t1_qc_img.exists():
            t1_rel_path = os.path.relpath(t1_qc_img, output_dir)
            t1_url_path = quote(t1_rel_path.replace('\\', '/'))
            html_content += f"""
            <div class="image-container">
                <h3>T1w Skull Stripping</h3>
                <img src="{t1_url_path}" alt="T1w Skull Stripping QC" style="width: 100%; max-width: 600px;" onclick="openModal(this.src, this.alt)">
            </div>
"""
        else:
            html_content += """
            <div class="image-container">
                <h3>T1w Skull Stripping</h3>
                <p>❌ QC image not found</p>
            </div>
"""
        
        # Add T2w skull stripping visualization
        if t2_qc_img.exists():
            t2_rel_path = os.path.relpath(t2_qc_img, output_dir)
            t2_url_path = quote(t2_rel_path.replace('\\', '/'))
            html_content += f"""
            <div class="image-container">
                <h3>T2w Skull Stripping</h3>
                <img src="{t2_url_path}" alt="T2w Skull Stripping QC" style="width: 100%; max-width: 600px;" onclick="openModal(this.src, this.alt)">
            </div>
"""
        else:
            html_content += """
            <div class="image-container">
                <h3>T2w Skull Stripping</h3>
                <p>❌ QC image not found</p>
            </div>
"""
        
        html_content += """
        </div>
        
        <h2>Registration Visualization</h2>
        <div style="text-align: center; margin: 20px 0;">
"""
        
        # Add registration visualization (centered)
        if multislice_viz.exists():
            multi_rel_path = os.path.relpath(multislice_viz, output_dir)
            multi_url_path = quote(multi_rel_path.replace('\\', '/'))
            html_content += f"""
            <div class="image-container">
                <h3>MNI Registration Quality</h3>
                <img src="{multi_url_path}" alt="MNI Registration" style="width: 100%; max-width: 800px; border: 2px solid #3498db;" onclick="openModal(this.src, this.alt)">
            </div>
"""
        else:
            html_content += """
            <div class="image-container">
                <h3>MNI Registration Quality</h3>
                <p>❌ Registration visualization not found</p>
            </div>
"""
        
        html_content += f"""
        </div>
        
        <h2>📋 Detailed Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Status</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>T1w Brain Volume</td>
                    <td>{subject_data['T1'].get('volume', 'N/A')} ml</td>
                    <td><span class="status-{status_class(subject_data['T1'].get('status'))}">{subject_data['T1'].get('status', 'N/A')}</span></td>
                    <td>Should be between {self.brain_volume_min}-{self.brain_volume_max} ml</td>
                </tr>
                <tr>
                    <td>T2w Brain Volume</td>
                    <td>{subject_data['T2'].get('volume', 'N/A')} ml</td>
                    <td><span class="status-{status_class(subject_data['T2'].get('status'))}">{subject_data['T2'].get('status', 'N/A')}</span></td>
                    <td>Should be between {self.brain_volume_min}-{self.brain_volume_max} ml</td>
                </tr>
        """
        
        # Get actual Dice coefficients for this subject
        dice_results = self.calculate_registration_dice(subject, session)
        
        def format_dice_status(dice_value):
            if not isinstance(dice_value, numbers.Real) or np.isnan(dice_value):
                return '<span class="status-unknown">UNKNOWN</span>'
            return '<span class="status-pass">PASS</span>' if dice_value >= config.DICE_PASS_THRESHOLD else '<span class="status-fail">FAIL</span>'
        
        def format_dice_value(dice_value):
            if isinstance(dice_value, numbers.Real):
                return '—' if np.isnan(dice_value) else f"{dice_value:.3f}"
            return 'N/A'
        
        html_content += f"""
                <tr>
                    <td>T1 → MNI Registration Dice</td>
                    <td>{format_dice_value(dice_results['T1_Dice_MNI'])}</td>
                    <td>{format_dice_status(dice_results['T1_Dice_MNI'])}</td>
                    <td>Should be ≥ {config.DICE_PASS_THRESHOLD}</td>
                </tr>
                <tr>
                    <td>T2 → MNI Registration Dice</td>
                    <td>{format_dice_value(dice_results['T2_Dice_MNI'])}</td>
                    <td>{format_dice_status(dice_results['T2_Dice_MNI'])}</td>
                    <td>Should be ≥ {config.DICE_PASS_THRESHOLD}</td>
                </tr>
            </tbody>
        </table>
        
        <h2>Generated Files</h2>
        <div class="info-grid">
            <div class="info-card">
                <h3>T1w Outputs</h3>
                <ul style="text-align: left; padding-left: 20px;">
"""
        
        # Add T1 files
        for key, filepath in reg_files.items():
            if key.startswith('T1_'):
                if filepath.exists():
                    rel_path = os.path.relpath(filepath, output_dir)
                    url_path = quote(rel_path.replace('\\', '/'))
                    filename = filepath.name
                    html_content += f'<li><a href="{url_path}" style="color: #3498db;">{filename}</a> ✅</li>\n'
                else:
                    filename = filepath.name
                    html_content += f'<li>{filename} ❌</li>\n'
        
        html_content += """
                </ul>
            </div>
            
            <div class="info-card">
                <h3>T2w Outputs</h3>
                <ul style="text-align: left; padding-left: 20px;">
"""
        
        # Add T2 files  
        for key, filepath in reg_files.items():
            if key.startswith('T2_'):
                if filepath.exists():
                    rel_path = os.path.relpath(filepath, output_dir)
                    url_path = quote(rel_path.replace('\\', '/'))
                    filename = filepath.name
                    html_content += f'<li><a href="{url_path}" style="color: #3498db;">{filename}</a> ✅</li>\n'
                else:
                    filename = filepath.name
                    html_content += f'<li>{filename} ❌</li>\n'
        
        html_content += f"""
                </ul>
            </div>
        </div>
        
        <div class="navigation">
            <a href="../structural_qc_summary.html" class="nav-button">← Back to Summary</a>
            <a href="{os.path.relpath(registration_base, output_dir)}" class="nav-button">Browse Files</a>
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
            <p>Design & Develop by <a href="https://stai.stanford.edu/" style="color: #3498db;">Stanford Translational AI (STAI)</a></p>
        </div>
    </div>

    <!-- Modal for image popup -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function openModal(src, alt) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = src;
            modalImg.alt = alt;
        }}

        function closeModal() {{
            const modal = document.getElementById('imageModal');
            modal.style.display = 'none';
        }}

        // Close modal when clicking outside the image
        window.onclick = function(event) {{
            const modal = document.getElementById('imageModal');
            if (event.target == modal) {{
                modal.style.display = 'none';
            }}
        }}

        // Close modal with ESC key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});
    </script>
</body>
</html>
"""
        
        return html_content
    
    def generate_summary_html(self, df, output_path):
        """Generate summary HTML report with all subjects"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Structural MRI QC Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #3498db; }}
        .stat-card h3 {{ margin-top: 0; color: #2c3e50; font-size: 1.1em; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #ddd; font-size: 0.9em; }}
        th {{ background-color: #3498db; color: white; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #e8f4f8; }}
        .subject-merged {{ vertical-align: middle; font-weight: bold; }}
        .status-pass {{ color: #27ae60; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        .status-warning {{ color: #f39c12; font-weight: bold; }}
        .status-unknown {{ color: #95a5a6; font-weight: bold; }}
        .subject-separator {{ border-top: 3px solid #3498db; }}
        .filter-container {{ margin: 20px 0; text-align: center; }}
        .filter-container input {{ padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin: 0 5px; }}
        .filter-container select {{ padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin: 0 5px; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .table-container {{ overflow-x: auto; max-height: 600px; }}
    </style>
    <script>
        function filterTable() {{
            const subjectFilter = document.getElementById('subjectFilter').value.toLowerCase();
            const statusFilter = document.getElementById('statusFilter').value;
            const table = document.getElementById('summaryTable');
            const tbody = table.getElementsByTagName('tbody')[0];
            const allRows = Array.from(tbody ? tbody.rows : []);

            // Build subject groups (handles rowspan)
            let groups = [];
            for (let i = 0; i < allRows.length;) {{
                const row = allRows[i];
                const firstCell = row.cells[0];
                if (firstCell && firstCell.textContent.trim() !== '') {{
                    const subjectText = firstCell.textContent.trim().toLowerCase();
                    const rs = firstCell.getAttribute('rowspan');
                    const sessionCount = rs ? parseInt(rs) : 1;
                    const groupRows = [];
                    for (let j = 0; j < sessionCount && (i + j) < allRows.length; j++) {{
                        groupRows.push(allRows[i + j]);
                    }}
                    groups.push({{ subject: subjectText, rows: groupRows }});
                    i += sessionCount;
                }} else {{
                    if (groups.length > 0) {{
                        groups[groups.length - 1].rows.push(row);
                    }}
                    i += 1;
                }}
            }}

            // Apply filters per group to avoid column misalignment
            groups.forEach(group => {{
                let showGroup = true;

                if (subjectFilter && !group.subject.includes(subjectFilter)) {{
                    showGroup = false;
                }}

                if (showGroup && statusFilter && statusFilter !== 'ALL') {{
                    let anyMatch = false;
                    for (let r of group.rows) {{
                        const overallIdx = r.cells.length >= 10 ? 8 : 7;
                        const statusCell = r.cells[overallIdx];
                        const span = statusCell ? statusCell.querySelector('span') : null;
                        const text = span ? span.textContent.trim() : (statusCell ? statusCell.textContent.trim() : '');
                        if (text === statusFilter) {{
                            anyMatch = true;
                            break;
                        }}
                    }}
                    if (!anyMatch) {{
                        showGroup = false;
                    }}
                }}

                group.rows.forEach(r => {{
                    r.style.display = showGroup ? '' : 'none';
                }});
            }});
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>{self.dataset_name} Structural MRI Quality Control Summary</h1>
        <p style="text-align: center; color: #7f8c8d;">{self.dataset_name} Dataset Preprocessing QC Report</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Subjects & Sessions</h3>
                <div style="font-size: 1.2em; color: #3498db; margin: 5px 0;">
                    <strong>{len(df['Subject'].unique()) if not df.empty else 0}</strong> Subjects
                </div>
                <div style="font-size: 1.2em; color: #3498db;">
                    <strong>{len(df) if not df.empty else 0}</strong> Sessions
                </div>
            </div>
            <div class="stat-card">
                <h3>T1 & T2 Available</h3>
                <div style="font-size: 1.2em; color: #3498db; margin: 5px 0;">
                    <strong>{len(df[df['T1_SkullStrip_QC'] != 'N/A'])}</strong> T1 Available
                </div>
                <div style="font-size: 1.2em; color: #9b59b6;">
                    <strong>{len(df[df['T2_SkullStrip_QC'] != 'N/A'])}</strong> T2 Available
                </div>
            </div>
            <div class="stat-card">
                <h3>PASS</h3>
                <div class="stat-value" style="color: #27ae60;">{len(df[df['Overall_Status'] == 'PASS'])}</div>
            </div>
            <div class="stat-card">
                <h3>WARNING</h3>
                <div class="stat-value" style="color: #f39c12;">{len(df[df['Overall_Status'] == 'WARNING'])}</div>
            </div>
            <div class="stat-card">
                <h3>FAIL</h3>
                <div class="stat-value" style="color: #e74c3c;">{len(df[df['Overall_Status'] == 'FAIL'])}</div>
            </div>
        </div>
        
        <div class="filter-container">
            <input type="text" id="subjectFilter" placeholder="Filter by subject..." onkeyup="filterTable()">
            <select id="statusFilter" onchange="filterTable()">
                <option value="ALL">All Status</option>
                <option value="PASS">PASS</option>
                <option value="WARNING">WARNING</option>
                <option value="FAIL">FAIL</option>
                <option value="UNKNOWN">UNKNOWN</option>
            </select>
        </div>
        
        <div class="table-container">
            <table id="summaryTable">
                <thead>
                    <tr>
                        <th>Subject</th>
                        <th>Session</th>
                        <th>T1 Skull Strip</th>
                        <th>T1 Volume (ml)</th>
                        <th>T2 Skull Strip</th>
                        <th>T2 Volume (ml)</th>
                        <th>T1 → MNI Dice</th>
                        <th>T2 → MNI Dice</th>
                        <th>Overall Status</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add table rows with subject column merging
        current_subject = None
        subject_sessions = df.groupby('Subject').size().to_dict()  # Count sessions per subject
        
        for idx, (_, row) in enumerate(df.iterrows()):
            # Format status with colors
            def format_status(status):
                if status == 'PASS':
                    return f'<span class="status-pass">{status}</span>'
                elif status == 'FAIL':
                    return f'<span class="status-fail">{status}</span>'
                elif status == 'WARNING':
                    return f'<span class="status-warning">{status}</span>'
                elif status == 'N/A' or pd.isna(status):
                    return '<span class="status-unknown">—</span>'
                else:
                    return f'<span class="status-unknown">{status}</span>'
            
            # Format numbers
            def format_number(value):
                if pd.isna(value) or value == 'N/A' or value in (None, ''):
                    return '—'
                try:
                    num = float(value)
                    if np.isnan(num):
                        return '—'
                    return str(int(num)) if float(num).is_integer() else f"{num:.3f}"
                except:
                    return '—' if value in (None, 'N/A', '') else str(value)
            
            # Check if this is a new subject or continuation
            is_new_subject = current_subject != row['Subject']
            subject_cell = ""
            row_class = ""
            
            if is_new_subject:
                if current_subject is not None:  # Not the first subject
                    row_class = ' class="subject-separator"'
                current_subject = row['Subject']
                session_count = subject_sessions[current_subject]
                if session_count > 1:
                    # Multi-session subject: add rowspan with vertical center alignment
                    subject_cell = f'<td rowspan="{session_count}" class="subject-merged">{row["Subject"]}</td>'
                else:
                    # Single session subject: normal cell
                    subject_cell = f'<td>{row["Subject"]}</td>'
            else:
                # Continuation of same subject: no subject cell (it's merged above)
                subject_cell = ""
            
            html_content += f"""
                    <tr{row_class}>
                        {subject_cell}
                        <td>{row['Session']}</td>
                        <td>{format_status(row['T1_SkullStrip_QC'])}</td>
                        <td>{format_number(row['T1_Brain_Volume_ml'])}</td>
                        <td>{format_status(row['T2_SkullStrip_QC'])}</td>
                        <td>{format_number(row['T2_Brain_Volume_ml'])}</td>
                        <td>{format_number(row['T1_Dice_MNI'])}</td>
                        <td>{format_number(row['T2_Dice_MNI'])}</td>
                        <td>{format_status(row['Overall_Status'])}</td>
                        <td><a href="subjects/{row['HTML_Report']}" target="_blank">📊 View</a></td>
                    </tr>
"""
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        
        <!-- Metadata Download Section -->
        <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-bottom: 15px; text-align: center;">📊 Download Metadata</h3>
            <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">"""
        
        # Add T1 CSV download if exists
        from config import CSV_METADATA_DIR, T1_CSV_FILENAME, T2_CSV_FILENAME
        import os
        
        t1_csv_path = os.path.join(CSV_METADATA_DIR, T1_CSV_FILENAME)
        t2_csv_path = os.path.join(CSV_METADATA_DIR, T2_CSV_FILENAME)
        
        if os.path.exists(t1_csv_path):
            t1_rel_path = os.path.relpath(t1_csv_path, os.path.dirname(output_path))
            html_content += f"""
                <a href="{t1_rel_path}" download style="display: inline-block; padding: 12px 24px; background-color: #3498db; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; margin: 5px;">
                    📋 T1 Metadata CSV
                </a>"""
        
        if os.path.exists(t2_csv_path):
            t2_rel_path = os.path.relpath(t2_csv_path, os.path.dirname(output_path))
            html_content += f"""
                <a href="{t2_rel_path}" download style="display: inline-block; padding: 12px 24px; background-color: #e74c3c; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; margin: 5px;">
                    📋 T2 Metadata CSV
                </a>"""
        
        html_content += """
            </div>
            <p style="text-align: center; margin-top: 15px; font-size: 0.9em; color: #7f8c8d;">
                CSV files contain file paths, demographics, and processing metadata for all subjects
            </p>
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
            <p>Designed & Developed by <a href="https://stai.stanford.edu/" style="color: #3498db;">Stanford Translational AI (STAI)</a></p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Summary HTML report saved to: {output_path}")
    
    def run_qc(self, subject_filter=None):
        """Run complete QC analysis"""
        
        logger.info("Starting Structural MRI QC analysis...")
        
        # Create summary dataframe
        df = self.create_summary_dataframe(subject_filter=subject_filter)
        if df is None:
            logger.error("Failed to create summary dataframe")
            return False
        
        logger.info(f"Generated QC summary for {len(df)} subjects")
        
        # Save CSV summary
        csv_path = self.qc_dir / "structural_qc_summary.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"CSV summary saved to: {csv_path}")
        
        # Create subjects directory for individual reports
        subjects_dir = self.qc_dir / "subjects"
        subjects_dir.mkdir(exist_ok=True)
        
        # Generate individual HTML reports
        for _, row in df.iterrows():
            subject = row['Subject']
            session = row['Session'] if row['Session'] else None

            html_content = self.generate_subject_html_report(subject, session, subjects_dir)
            html_path = subjects_dir / row['HTML_Report']

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            session_info = f" {session}" if session else ""
            logger.info(f"Generated HTML report for {subject}{session_info}")
        
        # Generate summary HTML
        summary_html_path = self.qc_dir / "structural_qc_summary.html"
        self.generate_summary_html(df, summary_html_path)
        
        logger.info("QC analysis completed successfully!")
        logger.info(f"Summary HTML: {summary_html_path}")
        logger.info(f"Individual reports: {subjects_dir}")
        
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Structural MRI Quality Control')
    parser.add_argument('--subject', help='Process specific subject (e.g., sub-ON62003)')
    
    args = parser.parse_args()
    
    # Initialize QC (always use config.py)
    qc = StructuralQC()
    
    # Run QC
    success = qc.run_qc(subject_filter=args.subject)
    
    if success:
        logger.info("QC completed successfully!")
        return 0
    else:
        logger.error("QC failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
