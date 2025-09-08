#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV Metadata Generator for OpenNeuro Dataset
Generates CSV files with subject metadata and file paths for T1 and T2 neuroimaging data.

This script creates structured CSV files containing:
- Subject identifiers
- File paths to processed brain images (MNI warped, Z-score normalized, cropped)
- Demographic information (age, encoded sex, encoded handedness)

Features:
- Multiprocessing support for fast execution
- Configurable file patterns and encoding schemes
- Automatic file existence validation
- Comprehensive logging and error handling

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""

import os
import pandas as pd
import glob
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import time
import fnmatch
from datetime import datetime

# Import configuration settings
try:
    from config import *
except ImportError:
    print("Error: Could not import config.py. Make sure it's in the same directory.")
    sys.exit(1)

# Setup logging with timestamped log files in LOG_DIR (like other scripts)
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_log_level = getattr(logging, getattr(sys.modules['config'], 'CSV_LOG_LEVEL', 'INFO'))
logging.basicConfig(
    level=csv_log_level,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'generate_csv_metadata_{timestamp}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CSVMetadataGenerator:
    """
    Generate CSV metadata files for neuroimaging data.
    
    This class handles the complete workflow of CSV generation:
    1. Loading participant demographics from TSV files
    2. Scanning directories for processed neuroimaging files
    3. Matching subjects with their corresponding files
    4. Encoding categorical variables (sex, handedness)
    5. Generating structured CSV outputs
    """
    
    def __init__(self):
        """Initialize the CSV metadata generator with configuration settings."""
        self.participants_data = None  # Will store loaded participant demographics
        self.output_dir = CSV_METADATA_DIR  # Directory for CSV output files
        self.registration_dir = REGISTRATION_DIR  # Directory containing processed images
        self.participants_tsv_path = PARTICIPANTS_TSV_PATH  # Path to demographics file
        
        # Performance settings from configuration
        self.use_multiprocessing = getattr(sys.modules['config'], 'CSV_USE_MULTIPROCESSING', False)
        self.max_workers = getattr(sys.modules['config'], 'CSV_MAX_WORKERS', 4)
        self.batch_size = getattr(sys.modules['config'], 'CSV_BATCH_SIZE', 10)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_participants_data(self) -> pd.DataFrame:
        """
        Load participants metadata from TSV file.
        
        Returns:
            pd.DataFrame: Cleaned participants data with demographics
        """
        try:
            logger.info(f"Loading participants data from: {self.participants_tsv_path}")
            
            if not os.path.exists(self.participants_tsv_path):
                raise FileNotFoundError(f"Participants file not found: {self.participants_tsv_path}")
            
            # Read TSV file with tab separator
            df = pd.read_csv(self.participants_tsv_path, sep='\t')
            logger.info(f"Loaded {len(df)} participants from TSV file")
            
            # Clean and standardize the data
            df = self._clean_participants_data(df)
            
            self.participants_data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading participants data: {str(e)}")
            raise
    
    def _clean_participants_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize participants data.
        
        Args:
            df: Raw participants dataframe from TSV file
            
        Returns:
            pd.DataFrame: Cleaned dataframe with standardized missing values
        """
        logger.info("Cleaning participants data...")
        
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Handle missing values by filling with standard missing data indicator
        df['age'] = df['age'].fillna(MISSING_DATA_VALUE)
        df['sex'] = df['sex'].fillna(MISSING_DATA_VALUE)
        df['handedness'] = df['handedness'].fillna(MISSING_DATA_VALUE)
        
        # Convert age to string to handle both numeric and 'n/a' values consistently
        df['age'] = df['age'].astype(str)
        
        logger.info(f"Cleaned data: {len(df)} participants")
        return df
    
    def get_subject_list(self) -> List[str]:
        """
        Get list of subjects from registration directory.
        
        Scans the registration directory for subdirectories matching the pattern 'sub-*'
        which indicates BIDS-formatted subject directories.
        
        Returns:
            List[str]: Sorted list of subject IDs (e.g., ['sub-ON00001', 'sub-ON00002', ...])
        """
        try:
            if not os.path.exists(self.registration_dir):
                raise FileNotFoundError(f"Registration directory not found: {self.registration_dir}")
            
            # Get all subject directories following BIDS naming convention
            subject_dirs = [d for d in os.listdir(self.registration_dir) 
                           if d.startswith('sub-') and os.path.isdir(os.path.join(self.registration_dir, d))]
            
            logger.info(f"Found {len(subject_dirs)} subjects in registration directory")
            return sorted(subject_dirs)
            
        except Exception as e:
            logger.error(f"Error getting subject list: {str(e)}")
            raise
    
    def find_processed_files_batch(self, subjects: List[str], modality: str) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Find processed files for multiple subjects at once (optimized for performance).
        
        This method is optimized to reduce filesystem I/O by:
        - Reading directory contents once per subject
        - Using string matching instead of glob patterns
        - Processing subjects in batches
        
        Args:
            subjects: List of subject IDs to process
            modality: Modality type ('T1' or 'T2')
            
        Returns:
            Dict[str, Dict[str, Optional[str]]]: Nested dict mapping subject_id -> file_type -> file_path
        """
        # Select appropriate file patterns based on modality
        if modality == 'T1':
            patterns = T1_PROCESSED_PATTERNS
        elif modality == 'T2':
            patterns = T2_PROCESSED_PATTERNS
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        results = {}
        
        # Process each subject in the batch
        for subject_id in subjects:
            file_paths = {}
            
            # Construct path to subject's anatomy directory (BIDS structure)
            subject_anat_dir = os.path.join(self.registration_dir, subject_id, "ses-01", "anat")
            
            # Skip subjects without anatomy directory
            if not os.path.exists(subject_anat_dir):
                results[subject_id] = {key: None for key in patterns.keys()}
                continue
            
            # Get all files in directory once (more efficient than multiple glob calls)
            try:
                all_files = os.listdir(subject_anat_dir)
            except OSError:
                results[subject_id] = {key: None for key in patterns.keys()}
                continue
            
            # Find files matching each pattern
            for file_type, pattern in patterns.items():
                # Use fnmatch for proper wildcard matching
                matching_files = [f for f in all_files if fnmatch.fnmatch(f, pattern)]
                
                # Store the first matching file (or None if no match)
                if matching_files:
                    file_path = os.path.join(subject_anat_dir, matching_files[0])
                    file_paths[file_type] = file_path
                else:
                    file_paths[file_type] = None
            
            results[subject_id] = file_paths
        
        return results
    
    def process_subject_batch(self, subjects: List[str], modality: str) -> List[Dict]:
        """
        Process a batch of subjects and generate CSV row data.
        
        Args:
            subjects: List of subject IDs to process
            modality: Modality type ('T1' or 'T2')
            
        Returns:
            List[Dict]: List of dictionaries, each representing a CSV row
        """
        csv_data = []
        
        # Get file paths for all subjects in this batch (batch processing for efficiency)
        all_file_paths = self.find_processed_files_batch(subjects, modality)
        
        # Process each subject in the batch
        for subject_id in subjects:
            # Get subject's demographic information with encoded categorical variables
            metadata = self.get_subject_metadata(subject_id)
            file_paths = all_file_paths.get(subject_id, {})
            
            # Only include subjects that have at least one processed file for this modality
            if any(path is not None for path in file_paths.values()):
                # Create row data following the defined column structure
                row_data = {
                    'subjectId': subject_id,
                    'MNI_Warped': file_paths.get('MNI_Warped', MISSING_DATA_VALUE),
                    'MNI_ZSCORE': file_paths.get('MNI_ZSCORE', MISSING_DATA_VALUE),
                    'MNI_Z_Cropped': file_paths.get('MNI_Z_Cropped', MISSING_DATA_VALUE),
                    'age': metadata['age'],
                    'sex': metadata['sex_encoded'],
                    'handedness': metadata['handedness_encoded']
                }
                
                # Replace any None values with the standard missing data indicator
                for key, value in row_data.items():
                    if value is None:
                        row_data[key] = MISSING_DATA_VALUE
                
                csv_data.append(row_data)
        
        return csv_data
    
    def get_subject_metadata(self, subject_id: str) -> Dict[str, str]:
        """
        Get metadata for a specific subject with encoded categorical variables.
        
        Args:
            subject_id: Subject ID (e.g., 'sub-ON99943')
            
        Returns:
            Dict[str, str]: Subject metadata including age and encoded sex/handedness
        """
        if self.participants_data is None:
            raise ValueError("Participants data not loaded. Call load_participants_data() first.")
        
        # Find subject in the participants dataframe
        subject_row = self.participants_data[self.participants_data['participant_id'] == subject_id]
        
        # Handle case where subject is not found in demographics file
        if subject_row.empty:
            return {
                'age': MISSING_DATA_VALUE,
                'sex_encoded': SEX_ENCODING.get(MISSING_DATA_VALUE, 0),
                'handedness_encoded': HANDEDNESS_ENCODING.get(MISSING_DATA_VALUE, 0)
            }
        
        # Extract raw demographic values
        age = str(subject_row.iloc[0]['age'])
        sex = str(subject_row.iloc[0]['sex'])
        handedness = str(subject_row.iloc[0]['handedness'])
        
        # Return metadata with encoded categorical variables for ML compatibility
        return {
            'age': age,
            'sex_encoded': SEX_ENCODING.get(sex, SEX_ENCODING.get(MISSING_DATA_VALUE, 0)),
            'handedness_encoded': HANDEDNESS_ENCODING.get(handedness, HANDEDNESS_ENCODING.get(MISSING_DATA_VALUE, 0))
        }
    
    def generate_csv_for_modality(self, modality: str) -> pd.DataFrame:
        """
        Generate CSV data for a specific modality using optimized batch processing.
        
        Args:
            modality: Modality type ('T1' or 'T2')
            
        Returns:
            pd.DataFrame: Complete CSV data for the specified modality
        """
        start_time = time.time()
        logger.info(f"Generating CSV for {modality} modality...")
        
        # Get list of all subjects
        subjects = self.get_subject_list()
        all_csv_data = []
        
        # Split subjects into batches for efficient processing
        batches = [subjects[i:i + self.batch_size] for i in range(0, len(subjects), self.batch_size)]
        
        # Use multiprocessing if enabled and beneficial
        if self.use_multiprocessing and len(batches) > 1:
            logger.info(f"Using multiprocessing with {self.max_workers} workers, {len(batches)} batches")
            
            # Use ThreadPoolExecutor for I/O bound tasks (file system operations)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_processor = partial(self.process_subject_batch, modality=modality)
                batch_results = list(executor.map(batch_processor, batches))
                
                # Flatten results from all batches
                for batch_data in batch_results:
                    all_csv_data.extend(batch_data)
        else:
            # Process sequentially for small datasets or when multiprocessing is disabled
            for i, batch in enumerate(batches):
                if i % 5 == 0:  # Progress update every 5 batches
                    logger.info(f"Processing batch {i+1}/{len(batches)}...")
                batch_data = self.process_subject_batch(batch, modality)
                all_csv_data.extend(batch_data)
        
        # Create final DataFrame with specified column order
        df = pd.DataFrame(all_csv_data, columns=CSV_COLUMNS)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {modality} CSV with {len(df)} subjects in {elapsed_time:.2f} seconds")
        
        return df
    
    def save_csv(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save DataFrame to CSV file with appropriate formatting.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            str: Full path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # Save without row indices for cleaner CSV format
            df.to_csv(output_path, index=False)
            logger.info(f"Saved CSV file: {output_path}")
            logger.info(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving CSV file {output_path}: {str(e)}")
            raise
    
    def generate_all_csvs(self) -> Tuple[str, str]:
        """
        Generate both T1 and T2 CSV files.
        
        Returns:
            Tuple[str, str]: Paths to generated T1 and T2 CSV files
        """
        logger.info("Starting CSV generation for all modalities...")
        
        # Load participant demographics data
        self.load_participants_data()
        
        # Generate T1 CSV
        logger.info("=" * 50)
        t1_df = self.generate_csv_for_modality('T1')
        t1_path = self.save_csv(t1_df, T1_CSV_FILENAME)
        
        # Generate T2 CSV
        logger.info("=" * 50)
        t2_df = self.generate_csv_for_modality('T2')
        t2_path = self.save_csv(t2_df, T2_CSV_FILENAME)
        
        # Print summary of generated files
        logger.info("=" * 50)
        logger.info("CSV Generation Summary:")
        logger.info(f"T1 CSV: {t1_path} ({len(t1_df)} subjects)")
        logger.info(f"T2 CSV: {t2_path} ({len(t2_df)} subjects)")
        
        return t1_path, t2_path
    
    def print_statistics(self):
        """Print comprehensive statistics about the dataset and generated CSV files."""
        logger.info("=" * 50)
        logger.info("Dataset Statistics:")
        
        if self.participants_data is not None:
            logger.info(f"Total participants in TSV: {len(self.participants_data)}")
            
            # Age statistics (only for numeric ages)
            ages = self.participants_data['age']
            numeric_ages = pd.to_numeric(ages, errors='coerce').dropna()
            if len(numeric_ages) > 0:
                logger.info(f"Age range: {numeric_ages.min():.0f} - {numeric_ages.max():.0f} years")
                logger.info(f"Mean age: {numeric_ages.mean():.1f} ± {numeric_ages.std():.1f} years")
            
            # Sex distribution
            sex_counts = self.participants_data['sex'].value_counts()
            logger.info(f"Sex distribution: {dict(sex_counts)}")
            
            # Handedness distribution
            hand_counts = self.participants_data['handedness'].value_counts()
            logger.info(f"Handedness distribution: {dict(hand_counts)}")

def main():
    """
    Main function to generate CSV metadata files.
    
    This function orchestrates the complete CSV generation workflow:
    1. Initialize the generator
    2. Generate CSV files for both modalities
    3. Print comprehensive statistics
    4. Handle any errors gracefully
    """
    try:
        start_time = time.time()
        logger.info("Starting CSV metadata generation...")
        logger.info(f"Configuration loaded from config.py")
        
        # Check if CSV generation is enabled in configuration
        if not GENERATE_CSV_METADATA:
            logger.info("CSV metadata generation is disabled in config. Exiting.")
            return
        
        # Create generator instance and run the complete workflow
        generator = CSVMetadataGenerator()
        
        # Generate CSV files for both T1 and T2 modalities
        t1_path, t2_path = generator.generate_all_csvs()
        
        # Print comprehensive dataset statistics
        generator.print_statistics()
        
        # Report total execution time
        total_time = time.time() - start_time
        logger.info(f"CSV metadata generation completed successfully in {total_time:.2f} seconds!")
        
    except Exception as e:
        logger.error(f"Error in CSV generation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 