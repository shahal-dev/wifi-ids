#!/usr/bin/env python3
"""
AWID3 Dataset Preprocessing Pipeline
===================================

This script provides a comprehensive preprocessing pipeline for the AWID3 dataset.
It combines multiple preprocessing steps into a single command-line tool.

Steps included:
1. Drop null values and extract common columns
2. Drop specified columns
3. Encode categorical features
4. Clean IP version column
5. Label encoding with consistent mapping
6. Fill missing values with smart strategies
7. Extract common features across all datasets
8. Generate detailed column inconsistency report

Usage:
    python awid3_preprocessor.py --input /path/to/awid3/CSV --output /path/to/processed --steps all
    python awid3_preprocessor.py --input /path/to/awid3/CSV --output /path/to/processed --steps 1,2,3
"""

import os
import pandas as pd
import csv
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Set, Dict
from tqdm import tqdm
from collections import Counter, defaultdict
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
COLUMNS_TO_DROP = [
"udp.payload",
"http.request.line",
"wlan.sa",
"frame.time",
"tcp.checksum",
"http.request.version",
"http.request.line",
"wlan.ra",
"wlan.ta",
"ip.dst",
"ip.src",
"wlan.bssid",
"radiotap.rxflags",
"udp.payload",
"http.host",
"http.request.full_uri",
"wlan.da",
"tcp.analysis",
"llc",
"ssdp",
"frame.encap_type",
"tcp.checksum.status",
"tcp.flags.syn",
"tcp.flags.reset",
"tcp.flags.fin",
"tcp.flags.push",
"tcp.flags.ack",

]

KNOWN_LABELS = [
    'Deauth', 'Disas', '(Re)Assoc', 'RogueAP', 'Krack', 'Kr00K', 'SSH', 'Botnet',
    'Malware', 'SQL_Injection', 'SSDP', 'Evil_Twin', 'Website_spoofing', 'Normal'
]

UNIVERSAL_COLUMNS = [
    # Frame Features
    'frame.len', 'frame.number', 'frame.time_delta', 'frame.time_delta_displayed',
    'frame.time_epoch', 'frame.time_relative',
    # Radio/Physical Layer
    'radiotap.channel.flags.cck', 'radiotap.channel.flags.ofdm', 'radiotap.channel.freq',
    'radiotap.dbm_antsignal', 'radiotap.length', 'radiotap.present.tsft',
    'radiotap.timestamp.ts', 'wlan_radio.channel', 'wlan_radio.data_rate',
    'wlan_radio.duration', 'wlan_radio.frequency', 'wlan_radio.phy', 'wlan_radio.signal_dbm',
    # WLAN/802.11 Features
    'wlan.duration', 'wlan.fc.ds', 'wlan.fc.frag', 'wlan.fc.moredata', 'wlan.fc.order',
    'wlan.fc.protected', 'wlan.fc.pwrmgt', 'wlan.fc.retry', 'wlan.fc.subtype', 'wlan.fc.type',
    # Target Label
    'Label'
]


class AWID3Preprocessor:
    def __init__(self, input_dir: str, output_dir: str, chunk_size: int = 10000, common_features_only: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.common_features_only = common_features_only
        self.label_encoder = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize label encoder once
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(KNOWN_LABELS)
        
    def list_csv_files(self, folder_path: str) -> List[str]:
        """Get all CSV files in a folder."""
        return [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    


    def clean_ip_version(self, val):
        """Clean IP version values."""
        if pd.isna(val):
            return None
        val_str = str(val).strip()

        if val_str in ['4', '4.0', '4-4', '4-4-4', '04-04-04']:
            return 4
        elif val_str in ['6', '6.0']:
            return 6
        elif '4' in val_str and all(c in '4-' for c in val_str):
            return 4
        elif '6' in val_str and all(c in '6-' for c in val_str):
            return 6
        else:
            return None

    def clean_label_column(self, val):
        """Clean label column values."""
        if pd.isna(val):
            return None
        return str(val).strip()

    def smart_fill(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Smart fill strategy for missing values."""
        # If entire column is missing
        if df[col].isna().sum() == len(df):
            logger.warning(f"Column '{col}' has 100% missing values. Filling with placeholder.")
            if pd.api.types.is_numeric_dtype(df[col]):
                return df[col].fillna(-999)
            else:
                return df[col].fillna("missing")

        # Try to convert to numeric if possible
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

        # Custom rules
        if 'radiotap.dbm' in col.lower() or 'dbm' in col.lower():
            return df[col].fillna(-999)

        # Smart fill strategy based on column name
        if 'time' in col.lower() or 'seq' in col.lower() or 'ack' in col.lower():
            return df[col].fillna(df[col].median())
        elif 'ttl' in col.lower() or 'proto' in col.lower() or 'len' in col.lower():
            return df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else df[col].median())
        elif 'port' in col.lower():
            return df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else df[col].median())
        elif 'signal' in col.lower():
            return df[col].fillna(df[col].median())
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                return df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                return df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "missing")



    def categorize_column(self, col_name: str) -> str:
        """Categorize columns by their feature type"""
        col_lower = col_name.lower()
        
        # Frame-level features
        if col_name.startswith('frame.'):
            return 'Frame Features'
        
        # Radio/Physical layer features
        elif col_name.startswith('radiotap.') or col_name.startswith('wlan_radio.'):
            return 'Radio/Physical Layer'
        
        # WLAN/802.11 features
        elif col_name.startswith('wlan.'):
            return 'WLAN/802.11 Features'
        
        # IP layer features
        elif col_name.startswith('ip.'):
            return 'IP Layer Features'
        
        # UDP features
        elif col_name.startswith('udp.'):
            return 'UDP Features'
        
        # TCP features
        elif col_name.startswith('tcp.'):
            return 'TCP Features'
        
        # HTTP features
        elif col_name.startswith('http.'):
            return 'HTTP Features'
        
        # DNS features
        elif col_name.startswith('dns.'):
            return 'DNS Features'
        
        # Label column
        elif col_name == 'Label':
            return 'Target Label'
        
        # One-hot encoded variations (with .0, .1 suffixes)
        elif re.search(r'\.\d+$', col_name):
            base_name = re.sub(r'\.\d+$', '', col_name)
            return f'One-Hot Variants ({self.categorize_column(base_name)})'
        
        else:
            return 'Other/Unknown'

    def generate_column_inconsistency_report(self):
        """Generate comprehensive column inconsistency report."""
        logger.info("Generating column inconsistency report...")
        
        # Dictionary to store column information for each folder and file
        folder_column_info = {}
        all_columns = set()
        total_files = 0
        
        # Analyze each folder in the OUTPUT directory (processed files)
        folders = [f for f in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, f))]
        folders.sort()
        
        for folder in folders:
            folder_path = os.path.join(self.output_dir, folder)
            csv_files = self.list_csv_files(folder_path)
            
            folder_column_info[folder] = {}
            
            for csv_file in csv_files:
                file_path = os.path.join(folder_path, csv_file)
                try:
                    # Read just the header to get column names
                    df_sample = pd.read_csv(file_path, nrows=0)
                    columns = list(df_sample.columns)
                    
                    folder_column_info[folder][csv_file] = columns
                    all_columns.update(columns)
                    total_files += 1
                    
                except Exception as e:
                    logger.error(f"Error reading {csv_file}: {e}")
        
        # Generate report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("AWID3 DATASET COLUMN INCONSISTENCY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Folders: {len(folders)}")
        report_lines.append(f"Total Files: {total_files}")
        report_lines.append(f"Total Unique Columns: {len(all_columns)}")
        report_lines.append("")
        
        # Column frequency analysis
        column_frequency = {}
        for folder, files in folder_column_info.items():
            for file, columns in files.items():
                for col in columns:
                    column_frequency[col] = column_frequency.get(col, 0) + 1
        
        # Universal vs Partial columns
        universal_cols = [col for col, freq in column_frequency.items() if freq == total_files]
        partial_cols = [col for col, freq in column_frequency.items() if freq < total_files]
        
        report_lines.append("COLUMN DISTRIBUTION SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Universal Columns (in all files): {len(universal_cols)}")
        report_lines.append(f"Partial Columns (not in all files): {len(partial_cols)}")
        report_lines.append("")
        
        # Universal columns list
        if universal_cols:
            report_lines.append("UNIVERSAL COLUMNS:")
            report_lines.append("-" * 20)
            for i, col in enumerate(sorted(universal_cols), 1):
                report_lines.append(f"{i:2d}. {col}")
            report_lines.append("")
        
        # Feature category analysis
        categories = {
            'Frame': ['frame.', 'Frame'],
            'Radio/Physical': ['radiotap.', 'phy.'],
            'WLAN/802.11': ['wlan.', 'ieee80211.'],
            'Network/IP': ['ip.', 'ipv6.', 'eth.', 'arp.'],
            'Transport': ['tcp.', 'udp.', 'icmp.'],
            'Application': ['http.', 'dns.', 'dhcp.', 'ssl.', 'tls.'],
            'Other': []
        }
        
        categorized_cols = {cat: [] for cat in categories}
        uncategorized = []
        
        for col in all_columns:
            categorized = False
            for cat, prefixes in categories.items():
                if cat == 'Other':
                    continue
                if any(col.startswith(prefix) for prefix in prefixes):
                    categorized_cols[cat].append(col)
                    categorized = True
                    break
            if not categorized:
                uncategorized.append(col)
        
        categorized_cols['Other'] = uncategorized
        
        report_lines.append("FEATURE CATEGORIES:")
        report_lines.append("-" * 20)
        for cat, cols in categorized_cols.items():
            if cols:
                report_lines.append(f"{cat}: {len(cols)} columns")
        report_lines.append("")
        
        # Folder-specific analysis
        report_lines.append("FOLDER-SPECIFIC ANALYSIS:")
        report_lines.append("-" * 30)
        for folder in sorted(folders):
            files = folder_column_info[folder]
            if not files:
                continue
            
            folder_all_cols = set()
            for file_cols in files.values():
                folder_all_cols.update(file_cols)
            
            report_lines.append(f"\n{folder}:")
            report_lines.append(f"  Files: {len(files)}")
            report_lines.append(f"  Unique columns: {len(folder_all_cols)}")
            report_lines.append(f"  Universal columns in folder: {len([c for c in folder_all_cols if c in universal_cols])}")
        
        # Column frequency details (top partial columns)
        report_lines.append("\n" + "="*50)
        report_lines.append("PARTIAL COLUMNS FREQUENCY (Top 20):")
        report_lines.append("="*50)
        partial_col_freq = [(col, freq) for col, freq in column_frequency.items() if freq < total_files]
        partial_col_freq.sort(key=lambda x: x[1], reverse=True)
        
        for col, freq in partial_col_freq[:20]:
            percentage = (freq / total_files) * 100
            report_lines.append(f"{col:40s} : {freq:3d}/{total_files} files ({percentage:5.1f}%)")
        
        # Save report
        report_file = os.path.join(self.output_dir, "column_inconsistency_report.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"📊 Column inconsistency report saved to: {report_file}")

    def process_single_folder(self, folder_name: str) -> None:
        """Process a single attack folder completely through all steps."""
        folder_path = os.path.join(self.input_dir, folder_name)
        output_folder = os.path.join(self.output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        logger.info(f"🎯 Processing folder: {folder_name}")
        
        # Get CSV files
        csv_files = self.list_csv_files(folder_path)
        if not csv_files:
            logger.warning(f"No CSV files in {folder_name}. Skipping.")
            return
        
        # Process each CSV file in the folder
        for csv_file in tqdm(csv_files, desc=f"Processing {folder_name}"):
            input_file_path = os.path.join(folder_path, csv_file)
            output_file_path = os.path.join(output_folder, csv_file)
            
            try:
                # Step 1: Load data (common features or full)
                if self.common_features_only:
                    # Load only universal columns
                    df_sample = pd.read_csv(input_file_path, nrows=1)
                    available_universal_cols = [col for col in UNIVERSAL_COLUMNS if col in df_sample.columns]
                    
                    if not available_universal_cols:
                        logger.warning(f"No universal columns found in {csv_file}. Skipping.")
                        continue
                    
                    chunks = pd.read_csv(input_file_path, usecols=available_universal_cols, chunksize=self.chunk_size)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    # Load all columns (full mode)
                    chunks = pd.read_csv(input_file_path, chunksize=self.chunk_size)
                    df = pd.concat(chunks, ignore_index=True)
                
                # Step 2: Drop specified columns (if they exist)
                cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
                
                # Step 3: Encode categorical features
                if 'http.request.method' in df.columns:
                    le = LabelEncoder()
                    df['http.request.method'] = le.fit_transform(df['http.request.method'].astype(str))

                if 'wlan.fc.ds' in df.columns:
                    hex_map = {'0x00000000': 0, '0x00000001': 1, '0x00000002': 2, '0x00000003': 3}
                    df['wlan.fc.ds'] = df['wlan.fc.ds'].map(hex_map)

                if 'radiotap.present.tsft' in df.columns:
                    df['radiotap.present.tsft'] = df['radiotap.present.tsft'].apply(
                        lambda x: 1 if str(x) == '1-0-0' else 0
                    )
                
                # Step 4: Clean IP version
                if 'ip.version' in df.columns:
                    df['ip.version'] = df['ip.version'].apply(self.clean_ip_version)
                
                # Step 5: Encode labels
                if 'Label' in df.columns:
                    original_count = len(df)
                    df['Label'] = df['Label'].map(self.clean_label_column)
                    df = df[df['Label'].isin(KNOWN_LABELS)].copy()
                    
                    if len(df) < original_count:
                        dropped = original_count - len(df)
                        logger.warning(f"Dropped {dropped} rows with unknown labels in {csv_file}")
                    
                    if not df.empty:
                        df['Label'] = self.label_encoder.transform(df['Label'])
                
                # Step 6: Fill missing values
                for col in df.columns:
                    if col.lower() == 'label':
                        continue
                    try:
                        df[col] = self.smart_fill(df, col)
                    except Exception as e:
                        logger.warning(f"Skipping column '{col}' in {csv_file}: {e}")
                
                # Save processed file
                df.to_csv(output_file_path, index=False)
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
        
        logger.info(f"✅ Completed folder: {folder_name}")

    def run_preprocessing(self) -> None:
        """Run preprocessing on all folders."""
        start_time = time.time()
        
        # Save label mapping
        label_map = {label: int(self.label_encoder.transform([label])[0]) for label in KNOWN_LABELS}
        logger.info("Label encoding map:")
        for k, v in label_map.items():
            logger.info(f"  {k}: {v}")
        
        mapping_file = os.path.join(self.output_dir, "label_mapping.txt")
        with open(mapping_file, "w") as f:
            for k, v in label_map.items():
                f.write(f"{k}: {v}\n")
        
        # Get all attack folders
        folders = [f for f in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, f))]
        folders.sort()
        
        logger.info(f"📁 Found {len(folders)} attack folders to process")
        
        # Process each folder completely
        for folder in folders:
            self.process_single_folder(folder)
        
        # Generate column inconsistency report
        logger.info("📊 Generating column inconsistency report...")
        self.generate_column_inconsistency_report()
        
        end_time = time.time()
        logger.info(f"\n🎉 All preprocessing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"📁 Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="AWID3 Dataset Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Steps (all performed automatically):
  1: Load data (common features or full)
  2: Drop specified columns
  3: Encode categorical features
  4: Clean IP version column
  5: Label encoding with consistent mapping
  6: Fill missing values with smart strategies
  7: Generate column inconsistency report

Examples:
  python awid3_preprocessor.py --input /path/to/awid3/CSV --output /path/to/processed
  python awid3_preprocessor.py --input /path/to/awid3/CSV --output /path/to/processed --common-features-only
  python awid3_preprocessor.py --input /path/to/awid3/CSV --output /path/to/processed --chunk-size 5000 --verbose
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing AWID3 CSV files'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for processed files'
    )
    
    parser.add_argument(
        '--chunk-size', '-c',
        type=int,
        default=10000,
        help='Chunk size for processing large files (default: 10000)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--common-features-only',
        action='store_true',
        help='Process only universal columns (30 columns) from the start - faster and more efficient'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not os.path.exists(args.input):
        parser.error(f"Input directory does not exist: {args.input}")
    
    logger.info("AWID3 Dataset Preprocessing Pipeline")
    logger.info("="*50)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Mode: {'🎯 Common Features Only (30 columns)' if args.common_features_only else '📊 Full Processing'}")
    logger.info("="*50)
    
    # Initialize preprocessor and run
    preprocessor = AWID3Preprocessor(args.input, args.output, args.chunk_size, args.common_features_only)
    preprocessor.run_preprocessing()


if __name__ == "__main__":
    main() 