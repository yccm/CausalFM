

import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

def normalize_and_save_datasets(input_pattern, output_dir, save_scalers=True):
    """
    Normalize all datasets and save them to disk once.
    This is a one-time preprocessing step.
    
    Args:
        input_pattern: Glob pattern for input CSV files
        output_dir: Directory to save normalized CSV files
        save_scalers: Whether to save the scalers for inverse transform
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    if save_scalers:
        os.makedirs(os.path.join(output_dir, 'scalers'), exist_ok=True)
    
    # Get all CSV files
    csv_files = glob.glob(input_pattern)
    print(f"Found {len(csv_files)} CSV files to normalize")
    
    # Process each file
    for idx, file in enumerate(tqdm(csv_files, desc="Normalizing datasets")):
        # Load data
        df = pd.read_csv(file)
        
        # Identify columns
        x_cols = [col for col in df.columns if col.startswith('x')]
        a_col = 'treatment'
        y_col = 'outcome'
        y0_col = 'y0'
        y1_col = 'y1'
        ite_col = 'ite'
        
        # Initialize scalers
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        # Normalize X features
        df_normalized = df.copy()
        df_normalized[x_cols] = x_scaler.fit_transform(df[x_cols].values)
        
        # Normalize Y values (fit on both y0 and y1)
        y_combined = np.concatenate([
            df[y0_col].values.reshape(-1, 1),
            df[y1_col].values.reshape(-1, 1)
        ])
        y_scaler.fit(y_combined)
        
        # Transform each y variable
        df_normalized[y_col] = y_scaler.transform(df[y_col].values.reshape(-1, 1)).flatten()
        df_normalized[y0_col] = y_scaler.transform(df[y0_col].values.reshape(-1, 1)).flatten()
        df_normalized[y1_col] = y_scaler.transform(df[y1_col].values.reshape(-1, 1)).flatten()
        
        # Recalculate ITE in normalized space
        df_normalized[ite_col] = df_normalized[y1_col] - df_normalized[y0_col]
        
        # Save normalized data
        base_name = os.path.basename(file)
        output_path = os.path.join(output_dir, f"normalized_{base_name}")
        df_normalized.to_csv(output_path, index=False)
        
        # Save scalers if requested
        if save_scalers:
            scaler_path = os.path.join(output_dir, 'scalers', f"scaler_{idx:05d}.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump({
                    'x_scaler': x_scaler,
                    'y_scaler': y_scaler,
                    'original_file': file,
                    'normalized_file': output_path
                }, f)
    
    print(f"Normalization complete! Normalized files saved to: {output_dir}")
    print(f"You can now use: {os.path.join(output_dir, 'normalized_*.csv')} for training")


class CausalDatasetFast(Dataset):
    """
    Faster dataset class that loads pre-normalized data.
    Use this after running normalize_and_save_datasets().
    """
    def __init__(self, csv_files, transform=None):
        """
        Args:
            csv_files (list or string): List of pre-normalized CSV file paths or a glob pattern
            transform (callable, optional): Optional transform to be applied on a sample
        """
        import torch
        from torch.utils.data import Dataset
        
        self.transform = transform
        
        # Handle both list of files or glob pattern
        if isinstance(csv_files, str):
            self.csv_files = glob.glob(csv_files)
        else:
            self.csv_files = csv_files
            
        print(f"Found {len(self.csv_files)} pre-normalized CSV files")
        
        # Load pre-normalized data
        self.sequences = []
        for idx, file in enumerate(self.csv_files):
            if idx % 1000 == 0:
                print(f"Loading file {idx + 1}/{len(self.csv_files)}")
            
            df = pd.read_csv(file)
            
            # Identify columns
            x_cols = [col for col in df.columns if col.startswith('x')]
            a_col = 'treatment'
            y_col = 'outcome'
            y0_col = 'y0'
            y1_col = 'y1'
            ite_col = 'ite'
            
            # Convert to tensors (data is already normalized)
            sequence = {
                'X': torch.FloatTensor(df[x_cols].values),
                'a': torch.FloatTensor(df[a_col].values).unsqueeze(1),
                'y': torch.FloatTensor(df[y_col].values).unsqueeze(1),
                'y0': torch.FloatTensor(df[y0_col].values).unsqueeze(1),
                'y1': torch.FloatTensor(df[y1_col].values).unsqueeze(1),
                'ite': torch.FloatTensor(df[ite_col].values).unsqueeze(1),
            }
            
            self.sequences.append(sequence)
        
        print(f"Loaded {len(self.sequences)} pre-normalized sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sample = self.sequences[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":

    normalize_and_save_datasets(
        input_pattern=PATH,
        output_dir=OUTPUT_DIR,
        save_scalers=True
    )
    