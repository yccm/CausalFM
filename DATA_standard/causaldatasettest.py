import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CausalTestDataset(Dataset):
    def __init__(self, csv_files, transform=None):

        self.transform = transform
        
        # Handle both list of files or glob pattern
        if isinstance(csv_files, str):
            self.csv_files = glob.glob(csv_files)
        else:
            self.csv_files = csv_files
            
        print(f"Found {len(self.csv_files)} test CSV files to load:")
        for file in self.csv_files:
            print(f"  - {file}")
            

        self.sequences = []
        for file in self.csv_files:
            df = pd.read_csv(file)
            print(f"Loaded {len(df)} rows from {file}")
            
            # Identify the columns by type (same as training)
            x_cols = [col for col in df.columns if col.startswith('x')]
            a_col = 'treatment'  
            y_col = 'outcome'   
            y0_col = 'y0' 
            y1_col = 'y1' 
            ite_col = 'ite' 
            
            # Convert to tensors - each CSV becomes one sequence
            sequence = {
                'X': torch.FloatTensor(df[x_cols].values),  # Shape: (seq_len, num_features)
                'a': torch.FloatTensor(df[a_col].values).unsqueeze(1),  # Shape: (seq_len, 1)
                'y': torch.FloatTensor(df[y_col].values).unsqueeze(1),  # Shape: (seq_len, 1)
                'y0': torch.FloatTensor(df[y0_col].values).unsqueeze(1),  # Shape: (seq_len, 1)
                'y1': torch.FloatTensor(df[y1_col].values).unsqueeze(1),  # Shape: (seq_len, 1)
                'ite': torch.FloatTensor(df[ite_col].values).unsqueeze(1),  # Shape: (seq_len, 1)
            }
            
            self.sequences.append(sequence)
        
        print(f"Total number of test sequences (CSV files): {len(self.sequences)}")
        if len(self.sequences) > 0:
            print(f"Each test sequence length: {len(self.sequences[0]['X'])}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sequences[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def test_collate_fn(batch):

    # Stack all sequences in the batch
    X = torch.stack([item['X'] for item in batch], dim=1)  # (seq_len, batch_size, num_features)
    a = torch.stack([item['a'] for item in batch], dim=1)  # (seq_len, batch_size, 1)
    y = torch.stack([item['y'] for item in batch], dim=1)  # (seq_len, batch_size, 1)
    y0 = torch.stack([item['y0'] for item in batch], dim=1)  # (seq_len, batch_size, 1)
    y1 = torch.stack([item['y1'] for item in batch], dim=1)  # (seq_len, batch_size, 1)
    ite = torch.stack([item['ite'] for item in batch], dim=1)  # (seq_len, batch_size, 1)
    
    return {
        'X': X,
        'a': a,
        'y': y,
        'y0': y0,
        'y1': y1,
        'ite': ite
    }


def create_test_data_loader(test_data_path, batch_size=32, shuffle=False, num_workers=4):
    """
    Create test data loader for separate test dataset
    
    Args:
        test_data_path (str): Path pattern for test CSV files 
        batch_size (int): Number of CSV files (sequences) to load in one batch
        shuffle (bool): Whether to shuffle the test data (typically False for testing)
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: test_loader
    """
    # Create test dataset
    test_dataset = CausalTestDataset(test_data_path)
    

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=test_collate_fn
    )
    
    return test_loader


