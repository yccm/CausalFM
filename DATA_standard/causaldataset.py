import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CausalDataset(Dataset):
    def __init__(self, csv_files, transform=None):
        """
        Args:
            csv_files (list or string): List of CSV file paths or a glob pattern
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        
        if isinstance(csv_files, str):
            self.csv_files = glob.glob(csv_files)
        else:
            self.csv_files = csv_files
            
        print(f"Found {len(self.csv_files)} CSV files to load:")
        for file in self.csv_files:
            print(f"  - {file}")
            
        self.sequences = []
        for file in self.csv_files:
            df = pd.read_csv(file)
            print(f"Loaded {len(df)} rows from {file}")
            
            # Identify the columns by type
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
        
        print(f"Total number of sequences (CSV files): {len(self.sequences)}")
        if len(self.sequences) > 0:
            print(f"Each sequence length: {len(self.sequences[0]['X'])}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sequences[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def collate_fn(batch):
    """
    Custom collate function to stack sequences into the required format:
    - X: (seq_len, batch_size, num_features)
    - a: (seq_len, batch_size, 1)
    - y: (seq_len, batch_size, 1)
    - y0: (seq_len, batch_size, 1)
    - y1: (seq_len, batch_size, 1)
    - ite: (seq_len, batch_size, 1)
    """
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


def create_causal_data_loaders(data_path, batch_size=32, val_split=0.2, test_split=0.1, 
                              shuffle=True, num_workers=4):
    
    # Create dataset
    dataset = CausalDataset(data_path)
    
    # Split dataset (splitting CSV files, not individual samples)
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, collate_fn=collate_fn)
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn)
    
    test_loader = None
    
    return train_loader, val_loader, test_loader

 