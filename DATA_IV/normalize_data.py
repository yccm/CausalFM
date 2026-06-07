

import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


def _is_binary_column(values):
    """Return True if a column only contains the values {0, 1} (ignoring NaNs)."""
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return False
    uniq = np.unique(finite)
    return np.all(np.isin(uniq, [0.0, 1.0]))


def normalize_and_save_datasets(input_pattern, output_dir, save_scalers=True,
                                normalize_instrument='auto'):
    """
    Normalize all IV datasets and save them to disk once.
    This is a one-time preprocessing step.

    Mirrors the standard CATE normalization logic and additionally handles the
    instrument `z`:
      - X features:    StandardScaler (per dataset).
      - Outcomes:       a single StandardScaler fitted on [y0, y1], then applied
                        to `outcome`, `y0`, `y1`; `ite` is recomputed as y1 - y0
                        in the normalized space.
      - Instrument z:   standardized only if it is continuous. A binary instrument
                        (values in {0, 1}) is left untouched.
      - Treatment:      left untouched (binary).
      - Unobserved confounders `u*` are passed through unchanged (the loaders
        ignore them).

    Args:
        input_pattern: Glob pattern for input CSV files
        output_dir: Directory to save normalized CSV files
        save_scalers: Whether to save the scalers for inverse transform
        normalize_instrument: 'auto' (standardize z only if continuous),
                              True (always standardize z),
                              False (never standardize z)
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
        z_col = 'z'  # Instrument variable
        a_col = 'treatment'
        y_col = 'outcome'
        y0_col = 'y0'
        y1_col = 'y1'
        ite_col = 'ite'

        # Initialize scalers
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        z_scaler = None

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

        # Normalize the instrument only when it is continuous
        if z_col in df.columns:
            z_values = df[z_col].values.astype(float)
            if normalize_instrument is True:
                do_norm_z = True
            elif normalize_instrument is False:
                do_norm_z = False
            else:  # 'auto'
                do_norm_z = not _is_binary_column(z_values)

            if do_norm_z:
                z_scaler = StandardScaler()
                df_normalized[z_col] = z_scaler.fit_transform(z_values.reshape(-1, 1)).flatten()

        # Treatment is binary -> left untouched.

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
                    'z_scaler': z_scaler,  # None if instrument was not normalized
                    'original_file': file,
                    'normalized_file': output_path
                }, f)

    print(f"Normalization complete! Normalized files saved to: {output_dir}")
    print(f"You can now use: {os.path.join(output_dir, 'normalized_*.csv')} for training")


class CausalDatasetFast(Dataset):
    """
    Faster dataset class that loads pre-normalized IV data.
    Use this after running normalize_and_save_datasets().
    """
    def __init__(self, csv_files, transform=None):
        """
        Args:
            csv_files (list or string): List of pre-normalized CSV file paths or a glob pattern
            transform (callable, optional): Optional transform to be applied on a sample
        """
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
            z_col = 'z'
            a_col = 'treatment'
            y_col = 'outcome'
            y0_col = 'y0'
            y1_col = 'y1'
            ite_col = 'ite'

            # Convert to tensors (data is already normalized)
            sequence = {
                'X': torch.FloatTensor(df[x_cols].values),
                'z': torch.FloatTensor(df[z_col].values).unsqueeze(1),
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

    # Binary instrument: z stays as {0, 1}, only X and Y are normalized.
    normalize_and_save_datasets(
        input_pattern="DATA_IV/binary_Z_conti_Y/*.csv",
        output_dir="DATA_IV/binary_Z_conti_Y_normalized",
        save_scalers=True,
        normalize_instrument='auto'
    )

    # Continuous instrument: z is standardized in addition to X and Y.
    normalize_and_save_datasets(
        input_pattern="DATA_IV/conti_Z_conti_Y/*.csv",
        output_dir="DATA_IV/conti_Z_conti_Y_normalized",
        save_scalers=True,
        normalize_instrument='auto'
    )
