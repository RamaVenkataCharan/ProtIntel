import gzip
import io
import os
from pathlib import Path
import numpy as np

def create_dummy_dataset(filename, num_samples):
    print(f"Creating dummy dataset {filename} with {num_samples} samples...")
    # Shape: (N, 700, 57)
    data = np.zeros((num_samples, 700, 57), dtype=np.float32)
    
    for i in range(num_samples):
        # Choose a random sequence length L
        L = np.random.randint(15, 60)
        
        # 1. Amino acid one-hot (cols 0-20)
        # For j < L, set one random AA index to 1.0
        aa_indices = np.random.randint(0, 20, size=L)
        for j, aa_idx in enumerate(aa_indices):
            data[i, j, aa_idx] = 1.0
            
        # 2. Sequence profile/PSSM (cols 21-34)
        data[i, :L, 21:35] = np.random.randn(L, 14)
        
        # 3. Q8 label one-hot (cols 35-42)
        # For j < L, set one random Q8 class index to 1.0
        q8_indices = np.random.randint(0, 8, size=L)
        for j, q8_idx in enumerate(q8_indices):
            data[i, j, 35 + q8_idx] = 1.0
            
        # 4. Noseq sentinel (col 43)
        # 0 for actual residue, 1 for padding/noseq
        data[i, :L, 43] = 0.0
        data[i, L:, 43] = 1.0
        
        # 5. Solvent accessibility etc (cols 44-56)
        data[i, :L, 44:57] = np.random.rand(L, 13)
        
    # Save as .npy.gz
    out_buf = io.BytesIO()
    np.save(out_buf, data)
    out_buf.seek(0)
    
    dest_path = Path("datasets/raw") / filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest_path, "wb") as f:
        f.write(out_buf.read())
        
    print(f"Saved to {dest_path} (size: {dest_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    np.random.seed(42)
    create_dummy_dataset("cullpdb+profile_6133_filtered.npy.gz", num_samples=16)
    create_dummy_dataset("cb513+profile_split1.npy.gz", num_samples=8)
    create_dummy_dataset("rs126+profile_split1.npy.gz", num_samples=8)
    print("Done generating dummy datasets!")
