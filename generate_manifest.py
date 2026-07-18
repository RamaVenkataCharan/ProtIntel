import numpy as np

def analyze_dataset(name, path):
    print(f"--- Dataset: {name} ---")
    try:
        data = np.load(path)
        print(f"Shape: {data.shape}")
        
        N, seq_len = data.shape[0], 700
        feature_dim = 57
        
        # Check if reshaped is needed
        if data.ndim == 2:
            data = data.reshape(-1, seq_len, feature_dim)
        
        num_proteins = data.shape[0]
        print(f"Total proteins: {num_proteins}")
        
        # find real sequence lengths
        noseq = data[:, :, 43]
        seq_lengths = (noseq == 0).sum(axis=1)
        
        print(f"Sequence lengths: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.1f}, median={np.median(seq_lengths):.1f}")
        
        return data
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

if __name__ == '__main__':
    cullpdb = analyze_dataset("CullPDB", "datasets/raw/cullpdb+profile_6133_filtered.npy.gz")
    cb513 = analyze_dataset("CB513", "datasets/raw/cb513+profile_split1.npy.gz")
    rs126 = analyze_dataset("RS126", "datasets/raw/rs126+profile_split1.npy.gz")

