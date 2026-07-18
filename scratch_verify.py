import numpy as np
import gzip
from pathlib import Path

datasets = {
    'cullpdb': 'datasets/raw/cullpdb+profile_6133_filtered.npy.gz',
    'cb513': 'datasets/raw/cb513+profile_split1.npy.gz',
    'rs126': 'datasets/raw/rs126+profile_split1.npy.gz'
}

for name, path in datasets.items():
    p = Path(path)
    if not p.exists():
        print(f"{name}: File not found")
        continue
    
    try:
        size = p.stat().st_size
        try:
            with gzip.open(p, 'rb') as f:
                arr = np.load(f)
        except gzip.BadGzipFile:
            arr = np.load(p)
        print(f"{name}: valid array, shape={arr.shape}, size={size} bytes")
    except Exception as e:
        print(f"{name}: Invalid file - {e}")
