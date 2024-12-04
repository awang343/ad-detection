import os
from pathlib import Path

import random
import numpy as np
from datetime import datetime
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, shots_dir, npy_dirname="npy-avg", 
                 seq_length, 
                 shot_begin_only= False,
                 shot_end_only = False):
        self.shots_dir = Path(shots_dir)
        self.npy_name = npy_dirname
        self.seq_length = seq_length

        self.shots = defaultdict(list)
        for video_dir in self.shots_dir.iterdir():
            video_id = video_dir.name
            npy_dir = video_dir / self.npy_name

            shots = sorted(list(npy_dir.glob("shot_segment_*.npy")),
                           key=lambda x: int(x.stem.split('_')[2]))
            self.shots[video_id] = shots

        self.sequences = []
        for video_id, shots in self.shots.items():
            if len(shots) < seq_length:
                continue
            if shot_begin_only:
                self.sequences.append((video_id, shots[:seq_length]))
            elif shot_end_only:
                self.sequences.append((video_id, shots[-seq_length:]))
            else:
                for shot_num in range(len(shots) - seq_length + 1):
                    self.sequences.append((video_id, shots[shot_num:shot_num+seq_length]))

    def __getitem__(self, idx):
        video_id, shot_paths = self.sequences[idx]
        return np.array([np.load(path) for path in shot_paths])

    def __len__(self):
        return len(self.sequences)
