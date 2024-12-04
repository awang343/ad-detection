import os
from pathlib import Path

import random
import numpy as np
from datetime import datetime
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, movie_shotdir, ads_shotdir, seq_length, npy_dirname="npy-avg"):
        self.movie_shotdir = Path(movie_shotdir)
        self.ad_shotdir = Path(ads_shotdir)

        self.npy_name = npy_dirname
        self.seq_length = seq_length

        movie_shots = defaultdict(list)
        for video_dir in self.movie_shotdir.iterdir():
            video_id = video_dir.name
            npy_dir = video_dir / self.npy_name

            shots = sorted(list(npy_dir.glob("shot_segment_*.npy")),
                           key=lambda x: int(x.stem.split('_')[2]))
            movie_shots[video_id] = shots

        ad_shots = defaultdict(list)
        for video_dir in self.ad_shotdir.iterdir():
            video_id = video_dir.name
            npy_dir = video_dir / self.npy_name

            shots = sorted(list(npy_dir.glob("shot_segment_*.npy")),
                           key=lambda x: int(x.stem.split('_')[2]))
            ad_shots[video_id] = shots

        self.get_sequence_types(ad_shots, movie_shots)

    def get_sequence_types(self, ad_shots, movie_shots):
        self.sequences = []

        for _, shots in movie_shots.items():
            if len(shots) < self.seq_length:
                continue
            for shot_num in range(len(shots) - self.seq_length + 1):
                movie_part = shots[shot_num:shot_num + self.seq_length*2]
                self.sequences.append((0, movie_part))

        for _, shots in ad_shots.items():
            if len(shots) < self.seq_length:
                continue
            for shot_num in range(len(shots) - self.seq_length + 1):
                ad_part = shots[shot_num:shot_num + self.seq_length * 2]
                self.sequences.append((3, ad_part))

        for _, a_shots in ad_shots.items():
            for _, m_shots in movie_shots.items():
                if len(m_shots) < self.seq_length or len(a_shots) < self.seq_length:
                    continue

                for shot_num in range(len(movie_shots) - self.seq_length + 1):
                    ad_start = a_shots[:self.seq_length]
                    ad_end = a_shots[-self.seq_length:]
                    movie_part = m_shots[shot_num:shot_num + self.seq_length]

                    self.sequences.append((2, movie_part+ad_start))
                    self.sequences.append((3, ad_end+movie_part))

    def __getitem__(self, idx):
        transition_type, shot_paths = self.sequences[idx]
        return np.array([np.load(path) for path in shot_paths]), transition_type

    def __len__(self):
        return len(self.sequences)
