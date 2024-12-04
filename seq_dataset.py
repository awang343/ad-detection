import os
from pathlib import Path

import random
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from shot_encoder import ShotEncoder

class SeqDataset(Dataset):
    def __init__(self, movie_shotdir, ad_shotdir, seq_length, encoder,
                 npy_dirname="npy-avg", encodings_dirname="encodings"):
        self.movie_shotdir = Path(movie_shotdir)
        self.ad_shotdir = Path(ad_shotdir)

        self.npy_name = npy_dirname
        self.encodings_name = encodings_dirname
        self.seq_length = seq_length

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder

        movie_shots = defaultdict(list)
        for video_dir in self.movie_shotdir.iterdir():
            video_id = video_dir.name

            frame_dir = video_dir / npy_dirname
            enc_dir = video_dir / encodings_dirname
            enc_dir.mkdir(exist_ok=True)

            shots = sorted(list(frame_dir.glob("shot_segment_*.npy")),
                           key=lambda x: int(x.stem.split('_')[2]))
            movie_shots[video_id] = shots

        ad_shots = defaultdict(list)
        for video_dir in self.ad_shotdir.iterdir():
            video_id = video_dir.name

            frame_dir = video_dir / npy_dirname
            enc_dir = video_dir / encodings_dirname
            enc_dir.mkdir(exist_ok=True)

            shots = sorted(list(frame_dir.glob("shot_segment_*.npy")),
                           key=lambda x: int(x.stem.split('_')[2]))
            ad_shots[video_id] = shots

        self.get_sequence_types(ad_shots, movie_shots)

    def get_sequence_types(self, ad_shots, movie_shots):
        self.sequences = []

        for m_id, shots in movie_shots.items():
            if len(shots) < self.seq_length * 2:
                continue

            for shot_num in range(len(shots) - self.seq_length * 2 + 1):
                movie_part = shots[shot_num:shot_num + self.seq_length * 2]
                self.sequences.append((0, movie_part))

        for a_id, shots in ad_shots.items():
            if len(shots) < self.seq_length * 2:
                continue

            for shot_num in range(len(shots) - self.seq_length * 2 + 1):
                ad_part = shots[shot_num:shot_num + self.seq_length * 2]
                self.sequences.append((3, ad_part))

        for a_id, a_shots in ad_shots.items():
            for m_id, m_shots in movie_shots.items():
                if len(m_shots) < self.seq_length or len(a_shots) < self.seq_length:
                    continue

                for shot_num in range(len(movie_shots) - self.seq_length + 1):
                    ad_start = a_shots[:self.seq_length]
                    ad_end = a_shots[-self.seq_length:]
                    movie_part = m_shots[shot_num:shot_num + self.seq_length]

                    self.sequences.append((2, movie_part+ad_start))
                    self.sequences.append((3, ad_end+movie_part))

    def _load_encoding(self, frame_path, encs_path):
        if frame_path.stem + ".npy" in os.listdir(encs_path):
            return np.load(encs_path + "/" + frame_path.stem + ".npy")

        inp = torch.tensor(np.expand_dims(np.load(frame_path), 0))
        encoded = self.encoder(inp.to(self.device)).detach().cpu().numpy()
        np.save(encs_path + "/" + frame_path.stem + ".npy", encoded)

        return encoded

    def __getitem__(self, idx):
        transition_type, shot_paths = self.sequences[idx]
        encodings = []
        for path in shot_paths:
            enc_path = os.path.dirname(os.path.dirname(path)) + "/" + self.encodings_name
            encodings.append(self._load_encoding(path, enc_path))
        return np.array(encodings), transition_type

    def __len__(self):
        return len(self.sequences)
