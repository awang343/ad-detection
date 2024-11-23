import os
from pathlib import Path

import random
import numpy as np
from datetime import datetime
from collections import defaultdict
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ShotDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.shots = defaultdict(list)
        for movie_dir in self.root_dir.glob("Movie_*"):
            movie_id = movie_dir.name
            shots = sorted(list(movie_dir.glob("shot_segment_*.mp4")),
                         key=lambda x: int(x.stem.split('_')[2]))
            self.shots[movie_id] = shots

        self.all_shots = [(movie_id, shot_path)
                         for movie_id, shots in self.shots.items()
                         for shot_path in shots]

    def _load_video_frame(self, video_path, save_path):
        if video_path.stem + ".npy" in os.listdir(save_path):
            return np.load(save_path + "/" + video_path.stem + ".npy")

        vidcap = cv2.VideoCapture(str(video_path))
        success = True
        count = 0

        frames = []
        success,image = vidcap.read()
        while success:
            if self.transform:
                image = self.transform(image)
            count += 1
            frames.append(image)
            success,image = vidcap.read()

        vidcap.release()
        shot_avg = np.mean(np.stack(frames[:-1]), axis=0)
        np.save(save_path + "/" + video_path.stem + ".npy", shot_avg)

        return shot_avg

    def _get_temporal_neighbors(self, movie_id, shot_idx, window_size=5):
        """Get temporally adjacent shots within window_size"""
        movie_shots = self.shots[movie_id]
        current_idx = next(i for i, shot in enumerate(movie_shots)
                         if shot.stem == shot_idx.stem)

        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(movie_shots), current_idx + window_size + 1)

        return movie_shots[start_idx:end_idx]

    def __getitem__(self, idx):
        movie_id, anchor_path = self.all_shots[idx]

        temporal_neighbors = self._get_temporal_neighbors(movie_id, anchor_path)

        anchor_frame = self._load_video_frame(anchor_path, os.path.dirname(anchor_path))
        temporal_frames = [self._load_video_frame(f, os.path.dirname(f)) for f in temporal_neighbors if f != anchor_path]

        return anchor_frame, temporal_frames

    def __len__(self):
        return len(self.all_shots)
