import shutil
import os
import sys
import cv2
import numpy as np
from typing import List

class ShotSegmentation:
    def __init__(self, threshold: float = 0.3, min_scene_len: int = 10):
        self.threshold = threshold
        self.min_scene_len = min_scene_len

# Histogram diff {{{
    def histogram_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 256])

        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)# }}}
# Segment Shots {{{ 
    def segment_shots(self, video_path: str, frame_div: int) -> List[int]:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        shot_frames = []
        prev_frame = None
        frame_count = 0
        last_shot_frame = -self.min_scene_len

        while True:
            for _ in range(frame_div):
                ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                diff = self.histogram_diff(prev_frame, frame)
                if (diff > self.threshold and
                    frame_count - last_shot_frame >= self.min_scene_len):
                    shot_frames.append(frame_count)
                    last_shot_frame = frame_count
            prev_frame = frame
            frame_count += frame_div

        cap.release()

        return shot_frames# }}}
# {{{ Extract frame ranges
    def extract_frame_ranges(self, video_path, frame_ranges, output_dir, frame_div, output_prefix='segment'):
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//frame_div
        fps = cap.get(cv2.CAP_PROP_FPS)//frame_div
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        processed_ranges = frame_ranges.copy()
        if processed_ranges[-1] is None:
            processed_ranges[-1] = total_frames

        sorted_ranges = sorted(processed_ranges)

        output_videos = []

        for i in range(len(sorted_ranges) - 1):
            start_frame = sorted_ranges[i]
            end_frame = sorted_ranges[i+1]

            if start_frame >= end_frame:
                print(f"Skipping invalid range: {start_frame} to {end_frame}")
                continue

            output_filename = f'shot_{output_prefix}_{i:06d}.mp4'
            output_path = os.path.join(output_dir, output_filename)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current_frame = start_frame
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                current_frame += frame_div

            out.release()
            output_videos.append(output_path)
        cap.release()

        return output_videos# }}}

    def __call__(self, video_path, export_path):
        shot_frames = self.segment_shots(video_path, 1)
        extracted_videos = self.extract_frame_ranges(video_path, shot_frames, export_path, 1)
        return extracted_videos

def main():
    movie_dir = "data/movies-raw/"
    ads_dir = "data/ads-raw/"
    movie_output = "data/movies/"
    ads_output = "data/ads/"

    segmenter = ShotSegmentation(threshold=0.3, min_scene_len=10)


    for movie in os.listdir(movie_dir):
        folder = movie.replace(".mp4", "/")
        if movie.replace(".mp4", "") not in os.listdir(movie_output):
            os.makedirs(movie_output + folder)
            try:
                segmenter(movie_dir + movie, movie_output+folder)
            except Exception as e:
                shutil.rmtree(movie_output + folder)
                print(f'Failed to parse movie - {movie_dir + movie}')

    for ad in os.listdir(ads_dir):
        folder = ad.replace(".mp4", "/")
        if ad.replace(".mp4", "") not in os.listdir(ads_output):
            os.makedirs(ads_output + folder)
            try:
                segmenter(ads_dir + ad, ads_output+folder)
            except Exception as e:
                shutil.rmtree(ads_output + folder)
                print(f'Failed to parse ad - {ads_dir + ad}')

        
if __name__ == "__main__":
    main()
