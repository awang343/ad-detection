# -*- coding: utf-8 -*-
"""Video-Shot-Detection-3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aNZ9FdSZffCTWgmPD3_MxZD258Qrd36Z
"""

import shutil
import os
import sys
import cv2
import numpy as np
from typing import List

class ShotSegmentation:
    def __init__(self, threshold: float = 0.3, min_scene_len: int = 10):
        """
        Initialize shot segmentation with configurable parameters.

        Args:
            threshold (float): Similarity threshold for detecting shot changes.
                               Lower values make detection more sensitive.
            min_scene_len (int): Minimum number of frames between detected shots.
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def histogram_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate color histogram difference between two frames using
        Bhattacharyya distance.

        Args:
            frame1 (np.ndarray): First input frame
            frame2 (np.ndarray): Second input frame

        Returns:
            float: Similarity metric between frames
        """
        # Convert frames to HSV color space
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        # Calculate histograms
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])

        # Normalize histograms
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        # Calculate Bhattacharyya distance
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    def segment_shots(self, video_path: str) -> List[int]:
        """
        Perform shot segmentation on a video file.

        Args:
            video_path (str): Path to the input video file

        Returns:
            List[int]: List of frame indices where shot changes occur
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Shot segmentation variables
        shot_frames = []
        prev_frame = None
        frame_count = 0
        last_shot_frame = -self.min_scene_len

        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Skip processing on first frame
            if prev_frame is not None:
                # Calculate histogram difference
                diff = self.histogram_diff(prev_frame, frame)

                # Check if shot change is detected
                if (diff > self.threshold and
                    frame_count - last_shot_frame >= self.min_scene_len):
                    shot_frames.append(frame_count)
                    last_shot_frame = frame_count

            # Update for next iteration
            prev_frame = frame
            frame_count += 1

        # Release video capture
        cap.release()

        return shot_frames

    @staticmethod
    def visualize_shots(video_path: str, shot_frames: List[int]) -> None:
        """
        Create a visualization of shot changes by extracting key frames.

        Args:
            video_path (str): Path to the input video file
            shot_frames (List[int]): List of frame indices with shot changes
        """
        cap = cv2.VideoCapture(video_path)

        # Create output directory
        import os
        os.makedirs('shot_keyframes', exist_ok=True)

        for idx, shot_frame in enumerate(shot_frames):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, shot_frame)
            ret, frame = cap.read()

            if ret:
                # Save key frame
                cv2.imwrite(f'shot_keyframes/shot_{idx}_frame_{shot_frame}.jpg', frame)

        cap.release()



shot_frames.append(None)
print(shot_frames)

def extract_frame_ranges(video_path, frame_ranges, output_dir=None, output_prefix='segment'):
    """
    Extract and save video segments based on specified frame ranges.

    :param video_path: Path to the input video file
    :param frame_ranges: List of frame indices defining the ranges to extract
                         Last element can be None to extract till the end of video
    :param output_dir: Directory to save extracted video segments
                       (creates directory if it doesn't exist)
    :param output_prefix: Prefix for output video files
    :return: List of paths to extracted video segments
    """
    # Create output directory if not exists
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), 'extracted_segments')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames and video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Replace None with total frames if it's the last element
    processed_ranges = frame_ranges.copy()
    if processed_ranges[-1] is None:
        processed_ranges[-1] = total_frames

    # Sorted frame ranges to ensure correct processing
    sorted_ranges = sorted(processed_ranges)

    # List to store output video paths
    output_videos = []

    # Process each range
    for i in range(len(sorted_ranges) - 1):
        start_frame = sorted_ranges[i]
        end_frame = sorted_ranges[i+1]

        # Validate frame ranges
        if start_frame >= end_frame:
            print(f"Skipping invalid range: {start_frame} to {end_frame}")
            continue

        # Output video writer
        output_filename = f'shot_{output_prefix}_{i:06d}.mp4'
        output_path = os.path.join(output_dir, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset video capture to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read and write frames for this segment
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()

            if not ret:
                break

            out.write(frame)
            current_frame += 1

        # Release the writer for this segment
        out.release()
        output_videos.append(output_path)

    # Release the video capture
    cap.release()

    return output_videos

extract_frame_ranges("./movie.mp4", shot_frames)

def main():
    video_path = "./movie.mp4"
    export_path = "./extracted_segments"
    segmenter = ShotSegmentation(threshold=0.3, min_scene_len=10)
    shot_frames = segmenter.segment_shots(video_path)
    extracted_videos = extract_frame_ranges(video_path, shot_frames, output_dir=export_path)
    print(f"Extracted video segments: {extracted_videos}")

if __name__ == "__main__":
    print(sys.argv)
    print('Hello World!')
    #main()