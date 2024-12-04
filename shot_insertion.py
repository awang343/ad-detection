import numpy as np
import pandas as pd

import cv2
import os
from pathlib import Path
import random

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from shot_encoder import ShotEncoder

def save_array_tuples(arr, filename):
    """
    Save array of tuples (label, numpy_array) to CSV file.

    Parameters:
    arr: List of tuples where each tuple is (label, numpy_array)
    filename: Output CSV filename
    """
    data = {
        "label": [t[0] for t in arr],
        "array": [",".join(map(str, t[1])) for t in arr],
    }

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def load_array_tuples(filename):
    """
    Load array of tuples from CSV file.

    Parameters:
    filename: Input CSV filename

    Returns:
    List of tuples (label, numpy_array)
    """
    # Read CSV file
    df = pd.DataFrame(pd.read_csv(filename))

    # Convert back to original format
    result = []
    for _, row in df.iterrows():
        label = row["label"]
        array = np.array([float(x) for x in row["array"].split(",")])
        result.append((label, array))

    return result

def load_npy_files(directory):
    """Loads all .npy files in a directory into a single numpy array.

    Args:
        directory: The path to the directory containing the .npy files.

    Returns:
        A numpy array containing the data from all .npy files,
        or None if no .npy files are found or an error occurs.
        The first dimension of the array indexes each original file.
    """
    file_paths = [f for f in os.listdir(directory) if f.endswith(".npy")]
    if not file_paths:
        print("No .npy files found in the specified directory.")
        return None

    try:
        all_arrays = []
        for file_path in file_paths:
            full_path = os.path.join(directory, file_path)
            arr = np.load(full_path)
            all_arrays.append(arr)
        return np.array(all_arrays)  # Stack arrays along a new axis

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def insert_tuples(enc_movie_list_of_tuples, enc_ads_list_of_tuples):
    """Inserts tuples from enc_ads_list_of_tuples into enc_movie_list_of_tuples at random locations."""

    insert_index = random.randint(int(0.3*len(enc_movie_list_of_tuples)), len(enc_movie_list_of_tuples))
    for ad_tuple in enc_ads_list_of_tuples:
        enc_movie_list_of_tuples.insert(insert_index, ad_tuple)
        insert_index = insert_index + 1

    return enc_movie_list_of_tuples

def main():
    movie_avg_shots = load_npy_files(
        "./data/movies/The.Godfather"
    )
    ads_avg_shots = load_npy_files(
        "./data/ads/temu"
    )

    movie_avg_shots_tensor = torch.tensor(movie_avg_shots, dtype=torch.float32)
    ads_avg_shots_tensor = torch.tensor(ads_avg_shots, dtype=torch.float32)

    enc = ShotEncoder()

    enc_movie_tensor = enc(movie_avg_shots_tensor)
    enc_ads_tensor = enc(ads_avg_shots_tensor)

    enc_movie_np = enc_movie_tensor.detach().numpy()
    enc_ads_np = enc_ads_tensor.detach().numpy()

    enc_movie_list_of_tuples = [
        (0, enc_movie_np[i]) for i in range(enc_movie_np.shape[0])
    ]
    enc_ads_list_of_tuples = [(1, enc_ads_np[i]) for i in range(enc_ads_np.shape[0])]

    proplery_inserted_list_of_tuples = insert_tuples(
        enc_movie_list_of_tuples, enc_ads_list_of_tuples
    )

    save_array_tuples(proplery_inserted_list_of_tuples, "./ads_spliced_into_movie.csv")

if __name__ == "__main__":
    main()
