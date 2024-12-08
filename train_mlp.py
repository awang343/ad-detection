from MLP import MLP

from seq_dataset import SeqDataset
from shot_encoder import ShotEncoder

from datetime import datetime
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import numpy as np

config = {
    'movies_dir': '/media/extra/data-dl/movie-shots',
    'ads_dir': '/media/extra/data-dl/ad-shots',
    'movies_test': '/media/extra/data-dl/test-movie-shots',
    'ads_test': '/media/extra/data-dl/test-ad-shots',

    'npy_dir': 'npy-avg',
    'encodings_dir': 'encoder-2',

    'save_dir': './models/mlp-model',
    'encoder_pth': './models/shotcol-model/checkpoint_epoch_2.pth',

    'epochs': 10,
    'encoding_size': 512,
    'batch_size': 64,
    'learning_rate': 1e-4,

    'seq_length': 7,

    'checkpoint_path': None
}

encoder = ShotEncoder(output_dim=config['encoding_size'], grad=False)
if config['encoder_pth']:
    encoder.load_model(config['encoder_pth'])

sequence_set = SeqDataset(config['movies_dir'], config['ads_dir'], 
                          config['seq_length'], encoder=encoder,
                          npy_dirname=config['npy_dir'], encodings_dirname=config["encodings_dir"])
seq_loader = DataLoader(
    sequence_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_set = SeqDataset(config['movies_test'], config['ads_test'], 
                          config['seq_length'], encoder=encoder,
                          npy_dirname=config['npy_dir'], encodings_dirname=config["encodings_dir"])
test_loader = DataLoader(
    test_set,
    batch_size=len(test_set),
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print(f"Configuration: {config}")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = Path(config['save_dir']) / timestamp
save_dir.mkdir(parents=True, exist_ok=True)

model = MLP(input_size = config['seq_length'] * 2 * 512, output_size = 4)

start_epoch = 0
if config['checkpoint_path']:
    start_epoch = model.load_model(config['checkpoint_path']) + 1


optimizer = torch.optim.Adam(model.parameters(), 
                            lr=config['learning_rate'])
metrics = {
    "train_acc": [], 
    "train_loss": [],
    "test_acc": [],
    "test_loss": []
}

for epoch in range(start_epoch, config['epochs']):
    print(f"Starting epoch {epoch + 1}/{config['epochs']}")
    avg_loss, avg_acc = model.train(seq_loader, optimizer)
    metrics["train_loss"].append(avg_loss)
    metrics["train_acc"].append(avg_acc)

    avg_loss, avg_acc = model.test(test_loader)
    metrics["test_loss"].append(avg_loss)
    metrics["test_acc"].append(avg_acc)

    print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    if (epoch + 1) % 1 == 0:
        check_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        model.save_model(epoch, check_path)
        print(f"Saved checkpoint model: {check_path}")

for key in metrics:
    np.save(save_dir / key, np.array(metrics[key]))

final_path = save_dir / 'model_final.pth'
model.save_model(epoch, final_path)

print(f"Saved final model: {final_path}")
