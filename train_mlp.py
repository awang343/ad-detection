from MLP import MLP

from seq_dataset import SeqDataset
from shot_encoder import ShotEncoder

from datetime import datetime
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

config = {
    'movies_dir': '/media/extra/data-dl/movie-shots',
    'ads_dir': '/media/extra/data-dl/ad-shots',
    'npy_dir': 'npy-avg',
    'save_dir': './models/',
    'encoder_pth': './models/checkpoint.pth',

    'epochs': 30,
    'encoding_size': 512,
    'batch_size': 16,
    'learning_rate': 3e-5,

    'seq_length': 4,

    'checkpoint_path': None
}

encoder = ShotEncoder(output_dim=config['encoding_size'])
if config['encoder_pth']:
    encoder.load_model(config['encoder_pth'])

sequence_set = SeqDataset(config['movies_dir'], config['ads_dir'], 
                          config['seq_length'], encoder=encoder,
                          npy_dirname=config['npy_dir'])
seq_loader = DataLoader(
    sequence_set,
    batch_size=config["batch_size"],
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


optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
for epoch in range(start_epoch, config['epochs']):
    print(f"Starting epoch {epoch + 1}/{config['epochs']}")
    avg_loss = model.train(seq_loader, optimizer)
    print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    if (epoch + 1) % 2 == 0:
        check_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        model.save_model(epoch, check_path)
        print(f"Saved checkpoint model: {check_path}")

final_path = save_dir / 'model_final.pth'
model.save_model(epoch, final_path)
print(f"Saved final model: {final_path}")
