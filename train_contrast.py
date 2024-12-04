from shot_dataset import ShotDataset
from shot_encoder import ShotEncoder
from shot_contrast import SceneBoundaryMoCo

from datetime import datetime
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

config = {
    'movies_dir': '/media/extra/data-dl/movie-shots',
    'ads_dir': '/media/extra/data-dl/ad-shots',
    'save_dir': './models/',

    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 3e-5,

    'queue_size': 32,
    'moco_dim': 512,
    'moco_momentum': 0.999,
    'temperature': 0.07,

    'checkpoint_path': None
}

movie_set = ShotDataset(config['movies_dir'])
movie_loader = DataLoader(
    movie_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

ad_set = ShotDataset(config['ads_dir'], is_ad=True)
ad_loader = DataLoader(
    ad_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = Path(config['save_dir']) / timestamp
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Configuration: {config}")

model = SceneBoundaryMoCo(
    K=config['queue_size'],
    m=config['moco_momentum'],
    T=config['temperature'],
    output_dim=config['moco_dim'],
    batch_size=config['batch_size']
)

start_epoch = 0
if config['checkpoint_path']:
    start_epoch = model.load_model() + 1

optimizer = torch.optim.Adam(model.encoder_q.parameters(), lr=config['learning_rate'])
for epoch in range(start_epoch, config['epochs']):
    print(f"Starting epoch {epoch + 1}/{config['epochs']}")
    avg_loss = model.train(movie_loader, ad_loader, optimizer)
    print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

    if (epoch + 1) % 2 == 0:
        check_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        model.save_model(epoch, check_path)
        print(f"Saved checkpoint model: {check_path}")

final_path = save_dir / 'model_final.pth'
model.save_model(epoch, final_path)
print(f"Saved final model: {final_path}")
