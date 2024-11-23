from shot_dataset import ShotDataset
from shot_encoder import ShotEncoder
from encoder_trainer import SceneBoundaryMoCo
from datetime import datetime
import os
from pathlib import Path
import torch

def main():
    config = {
        'data_dir': './data/movies',
        'save_dir': './experiments',

        'epochs': 30,
        'batch_size': 1,
        'learning_rate': 3e-5,

        'queue_size': 65536,
        'moco_dim': 512,
        'moco_momentum': 0.999,
        'temperature': 0.07,

        'checkpoint_path': None
    }

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
        checkpoint = torch.load(config['checkpoint_path'])
        model.encoder_q.load_state_dict(checkpoint['encoder_q'])
        model.encoder_k.load_state_dict(checkpoint['encoder_k'])
        model.queue = checkpoint['queue']
        model.queue_ptr = checkpoint['queue_ptr']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {config['checkpoint_path']}")

    for epoch in range(start_epoch, config['epochs']):
        print(f"Starting epoch {epoch + 1}/{config['epochs']}")

        model.train(
            train_dir=config['data_dir'],
            num_epochs=1,
            lr=config['learning_rate']
        )

        checkpoint = {
            'epoch': epoch,
            'encoder_q': model.encoder_q.state_dict(),
            'encoder_k': model.encoder_k.state_dict(),
            'queue': model.queue,
            'queue_ptr': model.queue_ptr
        }

        latest_path = save_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    final_path = save_dir / 'model_final.pth'
    torch.save({
        'encoder_q': model.encoder_q.state_dict(),
        'encoder_k': model.encoder_k.state_dict(),
        'queue': model.queue,
        'queue_ptr': model.queue_ptr
    }, final_path)
    print(f"Saved final model: {final_path}")

if __name__ == '__main__':
    main()
