import random
import numpy as np
from collections import defaultdict
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from shot_dataset import ShotDataset
from shot_encoder import ShotEncoder

class MLP(nn.Module):
    def __init__(self,
                 output_dim=128,
                 batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.encoder_q = ShotEncoder(output_dim=output_dim)

        self.register_buffer("queue", torch.randn(output_dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """MoCo momentum update for key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                  self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size-remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, im_ad):
        """Forward pass - required for nn.Module"""
        q = self.encoder_q(im_q)
        k = self.encoder_k(im_k)

        a = self.encoder_k(im_ad)

        best_dist = np.inf

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        self._dequeue_and_enqueue(a)

        return logits

    def training_step(self, batch, ad):
        """Single training step with MoCo"""
        im_q, im_k = [b.to(self.device) for b in batch]
        im_a = ad.to(self.device)

        logits = self(im_q, im_k, im_a)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        labels[0] = 1

        loss = F.cross_entropy(logits, labels)
        return loss

    def train(self, movie_loader, ad_loader, optimizer):
        """Train for one epoch"""
        total_loss = 0
        n_batches = 0

        ad_iter = iter(ad_loader)

        for batch in movie_loader:
            optimizer.zero_grad()

            ad = next(ad_iter, "eod")
            if ad == "eod":
                ad_iter = iter(ad_loader)
                ad = next(ad_iter)

            loss = self.training_step(batch, ad)
            print(f"Batch {n_batches}/{len(movie_loader)}", f"Loss: {loss}")
            loss.backward()
            optimizer.step()

            self._momentum_update_key_encoder()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def save_model(self, epoch, path):
        """Save query encoder"""
        torch.save({
            'epoch': epoch,
            'encoder_q': self.encoder_q.state_dict(),
            'encoder_k': self.encoder_k.state_dict(),
            'queue': self.queue,
            'queue_ptr': self.queue_ptr
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_model(self, path):
        """Load query encoder, returns epoch"""
        checkpoint = torch.load(path)

        self.encoder_q.load_state_dict(checkpoint['encoder_q'])
        self.encoder_k.load_state_dict(checkpoint['encoder_k'])
        self.queue = checkpoint['queue']
        self.queue_ptr = checkpoint['queue_ptr']
        print(f"Resumed from checkpoint: {config['checkpoint_path']}")

        return checkpoint['epoch']

