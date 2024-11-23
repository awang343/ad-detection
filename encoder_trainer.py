import random
import numpy as np
from collections import defaultdict
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from shot_dataset import ShotDataset
from shot_encoder import ShotEncoder

class SceneBoundaryMoCo(nn.Module):
    def __init__(self,
                 K=4096,  # queue size
                 m=0.999,  # momentum
                 T=0.07,   # temperature
                 output_dim=128,
                 batch_size=32):
        super(SceneBoundaryMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.batch_size = batch_size

        self.encoder_q = ShotEncoder(output_dim=output_dim)
        self.encoder_k = ShotEncoder(output_dim=output_dim)

        # Make sure the weights are the same initially
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                  self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

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

    def forward(self, im_q, im_k):
        """Forward pass - required for nn.Module"""
        q = self.encoder_q(im_q)

        best_dist = np.inf
        k = None

        for im in im_k:
            with torch.no_grad():
                k_im = self.encoder_k(im)
                dist = torch.dot(k_im[0], q[0])
                if dist < best_dist:
                    best_dist, k = dist, k_im

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        self._dequeue_and_enqueue(k)

        return logits

    def training_step(self, batch):
        """Single training step with MoCo"""
        im_q = batch[0].to(self.device)
        im_k = [b.to(self.device) for b in batch[1]]

        logits = self(im_q, im_k)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            loss = self.training_step(batch)
            print(f"Batch {n_batches}", f"Loss: {loss}")
            loss.backward()
            optimizer.step()

            self._momentum_update_key_encoder()

            total_loss += loss.item()
            n_batches += 1

        self.register_buffer("queue", torch.randn(output_dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        return total_loss / n_batches

    def train(self, train_dir, num_epochs=100, lr=3e-4):
        """Training loop"""
        dataset = ShotDataset(train_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        optimizer = torch.optim.Adam(self.encoder_q.parameters(), lr=lr)

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader, optimizer)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def save_model(self, path):
        """Save query encoder"""
        torch.save(self.encoder_q.state_dict(), path)

    def load_model(self, path):
        """Load query encoder"""
        self.encoder_q.load_state_dict(torch.load(path))

