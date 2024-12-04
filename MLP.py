import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=1024, output_size=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch):
        """Single training step with MoCo"""
        probs = self(batch[0])
        loss = F.cross_entropy(probs, batch[1].type(torch.FloatTensor))

        return loss

    def train(self, seq_loader, optimizer):
        """Train for one epoch"""
        total_loss = 0
        n_batches = 0

        for batch in seq_loader:
            mlp_inp = batch[0].view(batch[0].shape[0], -1)
            labels = torch.nn.functional.one_hot(batch[1], 4)

            optimizer.zero_grad()

            loss = self.training_step([mlp_inp, labels])
            print(f"Batch {n_batches}/{len(seq_loader)}", f"Loss: {loss}")
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

