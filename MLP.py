import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=1024, output_size=4, dropout_rate=0.35):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.lrelu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 128)
        self.lrelu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, output_size)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.lrelu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        x = self.softmax(x)
        return x

    def batch_step(self, batch):
        probs = self(batch[0])
        print(batch[1])
        loss = F.cross_entropy(probs, batch[1].type(torch.FloatTensor))

        preds = torch.argmax(probs, axis=1)
        labels = torch.argmax(batch[1].type(torch.FloatTensor), axis=1)
        
        correct_preds = torch.sum(preds == labels)
        total_preds = len(batch[1])
        accuracy = correct_preds / total_preds

        return loss, accuracy

    def train(self, seq_loader, optimizer):
        total_loss = 0
        total_acc = 0
        n_batches = 0

        for batch in seq_loader:
            mlp_inp = batch[0].view(batch[0].shape[0], -1)
            labels = torch.nn.functional.one_hot(batch[1], 4)

            optimizer.zero_grad()

            loss, accuracy = self.batch_step([mlp_inp, labels])
            print(f"Batch {n_batches+1}/{len(seq_loader)}", f"Loss: {loss:.4f}, Acc: {accuracy:.4f}")
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy
            n_batches += 1

        return total_loss / n_batches, total_acc / n_batches

    def test(self, seq_loader):
        total_loss = 0
        total_acc = 0
        n_batches = 0

        for batch in seq_loader:
            mlp_inp = batch[0].view(batch[0].shape[0], -1)
            labels = torch.nn.functional.one_hot(batch[1], 4)

            with torch.no_grad():
                loss, accuracy = self.batch_step([mlp_inp, labels])

            total_loss += loss.item()
            total_acc += accuracy
            n_batches += 1

        return total_loss / n_batches, total_acc / n_batches

    def save_model(self, epoch, path):
        """Save query encoder"""
        torch.save({
            'epoch': epoch,
            'model': self.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_model(self, path):
        """Load query encoder, returns epoch"""
        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint["model"])
        print(f"Resumed from checkpoint: {config['checkpoint_path']}")

        return checkpoint['epoch']

