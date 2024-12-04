# prompt: using torch take a tensor which is shape of (batch_size, 128, 8) which gets passed into a MLP. The MLP should take in a Tensor of size (batch_size, 1024)

import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_size=1024, output_size=4):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, output_size)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x

# Example usage:
batch_size = 1
input_tensor = torch.randn(batch_size, 128, 8)

# Flatten the input tensor
input_tensor = input_tensor.transpose(1, 2).contiguous().view(batch_size, -1)

# # Define the MLP
mlp = MLP()

# # Pass the input tensor through the MLP
output_tensor = mlp(input_tensor)

# # Print the shape of the output tensor
output_tensor.shape

