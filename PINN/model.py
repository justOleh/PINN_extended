import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden1 = nn.Linear(2, 64)  # Input layer with 2 inputs and 64 hidden units
        self.hidden2 = nn.Linear(64, 128) # Hidden layer with 64 inputs and 128 outputs
        self.hidden3 = nn.Linear(128, 64) # Hidden layer with 128 inputs and 64 outputs
        self.hidden4 = nn.Linear(64, 32)  # Hidden layer with 64 inputs and 32 outputs
        self.output = nn.Linear(32, 1)    # Output layer with 32 inputs and 1 output
        self.dropout = nn.Dropout(0.3)    # Dropout layer with a 30% drop rate

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = F.relu(self.hidden3(x))
        x = self.dropout(x)
        x = F.relu(self.hidden4(x))
        x = self.output(x)
        return x