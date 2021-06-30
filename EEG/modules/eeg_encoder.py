import torch
import torch.nn as nn 

class EEGEncoder(nn.Module):
    def __init__(self, n_inputs):
        super(EEGEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.LSTM(n_inputs, n_inputs, 1), # -1,32
            nn.Linear(n_inputs, n_inputs), # -1,32
            nn.ReLU() # -1,32
        )

        self.fc1 = nn.Linear(n_inputs, n_inputs)
        self.fc2 = nn.Linear(n_inputs, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.encoder(x)
        x = self.sigmoid(self.fc2(self.fc1(x)))
        return x
