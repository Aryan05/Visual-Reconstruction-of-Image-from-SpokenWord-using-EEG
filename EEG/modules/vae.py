import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VAE(nn.Module):
    def __init__(self, n_inputs=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), # -1,32,14,14
            nn.Conv2d(32,64,4,2,1), # -1,64,7,7
            nn.Conv2d(64,128,3,2,0), # -1,128,3,3
            nn.Conv2d(128,n_inputs,3,1,0), # -1,32,1,1
            Flatten() # -1,32
        )

        self.fc1 = nn.Linear(n_inputs, n_inputs)
        self.fc2 = nn.Linear(n_inputs, n_inputs)
        self.fc3 = nn.Linear(n_inputs,128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,256,4,2,1), # 2,2,256
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1), # 4,4,128
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), # 8,8,64
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), # 16,16,32
            nn.ReLU(),
            nn.ConvTranspose2d(32,1,2,2,2), # 28, 28, 1
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def bottleneck(self,h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self,x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return z, mu, logvar

    def decode(self,z):
        z = z.view((-1,128,1,1))
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
