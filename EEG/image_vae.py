import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms

from modules import VAE

from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

n_epochs = 1
batch_size = 32
LOAD_PATH = '../ckpt/image_vae.ckpt'
SAVE_PATH = '../ckpt/image_vae_2.ckpt'
MFF_DIR_PATH = '/home/91726/Desktop/data/'

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train_image_vae(vae, data_loader):
    optimizer = optim.Adam(vae.parameters(), lr = 1e-4)
    for epoch in range(n_epochs):
        for idx, (images, _) in enumerate(data_loader):
            optimizer.zero_grad()
            recon_images, mu, logvar = vae(images)
            loss = loss_fn(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}/{}] loss: {:.3f}".format(epoch+1,
                                n_epochs,idx, loss.data.item())
            print(to_print)
        torch.save(vae.state_dict(),SAVE_PATH)

    return vae

def main():
    # Load MNIST image dataset
    mnist_train_data = datasets.MNIST(
        '/home/ajays/Downloads/',download=True,transform=transforms.ToTensor()
    )
    mnist_test_data = datasets.MNIST('/home/ajays/Downloads/',train=False,download=True)

    train_loader = torch.utils.data.DataLoader(
        mnist_train_data, batch_size = batch_size, shuffle=True
    )

    # Instantiation
    vae = VAE(n_inputs=32)

    # *********************
    # IMAGE VAE TRAINING
    # *********************
    # plot before training
    # o_before, mu, logvar = vae(mnist_train_data[0][0].reshape((1,1,28,28)))
    # plt.imshow(o_before.detach().numpy().reshape((28,28)))
    # plt.show()

    # train
    vae.load_state_dict(torch.load(LOAD_PATH))
    #vae = train_image_vae(vae, train_loader)

    # After training
    # o_after, mu, logvar = vae(example[0].reshape((1,1,28,28)))
    o_after = vae.decode(torch.randn((128)))
    plt.imshow(o_after.detach().numpy().reshape((28,28)))
    plt.show()

if __name__ == '__main__':
    main()
