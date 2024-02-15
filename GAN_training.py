from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.utils.data import Dataset
import json

dataroot = "D:/Github/dungeonCrawlerMgr/LevelData/training_data_preprocessed_all/class1"
#dataroot = "D:/Github/dungeonCrawlerMgr/LevelData/training_data_preprocessed_copied/class1"
results_directory = "D:/Github/dungeonCrawlerMgr/LevelData/results_GAN/"
# 1024 images =>> 1 epoch = 8 iterations

# Hyperparameters
workers = 2   # the number of worker threads for loading the data with the help of DataLoader
batch_size = 128
image_size = 20
nc = 7  # number of channels in the input images
nz = 20  # length of latent vector
ngf = 15  # relates to the depth of feature maps
ndf = 15  # sets the depth of feature maps propagated through the discriminator
lr_gen = 0.001 #0.0002  # learning rate for generator
lr_discr = 0.0001  # learning rate for discriminator
beta1 = 0.5  # beta1 hyperparameter for Adam optimizers
kernel_size = 4
num_epochs = 20
save_step = 20

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d( nz, ngf * 8, kernel_size, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4       # input is Z
            nn.ConvTranspose2d(nz, ngf * 4, kernel_size, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8      # 80 x 4 x 4
            nn.ConvTranspose2d( ngf * 4, ngf * 2, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16      # 40 x 5 x 5
            nn.ConvTranspose2d( ngf * 2, ngf, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32      # 20 x 10 x 10
            nn.ConvTranspose2d( ngf, nc, kernel_size, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64      # 1 x 20 x 20
        )
    def forward(self, input):
        #print("G: ", input.shape)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64       # 1 x 20 x 20
            nn.Conv2d(nc, ndf, kernel_size, 2, 1, bias=False), #stride (przesunięcie) o 2, padding = 1
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32       # 20 x 10 x 10
            nn.Conv2d(ndf, ndf * 2, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16       # 40 x 5 x 5
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8        # 80 x 4 x 4
            nn.Conv2d(ndf * 4, 1, kernel_size, 1, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4        # 160 x 1 x 1
            # nn.Conv2d(ndf * 8, 1, kernel_size, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        #print("D: ", input.shape)
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train():
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(1, num_epochs + 1):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            #print(f"Shape before Discriminator: {real_cpu.shape}")  # Should be [128, 7, 20, 20]
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
                #output = output.reshape(128, 4).mean(dim=1)
            # Calculate loss on all-real batch
            #print("criterion output= ", output.size())
            #print("criterion label= ", label.size())
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1

        if epoch % save_step == 0 or epoch == num_epochs - 1:
            #save models
            dummy_input = torch.randn(1, nz, 1, 1, device=device)
            # Export the model
            torch.onnx.export(netG, dummy_input,
                              f"{results_directory}/generator_{epoch}.onnx",
                              dynamic_axes={'input': {0: 'batch_size'},
                                            'output': {0: 'batch_size'}})
            #torch.onnx.export(netG, (fixed_noise,), f"{results_directory}/generator_{epoch}.onnx")
            # Assuming one dummy input for Discriminator for ONNX export
            dummy_input_D = torch.randn(1, nc, image_size, image_size, device=device)
            torch.onnx.export(netD, (dummy_input_D,), f"{results_directory}/discriminator_{epoch}.onnx")
            print(f"Saved models at epoch {epoch}")

    return img_list, G_losses, D_losses

def plot_results(img_list, G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(results_directory + f"GAN_loss_plot.png")
    plt.show()
    plt.close()

class LevelTileDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with open(file_path, 'r') as file:
            data = json.load(file)
        level_array = torch.tensor(data['LevelTileArray'], dtype=torch.float)
        level_array = level_array.permute(2, 0, 1)#level_array.permute(0, 3, 1, 2)
        return level_array


# Main Function
if __name__ == "__main__":
    dataset = LevelTileDataset(dataroot)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    # for i, data in enumerate(dataloader, 0):
    #     print(f"Batch {i} shape: {data.shape}")  # Should be [batch_size, 7, 20, 20]
    #     if i >= 2:
    #         break
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.savefig(results_directory + f"GAN_img_plot.png")

    # Create the Generator
    netG = Generator(1).to(device)
    netG.apply(weights_init)
    print(netG)

    # Create the Discriminator
    netD = Discriminator(1).to(device)
    netD.apply(weights_init)
    print(netD)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(20, nz, 1, 1, device=device)
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr_discr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_gen, betas=(beta1, 0.999))

    img_list, G_losses, D_losses = train()
    plot_results(img_list, G_losses, D_losses)