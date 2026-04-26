import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import dataset
from nn import Generator, Discriminator
from loss import VGGLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-4
batch_size = 1
epochs = 1

gen = Generator(in_channels=3).to(device)
disc = Discriminator(in_channels=3).to(device)

opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.9, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.9, 0.999))

vgg_loss = VGGLoss(device)
pixel_loss = nn.L1Loss()
adv_loss = nn.BCEWithLogitsLoss()

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, real_img in enumerate(dataloader):
        real_img = real_img.to(device)
        
        low_res = nn.functional.interpolate(real_img, scale_factor=0.25, mode='bicubic')
        low_res = nn.functional.interpolate(low_res, size=(256, 256), mode='bicubic')

        opt_disc.zero_grad()
        fake_img = gen(low_res)
        disc_real = disc(real_img)
        disc_fake = disc(fake_img.detach())
        
        loss_disc_real = adv_loss(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = adv_loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        
        loss_disc.backward()
        opt_disc.step()

        opt_gen.zero_grad()
        disc_fake_for_gen = disc(fake_img)
        
        l_vgg = vgg_loss(fake_img, real_img)
        l_pixel = pixel_loss(fake_img, real_img)
        l_adv = adv_loss(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
        
        loss_gen = l_pixel + l_vgg + 0.005 * l_adv
        
        loss_gen.backward()
        opt_gen.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {i} | Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")