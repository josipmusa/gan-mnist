import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import config


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1, 1, 28, 28) #(batch, channel(hardcoded to 1), 28, 28)

    def generate(self):
        z = torch.randn(1, 100)  # batch of 1, 100-dim noise vector
        self.eval()
        with torch.no_grad():
            output = self(z)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.gen = Generator()
        self.disc = Discriminator()
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.gen_loss = nn.BCELoss()
        self.disc_loss = nn.BCELoss()

    def fit(self, train_loader, epochs):
        train_disc_losses, train_gen_losses = [], []
        for epoch in range(epochs):
            train_disc_loss = 0
            train_gen_loss = 0
            for x, in train_loader:
                #Discriminator training
                batch_size = x.size(0)
                real_labels = torch.ones(batch_size, 1) * 0.9 #avoid using straight up 1 for real labels, use 0.9
                fake_labels = torch.zeros(batch_size, 1) + 0.1 #avoid using straight up 0 for fake labels, use 0.1
                #Handle real images
                real_preds = self.disc(x)
                real_loss = self.disc_loss(real_preds, real_labels)
                #Handle fake images from generator
                z_d = torch.randn(batch_size, 100) # batch of 100-dim noise vectors
                fake_images_for_d = self.gen(z_d)
                fake_preds = self.disc(fake_images_for_d.detach().view(batch_size, -1)) #flatten generated images since they are [batch_size, 1, 28, 28] and discriminator expects input of [batch_size, 784]
                fake_loss = self.disc_loss(fake_preds, fake_labels)
                #Compute loss - backpropagation
                disc_loss = real_loss + fake_loss
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                self.disc_optimizer.step()

                #Generator training
                #Generate fake images
                z_g = torch.randn(batch_size, 100) # batch of 100-dim noise vectors
                fake_images_for_g = self.gen(z_g)
                disc_preds = self.disc(fake_images_for_g.view(batch_size, -1)) #flatten here also
                #Compute loss - backpropagation
                gen_loss = self.gen_loss(disc_preds, real_labels)
                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()

                train_disc_loss += disc_loss.item() * batch_size
                train_gen_loss += gen_loss.item() * batch_size

            train_disc_loss /= len(train_loader.dataset)
            train_gen_loss /= len(train_loader.dataset)
            train_disc_losses.append(train_disc_loss)
            train_gen_losses.append(train_gen_loss)
            print(f"Epoch {epoch}, Disc loss: {train_disc_loss:.4f}, Gen loss: {train_gen_loss:.4f}")
        torch.save(self.gen.state_dict(), config.MODEL_PATH)
        _plot_loss_curve(train_disc_losses, train_gen_losses)

def _plot_loss_curve(train_disc_losses, train_gen_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_disc_losses, label="train_disc")
    plt.plot(train_gen_losses, label="train_gen")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.LOSS_CURVE_PATH)
    plt.close()
    print(f"Saved loss curve to {config.LOSS_CURVE_PATH}")