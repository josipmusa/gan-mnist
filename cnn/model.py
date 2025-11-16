import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import config


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=100, out_features=256 * 7 * 7)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),  #7x7 -> 14x14
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=1), # 14x14 -> 28x28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256, 7, 7)
        x = self.layers(x)
        return x

    def generate(self):
        z = torch.randn(1, 100)
        self.eval()
        with torch.no_grad():
            output = self(z)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        num_filters = 256
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=4, stride=2, padding=1), #14x14
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=4, stride=2, padding=1), #7x7
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(num_filters * 2 * 7 * 7, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = Generator()
        self.disc = Discriminator()
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.gen_loss = nn.BCEWithLogitsLoss()
        self.disc_loss = nn.BCEWithLogitsLoss()

    def fit(self, train_loader, epochs = 20):
        train_gen_losses, train_disc_losses = [], []
        for epoch in range(epochs):
            epoch_gen_loss, epoch_disc_loss = 0, 0
            for x, in train_loader:
                batch_size = x.size(0)
                real_labels = torch.ones(batch_size, 1) * 0.9  # avoid using straight up 1 for real labels, use 0.9
                fake_labels = torch.zeros(batch_size, 1) + 0.1  # avoid using straight up 0 for fake labels, use 0.1

                #Discriminator training
                self.disc.train()
                real_disc_preds = self.disc(x)
                real_disc_loss = self.disc_loss(real_disc_preds, real_labels)

                z_disc = torch.randn(batch_size, 100) #generate noise vector
                fake_images_for_disc = self.gen(z_disc)
                fake_images_preds_for_disc = self.disc(fake_images_for_disc.detach())
                fake_disc_loss = self.disc_loss(fake_images_preds_for_disc, fake_labels)

                disc_loss = real_disc_loss + fake_disc_loss
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                self.disc_optimizer.step()

                #Generator training
                self.gen.train()
                z_gen = torch.randn(batch_size, 100)
                fake_images_for_gen = self.gen(z_gen)
                fake_images_preds_for_gen = self.disc(fake_images_for_gen)

                gen_loss = self.gen_loss(fake_images_preds_for_gen, real_labels)
                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()

                epoch_disc_loss += disc_loss.item() * batch_size
                epoch_gen_loss += gen_loss.item() * batch_size

            avg_disc_loss = epoch_disc_loss / len(train_loader.dataset)
            avg_gen_loss = epoch_gen_loss / len(train_loader.dataset)
            train_disc_losses.append(avg_disc_loss)
            train_gen_losses.append(avg_gen_loss)
            print(f"Epoch {epoch}, Disc loss: {avg_disc_loss: .4f}, Gen loss: {avg_gen_loss: .4f}")

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

