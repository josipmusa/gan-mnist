from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import config
from model import GAN


def _prepare_training_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # maps [0,1] → [-1,1]
    ])
    mnist_train = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
    data_loader = DataLoader(mnist_train, config.BATCH_SIZE, shuffle=True)
    return data_loader

def main():
    train_loader = _prepare_training_data()
    model = GAN()
    model.to(config.DEVICE)
    model.fit(train_loader, config.EPOCHS)

if __name__ == "__main__":
    main()