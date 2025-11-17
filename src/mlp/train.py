from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from model import GAN
import config


def _prepare_training_data():
    data_dir = config.SCRIPT_DIR.parent.parent / "data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # maps [0,1] → [-1,1]
    ])
    mnist_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    X_train = mnist_train.data.unsqueeze(1).float()
    X_train = X_train.view(X_train.shape[0], -1)
    dataset = TensorDataset(X_train)
    data_loader = DataLoader(dataset, config.BATCH_SIZE, shuffle=True)
    return data_loader

def main():
    train_loader = _prepare_training_data()
    model = GAN()
    model.fit(train_loader, config.EPOCHS)

if __name__ == "__main__":
    main()