import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image


def MNIST_Dataloader(batch_size=50):
    # Download MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

    # MNist Data Loader
    batch_size = batch_size
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":

    train_loader, test_loader = MNIST_Dataloader()
    print(len(train_loader), len(test_loader))
    for i, (img, label) in enumerate(train_loader):
        print(i, img.shape, label.shape)
        break

