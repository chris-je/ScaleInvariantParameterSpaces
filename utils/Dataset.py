from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# select a dataset according to the arguments and return it
def get_data_loader(dataset_name, batch_size):

    transform = {
        "cifar10": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "cifar100": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "mnist": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        "fashionmnist": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }[dataset_name.lower()]

    if dataset_name.lower() == "cifar10":
        train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        val_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    elif dataset_name.lower() == "cifar100":
        train_data = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
        val_data = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    elif dataset_name.lower() == "mnist":
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        val_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif dataset_name.lower() == "fashionmnist":
        train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
        val_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader