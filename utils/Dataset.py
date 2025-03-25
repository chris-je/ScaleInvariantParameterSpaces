from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(dataset_name, batch_size):
    """
    Loads the specified dataset and returns train and validation data loaders
    """

    dataset_name = dataset_name.lower()

    # Mapping of dataset names to torchvision datasets and normalization parameters
    dataset_map = {
        "cifar10": (datasets.CIFAR10, (0.5, 0.5, 0.5)),
        "cifar100": (datasets.CIFAR100, (0.5, 0.5, 0.5)),
        "mnist": (datasets.MNIST, (0.5,)),
        "cnn": (datasets.MNIST, (0.5,)),
        "fashionmnist": (datasets.FashionMNIST, (0.5,))
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_class, mean_std = dataset_map[dataset_name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_std, mean_std)
    ])

    train_data = dataset_class(root="data", train=True, download=True, transform=transform)
    val_data = dataset_class(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
