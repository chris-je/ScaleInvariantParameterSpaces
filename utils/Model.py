from models.Cifar10Net import Cifar10Net
from models.CifarNN import CifarNN
from models.MnistNet import MnistNet
from models.SimpleNN import SimpleNN


# Select a model according to the arguments and return it
def get_model(dataset_name):
    if dataset_name.lower() in ["mnist", "fashionmnist"]:
        # return SimpleNN()
        return MnistNet()
    elif dataset_name.lower() in ["cifar100"]:
        num_classes = 10 if dataset_name.lower() == "cifar10" else 100
        return CifarNN(num_classes)
    elif dataset_name.lower() in ["cifar10"]:
        return Cifar10Net()
    else:
        raise ValueError(f"Unsupported model: {dataset_name}")
