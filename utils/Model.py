from models.Cifar10Net import Cifar10Net
from models.MnistNet import MnistNet
from models.Simplecnn import Simplecnn
from models.GModel import GModel


def get_model(model_name, device):
    """
    Returns a model given it's name
    """
    if model_name.lower() in ["mnist", "fashionmnist"]:
        model = MnistNet()
    elif model_name.lower() in ["cifar10"]:
        model = Cifar10Net()
    elif model_name.lower() in ["cnn"]:
        model = Simplecnn()

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)