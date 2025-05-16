from torchvision import datasets, transforms


class PermutedMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = 42
        torch.manual_seed(seed)
        self.permutation = torch.randperm(28 * 28)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img = img.view(-1)[self.permutation].view(1, 28, 28)
        return img, target


class SequentialMNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img = img.view(1, 28, 28)
        return img, target
    

def get_datasets(model_name):
    ds = {}
    tf = {}
    
    if model_name in ["mlp", "cnn"]:
        ds = {
            "MNIST":        datasets.MNIST,
            "FashionMNIST": datasets.FashionMNIST,
            "CIFAR10":      datasets.CIFAR10,
            "CIFAR100":     datasets.CIFAR100
        }
        tf = {name: transforms.Compose([transforms.ToTensor()]) for name in ds}
        
    elif model_name == "transformer":
        ds = {
            'MNIST':        (datasets.MNIST, {'train': True}),
            'FashionMNIST': (datasets.FashionMNIST, {'train': True}),
            'CIFAR10':      (datasets.CIFAR10, {'train': True}),
            'QMNIST':       (datasets.QMNIST, {'train': True, 'what': 'train'})
        }

        for name in ds:
            if name == 'CIFAR10':
                mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
                tf[name] = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            else:
                mean, std = [0.1307], [0.3081]  # reused for grayscale datasets
                tf[name] = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(mean * 3, std * 3)
                ])
    
    elif model_name == "rnn":
        ds = {
            'MNIST': datasets.MNIST,
            'FashionMNIST': datasets.FashionMNIST,
            'PermutedMNIST': PermutedMNIST,
            'SequentialMNIST': SequentialMNIST
        }
        tf = {name: transforms.Compose([transforms.ToTensor()]) for name in ds}
        
    return ds, tf


def get_hyperparameters(model_name):
    lr_values    = [0.0005, 0.001, 0.005]
    batch_sizes  = [50, 100]
    eps_values   = [0.1, 0.15, 0.2]
    n_trials     = 100
    max_steps   = 500
    
    if model_name == "transformer":
        eps_values = [0.6, 0.5, 0.4]
    
    return (
        lr_values,
        batch_sizes,
        eps_values,
        n_trials,
        max_steps
    )