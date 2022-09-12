import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class CIFAR100Instance(torchvision.datasets.CIFAR100):
    """CIFAR100Instance Dataset."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

def get_training_CIFAR100(mean, std, batch_size=16, num_workers=2, shuffle=True, distributed=False):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    try:
        cifar100_training = CIFAR100Instance(root='/dev/shm/datasets/CIFAR/', train=True, download=False,
                                                          transform=transform_train)
    except:
        cifar100_training = CIFAR100Instance(root='datasets/CIFAR/', train=True, download=False,
                                                          transform=transform_train)

    if distributed:
        cifar100_training_loader = DataLoader(cifar100_training, num_workers=num_workers, batch_size=batch_size,
                                              pin_memory=True, sampler=DistributedSampler(cifar100_training))
    else:
        cifar100_training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_CIFAR100(mean, std, batch_size=16, num_workers=2, shuffle=True, distributed=False):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    try:
        cifar100_test = CIFAR100Instance(root='/dev/shm/datasets/CIFAR/', train=False, download=False, transform=transform_test)
    except:
        cifar100_test = CIFAR100Instance(root='datasets/CIFAR/', train=False, download=False, transform=transform_test)

    if distributed:
        cifar100_test_loader = DataLoader(cifar100_test, num_workers=num_workers, batch_size=batch_size,
                                          pin_memory=True, sampler=DistributedSampler(cifar100_test))
    else:
        cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader








