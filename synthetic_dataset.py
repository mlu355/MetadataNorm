from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils

class SyntheticDataset(Dataset):
    """Synthetic dataset."""

    def __init__(self, imgs, labels, cfs, transform=None):
        """
        Args:
            imgs (array of images): array of input images
            labels (array of [0, 1]): array of labels
            cfs (array of cfs): array of corresponding cfs 
            transform (cfs): Optional transform to be applied
                on a sample.
        """
        self.imgs = imgs
        self.labels = labels
        self.cfs = cfs
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]

        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        cf = self.cfs[idx]
        datum = {'image': image, 'label': int(label), 'cfs': cf}
        return datum