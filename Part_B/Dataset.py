import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

class iNaturalistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, valid_split=0.2, data_augmentation=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_split = valid_split
        self.data_augmentation = data_augmentation 

        # Mean and std from ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Train Transform with Data Augmentation
        if self.data_augmentation:
            self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(), 
                transforms.RandomRotation(10),  # Rotate within ±10 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust colors
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)  # Normalize
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

        # Validation & Test Transform (No Augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def setup(self, stage=None):
        # Load dataset
        full_train_dataset = torchvision.datasets.ImageFolder(root=f'{self.data_dir}/train', transform=self.train_transform)
        self.test_dataset = torchvision.datasets.ImageFolder(root=f'{self.data_dir}/val', transform=self.test_transform)

        # split train dataset into train 0.8 & validation 0.2
        total = len(full_train_dataset)
        val_size = int(total * self.valid_split)
        train_size = total - val_size
        self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])

        # Set validation transform
        self.val_dataset.dataset.transform = self.test_transform
        return self.train_dataset,self.val_dataset,self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
