import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from Dataset import iNaturalistDataModule

# Sweep Configuration
sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5]},
        "batch_size": {"values": [64, 128]},
        "denselayer_size": {"values": [64, 128]},
        "l_rate": {"values": [0.001, 0.0001]},
        "optimizer": {"values": ["Adam"]},
        "dropout": {"values": [0.2, 0.4]},
        "model_version": {"values": ["resnet50"]},
        "activation": {"values": ["relu", "leakyrelu"]}
    }
}

# Create Sweep
sweep_id = wandb.sweep(sweep_config, project="iNature_12K_by_pretrained_Resnet50-Model")

# Define Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class FineTuneModel(pl.LightningModule):
    def __init__(self, num_classes=10, model_version="resnet50", denselayer_size=128, dropout=0.4, l_rate=0.001, activation="relu"):
        super(FineTuneModel, self).__init__()
        self.learning_rate = l_rate
        self.activation_fn = nn.ReLU() if activation == "relu" else nn.LeakyReLU()

        self.model = models.__dict__[model_version](pretrained=True)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the final classification layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, denselayer_size),
            self.activation_fn,
            nn.Dropout(dropout),
            nn.Linear(denselayer_size, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        acc = (outputs.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        acc = (outputs.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        acc = (outputs.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

# Training Function for WandB Sweeps
def train():
    wandb.init()

    # Fetch hyperparameters
    config = wandb.config
    batch_size = config.batch_size
    epochs = config.epochs
    model_version = config.model_version
    denselayer_size = config.denselayer_size
    dropout = config.dropout
    l_rate = config.l_rate
    activation = config.activation

    # Load Data
    data_module = iNaturalistDataModule(
        data_dir='/kaggle/input/nature12k/inaturalist_12K',
        batch_size=batch_size
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Initialize Model
    model = FineTuneModel(num_classes=10, model_version=model_version, denselayer_size=denselayer_size, dropout=dropout,
                          l_rate=l_rate, activation=activation)

    # Set up WandB Logger
    wandb_logger = WandbLogger(project="iNature_12K_by_pretrained_Resnet50-Model")

    # Train Model
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )
    trainer.fit(model, train_loader, val_loader)

    # Test Best Model
    trainer.test(model, test_loader)

# Run the Sweep Agent
wandb.agent(sweep_id, function=train, count=20)