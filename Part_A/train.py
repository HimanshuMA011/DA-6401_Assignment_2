
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from smallCNN import SmallCNNModel
from Dataset import iNaturalistDataModule

# Sweep configuration for wandb hyperparameter search
sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'
    },
    'parameters': {
        'conv_filters': {
            'values': [[32, 32, 32, 32, 32], [512, 256, 128, 64, 32], [32, 64, 128, 256, 512]]
        },
        'kernel_size': {
            'values': [[3, 3, 3, 3, 3], [5, 3, 3, 3, 3]]
        },
        'conv_activation': {
            'values': ['ReLU', 'SiLU', 'GeLU','Mish']
        },
        'dense_neurons': {
            'values': [128, 256]
        },
        'dense_activation': {
            'values': ['ReLU', 'SiLU']
        },
        'dropout': {
            'values': [0.2, 0.3]
        },
        'batch_normalise': {
            'values': [True, False]
        },
        'lr': {
            'values': [1e-3, 1e-4]
        },
        'batch_size': {
            'values': [32, 64]
        },
        'data_augmentation': {
            'values': [True, False]
        }
    }
}

def train():
    # Initialize wandb run and get hyperparameters
    wandb.init(project="Small_cnn_model")  # Add project name
    config = wandb.config

    print("Loaded Config:", config)


    model = SmallCNNModel(
        conv_filters=config.get("conv_filters", [32, 32, 32, 32, 32]),  # Default filters
        kernel_size=config.get("kernel_size", [3, 3, 3, 3, 3]),
        conv_activation=config.get("conv_activation", "ReLU"),
        dense_neurons=config.get("dense_neurons", 128),
        dense_activation=config.get("dense_activation", "ReLU"),
        dropout=config.get("dropout", 0.2),
        batch_normalise=config.get("batch_normalise", False),
        lr=config.get("lr", 1e-3)
    )

    data_module = iNaturalistDataModule(
        data_dir='nature_12K',
        batch_size=config.get("batch_size", 64),
        data_augmentation=config.get("data_augmentation", False)
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Log training process with wandb
    wandb_logger = WandbLogger(project="Small_cnn_model", log_model="all")

    # Define Trainer pytorch_lighting method which automatically trains model and logs metrics
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=10,
        # accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")],
        log_every_n_steps=25
    )

    # Train the model
    trainer.fit(model, train_loader,val_loader)

sweep_id = wandb.sweep(sweep_config,project='Small_cnn_model')    
wandb.agent(sweep_id, train, count=1)



