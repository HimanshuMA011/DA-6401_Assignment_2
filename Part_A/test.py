
import torch
import wandb
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from smallCNN import SmallCNNModel
from Dataset import iNaturalistDataModule
from pytorch_lightning.loggers import WandbLogger


#  Initialize WandB Project
wandb.init(project="Small_cnn_model", name="best_model_eval_")

torch.cuda.empty_cache() # if gpu has not enough space

#  Load the best model from sweeps
best_model = SmallCNNModel(
    # best model from training :'batch_size': 32, 'conv_activation': 'SiLU', 'conv_filters': [512, 256, 128, 64, 32], 'kernel_size': [3, 3, 3, 3, 3], 'dense_activation': 'SiLU', 'dense_neurons': 128, 'dropout': 0.3, 'lr': 0.0001, 'data_augmentation': False,
    conv_filters=[512, 256, 128, 64, 32],
    kernel_size=[3, 3, 3, 3, 3],
    conv_activation="SiLU",
    dense_neurons=128,
    dense_activation="SiLU",
    dropout=0.3,
    batch_normalise=True,
    lr=0.0001
)

#  Load dataset
data_module = iNaturalistDataModule(
    data_dir='/kaggle/input/nature12k/inaturalist_12K',
    batch_size=32,
    data_augmentation=False
)
data_module.setup()

test_loader = data_module.test_dataloader()

# Load WandB Logger
wandb_logger = WandbLogger(project="Small_cnn_model", log_model="all")


trainer = pl.Trainer(
    max_epochs=10,
    logger=wandb_logger,
    accelerator="gpu"    
)

trainer.fit(best_model, data_module)

test_results = trainer.test(best_model, test_loader)
test_accuracy = test_results[0]['test_acc']


wandb.log({"Test Accuracy": test_accuracy})



def log_test_predictions(model, dataloader, num_classes=10, num_per_class=3):
    model.eval()
    class_images = {i: [] for i in range(num_classes)}  # Store images per class

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Collect 3 images per class
            for img, label, pred in zip(images, labels, preds):
                if len(class_images[label.item()]) < num_per_class:
                    class_images[label.item()].append((img, pred.item()))

            # Stop if we have enough samples
            if all(len(class_images[i]) == num_per_class for i in range(num_classes)):
                break

    # Create a grid
    fig, axes = plt.subplots(num_classes, num_per_class, figsize=(num_per_class * 3, num_classes * 3))

    for class_idx, ax_row in enumerate(axes):
        for img_idx, ax in enumerate(ax_row):
            if class_idx in class_images and len(class_images[class_idx]) > img_idx:
                img, pred_label = class_images[class_idx][img_idx]
                img = img.permute(1, 2, 0).cpu().numpy()

                ax.imshow(img)
                ax.set_title(f"Pred: {pred_label}", fontsize=10)
                ax.axis("off")

    plt.tight_layout()

    # Log to WandB
    wandb.log({"Test Predictions": wandb.Image(fig, caption="Per-Class Predictions (10x3)")})
    plt.close(fig)


test_dataloader = data_module.test_dataloader()
log_test_predictions(best_model, test_dataloader)

wandb.finish()
