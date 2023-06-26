# Standard Library
import sys; sys.path.append("../")
import os
from glob import glob

# External Libraries
import buteo as beo
import numpy as np
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast


class CNN_BasicEncoder(nn.Module):
    """
    Basic CNN encoder for the MNIST dataset

    Parameters
    ----------
    output_dim : int, optional
        The output dimension, default: 1
    
    Returns
    -------
    output : torch.Model
        The output model
    """
    def __init__(self, output_dim=1):
        super(CNN_BasicEncoder, self).__init__()
        self.conv1 = nn.LazyConv2d(32, 3, 1, 1)
        self.bn1 = nn.LazyBatchNorm2d(32)
        self.max_pool2d1 = nn.MaxPool2d(2)
        self.conv2 = nn.LazyConv2d(64, 3, 1, 1)
        self.bn2 = nn.LazyBatchNorm2d(64)
        self.max_pool2d2 = nn.MaxPool2d(2)
        self.fc1 = nn.LazyLinear(128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.LazyLinear(output_dim)

    def forward(self, x):
        # First layer: Conv -> Norm -> Activation -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.max_pool2d1(x)

        # Second Layer: Conv -> Norm -> Activation -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.max_pool2d2(x)

        # Flatten the layers and pass through the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x) # Dropout to prevent overfitting

        output = self.fc2(x)

        return output


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/model_zoo/data/patches/"

# Hyperparameters
NUM_EPOCHS = 250
PATIENCE = 25
VAL_SPLIT = 0.1
BATCH_SIZE = 16
NUM_WORKERS = 0 # Increase if you are working on a very large dataset. For small ones, multiple workers are slower.

# Cosine annealing scheduler with warm restarts
LEARNING_RATE = 0.001
T_0 = 15  # Number of iterations for the first restart
T_MULT = 2  # Multiply T_0 by this factor after each restart
ETA_MIN = 0.000001  # Minimum learning rate of the scheduler

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
else:
    print("No CUDA device available.")

def callback_preprocess(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    y = y.astype(np.float32, copy=False)

    return x_norm, y

def callback_postprocess(x, y):
    x = beo.channel_last_to_first(x)
    y = np.array([np.sum(y) / y.size], dtype=np.float32)

    return torch.from_numpy(x), torch.from_numpy(y)

def callback(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess(x, y)

    return x, y


def patience_calculator(epoch, t_0, t_m, max_patience=50):
    """ Calculate the patience for the scheduler. """
    if epoch <= t_0:
        return t_0

    p = [t_0 * t_m ** i for i in range(100) if t_0 * t_m ** i <= epoch][-1]
    if p > max_patience:
        return max_patience

    return p

x_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(FOLDER, "*train_s2.npy")))], shuffle=True)
y_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(FOLDER, "*train_label_area.npy")))])
y_train.set_shuffle_index(x_train.get_shuffle_index())

x_train, x_val = x_train.split(1 - VAL_SPLIT) # First 90% is training, last 10% is validation
y_train, y_val = y_train.split(1 - VAL_SPLIT) # First 90% is training, last 10% is validation

x_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(FOLDER, "*test_s2.npy")))])
y_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in sorted(glob(os.path.join(FOLDER, "*test_label_area.npy")))])

assert len(x_train) == len(y_train) and len(x_test) == len(y_test) and len(x_val) == len(y_val), "Lengths of x and y do not match."

# ds_train = beo.Dataset(x_train, y_train, callback=callback) # without augmentations
ds_train = beo.DatasetAugmentation(
    x_train, y_train,
    callback_pre_augmentation=callback_preprocess,
    callback_post_augmentation=callback_postprocess,
    augmentations=[
        beo.AugmentationRotationXY(p=0.2, inplace=True),
        beo.AugmentationMirrorXY(p=0.2, inplace=True),
        beo.AugmentationCutmix(p=0.2, inplace=True),
        beo.AugmentationNoiseNormal(p=0.2, inplace=True),
    ]
)
ds_test = beo.Dataset(x_test, y_test, callback=callback)
ds_val = beo.Dataset(x_val, y_val, callback=callback)

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS, drop_last=True, generator=torch.Generator(device='cuda'))
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS, generator=torch.Generator(device='cuda'))
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS, drop_last=True, generator=torch.Generator(device='cuda'))


if __name__ == "__main__":
    print("Starting training...")
    print("")

    model = CNN_BasicEncoder(output_dim=1)
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    # Save the initial learning rate in optimizer's param_groups
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = LEARNING_RATE

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0,
        T_MULT,
        ETA_MIN,
        last_epoch=NUM_EPOCHS - 1,
    )

    best_epoch = 0
    best_loss = None
    best_model_state = None
    epochs_no_improve = 0

    model.train()

    # Training loop
    for epoch in range(NUM_EPOCHS):

        # Initialize the running loss
        train_loss = 0.0

        # Initialize the progress bar for training
        train_pbar = tqdm(dl_train, total=len(dl_train), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Cast to bfloat16
            with autocast(dtype=torch.float16):

                # Forward pass
                outputs = model(images)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                # To avoid gradients either exploding or flushing to zero we use a gradient scaler.
                # If we were to use float32, instead of bfloat16, this would not be needed.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # loss.backward()
            # optimizer.step()

            # Update the scheduler
            scheduler.step()

            train_loss += loss.item()

            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
            })

            # Validate at the end of each epoch
            # This is done in the same scope to keep tqdm happy.
            if i == len(dl_train) - 1:

                # Validate every epoch
                with torch.no_grad():
                    model.eval()

                    val_loss = 0
                    for i, (images, labels) in enumerate(dl_val):
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)

                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                # Append val_loss to the train_pbar
                train_pbar.set_postfix({
                    "loss": f"{train_loss / len(dl_train):.4f}",
                    "val_loss": f"{val_loss / len(dl_val):.4f}",
                }, refresh=True)

                if best_loss is None:
                    best_loss = val_loss
                elif best_loss > val_loss:
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    best_epoch = epoch

                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == patience_calculator(epoch, T_0, T_MULT):
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    # Load the best weights
    model.load_state_dict(best_model_state)

    print("Finished Training. Best epoch: ", best_epoch + 1)
    print("")
    print("Starting Testing...")
    model.eval()

    # Test the model
    with torch.no_grad():
        test_loss = 0
        for i, (images, labels) in enumerate(dl_test):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

        print(f"Test Accuracy: {test_loss / (i + 1):.4f}")
