""" This is a debug script, used for ad-hoc testing. """
# disable all of pylint for this file only.
# pylint: disable-all

import sys; sys.path.append("../")
import os
import gc
from glob import glob
import numpy as np
import buteo as beo

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/ccai_tutorial/alexandria/patches/"
PATCH_SIZE = 32
PREPROCESS = False
SEED = 17
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

if PREPROCESS:
    test_indices = np.random.choice(78, 78 // 10)

    # Lists to hold our patches
    training_label = []
    training_s1 = []
    training_s2 = []
    training_dem = []

    testing_label = []
    testing_s1 = []
    testing_s2 = []
    testing_dem = []

    # Read and order the tiles in the temporary folder.
    total_patches_train = 0
    total_patches_test = 0
    for image in zip(
        sorted(glob(os.path.join(FOLDER, "label_*.tif"))),
        sorted(glob(os.path.join(FOLDER, "s1_*.tif"))),
        sorted(glob(os.path.join(FOLDER, "s2_*.tif"))),
        sorted(glob(os.path.join(FOLDER, "dem_*.tif"))),
    ):
        path_label, path_s1, path_s2, path_dem = image

        label_name = os.path.splitext(os.path.basename(path_label))[0]
        img_idx = int(label_name.split("_")[1])

        # Get the data from the tiles
        arr_label = beo.raster_to_array(path_label, filled=True, fill_value=0.0, cast=np.float32)
        arr_s1 = beo.raster_to_array(path_s1, filled=True, fill_value=0.0, cast=np.float32)
        arr_s2 = beo.raster_to_array(path_s2, filled=True, fill_value=0.0, cast=np.float32)
        arr_dem = beo.raster_to_array(path_dem, filled=True, fill_value=0.0, cast=np.float32)

        # Lets normalise the data
        arr_s1, _statdict = beo.scaler_truncate(arr_s1, -35.0, 5.0)
        arr_s2, _statdict = beo.scaler_truncate(arr_s2, 0.0, 10000.0)

        # Extract patches
        patches_label = beo.array_to_patches(arr_label, PATCH_SIZE)
        patches_s1 = beo.array_to_patches(arr_s1, PATCH_SIZE)
        patches_s2 = beo.array_to_patches(arr_s2, PATCH_SIZE)
        patches_dem = beo.array_to_patches(arr_dem, PATCH_SIZE)

        # Simple sanity check to ensure that the right images were chosen.
        assert patches_label.shape[0:3] == patches_s1.shape[0:3] == patches_s2.shape[0:3] == patches_dem.shape[0:3], "Patches do not align."

        if img_idx in test_indices:
            total_patches_test += patches_label.shape[0]
            testing_label.append(patches_label)
            testing_s1.append(patches_s1)
            testing_s2.append(patches_s2)
            testing_dem.append(patches_dem)
        else:
            total_patches_train += patches_label.shape[0]
            training_label.append(patches_label)
            training_s1.append(patches_s1)
            training_s2.append(patches_s2)
            training_dem.append(patches_dem)

    # Merge the patches back together
    shuffle_mask_train = np.random.permutation(total_patches_train)
    y_train = np.concatenate(training_label, axis=0).astype(np.float32, copy=False)[shuffle_mask_train]
    x_train_s1 = np.concatenate(training_s1, axis=0)[shuffle_mask_train]
    x_train_s2 = np.concatenate(training_s2, axis=0)[shuffle_mask_train]
    x_train_dem = np.concatenate(training_dem, axis=0)[shuffle_mask_train]

    shuffle_mask_test = np.random.permutation(total_patches_test)
    y_test = np.concatenate(testing_label, axis=0).astype(np.float32, copy=False)[shuffle_mask_test]
    y_test_s1 = np.concatenate(testing_s1, axis=0)[shuffle_mask_test]
    y_test_s2 = np.concatenate(testing_s2, axis=0)[shuffle_mask_test]
    y_test_dem = np.concatenate(testing_dem, axis=0)[shuffle_mask_test]

    # Lets save the data to our temporary folder and do some house cleaning.
    # This takes about 2 minutes, so go make some coffee!
    np.savez_compressed(os.path.join(FOLDER, "train.npz"), y=y_train, x_s1=x_train_s1, x_s2=x_train_s2, x_dem=x_train_dem)
    np.savez_compressed(os.path.join(FOLDER, "test.npz"), y=y_test, x_s1=y_test_s1, x_s2=y_test_s2, x_dem=y_test_dem)

    # Free the memory
    del training_label, training_s1, training_s2, training_dem
    del testing_label, testing_s1, testing_s2, testing_dem
    del y_train, x_train_s1, x_train_s2, x_train_dem
    del y_test, y_test_s1, y_test_s2, y_test_dem

    gc.collect()

# This is a Simple Convolutional Neural Network.
class SimpleConvNet(nn.Module):
    def __init__(self, in_channels, output_min, output_max):
        super(SimpleConvNet, self).__init__()
        self.output_min = output_min
        self.output_max = output_max

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        # Lets help the network
        x = torch.clamp(x, self.output_min, self.output_max)
        return x

# Load the data
x_train = np.load(os.path.join(FOLDER, "train.npz"))["x_s2"]
y_train = np.load(os.path.join(FOLDER, "train.npz"))["y"]

x_train, x_val, y_train, y_val, = beo.split_train_val(x_train, y_train, val_size=0.1, random_state=42)

# Define a model
model = SimpleConvNet(9, 0.0, 100.0)

# Lets train!
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def callback(x, y):
    return (
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float(),
    )
# Create the dataset and DataLoader
dataset = beo.AugmentationDataset(
    x_train,
    y_train,
    callback=callback,
    input_is_channel_last=True,
    output_is_channel_last=False,
    augmentations=[
        { "name": "rotation", "chance": 0.2},
        { "name": "mirror", "chance": 0.2 },
        # { "name": "channel_scale", "chance": 0.2 },
        # { "name": "noise", "chance": 0.2 },
        # { "name": "contrast", "chance": 0.2 },
        # { "name": "drop_pixel", "chance": 0.2 },
        # { "name": "drop_channel", "chance": 0.2 },
        # { "name": "blur_xy", "chance": 0.2 },
        # { "name": "sharpen_xy", "chance": 0.2 },
        # { "name": "cutmix", "chance": 0.2 },
        # { "name": "mixup", "chance": 0.2 },
    ],
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

x_val = torch.from_numpy(beo.channel_last_to_first(x_val)).float()
y_val = torch.from_numpy(beo.channel_last_to_first(y_val)).float()

# Training loop
for epoch in range(EPOCHS):
    running_loss = 0.0

    # Customize the progress bar with colors and fixed width
    bar_format = '{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'

    # Initialize the progress bar for training
    train_pbar = tqdm(dataloader, total=len(dataloader), ncols=140, position=0, leave=True, bar_format=bar_format)

    for i, (inputs, targets) in enumerate(train_pbar):
        # Move inputs and targets to the device (GPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Cast to bfloat16
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Print statistics
        current_loss = loss.item()
        running_loss += current_loss
        mean_loss = running_loss / (i + 1)

        train_pbar.set_description(f"Epoch: {epoch+1:03d}/{EPOCHS:03d}")
        print_dict = { "loss": f"{mean_loss:09.2f}" }

        if i == len(train_pbar) - 1:
            model.eval()  # Set model to evaluation mode

            with torch.no_grad():
                inputs, targets = x_val.to(device), y_val.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)

            print_dict["val_loss"] = f"{val_loss.item():09.2f}"

            model.train()  # Set model back to training mode

        train_pbar.set_postfix(print_dict)


# Clear GPU memory
torch.cuda.empty_cache()

# This way of doing metrics seem to give very high values - is it correct?
with torch.no_grad():
    x_test = beo.channel_last_to_first(np.load(os.path.join(FOLDER, "test.npz"))["x_s2"])
    y_test = beo.channel_last_to_first(np.load(os.path.join(FOLDER, "test.npz"))["y"])
    y_pred = model(torch.from_numpy(x_test).float().to(device)).cpu().detach().numpy()

print("RMSE:", np.sqrt(np.mean((y_test - y_pred) ** 2)))
print("MAE:", np.mean(np.abs(y_test - y_pred)))
print("MSE:", np.mean((y_test - y_pred) ** 2))

# torch.save(model.state_dict(), os.path.join(FOLDER, "example_model.pt"))

# BS 32, LR 0.0001, 100 epochs
# RMSE: 20.995897
# MAE: 8.981177
# MSE: 440.8277
