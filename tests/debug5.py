import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import numpy as np
import buteo as beo


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.e1 = encoder_block(num_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        self.b = conv_block(512, 1024)
        # self.b = conv_block(64, 128)
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        # b = self.b(p1)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        # d4 = self.d4(d2, s1)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)

        return outputs



class NumpyDataset(Dataset):
    def __init__(self, numpy_array):
        self.numpy_array = numpy_array

    def __len__(self):
        return len(self.numpy_array)

    def __getitem__(self, idx):
        # Transpose the numpy array from (height, width, channels) to (channels, height, width)
        image = self.numpy_array[idx].transpose((2, 0, 1))

        # copy the image to avoid modifying the original image
        image = np.divide(image, 10000.0, dtype=np.float32)

        # Here, I'm assuming the image is both the feature and target
        # If this is not the case, split the feature and target here
        X = torch.from_numpy(image[0:1, :, :])
        y = torch.from_numpy(image[1:2, :, :])

        return X, y


if __name__ == "__main__":
    training_data_path = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/patches_b08-b8a.npy"
    training_data = np.load(
        training_data_path,
        mmap_mode="r",
    )

    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001

    # training_data = training_data[0:10000, :, :, :]
    dataset = NumpyDataset(training_data)

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define the device for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model, optimizer and loss function
    model = Unet().to(device)

    # Load the model
    model.load_state_dict(torch.load("C:/Users/casper.fibaek/OneDrive - ESA/Desktop/model_unet.pt"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Define number of epochs
    n_epochs = 5

    # Training loop
    for epoch in range(n_epochs):
        model.train()  # set the model to training mode
        epoch_loss = 0

        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss/len(train_dataloader)}")
        torch.save(model.state_dict(), f"C:/Users/casper.fibaek/OneDrive - ESA/Desktop/model_unet_{epoch + 1}.pt")

    print(model)

    # Save the model
    torch.save(model.state_dict(), "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/model_unet_v2.pt")
