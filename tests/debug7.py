import os
import torch
import torch.nn as nn
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


def predict(arr):
    swap = beo.channel_last_to_first(arr)
    as_torch = torch.from_numpy(swap).float()
    on_device = as_torch.to(device)
    predicted = model(on_device)
    on_cpu = predicted.to('cpu')
    as_numpy = on_cpu.numpy()
    swap_back = beo.channel_first_to_last(as_numpy)

    return swap_back


if __name__ == "__main__":

    FOLDERS = [
        "C:/Users/philab/Desktop/train_guac/ISR/",
        "C:/Users/philab/Desktop/train_guac/ISR/",
    ]

    for FOLDER in FOLDERS:
        PATH_B08 = os.path.join(FOLDER, "B08.tif")

        # Define the device for computation
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # device = "cpu"

        # Instantiate the model, optimizer and loss function
        model = Unet().to(device)

        # Load the model
        model.load_state_dict(torch.load(os.path.join(FOLDER, "model_unet_4.pt")))

        model.eval()
        with torch.no_grad():
            idx = 0
            for arr, offset in beo.raster_to_array_chunks(PATH_B08, chunks=12, filled=True, fill_value=0, cast=np.float32):
                np.divide(arr, 10000.0, out=arr, dtype=np.float32)

                torch.cuda.empty_cache()
                predicted = beo.predict_array(
                    arr,
                    predict,
                    tile_size=128,
                    n_offsets=3,
                )
                np.multiply(predicted, 10000.0, out=predicted, dtype=np.float32)
                np.rint(predicted, out=predicted)
                predicted = predicted.astype(np.uint16)

                beo.array_to_raster(
                    predicted,
                    reference=PATH_B08,
                    pixel_offsets=offset,
                    out_path=os.path.join(FOLDER, f"B8A_{idx}.tif"),
                )

                idx += 1
