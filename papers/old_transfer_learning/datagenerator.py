from tensorflow.keras.utils import Sequence
import numpy as np

# x=DataGenerator(munis=test_munis, batch_size=bs[0], target="area"),


class DataGenerator(Sequence):
    """
    Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, batch_size=32, target=0):
        """Initialization
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """

        self.folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/machine_learning_data/"

        self.regions = ["001", "002", "003", "004"]
        # self.regions = [779, 707]
        self.region_lengths = []
        self.batch_size = batch_size
        self.length = 0
        self.target = target

        for region in self.regions:
            self.region_lengths.append(
                np.load(self.folder + f"{region}_SWIR.npy").shape[0]
            )

        self.region_lengths_cumsum = np.cumsum(self.region_lengths)

        for length in self.region_lengths:
            self.length += int(np.floor(length / self.batch_size))

        self.loaded = 0
        self.load = 0
        self.rgbn = np.load(self.folder + f"{self.regions[self.loaded]}_RGBN.npy")
        self.swir = np.load(self.folder + f"{self.regions[self.loaded]}_SWIR.npy")
        self.sar = np.load(self.folder + f"{self.regions[self.loaded]}_SAR.npy")
        self.labels = np.load(self.folder + f"{self.regions[self.loaded]}_LABELS.npy")[
            :, :, :, self.target
        ]

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return self.length

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        cumsum = index * self.batch_size

        if index == 0:
            self.load = 0

        for idx, region in enumerate(self.region_lengths_cumsum):
            if cumsum < region:
                self.load = idx
                break

        if self.load != self.loaded:
            self.rgbn = np.load(self.folder + f"{self.regions[self.loaded]}_RGBN.npy")
            self.swir = np.load(self.folder + f"{self.regions[self.loaded]}_SWIR.npy")
            self.sar = np.load(self.folder + f"{self.regions[self.loaded]}_SAR.npy")
            self.labels = np.load(
                self.folder + f"{self.regions[self.loaded]}_LABELS.npy"
            )[:, :, :, self.target]

            self.loaded = self.load

        if self.loaded != 0:
            region_start = self.region_lengths_cumsum[self.loaded - 1]
        else:
            region_start = 0

        region_end = self.region_lengths_cumsum[self.loaded]

        start = (index * self.batch_size) - region_start

        end = start + self.batch_size

        if end > region_end:
            end = region_end

        return (
            [self.rgbn[start:end], self.swir[start:end], self.sar[start:end]],
            self.labels[start:end],
        )

        # for idx, region_sum in enumerate(self.region_lengths_cumsum):
        #     if low < region_sum:
        #         if idx > 0 and low < (self.region_lengths_cumsum[idx - 1] + self.batch_size)

        #         if idx != self.loaded:
        #             if idx == 0:
        #                 self.loaded = 0
        #             else:
        #                 self.loaded += 1

        #             self.rgbn = np.load(
        #                 self.folder + f"{self.regions[self.loaded]}_RGBN.npy"
        #             )
        #             self.swir = np.load(
        #                 self.folder + f"{self.regions[self.loaded]}_SWIR.npy"
        #             )
        #             self.sar = np.load(
        #                 self.folder + f"{self.regions[self.loaded]}_SAR.npy"
        #             )
        #             self.labels = np.load(
        #                 self.folder + f"{self.regions[self.loaded]}_LABELS.npy"
        #             )[:, :, :, self.target]

        #         if high >= region_sum:
        #             high = region_sum

        #         if self.loaded != 0:
        #             import pdb

        #             pdb.set_trace()

        #         return (
        #             [self.rgbn[low:high], self.swir[low:high], self.sar[low:high]],
        #             self.labels[low:high],
        #         )

        raise Exception("Should not get here.")
