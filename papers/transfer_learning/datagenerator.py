# x=DataGenerator(munis=test_munis, batch_size=bs[0], target="area"),


class DataGenerator(Sequence):
    """
    Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, munis=[], batch_size=64, target="area"):
        """Initialization
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """

        self.folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/machine_learning_data/"

        self.images = []
        self.length = 0
        self.first_run = True
        self.current_muni = 0
        self.current_batch = 0
        self.batches_on_image = 0
        self.x_array = None
        self.y_array = None
        self.target = target
        self.use_munis = munis
        self.batch_size = batch_size
        self.on_epoch_end()

        self.all_munis = np.load(self.folder + "municipalities.npy")

        for muni in self.all_munis:
            if muni not in self.use_munis:
                continue

            labels = np.load(self.folder + f"{muni}_LABELS.npy")
            batch_count = int(np.ceil(labels.shape[0] / self.batch_size))

            self.images.append(
                {
                    "muni": muni,
                    "length": labels.shape[0],
                    "batches": batch_count,
                    "rgbn": self.folder + f"{muni}_RGBN.npy",
                    "swir": self.folder + f"{muni}_SWIR.npy",
                    "sar": self.folder + f"{muni}_SAR.npy",
                    "labels": self.folder + f"{muni}_LABELS.npy",
                }
            )
            self.length += batch_count
            labels = None

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

        current_muni = 0
        label_index = {
            "area": 0,
            "volume": 1,
            "people": 2,
        }

        if self.first_run:
            x_arr = preprocess_optical(
                random_scale_noise(np.load(self.images[0]["rgbn"]))
            )
            x_arr, y_arr = rotate_shuffle(
                [
                    x_arr,
                    np.load(self.images[0]["labels"])[
                        :, :, :, label_index[self.target]
                    ],
                ]
            )

            self.x_array = x_arr
            self.y_array = y_arr

            self.batches_on_image = self.images[0]["batches"]
            self.first_run = False

        if self.batches_on_image <= 0:
            current_muni += 1
            x_arr = preprocess_optical(
                random_scale_noise(np.load(self.images[current_muni]["rgbn"]))
            )
            x_arr, y_arr = rotate_shuffle(
                [
                    x_arr,
                    np.load(self.images[current_muni]["labels"])[
                        :, :, :, label_index[self.target]
                    ],
                ]
            )

            self.x_array = x_arr
            self.y_array = y_arr
            self.batches_on_image = self.images[current_muni]["batches"]
            self.current_batch = 0

        low = self.current_batch * self.batch_size
        high = (self.current_batch + 1) * self.batch_size

        if high > self.x_array.shape[0]:
            high = self.x_array.shape[0]

        X = self.x_array[low:high]
        y = self.y_array[low:high]

        self.batches_on_image -= 1
        self.current_batch += 1

        return X, y
