from typing import Union, List, Tuple, Optional

import numpy as np


class MultiArray:
    def __init__(self,
        array_list: List[Union[np.ndarray, np.memmap]],
        shuffle: bool = False,
        random_sampling: bool = False,
        seed: int = 42,
        _idx_start: Optional[int] = None,
        _idx_end: Optional[int] = None,
        _is_subarray: bool = False
    ):
        """
        This is a class that takes in a tuple of list of arrays and glues them together
        without concatenating them. This is useful for when you have a large
        dataset that you want to load into memory, but you don't want to
        concatenate them because that would take up too much memory.

        The function works for saved numpy arrays loading using mmap_mode="r".

        Parameters
        ----------
        array_list : List[Union[np.ndarray, np.memmap]]
            A list of numpy arrays to load as one. Can be mmaped or not.
        
        shuffle : bool, default: False
            Whether to shuffle the data or not. Cannot be used together with random_sampling.
            This is different from random sampling in that it ensures that all data will be used.
            It creates an index array that is shuffled and then uses that to index the MultiArray.
        
        random_sampling : bool, default: False
            Whether to use random sampling or not. If True, the returned data will be randomly sampled from the MultiArray.
            Cannot be used together with shuffling.
            Does not ensure that all data will be used, but it will be randomly (uniform) sampled.

        Returns
        -------
        MultiArray
            A class that can handle multiple memmapped numpy arrays.

        Examples
        --------
        ```python
        >>> from glob import glob
        >>> folder = "./data_patches/"

        >>> patches = sorted(glob(folder + "train*.npy"))
        >>> multi_array = MultiArray([np.load(p, mmap_mode="r") for p in patches])
        >>> single_image = multi_array[0]

        >>> print(single_image.shape)
        (128, 128, 10)
        >>> print(len(multi_array))
        32
        ```
        """
        self.array_list = array_list
        self.is_subarray = _is_subarray
        
        assert isinstance(self.array_list, list), "Input should be a list of numpy arrays."
        assert len(self.array_list) > 0, "Input list is empty. Please provide a list with numpy arrays."
        assert all(isinstance(item, (np.ndarray, np.memmap)) for item in self.array_list), "Input list should only contain numpy arrays."

        self.cumulative_sizes = np.cumsum([0] + [arr.shape[0] for arr in self.array_list])

        self._idx_start = int(_idx_start) if _idx_start is not None else 0
        self._idx_end = int(_idx_end) if _idx_end is not None else int(self.cumulative_sizes[-1])

        assert isinstance(self._idx_start, int), "Minimum length should be an integer."
        assert isinstance(self._idx_end, int), "Maximum length should be an integer."
        assert self._idx_start < self._idx_end, "Minimum length should be smaller than maximum length."

        self.total_length = int(min(self.cumulative_sizes[-1], self._idx_end - self._idx_start))  # Store length for faster access

        if shuffle and random_sampling:
            raise ValueError("Cannot use both shuffling and resevoir sampling at the same time.")

        # Shuffling
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_indices = None
        self.random_sampling = random_sampling
        self.rng = np.random.default_rng(seed)

        if self.shuffle:
            self.shuffle_indices = self.rng.permutation(range(self._idx_start, self._idx_end))

    def _load_item(self, idx: int):
        """ Load an item from the array list. """
        array_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1

        calculated_idx = idx - self.cumulative_sizes[array_idx]
        if calculated_idx < 0 or calculated_idx >= self.array_list[array_idx].shape[0]:
            raise IndexError(f'Index {idx} out of bounds for MultiArray with length {self.__len__()}')

        output = self.array_list[array_idx][calculated_idx]

        return output


    def set_shuffle_index(self, shuffle_indices: np.ndarray) -> None:
        """ Set the shuffle indices and enable shuffling. """
        self.shuffle_indices = shuffle_indices
        self.shuffle = True
        assert len(self.shuffle_indices) == len(self), "Length of shuffle indices should be equal to the length of the MultiArray."

    
    def get_shuffle_index(self) -> np.ndarray:
        """ Get the shuffle indices. """
        return self.shuffle_indices
    
    
    def shuffle_index(self) -> None:
        """ Shuffle the MultiArray. """
        self.shuffle_indices = self.rng.permutation(range(self._idx_start, self._idx_end))
        self.shuffle = True

    
    def disable_shuffle(self) -> None:
        """ Disable shuffling. """
        self.shuffle = False


    def __getitem__(self, idx: int):
        if idx < 0:
            idx = self._idx_end + idx

        if self.random_sampling:
            idx = np.random.randint(self._idx_start, self._idx_end)
        elif self.shuffle:
            idx = self.shuffle_indices[idx]
        else:
            idx = idx + self._idx_start

        if idx < self._idx_start or idx >= self._idx_end:
            raise IndexError("Index out of range")

        return self._load_item(idx)


    def split(self, split_point: Union[int, float]) -> Tuple["MultiArray", "MultiArray"]:
        """
        Split the MultiArray into two MultiArrays.

        Parameters
        ----------
        split_point : int or float
            The split point for the multi array. If float, it will be interpreted as a percentage.
            If int, it will be interpreted as an index.

        Returns
        -------
        MultiArray, MultiArray
            Two MultiArray objects. (before_split_point, after_split_point)
        """
        if isinstance(split_point, float):
            split_point = int(split_point * len(self))

        if split_point > len(self):
            raise ValueError("Split point is larger than the length of the MultiArray.")
            
        if self.is_subarray:
            raise ValueError("Cannot split an array that has already been split.")

        before_split_point = MultiArray(self.array_list, shuffle=False, random_sampling=self.random_sampling, seed=self.seed, _idx_start=self._idx_start, _idx_end=self._idx_start + split_point, _is_subarray=True)
        after_split_point = MultiArray(self.array_list, shuffle=False, random_sampling=self.random_sampling, seed=self.seed, _idx_start=self._idx_start + split_point, _idx_end=self._idx_end, _is_subarray=True)

        return before_split_point, after_split_point


    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]


    def __len__(self):
        return self.total_length



if __name__ == "__main__":
    import os

    FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/model_zoo/data/patches/"

    paths = [
        os.path.join(FOLDER, "copenhagen_train_s2.npy"),
        os.path.join(FOLDER, "naestved_test_s2.npy"),
        os.path.join(FOLDER, "odense_train_s2.npy"),
        os.path.join(FOLDER, "copenhagen_train_s2.npy"),
        os.path.join(FOLDER, "naestved_test_s2.npy"),
        os.path.join(FOLDER, "odense_train_s2.npy"),
        os.path.join(FOLDER, "copenhagen_train_s2.npy"),
        os.path.join(FOLDER, "naestved_test_s2.npy"),
        os.path.join(FOLDER, "odense_train_s2.npy"),
    ]

    multi_array = MultiArray([np.load(p, mmap_mode="r") for p in paths], shuffle=True)

    print(len(multi_array))

    for _ in range(100000):
        for i in range(len(multi_array)):
            bob = multi_array[i]
        multi_array.shuffle_index()

    print(bob)
