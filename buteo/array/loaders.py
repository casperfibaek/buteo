import numpy as np


class MultiArray:
    """
    This is a class that takes in a tuple of list of arrays and glues them together
    without concatenating them. This is useful for when you have a large
    dataset that you want to load into memory, but you don't want to
    concatenate them because that would take up too much memory.

    The function works for saved numpy arrays loading using mmap_mode="r".

    Parameters
    ----------
    array_or_list_of_arrays : array or list of arrays
        The arrays to glue together. Can be len 1.

    Returns
    -------
    MultiArray
        The multi array. Lazily loaded.

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
    >>> print(len(multi_array).shape)
    (32, 128, 128, 10)
    ```
    """
    def __init__(self, array_or_list_of_arrays):
        self.input_was_array = isinstance(array_or_list_of_arrays, (np.ndarray, np.memmap))
        self.list_of_arrays = array_or_list_of_arrays if not self.input_was_array else [array_or_list_of_arrays]

        self.one_deep = False
        if isinstance(self.list_of_arrays[0], (np.ndarray, np.memmap)):
            self.one_deep = True
            self.list_of_arrays = [self.list_of_arrays]

        if not self.one_deep:
            # All arrays in the list, must be the same length.
            first_len = 0
            for idx, lst in enumerate(self.list_of_arrays):
                if isinstance(lst, (np.ndarray, np.memmap)):
                    lst = [lst]

                assert isinstance(lst, list), "All arrays must be in a list."

                tup_len = 0
                for arr in lst:
                    tup_len += arr.shape[0]

                if idx == 0:
                    first_len = tup_len
                else:
                    assert tup_len == first_len, "All arrays must have the same length."

        # Calculate the cumulative sizes once, to speed up future calculations
        self.cumulative_sizes = np.cumsum([0] + [arr.shape[0] for arr in self.list_of_arrays[0]])
        self._shape = (self.cumulative_sizes[-1],) + self.list_of_arrays[0][0].shape[1:]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_single_item(idx)

        raise TypeError("Invalid argument type.")

    def _get_single_item(self, idx):
        if idx < 0:  # added this block to support negative indexing
            idx = self.__len__() + idx

        array_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        array_tuple = [arrays[array_idx] for arrays in self.list_of_arrays]
        idx_within_array = idx - self.cumulative_sizes[array_idx]

        return_list = [array[idx_within_array] for array in array_tuple]

        if self.one_deep:
            return_list = return_list[0]

        if self.input_was_array:
            return_list = return_list[0]

        return return_list
    
    def __len__(self):
        return self.cumulative_sizes[-1]
