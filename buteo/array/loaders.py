from typing import Union, List, Tuple

import numpy as np


class MultiArray:
    def __init__(self,
        array_list: Union[np.ndarray, List[Union[np.ndarray, List[np.ndarray]]]],
        shuffle: bool = False,
        seed: int = 42,
    ):
        """
        This is a class that takes in a tuple of list of arrays and glues them together
        without concatenating them. This is useful for when you have a large
        dataset that you want to load into memory, but you don't want to
        concatenate them because that would take up too much memory.

        The function works for saved numpy arrays loading using mmap_mode="r".

        Parameters
        ----------
        array_list : Union[np.ndarray, List[Union[np.ndarray, List[np.ndarray]]]]
            The input can be multimodal data in the following formats:
                1. A single numpy array. 
                2. A list of numpy arrays. (single modality) - will be iterable
                3. A list of lists of numpy arrays. (multi-modality) - Each list will be iterable, returning a tuple of arrays.

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
        self.list_list_arrays, self.original_format = self._convert_input(array_list)
        self.cumulative_lens = []
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_incides = None

        # Calculate the cumulative lengths once for all the arrays
        for i in range(len(self.list_list_arrays)):
            self.cumulative_lens.append(
                np.cumsum([0] + [arr.shape[0] for arr in self.list_list_arrays[i]])
            )

        # Check that all arrays have the same length (across modalities)
        if len(self.cumulative_lens) > 1:
            for i in range(1, len(self.cumulative_lens)):
                if not np.array_equal(self.cumulative_lens[i], self.cumulative_lens[i-1]):
                    raise ValueError("All arrays should have the same length.")

        # They should all be the same, so just take the first one
        self.cumulative_sizes = self.cumulative_lens[0]

        if self.shuffle:
            np.random.seed(self.seed)
            self.shuffle_incides = np.random.permutation(self.__len__())


    def _convert_input(self,
        array_list: Union[np.ndarray, List[Union[np.ndarray, List[np.ndarray]]]],
    ) -> Tuple[List[List[np.ndarray]], str]:
        """
        The input can be:
            1. A single numpy array.
            2. A list of numpy arrays 
            3. A list of lists of numpy arrays

        All inputs will be converted to format 3. The output will be in the original format.
        """
        # Case 1: single numpy array
        if isinstance(array_list, np.ndarray):
            return [[array_list]], 'array'  

        elif isinstance(array_list, list):
            if not array_list:
                raise ValueError("Input list is empty. Please provide a list with numpy arrays.")

            # Case 2: list of numpy arrays
            if all(isinstance(item, np.ndarray) for item in array_list):
                return [array_list], 'list'

            # Case 3: list of lists of numpy arrays
            elif all(isinstance(item, list) and all(isinstance(sub_item, np.ndarray) for sub_item in item) for item in array_list):
                return array_list, 'list_list'

            else:
                raise ValueError("Input list should either contain numpy arrays or lists of numpy arrays.")
        else:
            raise TypeError("Invalid input type. Input should either be a numpy array or a list of numpy arrays or a list of lists of numpy arrays.")


    def _revert_input(self,
        converted_array_list: List[List[np.ndarray]], original_format: str,
    ) -> Union[np.ndarray, List[Union[np.ndarray, List[np.ndarray]]]]:
        """ Convert the list of lists back to the original format. """
        if original_format == 'array':
            return converted_array_list[0][0]

        elif original_format == 'list':
            return converted_array_list[0]

        elif original_format == 'list_list':
            return converted_array_list
        else:
            raise ValueError("Invalid format string. It should be one of the following: 'array', 'list', 'list_list'.")


    def set_shuffle_index(self, shuffle_indices: np.ndarray) -> None:
        """ Set the shuffle indices and enable shuffling. """
        self.shuffle_incides = shuffle_indices
        self.shuffle = True
        assert len(self.shuffle_incides) == len(self), "Length of shuffle indices should be equal to the length of the MultiArray."

    
    def get_shuffle_index(self) -> np.ndarray:
        """ Get the shuffle indices. """
        return self.shuffle_incides


    def __getitem__(self, idx: int):
        if idx < 0:  # support negative indexing
            idx = self.__len__() + idx

        if self.shuffle:
            idx = self.shuffle_incides[idx]

        if idx >= self.__len__():
            raise IndexError("Index out of range")

        array_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1

        output = [
            self.list_list_arrays[i][array_idx][idx - self.cumulative_sizes[array_idx]]
            for i in range(len(self.list_list_arrays))
        ]

        return self._revert_input(output, self.original_format)


    def __len__(self):
        return self.cumulative_sizes[-1]


class MultiArraySubset:
    """
    A subset of a MultiArray. Usually used for train/val/test splits.
    """
    def __init__(self,
        multi_array: MultiArray,
        idx_start: int,
        idx_end: int,
    ):
        self.multi_array = multi_array
        self.idx_start = idx_start
        self.idx_end = idx_end

        assert self.idx_start < self.idx_end, "Start index should be smaller than end index."
        assert self.idx_start >= 0, "Start index should be larger than or equal to 0."
        assert self.idx_end <= len(self.multi_array), "End index should be smaller than or equal to the length of the MultiArray."

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = self.__len__() + idx
        
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        
        return self.multi_array[self.idx_start + idx]
    
    def __len__(self):
        return self.idx_end - self.idx_start


def split_multi_array(multi_array, split_point=0.9):
    """
    Split a MultiArray into two MultiArraySubsets.

    Parameters
    ----------
    multi_array : MultiArray
        A MultiArray object.

    split_point : float or int. Default: 0.9
        The split point for the multi array. If float, it will be interpreted as a percentage.
        If int, it will be interpreted as an index.

    Returns
    -------
    MultiArraySubset, MultiArraySubset
        Two MultiArraySubset objects. (before_split_point, after_split_point)
    """
    if isinstance(split_point, float):
        split_point = int(split_point * len(multi_array))

    if split_point > len(multi_array):
        raise ValueError("Split point is larger than the length of the MultiArray.")
    
    before_split_point = MultiArraySubset(multi_array, 0, split_point)
    after_split_point = MultiArraySubset(multi_array, split_point, len(multi_array))

    return before_split_point, after_split_point
