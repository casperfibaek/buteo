import numpy as np



class CachedMultiArray:
    """
    This is a class that takes in a list of arrays and glues them together
    without concatenating them. This is useful for when you have a large
    dataset that you want to load into memory, but you don't want to
    concatenate them because that would take up too much memory.

    This class also caches the last accessed array, so that if you are
    accessing the arrays sequentially, it will be fast.

    The function also worked for saved numpy arrays loading using mmap_mode="r".
    Uses reservoir sampling to sample random batches. As such, it is not
    guaranteed that all samples will be used. The sampling strategy is uniform.

    Parameters
    ----------
    arrays : list of np.ndarray
        The arrays to glue together.

    cache_size : int
        The size of the cache. If you are accessing the arrays sequentially,
    
    enable_cache : bool. Default: False
        Whether to enable the cache. If you are accessing the arrays

    Returns
    -------
    CachedMultiArray
        The cached multi array. Lazily loaded.

    Examples
    --------
    ```python
    >>> from glob import glob
    >>> folder = "./data_patches/"

    >>> patches = sorted(glob(folder + "train*.npy"))
    >>> multi_array = CachedMultiArray([np.load(p, mmap_mode="r") for p in patches], enable_cache=False)
    >>> single_batch = next(multi_array.random_batches(32))

    >>> print(multi_array.shape)
    (6474, 128, 128, 10)
    >>> print(single_batch.shape)
    (32, 128, 128, 10)
    ```
    """
    def __init__(self, arrays, cache_size=1024, enable_cache=False):
        self.arrays = arrays
        self.sizes = np.cumsum([0] + [a.shape[0] for a in arrays])
        self.enable_cache = enable_cache
        self.array = None  # Define the array variable here. This is for the cache.

        if enable_cache:
            self.cache_size = cache_size
            self.cache = None
            self.cache_idx = None
        else:
            self.cache_size = None
            self.cache = None
            self.cache_idx = None

        # Save the shape of the first array, except its first dimension.
        self._shape = (self.sizes[-1],) + self.arrays[0].shape[1:]

    def _load_cache(self, idx):
        array_idx = np.searchsorted(self.sizes, idx, side='right') - 1
        self.array = self.arrays[array_idx]
        within_array_idx = idx - self.sizes[array_idx]
        start = max(0, within_array_idx - self.cache_size // 2)
        stop = start + self.cache_size

        self.cache = self.array[start:stop]
        self.cache_idx = start

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, *rest = idx
        else:
            rest = ()

        if self.enable_cache:
            if self.cache is None or not self.cache_idx <= idx < self.cache_idx + len(self.cache):
                self._load_cache(idx)
            return self.cache[idx - self.cache_idx][tuple(rest)]

        else:
            array_idx = np.searchsorted(self.sizes, idx, side='right') - 1
            array = self.arrays[array_idx]
            within_array_idx = idx - self.sizes[array_idx]

            return array[within_array_idx][tuple(rest)]

    @property
    def shape(self):
        """ The shape of the array. """
        return self._shape

    # Reservoir sampling
    def random_batches(self, batch_size):
        """
        Returns a generator that yields random batches of the array.
        Consume like so: next(multi_array.random_batches(32))

        Parameters
        ----------
        batch_size : int
            The size of the batches.

        Returns
        -------
        generator
            A generator that yields random batches of the array.
        """
        total_size = self.sizes[-1]
        n_batches = total_size // batch_size
        remainder = total_size % batch_size

        for _ in range(n_batches):
            batch_indices = np.random.choice(total_size, size=batch_size, replace=False)
            yield np.array([self[i] for i in batch_indices])

        if remainder != 0:
            batch_indices = np.random.choice(total_size, size=remainder, replace=False)
            yield np.array([self[i] for i in batch_indices])

    def __len__(self):
        return self.sizes[-1]
