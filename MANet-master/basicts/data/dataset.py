import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import load_pkl


class TimeSeriesForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path)
        # read raw data (normalized)
        data = np.load(data_file_path)
        index = np.load(index_file_path)
        self.data = data
        self.index = index
        self.mode = mode
        # read index
        # self.index = load_pkl(index_file_path)[mode]

    def _check_if_file_exists(self, data_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))

    def __getitem__(self, mode: str) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        # idx = list(self.index[index])
        # if isinstance(idx[0], int):
        #     # continuous index
        #     history_data = self.data[idx[0]:idx[1]]
        #     future_data = self.data[idx[1]:idx[2]]
        # else:
        #     # discontinuous index or custom index
        #     # NOTE: current time $t$ should not included in the index[0]
        #     history_index = idx[0]    # list
        #     assert idx[1] not in history_index, "current time t should not included in the idx[0]"
        #     history_index.append(idx[1])
        #     history_data = self.data[history_index]
        #     future_data = self.data[idx[1], idx[2]]
        if self.mode == "train":
            history_data = self.data['train_x']
            future_data = self.data['train_target']
            history_index = self.index['train_x']
            return future_data, history_data, history_index
        elif self.mode == "valid":
            history_data = self.data['val_x']
            future_data = self.data['val_target']
            history_index = self.index['val_x']
            return future_data, history_data, history_index
        elif self.mode == "test":
            history_data = self.data['test_x']
            future_data = self.data['test_target']
            history_index = self.index['test_x']
            return future_data, history_data, history_index

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        # return len(self.index)
        return self.data['train_x'].shape[0]+self.data['val_x'].shape[0]+self.data['test_x'].shape[0]
