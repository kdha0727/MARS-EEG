from collections import OrderedDict
import os
import pandas as pd
from scipy import io
import torch
import torch.utils.data


class LegacyVnsEEGDataset(torch.utils.data.Dataset):

    def __init__(self, original_csv_path, ext='.mat'):
        # original_csv_path: EEG_original.csv
        datum = OrderedDict()
        metadata = pd.read_csv(original_csv_path)
        for _, entity in metadata.iterrows():
            if not os.path.isdir(entity.path):
                continue
            for filename in os.listdir(entity.path):
                filename_ = os.path.join(entity.path, filename)
                if os.path.splitext(filename)[-1] == ext and os.path.isfile(filename_):
                    datum[filename_] = entity.target
        self.datum = list(datum.items())

    def __getitem__(self, index):
        path, target = self.datum[index]
        mat = io.loadmat(path)
        key = os.path.splitext(os.path.split(path)[-1])[0]
        data = torch.as_tensor(mat[key][0][0][15], dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int32)
        return data, target

    def __len__(self):
        return len(self.datum)


class VnsEEGDataset(torch.utils.data.Dataset):

    _window_size: int

    def __init__(self, csv_path, window_size=30000, channel_truncation=22):
        # csv_path: EEG_processing.csv (by output of `convert_csv.py`)
        self.metadata = pd.read_csv(csv_path)
        self.channel_truncation = channel_truncation
        self.window_size = window_size

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        self._window_size = value
        self._prepare()

    def _prepare(self):
        datum = [...] * self.metadata['T'].map(lambda x: x // self._window_size).sum()
        count = 0
        for _, entity in self.metadata.iterrows():
            num_windows = entity['T'] // self._window_size
            for iteration in range(num_windows):
                datum[count] = entity.path, iteration * self._window_size, entity.target  # path, start, target
                count += 1
        self.datum = datum

    def __getitem__(self, index):
        path, start, target = self.datum[index]
        mat = io.loadmat(path)
        key = os.path.splitext(os.path.split(path)[-1])[0]
        data = mat[key][0][0][15][:self.channel_truncation, start:start + self.window_size]
        data = torch.as_tensor(data, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int32)
        return data, target

    def __len__(self):
        return len(self.datum)


# class EEGDataLoader(torch.utils.data.DataLoader):
#     def __init__(self, dataset, window_size=30000, batch_size=1, shuffle=False, num_workers=0):
#         self.window_size = window_size
#
#         def collate_fn(batch):
#             data, target = zip(*batch)
#             data = torch.stack(data)
#             target = torch.stack(target)
#             num_windows = data.shape[2] // self.window_size
#
#             # Truncate the data to have integer multiples of the window size
#             data = data[:, :, :num_windows * self.window_size]
#
#             # Reshape data to split into windows
#             data_windows = data.reshape(-1, num_windows, data.shape[1], self.window_size)
#             target = target.repeat(num_windows)
#
#             return data_windows[0], target
#
#         super(EEGDataLoader, self).__init__(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             collate_fn=collate_fn,
#             num_workers=num_workers
#         )


if __name__ == '__main__':
    dataset = VnsEEGDataset(os.path.join(os.path.dirname(__file__), 'EEG_processing.csv'))
    print(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,  # True if num_workers > 0
        drop_last=True,
    )
    dat, trg = next(iter(loader))
    print(dat.shape, dat.dtype)
    print(trg)
    print(len(loader))
