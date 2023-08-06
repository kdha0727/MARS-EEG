from collections import OrderedDict
import os
import pandas as pd
from scipy import io
import torch
import torch.utils.data


class VnsEEGDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, ext='.mat'):
        datum = OrderedDict()
        metadata = pd.read_csv(csv_path)
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
    

class EEGDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, window_size=30000, batch_size=1, shuffle=False, num_workers=0):
        self.window_size = window_size

        def collate_fn(batch):
            data, target = zip(*batch)
            data = torch.stack(data)
            target = torch.stack(target)
            num_windows = data.shape[2] // self.window_size

            # Truncate the data to have integer multiples of the window size
            data = data[:, :, :num_windows * self.window_size]

            # Reshape data to split into windows
            data_windows = data.reshape(-1, num_windows, data.shape[1], self.window_size)
            target = target.repeat(num_windows)

            return data_windows[0], target

        super(EEGDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers
        )


if __name__ == '__main__':
    print(len(VnsEEGDataset(os.path.join(os.path.dirname(__file__), 'EEG_processing.csv'))))