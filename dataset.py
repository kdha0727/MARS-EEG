from collections import OrderedDict
import os
import pandas as pd
import scipy
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
        mat = scipy.io.loadmat(path)
        key = os.path.splitext(os.path.split(path)[-1])[0]
        data = torch.as_tensor(mat[key][0][0][15], dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int32)
        return data, target

    def __len__(self):
        return len(self.datum)


if __name__ == '__main__':
    print(len(VnsEEGDataset(os.path.join(os.path.dirname(__file__), 'EEG_processing.csv'))))
