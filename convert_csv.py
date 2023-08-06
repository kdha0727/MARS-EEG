from tqdm.std import tqdm
from collections import OrderedDict
import os
import pandas as pd
import scipy


def convert_csv(original_csv_path, ext='.mat'):
    # csv from xlsx (folder, target) -> another csv (filepath, C, T, target)
    datum = OrderedDict()
    metadata = pd.read_csv(original_csv_path)
    cnt = 0
    outer_loop = tqdm(metadata.iterrows(), leave=False, position=0, total=len(metadata))
    for _, entity in outer_loop:
        outer_loop.set_description(os.path.split(entity.path)[-1])
        if not os.path.isdir(entity.path):
            continue
        inner_loop = tqdm(os.listdir(entity.path), leave=False, position=1)
        for filename in inner_loop:
            inner_loop.set_description(filename)
            filename_ = os.path.join(entity.path, filename)
            if os.path.splitext(filename)[-1] == ext and os.path.isfile(filename_):
                mat = scipy.io.loadmat(filename_)
                shape = mat[os.path.splitext(filename)[0]][0][0][15].shape  # C, T
                datum[cnt] = dict(path=filename_, C=shape[0], T=shape[1], target=entity.target)
            cnt += 1
    return pd.DataFrame(datum.values())


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), 'EEG_original.csv')
    convert_csv(csv_path).to_csv(os.path.join(os.path.dirname(__file__), 'EEG_processing.csv'), index=False)
