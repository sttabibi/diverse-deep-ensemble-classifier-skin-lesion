from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

import cv2

from torch.utils.data import Dataset


class SkinDataset(Dataset):
    def __init__(self, df: DataFrame, root: Path, n_sample: Union[int, None] = 115, transform=None):
        self.transform = transform
        self.root = root
        self.df = df.copy().sample(frac=1)
        if n_sample is not None:
            one_hots = self.df.iloc[:, 1:].to_numpy()
            n_classes = one_hots.shape[1]
            class_agg = []
            for class_idx in range(n_classes):
                src = one_hots[:, class_idx]
                target_idx = np.where(src == 1)[0]
                target_filenames = self.df.iloc[target_idx, :]
                tr_size = target_filenames.shape[0]
                tr_size_select = min(tr_size, n_sample)
                tr_idx = np.arange(tr_size)
                for _ in range(5):
                    np.random.shuffle(tr_idx)
                target_select_idx = np.random.choice(tr_idx, size=tr_size_select)
                tr_select_filenames = target_filenames.iloc[target_select_idx, :]
                tr_one_hot_classes = one_hots[target_select_idx, :]
                assert tr_one_hot_classes.shape[0] == tr_select_filenames.shape[0]
                class_agg.append(tr_select_filenames)

            self.df = pd.concat(class_agg).sample(frac=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        cls_idx = np.argmax(sample[1:].tolist())
        img_path = self.root.joinpath(f'{sample[0]}.jpg')
        im = cv2.imread(str(img_path))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            im = self.transform(im)
        return im, cls_idx


if __name__ == '__main__':
    train_path = "/home/jericho/Project/SkinClassification/data/Data/Train/ISIC2018_Task3_Training_GroundTruth.csv"
    train_df = pd.read_csv(train_path)
    ds = SkinDataset(df=train_df, root=Path('/home/jericho/Project/SkinClassification/data/train_center'))

    print(ds[0])
