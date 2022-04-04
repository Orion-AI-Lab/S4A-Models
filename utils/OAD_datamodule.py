from pathlib import Path
from typing import Tuple, Union, Optional
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl


class OADDataset(Dataset):

    def __init__(
            self,
            file: Union[str, Path],
            is_train: bool = True,
            num_bands: int = 13,
            metrics_per_band: int = 2,
            linear_encoder: dict = None
    ):

        # Keep local copies
        self.file = Path(file)
        self.is_train = is_train
        self.num_bands = num_bands
        self.metrics_per_band = metrics_per_band
        self.linear_encoder = linear_encoder

        # Load data
        self.data = pd.read_csv(self.file)

        def rename_columns(col: str) -> str:
            splited = col.split('_')

            if len(splited) <= 2:
                return col
            else:
                return f'{splited[1]}_{splited[0]}_{splited[2]}'

        # Rename columns so timestep comes first
        self.data = self.data.rename(mapper=rename_columns, axis='columns')

        # Sort based on label index
        self.data = self.data.sort_index(axis=1)

        # apply encoding
        if self.linear_encoder is not None:
            self.data['label'] = self.data['label'].map(self.linear_encoder)
            self.data.dropna(axis=0, how='any', inplace=True)

        # Convert to numpy
        self.labels = self.data.loc[:, 'label'].values

        # Manually select from 4th month to 10th
        self.data = self.data.iloc[:, 4 * self.num_bands * self.metrics_per_band:-(self.num_bands * self.metrics_per_band + 3)].values

        self.total_items = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        data = self.data[idx].reshape(-1, self.num_bands * self.metrics_per_band).astype('float32')
        label = self.labels[idx].astype('int64')
        return data, label


class OADDataModule(pl.LightningDataModule):

    def __init__(
            self,
            file: Union[str, Path],
            batch_size: int = 64,
            num_workers: int = 4,
            linear_encoder: dict = None
    ) -> None:

        super().__init__()

        self.file = file
        self.batch_size = batch_size
        self.linear_encoder = linear_encoder
        self.num_workers = num_workers

        self.dataset_train = None
        self.dataset_eval = None
        self.dataset_test = None

    def setup(self, stage: Optional[str] = None):
        # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#prepare-data
        # stage (Optional[str]) – either 'fit', 'validate', 'test', or 'predict'
        if stage == 'fit':
            self.dataset_train = OADDataset(
                file=self.file.parent / f'{self.file.stem}_test.csv.gz',
                linear_encoder=self.linear_encoder
            )

            self.dataset_eval = OADDataset(
                file=self.file.parent / f'{self.file.stem}_test.csv.gz',
                linear_encoder=self.linear_encoder
            )

        elif stage == 'validate':
            self.dataset_eval = OADDataset(
                file=self.file.parent / f'{self.file.stem}_test.csv.gz',
                linear_encoder=self.linear_encoder
            )

        elif stage == 'test':
            self.dataset_test = OADDataset(
                file=self.file.parent / f'{self.file.stem}_test.csv.gz',
                linear_encoder=self.linear_encoder
            )

        elif stage == 'predict':
            # TODO: Replace with Predict Dataset
            self.dataset_test = OADDataset(
                file=self.file.parent / f'{self.file.stem}_test.csv.gz',
                linear_encoder=self.linear_encoder
            )

        else:
            raise Exception(f'{stage} not supported!')

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True
                          )

    def val_dataloader(self):
        return DataLoader(self.dataset_eval,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True
                          )

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True
                          )
