import numpy as np
import time
import torch
from typing import Any, Union
from pathlib import Path
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .settings.config import RANDOM_SEED, IMG_SIZE
from .PAD_dataset import PADDataset

# Set seed for everything
pl.seed_everything(RANDOM_SEED)


class PADDataModule(pl.LightningDataModule):
    # Documentation: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    '''
    PyTorch Lightning DataModule Wrapper for PADDataset
    '''

    def __init__(
            self,
            root_path_coco: Union[str, Path] = Path(),
            path_train: Union[str, Path] = Path(),
            path_val: Union[str, Path] = Path(),
            path_test: Union[str, Path] = Path(),
            bands: list = None,
            transforms=None,
            compression: str = 'gzip',
            group_freq: str = '1MS',
            saved_medians: bool = False,
            linear_encoder: dict = None,
            prefix: str = None,
            window_len: int = 12,
            fixed_window: bool = False,
            requires_norm: bool = True,
            return_masks: bool = False,
            clouds: bool = True,
            cirrus: bool = True,
            shadow: bool = True,
            snow: bool = True,
            output_size: tuple = None,
            batch_size: int = 64,
            num_workers: int = 4,
            binary_labels: bool = False,
            return_parcels: bool = False
    ) -> None:
        '''
        Parameters
        ----------
        root_path_coco: Path or str
            The path containing the COCO files.
        path_train: Path or str, default Path('coco_train.json')
            The file path containing the training data.
        path_val: Path or str, default Path('coco_val.json')
            The file path containing the validation data.
        path_test: Path or str, default Path('coco_test.json')
            The file path containing the testing data.
        bands: list of str, default None
            A list of the bands to use. If None, then all available bands are
            taken into consideration. Note that the bands are given in a two-digit
            format, e.g. '01', '02', '8A', etc.
        transforms: list of pytorch Transforms, default None
            A list of pytorch Transforms to use. To be implemented.
        compression: str, default 'gzip'
            The type of compression to use for the produced index file.
        group_freq: str, default '1MS'
            The frequency to use for binning. All Pandas offset aliases are supported.
            Check: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        saved_medians: boolean, default False
            Whether to precompute and save all medians. This saves on computation
            time during batching.
        linear_encoder: dict, default None
            Maps arbitrary crop_ids to range 0-len(unique(crop_id)).
        prefix: str, default None
            A prefix to use for all exported files. If None, then the current
            timestamp is used.
        window_len: integer, default 12
            If a value is passed, then a rolling window of this length is applied
            over the data. E.g. if `window_len` = 6 and `group_freq` = '1M', then
            a 6-month rolling window will be applied and each batch will contain
            6 months of training data and the corresponding label.
        fixed_window: boolean, default False
            If True, then a fixed window including months 4 (April) to 9 (September) is used
            instead of a rolling one.
        requires_norm: boolean, default True
            If True, then it normalizes the dataset to [0, 1] range.
        return_masks: boolean, default False
            based: https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/hollstein
            If True, returns Logical OR of all hollstein masks, see below.
        clouds: boolean, default True
            If True and return_masks=True, returns mask for clouds
        cirrus: boolean, default True
            If True and return_masks=True, returns mask for cirrus
        shadow: boolean, default True
            If True and return_masks=True, returns mask for shadow
        snow: boolean, default True
            If True and return_masks=True, returns mask for snow
        output_size: tuple of int, default None
            If a tuple (H, W) is given, then the output images will be divided
            into non-overlapping subpatches of size (H, W). Otherwise, the images
            will retain their original size.
        batch_size: int, default 64
            The batch size to use.
        num_workers: int, default 4
            The number of workers to use.
        binary_labels: bool, default False
            Map categories to 0 background, 1 parcel.
        return_parcels: bool, default False
            If True, then a boolean mask for the parcels is also returned.
        '''

        super().__init__()

        self.root_path_coco = root_path_coco

        self.path_train = Path(path_train)
        self.path_val = Path(path_val)
        self.path_test = Path(path_test)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.binary_labels = binary_labels

        # Initialize parameters required for Patches Dataset
        self.prefix = prefix
        self.bands = bands
        self.linear_encoder = linear_encoder
        self.saved_medians = saved_medians
        self.window_len = window_len
        self.fixed_window = fixed_window
        self.requires_norm = requires_norm
        self.return_masks = return_masks
        self.clouds = clouds
        self.cirrus = cirrus
        self.shadow = shadow
        self.snow = snow
        self.output_size = output_size
        self.group_freq = group_freq
        self.compression = compression
        self.return_parcels = return_parcels

        self.num_bands = len(bands)
        self.img_size = IMG_SIZE

        if output_size is None:
            self.dims = (self.batch_size, self.window_len, self.num_bands, self.img_size, self.img_size)
        else:
            self.dims = (self.batch_size, self.window_len, self.num_bands, self.output_size[0], self.output_size[1])

        self.dataset_train = None
        self.dataset_eval = None
        self.dataset_test = None


    def setup(self, stage=None):
        # called on every GPU
        # Create train/val/test loaders
        assert stage in ['fit', 'test'], f'Stage : "{stage}" must be fit or test!'

        # Check everything is ok
        if stage == 'fit':
            assert self.path_train is not None, \
                f'Train path cannot be None when training'

            assert self.path_val is not None, \
                f'Validation path cannot be None when training'

            assert self.path_train.is_file(), f'"{self.path_train}" is not a valid file.'
            assert self.path_val.is_file(), f'"{self.path_val}" is not a valid file.'

        else:
            assert self.path_test is not None, \
                f'Test path cannot be None when testing.'

            assert self.path_test.is_file(), f'"{self.path_test}" is not a valid file.'

        # Load to COCO Objects using COCO API
        # https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools

        if stage == 'fit':
            # Setup datasets for training
            coco_train = COCO(self.path_train)
            coco_val = COCO(self.path_val)

            self.dataset_train = PADDataset(root_path_coco=self.root_path_coco,
                                                coco=coco_train,
                                                # transforms=transforms,
                                                group_freq=self.group_freq,
                                                compression=self.compression,
                                                prefix=self.prefix,
                                                bands=self.bands,
                                                linear_encoder=self.linear_encoder,
                                                saved_medians=self.saved_medians,
                                                window_len=self.window_len,
                                                fixed_window=self.fixed_window,
                                                requires_norm=self.requires_norm,
                                                return_masks=self.return_masks,
                                                clouds=self.clouds,
                                                cirrus=self.cirrus,
                                                shadow=self.shadow,
                                                snow=self.snow,
                                                output_size=self.output_size,
                                                binary_labels=self.binary_labels,
                                                mode='train',
                                                return_parcels=self.return_parcels
                                                )

            self.dataset_eval = PADDataset(root_path_coco=self.root_path_coco,
                                               coco=coco_val,
                                               group_freq=self.group_freq,
                                               compression=self.compression,
                                               prefix=self.prefix,
                                               bands=self.bands,
                                               linear_encoder=self.linear_encoder,
                                               saved_medians=self.saved_medians,
                                               window_len=self.window_len,
                                               fixed_window=self.fixed_window,
                                               requires_norm=self.requires_norm,
                                               return_masks=self.return_masks,
                                               clouds=self.clouds,
                                               cirrus=self.cirrus,
                                               shadow=self.shadow,
                                               snow=self.snow,
                                               output_size=self.output_size,
                                               binary_labels=self.binary_labels,
                                               mode='val',
                                               return_parcels=self.return_parcels
                                               )

        else:
            # Setup datasets for testing
            coco_test = COCO(self.path_test)

            self.dataset_test = PADDataset(root_path_coco=self.root_path_coco,
                                               coco=coco_test,
                                               group_freq=self.group_freq,
                                               compression=self.compression,
                                               prefix=self.prefix,
                                               bands=self.bands,
                                               linear_encoder=self.linear_encoder,
                                               saved_medians=self.saved_medians,
                                               window_len=self.window_len,
                                               fixed_window=self.fixed_window,
                                               requires_norm=self.requires_norm,
                                               return_masks=self.return_masks,
                                               clouds=self.clouds,
                                               cirrus=self.cirrus,
                                               shadow=self.shadow,
                                               snow=self.snow,
                                               output_size=self.output_size,
                                               binary_labels=self.binary_labels,
                                               mode='test',
                                               return_parcels=self.return_parcels
                                               )

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
