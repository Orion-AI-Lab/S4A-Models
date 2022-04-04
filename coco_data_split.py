'''
Splits the data into train/val/test sets in a stratified or random way and exports
the corresponding COCO json files into the specified folder.

Stratification is applied on the unique labels contained in each image, not
on a pixel basis.

The user can choose among three experiment settings:
 - Experiment 1: All tiles over all years are sampled randomly
 - Experiment 2: Train/Val sets contain only Catalonia tiles for both years,
                 and test set contains only France tiles for a single year
 - Experiment 3: Train/Val sets contain only France tiles for a single year,
                 and test set contains only Catalonia tiles for a different year.
'''

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from pycocotools.coco import COCO
import netCDF4
import xarray as xr
import numpy as np
import random

from utils.tools import keep_tile, common_labels
from utils.coco_tools import create_coco_dataframe, create_coco_netcdf
from utils.settings.mappings.mappings_cat import SAMPLE_TILES as CAT_TILES
from utils.settings.mappings.mappings_fr import SAMPLE_TILES as FR_TILES
from utils.settings.mappings.encodings_en import CROP_ENCODING

from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection.iterative_stratification import IterativeStratification

netCDF4.default_encoding = 'utf-8'


def plot_label_frequencies(coco, data_path, title, ax, labels_common=None):
    if labels_common is not None:
        label_freqs = {l: 0 for l in labels_common}
    else:
        label_freqs = {l: 0 for l in CROP_ENCODING.values()}

    label_freqs[0] = 0
    for img in coco.imgs.values():
        fname = Path(img['file_name']).name
        patch_netcdf = netCDF4.Dataset(data_path / fname, 'r')
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(patch_netcdf['labels']))
        for label in list(np.unique(labels.labels.data)):
            label_freqs[label] += 1

    ax.bar(list(range(1, len(label_freqs) + 1)), label_freqs.values())
    ax.set_xticks(list(range(1, len(label_freqs) + 1)))
    ax.set_xticklabels(list(label_freqs.keys()), rotation=90, fontsize=7)
    ax.set_title(title)


def create_dataframe(data_path, tiles, years, common_labels=None):
    '''
    Reads the labels from the netCDF files and inserts them into a dataframe.
    '''
    data = pd.DataFrame(columns=['patch_path', 'labels'])
    patch_paths = list(data_path.glob('*.nc'))
    random.shuffle(patch_paths)

    for i, patch_path in enumerate(patch_paths):
        year, tile = patch_path.stem.split('_')[:2]

        if not keep_tile(tile, year, tiles, years): continue

        patch_netcdf = netCDF4.Dataset(patch_path, 'r')
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(patch_netcdf['labels']))

        unique_labels = set(np.unique(labels.labels.data))

        if common_labels is not None:
            # We want the train/val and test tiles to have common labels
            if unique_labels.isdisjoint(common_labels): continue

            # Note only the common labels, we don't want to stratify on the non-common
            data.loc[i] = [patch_path, list(unique_labels & common_labels)]
        else:
            # Data comes from all tiles so all labels are common
            data.loc[i] = [patch_path, list(unique_labels)]
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--how', type=str, required=True, help='Perform a stratified split.',
                        choices=['stratified', 'random'])
    parser.add_argument('--ann_limit', type=int, required=False, default=None,
                        help='Maximum number of annotations per patch. Patches exceeding this limit, are dismissed')
    parser.add_argument('--data_path', type=str, default='dataset/netcdf/', required=False,
                        help='The path containing the data in netCDF format. Default "dataset/netcdf/".')
    parser.add_argument('--data_ann', type=str, default='dataset/patches/', required=False,
                        help='The path containing subfolders with the annotations files. Default "dataset/patches/".')
    parser.add_argument('--coco_path', type=str, default='dataset/', required=False,
                        help='The path to export the COCO files into. Default "dataset/"')
    parser.add_argument('--ratio', nargs='+', default=['60', '20', '20'], required=False,
                        help='The train/val/test ratio. Default is 60/20/20.')
    parser.add_argument('--prefix', type=str, default=None, required=False,
                        help='The prefix to use for the exported files. If none is given, \
                        then the current timestamp is used.')
    parser.add_argument('--plot_distros', action='store_true', default=False, required=False,
                        help='Plot label distributions.')
    parser.add_argument('--tiles', nargs='+', default='all', required=False,
                        help='space-separated list of tiles to use, e.g. "31TCG 31TDG". If none given, \
                        all tiles found will be used.')
    parser.add_argument('--years', nargs='+', default='all', required=False,
                        help='space-separated list of years to use, e.g. "2019 2020". If none given, \
                        all years found will be used.')

    parser.add_argument('--num_patches', type=int, default=None, required=False,
                        help='The number of patches to use overall. Default all.')

    parser.add_argument('--experiment', type=int, choices=[1, 2, 3], default=None, required=False,
                        help='The type of experiment to create COCO files for. \
                            If it is given, any other specified tiles/years are ignored. \
                            Type 1: Train/val/test with all tiles and all years. \
                            Type 2: Train/val with Catalonia for all years, test with France. \
                            Type 3: Train/val with France for 2019, test with Catalonia for 2020.')

    parser.add_argument('--seed', type=int, default=None, required=False,
                        help='The seed to use for random patch selection. Defauly None (random).')

    args = parser.parse_args()

    # Define paths
    data_path = Path(args.data_path)
    coco_path = Path(args.coco_path)
    ann_path = Path(args.data_ann)

    # Ignore tile/year filtering in case an explicit experiment scheme is selected
    if args.experiment is not None:
        args.tiles, args.years = 'all', 'all'

    train_tiles, train_years, test_tiles, test_years, common_lbls = None, None, None, None, None

    # Define years and tiles based on the selected experiment
    if args.experiment == 1:
        # Train/val/test with all tiles/years  randomly
        train_tiles, train_years = set(FR_TILES + CAT_TILES), set(['2019', '2020'])
        test_tiles, test_years = set(FR_TILES + CAT_TILES), set(['2019', '2020'])
    elif args.experiment == 2:
        # Train/val with Catalonia for all years, test with France for 2019
        train_tiles, train_years = set(CAT_TILES), set(['2019', '2020'])
        test_tiles, test_years = set(FR_TILES), set(['2019'])

        common_lbls = common_labels(train_tiles | test_tiles)
    elif args.experiment == 3:
        # Train/val with France for 2019, test with Catalonia for 2020
        train_tiles, train_years = set(FR_TILES), set(['2019'])
        test_tiles, test_years = set(CAT_TILES), set(['2020'])

        common_lbls = common_labels(train_tiles | test_tiles)
    else:
        train_tiles, train_years = set(args.tiles), set(args.years)
        test_tiles, test_years = set(args.tiles), set(args.years)

        common_lbls = common_labels(train_tiles | test_tiles)

    # Define prefix
    if args.prefix is None:
        # No prefix given, use current timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        prefix = timestamp
    else:
        prefix = args.prefix

    # Define ratios
    train_r, val_r, test_r = [int(i) for i in args.ratio]

    if args.how == 'stratified':
        if args.seed is not None:
            # Set the seed
            random.seed(args.seed)

        # Create a dataframe containing patch paths and the labels of each patch
        if args.experiment == 1:
            data = create_dataframe(data_path, train_tiles | test_tiles, train_years | test_years, common_labels=common_lbls)
            all_patches = data.shape[0]
        else:
            data = create_dataframe(data_path, train_tiles, train_years, common_labels=common_lbls)
            test_data = create_dataframe(data_path, test_tiles, test_years, common_labels=common_lbls)
            all_patches = data.shape[0]
            test_patches = test_data.shape[0]

        # Convert labels to one-hot encoding
        mlb = MultiLabelBinarizer()
        labels_onehot = pd.DataFrame(mlb.fit_transform(data.labels), columns=mlb.classes_)

        data = data.drop(columns=['labels'])

        if args.num_patches is None:
            new_train_r = train_r
        else:
            new_train_r = (args.num_patches * train_r) / all_patches

        # Select a subset of tiles in a stratified way based on the given ratio
        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - (new_train_r / 100), new_train_r / 100])
        train_idx, val_test_idx = next(stratifier.split(data.values[:, np.newaxis], labels_onehot.values))

        # `IterativeStratification` returns a train-test split, so we must further
        # split the test set into test and validation
        X_val_test = data.iloc[val_test_idx, :]
        y_val_test = labels_onehot.iloc[val_test_idx, :]

        if args.num_patches is None:
            new_val_r = (X_val_test.shape[0] * val_r) / 100
        else:
            new_val_r = (args.num_patches * val_r) / X_val_test.shape[0]

        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - (new_val_r / 100), new_val_r / 100])
        val_idx, test_idx = next(stratifier.split(X_val_test.values[:, np.newaxis], y_val_test))

        if args.experiment in [2, 3]:
            # We will split the test data so that one of the splits will have the
            # appropriate test size
            if args.num_patches is None:
                test_size = (test_r / 100) * all_patches
                new_test_r = (test_size / test_patches) * 100
            else:
                new_test_r = (args.num_patches * test_r) / test_patches

            mlb = MultiLabelBinarizer()
            labels_onehot = pd.DataFrame(mlb.fit_transform(test_data.labels), columns=mlb.classes_)

            stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - (new_test_r / 100), new_test_r / 100])
            test_idx, _ = next(stratifier.split(test_data.values[:, np.newaxis], labels_onehot.values))

            X_train = data.iloc[train_idx, :]
            X_val = data.iloc[val_idx, :]
            X_test = test_data.iloc[test_idx, :]
        else:
            X_test = X_val_test.iloc[test_idx, :]
            y_test = y_val_test.iloc[test_idx, :]

            if args.num_patches is None:
                new_test_r = (X_test.shape[0] * test_r) / 100
            else:
                # We must further split the test set in order to obtain the required test size
                new_test_r = (args.num_patches * test_r) / X_test.shape[0]

            stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - (new_test_r / 100), new_test_r / 100])
            test_idx, _ = next(stratifier.split(X_test.values[:, np.newaxis], y_test))

            X_train = data.iloc[train_idx, :]
            X_val = X_val_test.iloc[val_idx, :]
            X_test = X_val_test.iloc[test_idx, :]

        # Export COCO files
        create_coco_dataframe(df=X_train,
                              path_coco=coco_path / f'{prefix}_coco_train.json',
                              ann_path=ann_path,
                              ann_limit=args.ann_limit,
                              keep_tiles=train_tiles,
                              keep_years=train_years,
                              common_labels=common_lbls
                              )

        create_coco_dataframe(df=X_val,
                              path_coco=coco_path / f'{prefix}_coco_val.json',
                              ann_path=ann_path,
                              ann_limit=args.ann_limit,
                              keep_tiles=train_tiles,
                              keep_years=train_years,
                              common_labels=common_lbls
                              )

        create_coco_dataframe(df=X_test,
                              path_coco=coco_path / f'{prefix}_coco_test.json',
                              ann_path=ann_path,
                              ann_limit=args.ann_limit,
                              keep_tiles=test_tiles,
                              keep_years=test_years,
                              common_labels=common_lbls
                              )

    elif args.how == 'random':
        create_coco_netcdf(netcdf_path=data_path,
                           ann_path=ann_path,
                           path_train=coco_path / f'{prefix}_coco_train.json',
                           path_test=coco_path / f'{prefix}_coco_test.json',
                           path_val=coco_path / f'{prefix}_coco_val.json',
                           having_annotations=False,
                           train_r=train_r,
                           val_r=val_r,
                           ann_limit=args.ann_limit,
                           keep_tiles=args.tiles,
                           keep_years=args.years,
                           experiment=args.experiment,
                           train_tiles=train_tiles,
                           test_tiles=test_tiles,
                           train_years=train_years,
                           test_years=test_years,
                           common_labels=common_lbls,
                           num_patches=args.num_patches
                           )

    # Plot label distributions of the produced files.
    if args.plot_distros:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2)

        if args.experiment == 1:
            plot_label_frequencies(COCO(coco_path / f'{prefix}_coco_train.json'),
                                   data_path,
                                   "Train set",
                                   axes[0, 0])
            plot_label_frequencies(COCO(coco_path / f'{prefix}_coco_val.json'),
                                   data_path,
                                   "Validation set",
                                   axes[1, 0])
            plot_label_frequencies(COCO(coco_path / f'{prefix}_coco_test.json'),
                                   data_path,
                                   "Test set",
                                   axes[1, 1])
        else:
            plot_label_frequencies(COCO(coco_path / f'{prefix}_coco_train.json'),
                                   data_path,
                                   "Train set",
                                   axes[0, 0],
                                   labels_common=common_lbls)
            plot_label_frequencies(COCO(coco_path / f'{prefix}_coco_val.json'),
                                   data_path,
                                   "Validation set",
                                   axes[1, 0],
                                   labels_common=common_lbls)
            plot_label_frequencies(COCO(coco_path / f'{prefix}_coco_test.json'),
                                   data_path,
                                   "Test set",
                                   axes[1, 1],
                                   labels_common=common_lbls)

        plt.show()
