import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import copy

from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection.iterative_stratification import IterativeStratification

from utils.settings.config import RANDOM_SEED, NORMALIZATION_DIV

from utils.settings.mappings.mappings_cat import CLASSES_MAPPING as CAT_CLASSES
from utils.settings.mappings.mappings_fr import CLASSES_MAPPING as FR_CLASSES
from utils.settings.mappings.encodings_en import CROP_ENCODING
from utils.settings.mappings.mappings_cat import SAMPLE_TILES as CAT_TILES
from utils.settings.mappings.mappings_fr import SAMPLE_TILES as FR_TILES

np.random.seed(RANDOM_SEED)


class font_colors:
    '''
    Colors for printing messages to stdout.
    '''
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def split_data(prefix, loaders_path, data_path, train_years, train_ratio=0.8):
    '''
    Splits the data into train/val sets in a stratified way and exports
    the corresponding .csv files into the specified folder.

    The splitting is done in the following manner:
    - Train set: a subset of patches for the given train years
    - Val set: the rest of the patches for the given train years

    NOTES:
        - We try to ensure that label distribution is the same in both sets.
        - We assume that all tiles have observations for all years.

    Parameters
    ----------
    prefix: str
        The prefix to use for the exported files.
    loaders_path: Path
        The path to export files into.
    data_path: Path
        The path containing the data.
    train_years: list of str
        A list of the years to use for the train set.
    train_ratio: float, default 0.8
        The ratio to use for the train data patches.

    Returns
    -------
    train_set, val_set: Two lists containing image paths for the
        train/val sets respectively.
    '''
    # Create a dataframe containing patch paths and the labels of each patch
    # Only the chosen training years are taken into account
    data = pd.DataFrame(columns=['patch_path', 'labels'])
    for i, patch_dir in enumerate(sorted(list(data_path.glob('*/*/patch_*')))):
        if patch_dir.parts[-3] not in train_years: continue

        with rasterio.open(os.path.join(patch_dir, 'labels.tiff')) as f:
            data.loc[i] = [patch_dir, list(np.unique(f.read(1)))]

    # Convert labels to one-hot encoding
    mlb = MultiLabelBinarizer()
    labels_onehot = pd.DataFrame(mlb.fit_transform(data.labels), columns=mlb.classes_)

    data = data.drop(columns=['labels'])

    # Select a subset of tiles based on train_ratio in a stratified way
    stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - train_ratio, train_ratio])
    train_indexes, val_indexes = next(stratifier.split(data.values[:, np.newaxis], labels_onehot.values))

    # Construct separate sets
    # X_train, y_train = data.iloc[train_indexes, :], labels_onehot.iloc[train_indexes, :]
    # X_val, y_val = data.iloc[val_indexes, :], labels_onehot.iloc[val_indexes, :]

    X_train = data.iloc[train_indexes, :]
    X_val = data.iloc[val_indexes, :]

    # Export files
    # pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(loaders_path, f'{prefix}_train.csv'), index=False)
    # pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(loaders_path, f'{prefix}_val.csv'), index=False)

    X_train.to_csv(os.path.join(loaders_path, f'{prefix}_train.csv'), index=False)
    X_val.to_csv(os.path.join(loaders_path, f'{prefix}_val.csv'), index=False)

    # Convert Path to string
    X_train = [str(path) for path in X_train.patch_path.to_list()]
    X_val = [str(path) for path in X_val.patch_path.to_list()]

    return X_train, X_val


def common_labels(tiles):
    '''
    Returns the common labels among the given tiles.

    Parameters
    ----------
    tiles: set of str
        The tiles to find common labels for.

    Returns
    -------
    set of int: the common labels
    '''
    all_classes = set()
    if not tiles.isdisjoint(CAT_TILES):
        all_classes = all_classes | set(CAT_CLASSES.values())
    if not tiles.isdisjoint(FR_TILES):
        all_classes = all_classes | set(FR_CLASSES.values())

    return set([CROP_ENCODING[k] for k in all_classes])


def keep_tile(parsed_tile, parsed_year, tiles, years):
    '''
    Checks whether the given tile should be handled or not.
    '''
    if (years != 'all') and (parsed_year not in years):
        return False

    if (tiles != 'all') and (parsed_tile not in tiles):
        return False

    return True


def hollstein_mask(bands, clouds=True, cirrus=False, shadows=False, snow=False, requires_norm=False,
                   reference_bands=None):

    # BASED: https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/hollstein
    bands = copy.deepcopy(bands)

    # If ndarray is given, assume NxBxWxH (Time bins, Bands, Width, Height) and convert it to dictionary
    # else assume it is a dictionary
    if not isinstance(bands, dict):
        assert reference_bands is not None, 'Hollstein: ndarray was given, but reference bands were empty'
        bands = {band: bands[:, i, :, :] for i, band in enumerate(reference_bands)}

    # This should be the reverse of requires_norm in dataloader
    if not requires_norm:
        bands = {key:  band / NORMALIZATION_DIV for key, band in bands.items()}

    out = {}

    if shadows:
        shadows_cond = ((bands['B03'] < 0.319) & (bands['B8A'] < 0.166) & (
                ((bands['B03'] - bands['B07'] < 0.027) & (bands['B09'] - bands['B11'] >= -0.097)) |
                ((bands['B03'] - bands['B07'] >= 0.027) & (bands['B09'] - bands['B11'] >= 0.021))
        )) | \
                       ((bands['B03'] >= 0.319) & (np.divide(bands['B05'], bands['B11']) >= 4.33) &
                        (bands['B03'] < 0.525) & (np.divide(bands['B01'], bands['B05']) >= 1.184))

        out['shadow_mask'] = shadows_cond

    if clouds:
        clouds_cond = (
                (bands['B03'] >= 0.319) & (bands['B05'] / bands['B11'] < 4.33) &
                (
                        ((bands['B11'] - bands['B10'] < 0.255) & (bands['B06'] - bands['B07'] < -0.016)) |
                        ((bands['B11'] - bands['B10'] >= 0.255) & (bands['B01'] >= 0.3))
                )
        )

        out['cloud_mask'] = clouds_cond

    if cirrus:
        cirrus_cond = (
            (
                (bands['B03'] < 0.319) & (bands['B8A'] >= 0.166) & (np.divide(bands['B02'], bands['B10']) < 14.689) &
                (np.divide(bands['B02'], bands['B09']) >= 0.788)
            ) |
            (
                (bands['B03'] >= 0.319) & (np.divide(bands['B05'], bands['B11']) < 4.33) &
                (bands['B11'] - bands['B10'] < 0.255) & (bands['B06'] - bands['B07'] >= -0.016)
            )
        )

        out['cirrus_mask'] = cirrus_cond

    if snow:
        snow_cond = (
                (bands['B03'] >= 0.319) & (np.divide(bands['B05'], bands['B11']) >= 4.33) & (bands['B03'] >= 0.525)
        )

        out['snow_mask'] = snow_cond

    # Logical OR between all masks is the final mask
    out = np.any(np.array(list(out.values())), axis=0)

    return out


def NVDI(B04, B08):
    # https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index
    # NVDI = (NIR - R) / (NIR + R)
    return (B08 - B04) / (B08 + B04)
