# BASED
# https://www.immersivelimit.com/create-coco-annotations-from-scratch
# and https://github.com/akarazniewicz/cocosplit

import funcy
from sklearn.model_selection import train_test_split
import json
from datetime import date
import pandas as pd
import ast

# Custom Tools
from utils.settings.config import RANDOM_SEED, IMG_SIZE, DATASET_VERSION, CROP_ENCODING, LICENSES, AUTHORS, LINEAR_ENCODER
from utils.tools import keep_tile

import time
from pathlib import Path


def init_coco():
    coco_out = {

        'info': {
            'description': 'SAgNet 2021 Dataset',
            'url': '',
            'version': f'{DATASET_VERSION}',
            'year': date.today().year,
            'contributor': f'{AUTHORS}',
            'date_created': date.today().strftime('%Y/%m/%d')
        },

        # Add licenses in config file
        'licenses': LICENSES,
        'images': [],
        'annotations': [],
        'categories': []
    }

    return coco_out


def create_coco_dataframe(df, path_coco, keep_tiles='all', keep_years='all', common_labels=None):
    '''
    Creates and exports a COCO file with the data provided in a given dataframe.

    Parameters
    ----------
    df: pandas DataFrame
        A pandas dataframe containing the paths of the data.
    path_coco: str or Path
        The file path to export COCO file into.
    keep_tiles: list of str, default 'all'
        The tiles to use for train/val/test. Default 'all'.
    keep_years: list of str, default 'all'
        The years to use for train/val/test. Default 'all'.
    common_labels: set of int
        The common labels of the selected tiles.
    '''
    path_coco = Path(path_coco)

    # Initializations
    image_id = 1
    coco = init_coco()

    if common_labels is not None:
        linear_enc = {val: i + 1 for i, val in enumerate(sorted(list(common_labels)))}
        linear_enc[0] = 0

        # Add only the common labels
        coco['categories'] = [
            {
                'supercategory': 'Crop',
                'name': crop_name,
                'id': linear_enc[crop_id],
            } for crop_name, crop_id in CROP_ENCODING.items() if crop_id in common_labels
        ]
    else:
        linear_enc = LINEAR_ENCODER

        # Add all categories from config file
        coco['categories'] = [
            {
                'supercategory': 'Crop',
                'name': crop_name,
                'id': linear_enc[crop_id],
            } for crop_name, crop_id in CROP_ENCODING.items()
        ]

    time_start = time.time()

    for path in df.itertuples(index=False):
        # Grab year, tile and patch from path
        path = path.patch_path
        base_name = path.stem.split('_')
        year, tile, patch = base_name[0], base_name[1], '_'.join(base_name[2:])

        # Check against parsed tiles/years
        if not keep_tile(tile, year, keep_tiles, keep_years): continue

        # file_name: should be current netcdf name and parent folder
        file_name = Path(path.parts[-2]) / path.parts[-1]
        coco['images'].append({
            'license': 1,
            'file_name': str(file_name),
            'height': IMG_SIZE,
            'width': IMG_SIZE,
            'date_captured': year,
            'id': image_id
        })

        image_id += 1

    print(f'Done. Time Elapsed: {(time.time() - time_start) / 60:0.2f} min(s).')

    # Dump COCO file to disk
    with open(path_coco, 'wt', encoding='UTF-8') as file:
        json.dump(coco, file)


def create_coco_netcdf(netcdf_path, path_train, path_test, path_val, train_r=60, val_r=30,
                       keep_tiles='all', keep_years='all', experiment=None,
                       train_tiles=None, test_tiles=None, train_years=None, test_years=None,
                       common_labels=None, num_patches=None):
    netcdf_path = Path(netcdf_path)

    # Initializations
    image_id = 1
    coco = init_coco()

    if common_labels is not None:
        linear_enc = {val: i + 1 for i, val in enumerate(sorted(list(common_labels)))}
        linear_enc[0] = 0

        # Add only the common labels
        coco['categories'] = [
            {
                'supercategory': 'Crop',
                'name': crop_name,
                'id': linear_enc[crop_id],
            } for crop_name, crop_id in CROP_ENCODING.items() if crop_id in common_labels
        ]

        if experiment in [2, 3]:
            coco_test = init_coco()
            test_image_id = 1

            coco_test['categories'] = [
                {
                    'supercategory': 'Crop',
                    'name': crop_name,
                    'id': linear_enc[crop_id],
                } for crop_name, crop_id in CROP_ENCODING.items() if crop_id in common_labels
            ]
    else:
        linear_enc = LINEAR_ENCODER

        # Add all categories from config file
        coco['categories'] = [
            {
                'supercategory': 'Crop',
                'name': crop_name,
                'id': linear_enc[crop_id],
            } for crop_name, crop_id in CROP_ENCODING.items() if crop_id in linear_enc.keys()
        ]

    print(f'\nReading Netcdfs from: "{netcdf_path}".')
    time_start = time.time()

    for path in netcdf_path.rglob('*.nc'):
        # Grab year, tile and patch from path
        base_name = path.stem.split('_')
        year, tile, patch = base_name[0], base_name[1], '_'.join(base_name[2:])

        # Check against parsed tiles/years
        if (experiment is None) and not keep_tile(tile, year, keep_tiles, keep_years): continue

        if (experiment not in [2, 3]) or \
            ((experiment in [2, 3]) and ((tile in train_tiles) and (year in train_years))):
                # file_name: should be current netcdf name and parent folder
                file_name = Path(path.parts[-2]) / path.parts[-1]
                coco['images'].append({
                    'license': 1,
                    'file_name': str(file_name),
                    'height': IMG_SIZE,
                    'width': IMG_SIZE,
                    'date_captured': year,
                    'id': image_id
                })

                image_id += 1
        elif (tile in test_tiles) and (year in test_years):
            # file_name: should be current netcdf name and parent folder
            file_name = Path(path.parts[-2]) / path.parts[-1]
            coco_test['images'].append({
                'license': 1,
                'file_name': str(file_name),
                'height': IMG_SIZE,
                'width': IMG_SIZE,
                'date_captured': year,
                'id': test_image_id
            })

            test_image_id += 1

    print(f'Done. Time Elapsed: {(time.time() - time_start) / 60:0.2f} min(s).')

    print('Splitting COCO into train/val/test sets')

    if experiment in [2, 3]:
        # Split COCO into train/test
        if num_patches is None:
            new_train_r = train_r
        else:
            new_train_r = (train_r * num_patches) / len(coco['images'])

        coco_train, coco_val = split_coco(coco,
                                          train_size=new_train_r / 100,
                                          random_state=RANDOM_SEED)

        if num_patches is not None:
            new_val_r = (num_patches * val_r) / len(coco_val['images'])

            coco_val, _ = split_coco(coco_val,
                                     train_size=new_val_r / 100,
                                     random_state=RANDOM_SEED)

        # Randomly select samples from test COCO
        if num_patches is None:
            new_test_r = 100 - (train_r + val_r)
        else:
            new_test_r = ((100 - train_r - val_r) * num_patches) / len(coco_test['images'])

        coco_test, _ = split_coco(coco_test,
                                  train_size=new_test_r / 100,
                                  random_state=RANDOM_SEED)
    else:
        # Split big COCO into train/test
        if num_patches is None:
            new_train_r = train_r
        else:
            new_train_r = (train_r * num_patches) / len(coco['images'])


        coco_train, coco_val_test = split_coco(coco,
                                               train_size=new_train_r / 100,
                                               random_state=RANDOM_SEED)
        # Split val_test to val/test
        if num_patches is None:
            new_val_r = val_r / (100 - train_r) * 100
        else:
            new_val_r = (num_patches * val_r) / len(coco_val_test['images'])

        coco_val, coco_test = split_coco(coco_val_test,
                                         train_size=new_val_r / 100,
                                         random_state=RANDOM_SEED)
        if num_patches is not None:
            new_test_r = ((100 - train_r - val_r) * num_patches) / len(coco_test['images'])

            coco_test, _ = split_coco(coco_test,
                                      train_size=new_test_r / 100,
                                      random_state=RANDOM_SEED)

    # Dump them to disk
    print('Saving to disk...')

    with open(path_train, 'wt', encoding='UTF-8') as file:
        json.dump(coco_train, file)

    with open(path_val, 'wt', encoding='UTF-8') as file:
        json.dump(coco_val, file)

    with open(path_test, 'wt', encoding='UTF-8') as file:
        json.dump(coco_test, file)


def split_coco(coco, train_size=0.9, random_state=0):
    assert isinstance(coco, dict), f'Split Coco: given file is not in dict format'

    info = coco['info']
    licenses = coco['licenses']
    images = sorted(coco['images'], key=lambda img: img['id'])
    categories = coco['categories']

    images_x, images_y, _, _ = train_test_split(images, range(0, len(images)), train_size=train_size, random_state=random_state)

    x = {
        'info': info,
        'licenses': licenses,
        'images': images_x,
        'categories': categories
    }

    y = {
        'info': info,
        'licenses': licenses,
        'images': images_y,
        'categories': categories
    }

    return x, y
