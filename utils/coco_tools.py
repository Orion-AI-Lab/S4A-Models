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


def create_coco_dataframe(df, path_coco, ann_path, ann_limit=None,
                          keep_tiles='all', keep_years='all', common_labels=None):
    '''
    Creates and exports a COCO file with the data provided in a given dataframe.

    Parameters
    ----------
    df: pandas DataFrame
        A pandas dataframe containing the paths of the data.
    path_coco: str or Path
        The file path to export COCO file into.
    ann_path: str or Path
        The path containing the annotations of the data. Folder structure in this
        path should be: `ann_path/<year>/<tile>/<patch>/`
    ann_limit: int. Default None.
        Upper annotation limit, patches that exceed it are dismissed
    keep_tiles: list of str, default 'all'
        The tiles to use for train/val/test. Default 'all'.
    keep_years: list of str, default 'all'
        The years to use for train/val/test. Default 'all'.
    common_labels: set of int
        The common labels of the selected tiles.
    '''
    path_coco = Path(path_coco)
    ann_path = Path(ann_path)

    # Initializations
    annotation_id, image_id = 1, 1
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

        if not Path(ann_path / year / tile / patch / 'annotations.csv.gz').is_file():
            print(f'Skipping: "{path}", no annotations!')
            continue

        # Read annotations and insert them into COCO
        df = pd.read_csv(
            ann_path / year / tile / patch / 'annotations.csv.gz',
            converters={'segmentation': ast.literal_eval, 'bbox': ast.literal_eval}
        )

        # If annotations limit is passed as a parameter,
        # ignore dataframes that contain more entries than ann_limit
        if ann_limit is not None:
            if len(df) > ann_limit:
                print(f'Skipping: "{path}", {len(df)} annos!')
                continue

        for row in df.itertuples(index=False):
            coco['annotations'].append({
                'id': annotation_id,
                'iscrowd': 0,
                'image_id': image_id,
                'category_id': linear_enc[row.Crop_type],
                'segmentation': row.segmentation,
                'bbox': row.bbox,
                'area': row.area
            })
            annotation_id += 1

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


def create_coco_netcdf(netcdf_path, ann_path, path_train, path_test, path_val,
                       having_annotations=False, train_r=60, val_r=30, ann_limit=None,
                       keep_tiles='all', keep_years='all', experiment=None,
                       train_tiles=None, test_tiles=None, train_years=None, test_years=None,
                       common_labels=None, num_patches=None):
    print(f'Creating Coco Attribute file..')
    netcdf_path = Path(netcdf_path)
    ann_path = Path(ann_path)

    # Initializations
    annotation_id, image_id = 1, 1
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
            test_annotation_id, test_image_id = 1, 1

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
            } for crop_name, crop_id in CROP_ENCODING.items()
        ]

    print(f'\nReading Netcdfs from: "{netcdf_path}".')
    print(f'Reading Annotations from: "{ann_path}".')
    time_start = time.time()

    for path in netcdf_path.rglob('*.nc'):
        # Grab year, tile and patch from path
        base_name = path.stem.split('_')
        year, tile, patch = base_name[0], base_name[1], '_'.join(base_name[2:])

        # Check against parsed tiles/years
        if (experiment is None) and not keep_tile(tile, year, keep_tiles, keep_years): continue

        if not Path(ann_path / year / tile / patch / 'annotations.csv.gz').is_file():
            print(f'Skipping: "{path}", no annotations!')
            continue

        df = pd.read_csv(
            ann_path / year / tile / patch / 'annotations.csv.gz',
            converters={'segmentation': ast.literal_eval, 'bbox': ast.literal_eval}
        )

        # If annotations limit is passed as a parameter,
        # ignore dataframes that contain more entries than ann_limit
        if ann_limit is not None:
            if len(df) > ann_limit:
                print(f'Skipping: "{path}", {len(df)} annos!')
                continue

        if (experiment not in [2, 3]) or \
            ((experiment in [2, 3]) and ((tile in train_tiles) and (year in train_years))):
                for row in df.itertuples(index=False):
                    coco['annotations'].append({
                        'id': annotation_id,
                        'iscrowd': 0,
                        'image_id': image_id,
                        'category_id': linear_enc[row.Crop_type],
                        'segmentation': row.segmentation,
                        'bbox': row.bbox,
                        'area': row.area
                    })
                    annotation_id += 1

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
            for row in df.itertuples(index=False):
                coco_test['annotations'].append({
                    'id': test_annotation_id,
                    'iscrowd': 0,
                    'image_id': test_image_id,
                    'category_id': linear_enc[row.Crop_type],
                    'segmentation': row.segmentation,
                    'bbox': row.bbox,
                    'area': row.area
                })
                test_annotation_id += 1

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
                                          random_state=RANDOM_SEED,
                                          having_annotations=having_annotations
                                          )

        if num_patches is not None:
            new_val_r = (num_patches * val_r) / len(coco_val['images'])

            coco_val, _ = split_coco(coco_val,
                                     train_size=new_val_r / 100,
                                     random_state=RANDOM_SEED,
                                     having_annotations=having_annotations)

        # Randomly select samples from test COCO
        if num_patches is None:
            new_test_r = 100 - (train_r + val_r)
        else:
            new_test_r = ((100 - train_r - val_r) * num_patches) / len(coco_test['images'])

        coco_test, _ = split_coco(coco_test,
                                  train_size=new_test_r / 100,
                                  random_state=RANDOM_SEED,
                                  having_annotations=having_annotations)
    else:
        # Split big COCO into train/test
        if num_patches is None:
            new_train_r = train_r
        else:
            new_train_r = (train_r * num_patches) / len(coco['images'])


        coco_train, coco_val_test = split_coco(coco,
                                               train_size=new_train_r / 100,
                                               random_state=RANDOM_SEED,
                                               having_annotations=having_annotations
                                               )
        # Split val_test to val/test
        if num_patches is None:
            new_val_r = val_r / (100 - train_r) * 100
        else:
            new_val_r = (num_patches * val_r) / len(coco_val_test['images'])

        coco_val, coco_test = split_coco(coco_val_test,
                                         train_size=new_val_r / 100,
                                         random_state=RANDOM_SEED,
                                         having_annotations=having_annotations
                                         )
        if num_patches is not None:
            new_test_r = ((100 - train_r - val_r) * num_patches) / len(coco_test['images'])

            coco_test, _ = split_coco(coco_test,
                                      train_size=new_test_r / 100,
                                      random_state=RANDOM_SEED,
                                      having_annotations=having_annotations)

    # Dump them to disk
    print('Saving to disk...')

    with open(path_train, 'wt', encoding='UTF-8') as file:
        json.dump(coco_train, file)

    with open(path_val, 'wt', encoding='UTF-8') as file:
        json.dump(coco_val, file)

    with open(path_test, 'wt', encoding='UTF-8') as file:
        json.dump(coco_test, file)


def split_coco(coco, train_size=0.9, having_annotations=False, random_state=0):
    assert isinstance(coco, dict), f'Split Coco: given file is not in dict format'

    info = coco['info']
    licenses = coco['licenses']
    images = sorted(coco['images'], key=lambda img: img['id'])
    annotations = sorted(coco['annotations'], key=lambda ann: ann['image_id'])
    categories = coco['categories']

    if having_annotations:
        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

    images_x, images_y, annotations_x, annotations_y = train_test_split(images, annotations, train_size=train_size, random_state=random_state)

    x = {
        'info': info,
        'licenses': licenses,
        'images': images_x,
        'annotations': annotations_x,
        'categories': categories
    }

    y = {
        'info': info,
        'licenses': licenses,
        'images': images_y,
        'annotations': annotations_y,
        'categories': categories
    }

    return x, y
