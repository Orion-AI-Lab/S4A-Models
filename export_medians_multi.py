import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from tqdm.contrib.concurrent import process_map
from functools import partial

import xarray as xr
from pycocotools.coco import COCO
import netCDF4


IMG_SIZE = 366
BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}

# Extract patches based on this band
REFERENCE_BAND = 'B02'

def process_patch(out_path, mode, num_buckets, root_coco_path, bands, padded_patch_height,
                  padded_patch_width, medians_dtype, label_dtype, group_freq, output_size,
                  pad_top, pad_bot, pad_left, pad_right, patch):
    patch_id, patch_info = patch
    patch_dir = out_path / mode / f'{patch_id}'
    patch_dir.mkdir(exist_ok=True, parents=True)

    # if len(list(patch_dir.iterdir())) == num_buckets + 1:
    #     return

    # Calculate medians
    netcdf = netCDF4.Dataset(root_coco_path / patch_info['file_name'], 'r')
    medians = get_medians(netcdf, 0, num_buckets, group_freq, bands, padded_patch_height,
                          padded_patch_width, output_size, pad_top, pad_bot,
                          pad_left, pad_right, medians_dtype)

    num_bins, num_bands = medians.shape[:2]

    medians = sliding_window_view(medians, [num_bins, num_bands, output_size[0], output_size[1]], [1, 1, output_size[0], output_size[1]]).squeeze()
    # shape: (subpatches_in_row, subpatches_in_col, bins, bands, height, width)

    # Save medians
    bins_pad = len(str(medians.shape[-4]))
    subs_pad = len(str(medians.shape[0] * medians.shape[1]))
    sub_idx = 0
    for i in range(medians.shape[0]):
        for j in range(medians.shape[1]):
            for t in range(num_bins):
                np.save(patch_dir / f'sub{str(sub_idx).rjust(subs_pad, "0")}_bin{str(t).rjust(bins_pad, "0")}', medians[i, j, t, :, :, :].astype(medians_dtype))
            sub_idx += 1

    # Save labels
    labels = get_labels(netcdf, output_size, pad_top, pad_bot, pad_left, pad_right)
    labels = sliding_window_view(labels, output_size, output_size)
    labels = labels.squeeze()  # shape: (subpatches_in_row, subpatches_in_col, height, width)

    lbl_idx = 0
    lbl_pad = len(str(labels.shape[0] * labels.shape[1]))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            np.save(patch_dir / f'labels_sub{str(lbl_idx).rjust(lbl_pad, "0")}', labels[i, j, :, :].astype(label_dtype))
            lbl_idx += 1

def sliding_window_view(arr, window_shape, steps):
    '''
    Code taken from:
        https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a

    Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary.
        Parameters
        ----------
        arr : numpy.ndarray, shape=(...,[x, (...), z])
            The array to slide the window over.
        window_shape : Sequence[int]
            The shape of the window to raster: [Wx, (...), Wz],
            determines the length of [x, (...), z]
        steps : Sequence[int]
            The step size used when applying the window
            along the [x, (...), z] directions: [Sx, (...), Sz]
        Returns
        -------
        view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
            Where X = (x - Wx) // Sx + 1
    '''
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)


def get_medians(netcdf, start_bin, window, group_freq, bands,
                padded_patch_height, padded_patch_width, output_size,
                pad_top, pad_bot, pad_left, pad_right, medians_dtype):
    # Grab year from netcdf4's global attribute
    year = netcdf.patch_year

    # output intervals
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{int(year) + 1}-01-01', freq=group_freq)

    # out, aggregated array
    medians = np.empty((len(bands), window, padded_patch_height, padded_patch_width), dtype=medians_dtype)

    for band_id, band in enumerate(bands):
        # Load band data
        band_data = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf[band]))

        # Aggregate into time bins
        band_data = band_data.groupby_bins(
            'time',
            bins=date_range,
            right=True,
            include_lowest=False,
            labels=date_range[:-1]
        ).median(dim='time')

        # Upsample so months without data are initiated with NaN values
        band_data = band_data.resample(time_bins=group_freq).median(dim='time_bins')

        # Fill:
        # NaN months with linear interpolation
        # NaN months outsize of range (e.x month 12) using extrapolation
        band_data = band_data.interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')

        # Keep values within requested time window
        band_data = band_data.isel(time_bins=slice(start_bin, start_bin + window))

        # Convert to numpy array
        band_data = band_data[f'{band}'].values

        # If expand ratio is 1, that means current band has the same resolution as reference band
        expand_ratio = int(BANDS[band] / BANDS[REFERENCE_BAND])

        # If resolution does not match reference band, stretch it
        if expand_ratio != 1:
            band_data = np.repeat(band_data, expand_ratio, axis=1)
            band_data = np.repeat(band_data, expand_ratio, axis=2)

        # Add padding if needed
        if  (output_size[0] < band_data.shape[1]) or (output_size[1] < band_data.shape[2]):
            band_data = np.pad(band_data,
                                pad_width=((0, 0), (pad_top, pad_bot), (pad_left, pad_right)),
                                mode='constant',
                                constant_values=0)

        medians[band_id, :, :, :] = np.expand_dims(band_data, axis=0)

    # Reshape so window length is first
    return medians.transpose(1, 0, 2, 3)   # (T, B, H, W)


def get_labels(netcdf, output_size, pad_top, pad_bot, pad_left, pad_right):
    # Load and Convert to numpy array
    labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['labels']))['labels'].values

    # Add padding if needed
    if (output_size[0] < labels.shape[0]) or (output_size[1] < labels.shape[1]):
        labels = np.pad(labels,
                        pad_width=((pad_top, pad_bot), (pad_left, pad_right)),
                        mode='constant',
                        constant_values=0
                        )

    return labels


def get_padding_offset(patch_height, patch_width, output_size):
    img_size_x = patch_height
    img_size_y = patch_width

    output_size_x = output_size[0]
    output_size_y = output_size[1]

    # Calculate padding offset
    if img_size_x >= output_size_x:
        pad_x = int(output_size_x - img_size_x % output_size_x)
    else:
        # For bigger images, is just the difference
        pad_x = output_size_x - img_size_x

    if img_size_y >= output_size_y:
        pad_y = int(output_size_y - img_size_y % output_size_y)
    else:
        # For bigger images, is just the difference
        pad_y = output_size_y - img_size_y

    # Number of rows that need to be padded (top and bot)
    if not pad_x == output_size_x:
        pad_top = int(pad_x // 2)
        pad_bot = int(pad_x // 2)

        # if padding is not equally divided, pad +1 row to the top
        if not pad_x % 2 == 0:
            pad_top += 1
    else:
        pad_top = 0
        pad_bot = 0

    # Number of rows that need to be padded (left and right)
    if not pad_y == output_size_y:
        pad_left = int(pad_y // 2)
        pad_right = int(pad_y // 2)

        # if padding is not equally divided, pad +1 row to the left
        if not pad_y % 2 == 0:
            pad_left += 1
    else:
        pad_left = 0
        pad_right = 0

    return pad_top, pad_bot, pad_left, pad_right


def calculate_subpatches(output_size):
    assert output_size[0] == output_size[1], \
        f'Only square sub-patch size is supported. Mismatch: {output_size[0]} != {output_size[1]}.'

    patch_width, patch_height = IMG_SIZE, IMG_SIZE
    padded_patch_width, padded_patch_height = IMG_SIZE, IMG_SIZE

    # Calculate number of sub-patches in each dimension, check if image needs to be padded
    if (output_size[0] == patch_height) or (output_size[1] == patch_width):
        return patch_height, patch_width, 0, 0, 0, 0

    # Calculating padding offsets if there is a need to
    if (patch_height % output_size[0] != 0) or (patch_width % output_size[1] != 0):
        requires_pad = True
        pad_top, pad_bot, pad_left, pad_right = get_padding_offset(patch_height, patch_width, output_size)

        # patch_height should always match patch_width because we have square images,
        # but doing it like this ensures expandability
        padded_patch_height += (pad_top + pad_bot)
        padded_patch_width += (pad_left + pad_right)
    else:
        pad_top, pad_bot, pad_left, pad_right = 0, 0, 0, 0

    # num_subpatches = (padded_patch_height // output_size[0]) * (padded_patch_width // output_size[1])

    return padded_patch_height, padded_patch_width, pad_top, pad_bot, pad_left, pad_right



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and export median files for a given S2 dataset')
    parser.add_argument('--data', type=str, default='dataset/netcdf', required=False,
                        help='Path to the netCDF files. Default "dataset/netcdf/".')
    parser.add_argument('--root_coco_path', type=str, default='dataset/', required=False,
                        help='Root path for coco file. Default "dataset/".')
    parser.add_argument('--prefix_coco', type=str, default=None, required=False,
                        help='The prefix to use for the COCO file. Default none.')
    parser.add_argument('--out_path', type=str, default='logs/medians', required=False,
                        help='Path to export the medians into. Default "logs/medians/".')
    parser.add_argument('--group_freq', type=str, default='1MS', required=False,
                        help='The frequency to aggregate medians with. Default "1MS".')
    parser.add_argument('--output_size', nargs='+', default=None, required=False,
                        help='The size of the medians. If none given, the output will be of the same size.')
    parser.add_argument('--bands', nargs='+', default=None, required=False,
                        help='The bands to use. Default all.')
    parser.add_argument('--num_workers', type=int, default=8, required=False,
                        help='The number of workers to use for parallel computation. Default 8.')
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out_path)
    root_coco_path = Path(args.root_coco_path)

    medians_dtype = np.float32
    label_dtype = np.int16

    if args.bands is None:
        bands = BANDS.keys()
    else:
        bands = args.bands

    bands = sorted(bands)

    if args.output_size is None:
        output_size = [366, 366]
    else:
        output_size = [int(x) for x in args.output_size]

    num_buckets = len(pd.date_range(start=f'2020-01-01', end=f'2021-01-01', freq=args.group_freq)) - 1

    padded_patch_height, padded_patch_width, pad_top, pad_bot, pad_left, pad_right = calculate_subpatches(output_size)

    # Create medians folder if it doesn't exist
    out_path.mkdir(exist_ok=True, parents=True)

    print(f'Saving into: {out_path}.')

    print(f'\nStart process...')

    for mode in ['train', 'val', 'test']:
        if args.prefix_coco is not None:
            coco_path = root_coco_path / f'{args.prefix_coco}_coco_{mode}.json'
        else:
            coco_path = root_coco_path / f'coco_{mode}.json'
        coco = COCO(coco_path)

        func = partial(process_patch, out_path, mode, num_buckets, root_coco_path,
                       bands, padded_patch_height, padded_patch_width, medians_dtype,
                       label_dtype, args.group_freq, output_size, pad_top, pad_bot, pad_left, pad_right)

        process_map(func, list(coco.imgs.items()), max_workers=args.num_workers)

    print('Medians saved.\n')
