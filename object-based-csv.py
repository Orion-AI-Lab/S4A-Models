from multiprocessing import Pool, cpu_count
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union
import netCDF4
from pycocotools.coco import COCO
from tqdm import tqdm

import time
import platform

RANDOM_SEED = 16
IMG_SIZE = 366
# Extract patches based on this band
REFERENCE_BAND = 'B02'

# Band names and their resolutions
BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}

np.random.seed(RANDOM_SEED)


def extract_metrics(
        file: Union[str, Path],
        verbose: bool = True,
        freq: str = '1MS',
        save_path: Union[str, Path] = 'data/oad'
) -> bool:

    if verbose:
        print(f'Working : {file}')

    file = Path(file)
    save_path = Path(save_path)

    dump_path = save_path / 'temp' / (file.stem + '.csv.gz')

    if dump_path.exists():
        print(f'Exists: {file}')
        return True

    bands = sorted(BANDS.keys())

    # Load netcdf as xarray
    netcdf = netCDF4.Dataset(file, 'r')

    # Grab year from netcdf4's global attribute
    year = netcdf.patch_year

    # output intervals
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{int(year) + 1}-01-01', freq=freq)

    # out, aggregated array
    medians = np.empty((len(bands), 12, IMG_SIZE, IMG_SIZE), dtype='float64')

    for band_id, band in enumerate(bands):

        # Load band data
        data = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf[band]))

        # Aggregate into time bins
        data = data.groupby_bins(
            'time',
            bins=date_range,
            right=True,
            include_lowest=False,
            labels=date_range[:-1]
        ).median(dim='time')

        # Upsample so months without data are initiated with NaN values
        data = data.resample(time_bins=freq).median(dim='time_bins')

        # Fill:
        # NaN months with linear interpolation
        # NaN months outsize of range (e.x month 12) using extrapolation
        data = data.interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')

        # Convert to numpy array
        data = data[f'{band}'].values

        # If expand ratio is 1, that means current band has the same resolution as reference band
        expand_ratio = int(BANDS[band] / BANDS[REFERENCE_BAND])

        # If resolution does not match reference band, stretch it
        if expand_ratio != 1:
            data = np.repeat(data, expand_ratio, axis=1)
            data = np.repeat(data, expand_ratio, axis=2)

        medians[band_id, :, :, :] = np.expand_dims(data, axis=0)

    csv_data = []

    # After aggregating data, calculate metrics (mean, std, count) for each unique parcel
    parcels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['parcels'])).to_array().squeeze().values
    labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['labels'])).to_array().squeeze().values

    for parcel in np.unique(parcels):

        # Mask of this unique parcel
        mask = (parcels == parcel)
        csv_band = {
            'parcel_id': int(parcel),
            'label': int(np.unique(labels[mask])[0]),
            'counts': int(np.count_nonzero(mask))
        }

        # Loop over bands to create csv
        for band_id, band in enumerate(bands):

            for interval in range(medians.shape[1]):
                data = medians[band_id, interval, :, :]
                data = data[mask]

                csv_band[f'{band}_{interval:02d}_mean'] = float(np.nanmean(data))
                csv_band[f'{band}_{interval:02d}_std'] = float(np.nanstd(data))

        csv_data.append(csv_band)

    # Concat to dataframe and write to disk

    pd.DataFrame(csv_data).to_csv(
        dump_path,
        compression='gzip',
        index=False
    )

    if verbose:
        print(f'Finished: {str(file)}')

    return True


def main():

    # Fork is faster and more memory efficient for our task, default for UNIX, but making sure
    assert platform.system().lower() == 'linux', f'This system is using fork() as a method for multiprocessing,' \
                                                 f'fork is only available in UNIX systems. You are running on: ' \
                                                 f'"{platform.system()}.'

    # Parse cli argumnets
    args = parse_args()

    # If a value not between 1-max_cores is given, change it to use max_cores-2
    if args.num_processes not in range(1, cpu_count() + 1):
        args.num_processes = cpu_count()

    # Keep min between max available cores and parsed ones
    # Always leave 2 cores out for synchronization
    args.num_processes = min(args.num_processes, cpu_count() - 2)

    # Convert to Pathlib objects
    coco_path = Path(args.coco_path)
    coco_root = Path(args.coco_root)
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # temp path
    Path(save_path / 'temp').mkdir(exist_ok=True, parents=True)

    time_start = time.time()

    def log_result(result):
        # This is called whenever my_func() returns a result.
        # result_list is modified only by the main process, not the pool workers.
        result_list.append(result)

    def errorhandler(exc):
        print('Exception:', exc)

    if args.num_processes > 1:
        # Run in parallel
        pool = Pool(processes=args.num_processes, maxtasksperchild=1)

    total_files = 0
    result_list = []

    coco = COCO(coco_path)

    patches = sorted([patch['file_name'] for patch in list(coco.imgs.values())])

    for patch in patches:

        patch_path = coco_root / patch

        if args.num_processes > 1:
            # Launch n processes, where n (args.num_processes)
            _ = pool.apply_async(extract_metrics,
                                 args=(patch_path, args.verbose, '1MS', save_path),
                                 callback=log_result,
                                 error_callback=errorhandler
                                 )
        else:
            result = extract_metrics(file=patch_path, verbose=args.verbose, freq='1MS', save_path=save_path)
            result_list.append(result)

        total_files += 1

    if args.num_processes > 1:
        pool.close()
        pool.join()

    print(f'Successfully completed {sum(result_list)}/{total_files} file(s).')

    csvs = [file for file in save_path.rglob('temp/*patch*.gz')]

    # Open dfs
    dfs = [pd.read_csv(csv) for csv in tqdm(csvs, ncols=75, desc='Merging CSV(s).')]
    # Concat and write
    pd.concat(dfs, ignore_index=True).to_csv(save_path / (coco_path.stem + '.csv.gz'), index=False, compression='gzip')

    # Delete
    _ = [csv.unlink() for csv in tqdm(csvs, ncols=75, desc='Deleting..')]

    # Remove dir
    Path(save_path / 'temp').rmdir()

    print(f'\nDone. Time Elapsed: {(time.time() - time_start) / 60:0.2f} min(s).\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default='exp1_patches2000_strat_coco_val.json')
    parser.add_argument('--coco_root', type=str, default='coco_files/')
    parser.add_argument('--save_path', type=str, default='data/oad/')
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', default=True, required=False)

    return parser.parse_args()


if __name__ == '__main__':
    main()
