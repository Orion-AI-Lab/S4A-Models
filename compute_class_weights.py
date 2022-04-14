import argparse
import numpy as np
from pathlib import Path
import pickle

from pycocotools.coco import COCO

from utils.PAD_datamodule import PADDataModule
from utils.settings.config import CROP_ENCODING, LINEAR_ENCODER


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes the weights of each class.')

    parser.add_argument('--coco_path', type=str, default='coco_files/', required=False,
                        help='Path of COCO files. Default "coco_files/".')
    parser.add_argument('--coco_prefix', type=str, default='exp1_patches2000_strat', required=False,
                        help='Prefix of the COCO file. Default "exp1_patches2000_strat".')
    parser.add_argument('--medians_prefix', type=str, default='exp1_patches2000_strat_61x61', required=False,
                        help='Prefix of the medians directory. Default "exp1_patches2000_strat_61x61".')
    parser.add_argument('--out_prefix', type=str, required=False,
                        help='The prefix to use for the class weights file. Default none.')
    parser.add_argument('--ignore_zero', default=False, action='store_true', required=False,
                        help='Ignore the zero class.')
    parser.add_argument('--fixed_window', action='store_true', default=False, required=False,
                            help='Use a fixed window including months 4 (April) to 9 (September).')
    args = parser.parse_args()

    # Define paths
    root_path_coco = Path(args.coco_path)
    coco_train = root_path_coco / f'{args.coco_prefix}_coco_train.json'
    coco_val = root_path_coco / f'{args.coco_prefix}_coco_val.json'

    if args.out_prefix is not None:
        out_name = f'{args.out_prefix}_class_weights.pkl'
    else:
        out_name = 'class_weights.pkl'

    if args.out_prefix is not None:
        pixel_cnts_name = f'{args.out_prefix}_class_pixel_counts.pkl'
    else:
        pixel_cnts_name = 'class_pixel_counts.pkl'

    if Path(pixel_cnts_name).is_file():
        class_pixel_counts = pickle.load(open(pixel_cnts_name, 'rb'))
    else:
        # Create Data Module
        dm = PADDataModule(
            root_path_coco=root_path_coco,
            path_train=coco_train,
            path_val=coco_val,
            group_freq='1MS',
            prefix=args.medians_prefix,
            bands=['B02', 'B03', 'B04', 'B08'],
            linear_encoder=LINEAR_ENCODER,
            saved_medians=True,
            window_len=12,
            fixed_window=args.fixed_window,
            requires_norm=True,
            return_masks=False,
            clouds=False,
            cirrus=False,
            shadow=False,
            snow=False,
            output_size=[61, 61],
            batch_size=1,
            num_workers=2,
            binary_labels=False,
            return_parcels=True
        )

        # TRAINING
        dm.setup('fit')

        # Count pixels for each class
        class_pixel_counts = {c: 0 for c in LINEAR_ENCODER.values()}

        if args.ignore_zero:
            del class_pixel_counts[0]

        for idx in range(len(dm.dataset_train)):
            try:
                batch = dm.dataset_train.__getitem__(idx)
            except:
                print(f'IDX: {idx}')
                break
            labels = batch['labels']
            parcels = batch['parcels']

            # Keep only parcels in labels
            labels = labels[parcels]

            values, counts = np.unique(labels, return_counts=True)
            for i in range(len(values)):
                if values[i] not in class_pixel_counts.keys(): continue

                class_pixel_counts[values[i]] += counts[i]

        pickle.dump(class_pixel_counts, open(pixel_cnts_name, 'wb'))

    # Compute weights for each class
    all_counts = sum(list(class_pixel_counts.values()))
    n_classes = len(class_pixel_counts)
    class_weights = {k: all_counts / (n_classes * v) for k, v in class_pixel_counts.items()}

    pickle.dump(class_weights, open(out_name, 'wb'))
