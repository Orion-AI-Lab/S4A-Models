import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.axes_grid1 import ImageGrid
from pycocotools.coco import COCO

from model.convLSTM_lightning import ConvLSTM
from model.tempCNN_lightning import TempCNN
from model.convSTAR_lightning import ConvSTAR

import pytorch_lightning as pl
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.tools import font_colors
from utils.PAD_datamodule import PADDataModule
from utils.settings.config import CROP_ENCODING, IMG_SIZE, LINEAR_ENCODER, BANDS


def get_window(idx, window_len, image_size, coco_file):
        '''
        Returns the DataSet index for a given patch id and the
        number of subpatches belonging to this patch.

        Patch indexing starts from 0.
        '''
        num_patches = len(COCO(coco_file).imgs)

        num_subpatches = ((IMG_SIZE // image_size[0]), (IMG_SIZE // image_size[1]))
        subpatch_id = idx * (num_subpatches[0] * num_subpatches[1])

        return subpatch_id, num_subpatches


if __name__ == '__main__':
    # Parse user arguments
    parser = argparse.ArgumentParser(description='Download MODIS and Sentinel-2 images based on given CSV files.')

    parser.add_argument('--checkpoints', nargs='+', required=True,
                            help='The checkpoints to load for every model. First the ConvLSTM, second the ConvSTAR.')
    parser.add_argument('--exp', type=int, required=True,
                            help='The experiment the models were trained on. Used in the exported figure file.')
    parser.add_argument('--out_path', type=str, required=True,
                            help='The path to export figures into.')

    parser.add_argument('--image_idx', nargs='+', required=False,
                            help='A list of indices of the image batches to evaluate on. If not given, a random one is chosen.')

    parser.add_argument('--parcel_loss', action='store_true', default=False, required=False,
                            help='Use a loss function that takes into account parcel pixels only.')

    parser.add_argument('--binary_labels', action='store_true', default=False, required=False,
                             help='Map categories to 0 background, 1 parcel. Default False')

    parser.add_argument('--root_path_coco', type=str, default='dataset/', required=False,
                             help='root path until coco file')
    parser.add_argument('--prefix_coco', type=str, default=None, required=False,
                             help='The prefix to use for the COCO file. Default none.')

    parser.add_argument('--prefix', type=str, default=None, required=False,
                             help='The prefix to use for dumping data files. If none, the current timestamp is used')

    parser.add_argument('--group_freq', type=str, required=False, default='1MS',
                             help='The frequency to use for binning. All Pandas offset aliases are supported.'
                                  'Check: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases ')
    parser.add_argument('--window_len', type=int, default=6, required=False,
                             help='The length of the rolling window to be used. Default 6')

    parser.add_argument('--bands', nargs='+', default=sorted(list(BANDS.keys())),
                             help='The image bands to use. Must be space separated')
    parser.add_argument('--saved_medians', action='store_true', default=False, required=False,
                             help='Precompute and export the image medians')
    parser.add_argument('--img_size', nargs='+', required=False,
                             help='The size of the subpatch to use as model input. Must be space separated')
    parser.add_argument('--requires_norm', action='store_true', default=False, required=False,
                             help='Normalize data to 0-1 range. Default False')

    parser.add_argument('--num_workers', type=int, default=6, required=False,
                             help='Number of workers to work on dataloader. Default 6')
    parser.add_argument('--num_gpus', type=int, default=1, required=False,
                             help='Number of gpus to use (per node). Default 1')
    parser.add_argument('--num_nodes', type=int, default=1, required=False,
                             help='Number of nodes to use. Default 1')
    args = parser.parse_args()

    # Try convert args.img_size to int tuple
    if args.img_size is not None:
        try:
            args.img_size = tuple(map(int, args.img_size))
        except:
            print(f'argument img_size should be castable to int but instead "{args.img_size}" was given!')
            exit(1)

    # Normalize paths for different OSes
    root_path_coco = Path(args.root_path_coco)
    out_path = Path(args.out_path)

    # Check existence of data folder
    if not root_path_coco.is_dir():
        print(f'{font_colors.RED}Coco path doesn\'t exist!{font_colors.ENDC}')
        exit(1)

    convlstm_chkpt, convstar_chkpt = args.checkpoints
    run_path_convlstm = Path(*Path(convlstm_chkpt).parts[:-2])
    run_path_convstar = Path(*Path(convstar_chkpt).parts[:-2])

    if args.binary_labels:
        n_classes = 2
    else:
        n_classes = len(list(CROP_ENCODING.values())) + 1

    args.img_size = [int(dim) for dim in args.img_size]

    convlstm = ConvLSTM(run_path_convlstm, LINEAR_ENCODER, parcel_loss=args.parcel_loss)

    # Load the model for testing
    checkpoint_epoch = Path(convlstm_chkpt).stem.split('=')[1].split('-')[0]
    convlstm = ConvLSTM.load_from_checkpoint(convlstm_chkpt,
                                          map_location=torch.device('cpu'),
                                          run_path=run_path_convlstm,
                                          linear_encoder=LINEAR_ENCODER,
                                          checkpoint_epoch=checkpoint_epoch)

    convstar = ConvSTAR(run_path_convstar, LINEAR_ENCODER, parcel_loss=args.parcel_loss)

    # Load the model for testing
    checkpoint_epoch = Path(convstar_chkpt).stem.split('=')[1].split('-')[0]
    convstar = ConvSTAR.load_from_checkpoint(convstar_chkpt,
                                          map_location=torch.device('cpu'),
                                          run_path=run_path_convstar,
                                          linear_encoder=LINEAR_ENCODER,
                                          checkpoint_epoch=checkpoint_epoch)

    if args.prefix_coco is not None:
        path_test = root_path_coco / f'{args.prefix_coco}_coco_test.json'
    else:
        path_test = root_path_coco / 'coco_test.json'

    # Create Data Module for testing images
    dm = PatchesDataModule(
        root_path_coco=root_path_coco,
        path_test=path_test,
        group_freq=args.group_freq,
        prefix=args.prefix,
        bands=args.bands,
        linear_encoder=LINEAR_ENCODER,
        saved_medians=args.saved_medians,
        window_len=args.window_len,
        requires_norm=args.requires_norm,
        return_masks=False,
        clouds=False,
        cirrus=False,
        shadow=False,
        snow=False,
        output_size=args.img_size,
        batch_size=1,
        num_workers=args.num_workers,
        binary_labels=args.binary_labels,
        return_parcels=args.parcel_loss
    )

    # Create a second DataModule for non-normalized Sentinel-2 images
    dm_nonNorm = PatchesDataModule(
        root_path_coco=root_path_coco,
        path_test=path_test,
        group_freq=args.group_freq,
        prefix=args.prefix,
        bands=args.bands,
        linear_encoder=LINEAR_ENCODER,
        saved_medians=args.saved_medians,
        window_len=args.window_len,
        requires_norm=True,
        return_masks=False,
        clouds=False,
        cirrus=False,
        shadow=False,
        snow=False,
        output_size=args.img_size,
        batch_size=1,
        num_workers=args.num_workers,
        binary_labels=args.binary_labels,
        return_parcels=args.parcel_loss
    )

    # TESTING
    dm.setup('test')
    dm_nonNorm.setup('test')

    convlstm.cuda()
    convstar.cuda()

    # Test model
    convlstm.eval()
    convstar.eval()

    if args.image_idx is None:
        total_images = len(COCO(path_test).imgs)
        image_idx = [np.random.randint(0, total_images)]
    else:
        image_idx = [int(x) for x in args.image_idx]

    title_font = {'size':'28'}
    months = ['April', 'May', 'June', 'July', 'August', 'September']

    # Visualize results
    for image_id in image_idx:
        subpatch_id, num_subpatches = get_window(image_id, args.window_len, args.img_size, path_test)

        # Export the Sentinel-2 time series
        fig, axes = plt.subplots(1, 6, figsize=(60, 15))
        grids = [
            ImageGrid(fig, 161, nrows_ncols=(num_subpatches[0], num_subpatches[1]), axes_pad=0.0),
            ImageGrid(fig, 162, nrows_ncols=(num_subpatches[0], num_subpatches[1]), axes_pad=0.0),
            ImageGrid(fig, 163, nrows_ncols=(num_subpatches[0], num_subpatches[1]), axes_pad=0.0),
            ImageGrid(fig, 164, nrows_ncols=(num_subpatches[0], num_subpatches[1]), axes_pad=0.0),
            ImageGrid(fig, 165, nrows_ncols=(num_subpatches[0], num_subpatches[1]), axes_pad=0.0),
            ImageGrid(fig, 166, nrows_ncols=(num_subpatches[0], num_subpatches[1]), axes_pad=0.0)
        ]

        for idx in range(num_subpatches[0] * num_subpatches[1]):
            batch_sen = dm_nonNorm.dataset_test.__getitem__(subpatch_id + idx)

            for month_i in range(len(months)):
                sentinel = grids[month_i][idx].imshow(np.moveaxis(batch_sen['medians'][month_i, :3, :, :].squeeze(), 0, -1), vmin=0, vmax=1)
                grids[month_i][idx].set_axis_off()

        for month_i in range(len(months)):
            axes[month_i].set_title(months[month_i], fontdict=title_font)
            axes[month_i].set_xticks([])
            axes[month_i].set_yticks([])

        plt.savefig(out_path / f'exp{args.exp}_comparative_evaluation_of_image_{image_id}_SENTINEL.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)

        # Export the label and predictions
        fig, axes = plt.subplots(1, 3, figsize=(50, 15))

        grid1 = ImageGrid(fig, 131, nrows_ncols=(num_subpatches[0], num_subpatches[1]), axes_pad=0.0)
        grid2 = ImageGrid(fig, 132, nrows_ncols=(num_subpatches[0], num_subpatches[1]), axes_pad=0.0)
        grid3 = ImageGrid(fig, 133, nrows_ncols=(num_subpatches[0], num_subpatches[1]), axes_pad=0.0)
        for idx in range(num_subpatches[0] * num_subpatches[1]):
            batch = dm.dataset_test.__getitem__(subpatch_id + idx)

            im = grid1[idx].imshow(batch['labels'].squeeze(), vmin=0, vmax=max(LINEAR_ENCODER.values()), cmap='tab20')
            grid1[idx].set_axis_off()

            # Make predictions
            inputs = batch['medians'][None, :, :, :, :]  # (B, T, C, H, W)
            inputs = torch.from_numpy(inputs).cuda()
            pred_convlstm = convlstm(inputs)

            pred_convstar = convstar(inputs)  # (B, K, H, W)

            pred_convlstm = pred_convlstm.to(torch.float32)
            pred_convstar = pred_convstar.to(torch.float32)

            parcels = torch.from_numpy(batch['parcels'])[None, :, :].cuda()  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred_convlstm.size(1), 1, 1)  # (B, K, H, W)
            #pred = torch.clamp(pred, 0, max(LINEAR_ENCODER.values()))

            label = torch.from_numpy(batch['labels'][None, :, :]).to(torch.long).cuda()  # (B, H, W)

            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred_convlstm.size(1), 1, 1) != 0)
            pred_convlstm[~mask_K] = 0
            pred_convstar[~mask_K] = 0

            pred_sparse_convlstm = pred_convlstm.argmax(axis=1).squeeze().cpu().detach().numpy()
            pred_sparse_convstar = pred_convstar.argmax(axis=1).squeeze().cpu().detach().numpy()

            grid2[idx].imshow(pred_sparse_convlstm, vmin=0, vmax=max(LINEAR_ENCODER.values()), cmap='tab20')
            grid2[idx].set_axis_off()

            grid3[idx].imshow(pred_sparse_convstar, vmin=0, vmax=max(LINEAR_ENCODER.values()), cmap='tab20')
            grid3[idx].set_axis_off()

        crop_encoding_rev = {v: k for k, v in CROP_ENCODING.items()}
        crop_encoding = {k: crop_encoding_rev[k] for k in LINEAR_ENCODER.keys() if k != 0}
        crop_encoding[0] = 'Background/Other'

        crop_ids = sorted(LINEAR_ENCODER.keys())
        colors = [im.cmap(im.norm(LINEAR_ENCODER[crop_id])) for crop_id in crop_ids]
        patches = [mpatches.Patch(color=colors[LINEAR_ENCODER[crop_id]], label=f'{crop_id} ({crop_encoding[crop_id]})') for crop_id in crop_ids]
        axes[2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-large')

        axes[0].set_title('Label', fontdict=title_font)
        axes[1].set_title('ConvLSTM Prediction', fontdict=title_font)
        axes[2].set_title('ConvSTAR Prediction', fontdict=title_font)

        # Remove all axis labels
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        plt.savefig(out_path / f'exp{args.exp}_comparative_evaluation_of_image_{image_id}_PREDS.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
