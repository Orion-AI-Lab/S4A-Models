from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.settings.config import RANDOM_SEED
from utils.OAD_datamodule import OADDataModule
from model.OAD_LSTM import LSTM
from model.OAD_Transformer import Transformer
from model.OAD_TempCNN import TempCNN
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Set seed for everything
pl.seed_everything(RANDOM_SEED)


def main():
    args = parse_args()
    print(args)

    # Create folders for saving and/or retrieving useful files for dataloaders
    log_path = Path('logs')
    log_path.mkdir(exist_ok=True, parents=True)

    # Trainer callbacks
    callbacks = []
    monitor = 'val_loss'
    mode = 'min'

    run_path = log_path / args.model / f'{args.prefix}'
    run_path.mkdir(exist_ok=True, parents=True)

    min_epochs = 1
    max_epochs = args.num_epochs + 1
    check_val_every_n_epoch = 1

    tb_logger = pl_loggers.TensorBoardLogger(run_path / 'tensorboard')

    selected_classes = {
        110:  'Wheat',
        120:  'Maize',
        140:  'Sorghum',
        150:  'Barley',
        160:  'Rye',
        170:  'Oats',
        330:  'Grapes',
        435:  'Rapeseed',
        438:  'Sunflower',
        510:  'Potatoes',
        770:  'Peas'
    }

    linear_encoder = {key: i for i, key in enumerate(sorted(selected_classes.keys()))}
    name_decoder = {str(i): str(selected_classes[key]) for i, key in enumerate(sorted(selected_classes.keys()))}
    id_decoder = {str(i): str(key) for i, key in enumerate(sorted(selected_classes.keys()))}

    # Load models
    if args.model.lower() == 'lstm':
        model = LSTM(
            num_classes=len(linear_encoder),
            name_decoder=name_decoder,
            id_decoder=id_decoder,
            lr=args.lr,
            input_size=26,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            bidirectional=args.bidirectional
        )

    elif args.model.lower() == 'transformer':
        model = Transformer(
            d_model=26,
            name_decoder=name_decoder,
            id_decoder=id_decoder,
            num_layers=args.num_layers,
            dim_feedforward=args.hidden_size,
            nhead=2,
            dropout=0.1,
            batch_size=args.batch_size,
            batch_first=True,
            norm_first=False,
            num_classes=len(linear_encoder)
        )

    elif args.model.lower() == 'tempcnn':
        model = TempCNN(
            input_size=26,
            sequencelength=7,
            hidden_size=128,
            kernel_size=7,
            name_decoder=name_decoder,
            id_decoder=id_decoder,
            batch_size=args.batch_size,
            num_classes=len(linear_encoder),
            lr=args.lr,
            dropout=0.2
        )
    else:
        print('Invalid model!')
        exit(1)

    if args.train:
        callbacks.append(
            LearningRateMonitor(logging_interval='step')
        )

        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode
            )
        )

        callbacks.append(
            ModelCheckpoint(
                dirpath=run_path / 'checkpoints',
                monitor=monitor,
                filename='model_best',
                mode=mode,
                save_top_k=1
            )
        )

        dm = OADDataModule(
            file=Path(args.file),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            linear_encoder=linear_encoder
        )

        trainer = pl.Trainer(gpus=[args.num_gpus],
                             num_nodes=args.num_nodes,
                             min_epochs=min_epochs,
                             max_epochs=max_epochs,
                             check_val_every_n_epoch=check_val_every_n_epoch,
                             callbacks=callbacks,
                             logger=tb_logger,
                             enable_checkpointing=True,
                             resume_from_checkpoint=args.checkpoint if args.checkpoint is not None else None,
                             fast_dev_run=args.fast_dev_run
                             )

        # Train model
        trainer.fit(model, datamodule=dm)

    else:
        dm = OADDataModule(
            file=Path(args.file),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            linear_encoder=linear_encoder
        )
        trainer = pl.Trainer(gpus=[args.num_gpus],
                             num_nodes=args.num_nodes,
                             logger=tb_logger
                             )

        # Load weights from checkpoint
        model = model.load_from_checkpoint(args.checkpoint)
        model.eval()
        trainer.test(model, datamodule=dm)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True, required=False)
    parser.add_argument('--checkpoint', type=str, default=None, required=False)
    parser.add_argument('--model', type=str, required=False, choices=['lstm', 'transformer', 'tempcnn'], default='tempcnn')
    parser.add_argument('--prefix', type=str, required=False, default='exp1')
    parser.add_argument('--file', type=str, required=False, default='dataset/data/oad/exp1_patches2000_strat_coco')
    parser.add_argument('--num_epochs', type=int, default=10, required=False)
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--lr', type=float, default=1e-3, required=False)
    parser.add_argument('--num_workers', type=int, default=6, required=False)
    parser.add_argument('--num_gpus', type=int, default=0, required=False)
    parser.add_argument('--num_nodes', type=int, default=1, required=False)
    parser.add_argument('--fast_dev_run', action='store_true', default=False, required=False)

    parser.add_argument('--hidden_size', type=int, default=512, required=False)
    parser.add_argument('--num_layers', type=int, default=3, required=False)
    parser.add_argument('--bidirectional', action='store_true', default=False, required=False)

    return parser.parse_args()


if __name__ == '__main__':
    main()
