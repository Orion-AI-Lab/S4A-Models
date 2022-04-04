'''
Implementation of the model proposed in:
- Xingjian Shi, et al. "Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting." (2015), arXiv:1506.04214v2.

Code adopted from:
https://github.com/jhhuang96/ConvLSTM-PyTorch.git
'''

import os

import numpy as np
from tqdm import tqdm
import copy
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from tensorboardX import SummaryWriter
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


def print_model_stats(model):
    '''
    Prints the number of parameters, the number of trainable parameters and
    the peak memory usage of a given Pytorch model.
    '''
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {pytorch_total_params}')
    print(f'Trainable params: {pytorch_train_params}')
    print(f'Peak memory usage: {(torch.cuda.max_memory_allocated() / (1024**2)):.2f} MB\n')


def tensor_size(t):
    size = t.element_size() * t.nelement()
    print(f'\nTensor memory usage: {size / (1024**2):.2f} MB')


def get_last_model_checkpoint(path):
    '''
    Browses through the given path and finds the last saved checkpoint of a
    model.

    Parameters
    ----------
    path: str or Path
        The path to search.

    Returns
    -------
    (Path, Path, int): the path of the last model checkpoint file, the path of the
    last optimizer checkpoint file and the corresponding epoch.
    '''
    model_chkp = [c for c in Path(path).glob('model_state_dict_*')]
    optimizer_chkp = [c for c in Path(path).glob('optimizer_state_dict_*')]
    model_chkp_per_epoch = {int(c.name.split('.')[0].split('_')[-1]): c for c in model_chkp}
    optimizer_chkp_per_epoch = {int(c.name.split('.')[0].split('_')[-1]): c for c in optimizer_chkp}

    last_model_epoch = sorted(model_chkp_per_epoch.keys())[-1]
    last_optimizer_epoch = sorted(optimizer_chkp_per_epoch.keys())[-1]

    assert last_model_epoch == last_optimizer_epoch, 'Error: Could not resume training. Optimizer or model checkpoint missing.'

    return model_chkp_per_epoch[last_model_epoch], optimizer_chkp_per_epoch[last_model_epoch], last_model_epoch


class CLSTM_cell(nn.Module):
    """
    The ConvLSTM cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # height, width
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))


    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)


class ConvLSTM(pl.LightningModule):
    def __init__(self, run_path, linear_encoder, learning_rate=1e-3, parcel_loss=False,
                 class_weights=None, crop_encoding=None, checkpoint_epoch=None):
        '''
        Parameters:
        -----------
        run_path: str or Path
            The path to export results into.
        linear_encoder: dict
            A dictionary mapping the true labels to the given labels.
            True labels = the labels in the mappings file.
            Given labels = labels ranging from 0 to len(true labels), which the
            true labels have been converted into.
        learning_rate: float, default 1e-3
            The initial learning rate.
        parcel_loss: boolean, default False
            If True, then a custom loss function is used which takes into account
            only the pixels of the parcels. If False, then all image pixels are
            used in the loss function.
        class_weights: dict, default None
            Weights per class to use in the loss function.
        crop_encoding: dict, default None
            A dictionary mapping class ids to class names.
        checkpoint_epoch: int, default None
            The epoch loaded for testing.
        '''
        super(ConvLSTM, self).__init__()

        self.linear_encoder = linear_encoder
        self.parcel_loss = parcel_loss

        self.epoch_train_losses = []
        self.epoch_valid_losses = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.best_loss = None

        num_discrete_labels = len(set(linear_encoder.values()))
        self.confusion_matrix = torch.zeros([num_discrete_labels, num_discrete_labels])

        self.class_weights = class_weights
        self.checkpoint_epoch = checkpoint_epoch

        if class_weights is not None:
            class_weights_tensor = torch.tensor([class_weights[k] for k in sorted(class_weights.keys())]).cuda()

            if self.parcel_loss:
                self.lossfunction = nn.NLLLoss(ignore_index=0, weight=class_weights_tensor, reduction='sum')
            else:
                self.lossfunction = nn.NLLLoss(ignore_index=0, weight=class_weights_tensor)
        else:
            if self.parcel_loss:
                self.lossfunction = nn.NLLLoss(ignore_index=0, reduction='sum')
            else:
                self.lossfunction = nn.NLLLoss(ignore_index=0)

        self.crop_encoding = crop_encoding

        self.run_path = Path(run_path)

        # Encoder
        # -------
        self.conv_1e = nn.Conv2d(in_channels=4,
                                out_channels=16,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.leakyrelu_1e = nn.LeakyReLU(negative_slope=0.2)
        self.clstm_1e = CLSTM_cell(shape=(61, 61), input_channels=16, filter_size=5, num_features=64)

        self.conv_2e = nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.leakyrelu_2e = nn.LeakyReLU(negative_slope=0.2)
        self.clstm_2e = CLSTM_cell(shape=(31, 31), input_channels=64, filter_size=5, num_features=96)

        self.conv_3e = nn.Conv2d(in_channels=96,
                                out_channels=96,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.leakyrelu_3e = nn.LeakyReLU(negative_slope=0.2)
        self.clstm_3e = CLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96)

        # Decoder
        # -------
        self.clstm_1d = CLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96)
        self.transconv_1d = nn.ConvTranspose2d(in_channels=96,
                                             out_channels=96,
                                             kernel_size=4, # 3
                                             stride=2,
                                             padding=3)  # 2
        self.leakyrelu_1d = nn.LeakyReLU(negative_slope=0.2)

        self.clstm_2d = CLSTM_cell(shape=(31, 31), input_channels=96, filter_size=5, num_features=96)
        self.transconv_2d = nn.ConvTranspose2d(in_channels=96,
                                             out_channels=96,
                                             kernel_size=4,
                                             stride=2,
                                             padding=3)
        self.leakyrelu_2d = nn.LeakyReLU(negative_slope=0.2)

        self.clstm_3d = CLSTM_cell(shape=(61, 61), input_channels=96, filter_size=5, num_features=64)
        self.transconv_3d = nn.ConvTranspose2d(in_channels=64,
                                             out_channels=16,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)
        self.leakyrelu_3d = nn.LeakyReLU(negative_slope=0.2)

        self.transconv_4d = nn.ConvTranspose2d(in_channels=16,
                                             out_channels=num_discrete_labels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)
        self.softmax = nn.LogSoftmax(dim=1)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.save_hyperparameters()


    def forward(self, input):
        input = input.transpose(0, 1)

        hidden_states = []

        # Encoder
        num_timesteps, batch_size, input_channels, height, width = input.size()
        input = torch.reshape(input, (-1, input_channels, height, width))
        x = self.leakyrelu_1e(self.conv_1e(input))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))
        x, hidden_state = self.clstm_1e(x, None, num_timesteps)
        hidden_states.append(hidden_state)

        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_2e(self.conv_2e(x))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))
        x, hidden_state = self.clstm_2e(x, None, num_timesteps)
        hidden_states.append(hidden_state)

        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_3e(self.conv_3e(x))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))
        x, hidden_state = self.clstm_3e(x, None, num_timesteps)
        hidden_states.append(hidden_state)

        # Decoder
        x, hidden_state = self.clstm_1d(None, hidden_states[-1], x.size(0))
        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_1d(self.transconv_1d(x))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))

        x, hidden_state = self.clstm_2d(None, hidden_states[-2], x.size(0))
        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_2d(self.transconv_2d(x))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))

        x, hidden_state = self.clstm_3d(None, hidden_states[-3], x.size(0))
        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_3d(self.transconv_3d(x))

        x = self.transconv_4d(x)
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))

        # Keep only the last output for an N-to-1 scheme
        x = x[-1, :, :, :, :]  # (T, B, K, H, W) -> (B, K, H, W)
        x = self.softmax(x)

        return x


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        pla_lr_scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.5,
                                                        patience=4,
                                                        verbose=True),
            'monitor': 'val_loss'
        }
        return [optimizer], [pla_lr_scheduler]


    def training_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, T, C, H, W)

        label = batch['labels']  # (B, H, W)
        label = label.to(torch.long)

        pred = self(inputs)  # (B, K, H, W)

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            # Note: a new masked array must be created in order to avoid inplace
            # operations on the label/pred variables. Otherwise the optimizer
            # will throw an error because it requires the variables to be unchanged
            # for gradient computation

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)

            label_masked = label.clone()
            label_masked[~mask] = 0

            pred_masked = pred.clone()
            pred_masked[~mask_K] = 0

            label = label_masked.clone()
            pred = pred_masked.clone()

            loss = self.lossfunction(pred, label)

            loss = loss / parcels.sum()
        else:
            loss = self.lossfunction(pred, label)

        # Compute total loss for current batch
        loss_aver = loss.item() * inputs.shape[0]

        self.epoch_train_losses.append(loss_aver)

        # torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=10.0)

        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, T, C, H, W)

        label = batch['labels']  # (B, H, W)
        label = label.to(torch.long)

        pred = self(inputs)  # (B, K, H, W)

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            # Note: a new masked array must be created in order to avoid inplace
            # operations on the label/pred variables. Otherwise the optimizer
            # will throw an error because it requires the variables to be unchanged
            # for gradient computation

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)

            label_masked = label.clone()
            label_masked[~mask] = 0

            pred_masked = pred.clone()
            pred_masked[~mask_K] = 0

            label = label_masked.clone()
            pred = pred_masked.clone()

            loss = self.lossfunction(pred, label)

            loss = loss / parcels.sum()
        else:
            loss = self.lossfunction(pred, label)

        # Compute total loss for current batch
        loss_aver = loss.item() * inputs.shape[0]

        self.epoch_valid_losses.append(loss_aver)

        return {'val_loss': loss}


    def test_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, T, C, H, W)

        label = batch['labels'].to(torch.long)  # (B, H, W)

        pred = self(inputs).to(torch.long)  # (B, K, H, W)

        # Reverse the logarithm of the LogSoftmax activation
        pred = torch.exp(pred)

        # Clip predictions larger than the maximum possible label
        pred = torch.clamp(pred, 0, max(self.linear_encoder.values()))

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)
            label[~mask] = 0
            pred[~mask_K] = 0

            pred_sparse = pred.argmax(axis=1)

            label = label.flatten()
            pred = pred_sparse.flatten()

            # Discretize predictions
            #bins = np.arange(-0.5, sorted(list(self.linear_encoder.values()))[-1] + 0.5, 1)
            #bins_idx = torch.bucketize(pred, torch.tensor(bins).cuda())
            #pred_disc = bins_idx - 1

        for i in range(label.shape[0]):
            self.confusion_matrix[label[i], pred[i]] += 1

        return


    def training_epoch_end(self, outputs):
        # Calculate average loss over an epoch
        train_loss = np.nanmean(self.epoch_train_losses)
        self.avg_train_losses.append(train_loss)

        with open(self.run_path / "avg_train_losses.txt", 'a') as f:
            f.write(f'{self.current_epoch}: {train_loss}\n')

        with open(self.run_path / 'lrs.txt', 'a') as f:
            f.write(f'{self.current_epoch}: {self.learning_rate}\n')

        self.log('train_loss', train_loss, prog_bar=True)

        # Clear list to track next epoch
        self.epoch_train_losses = []


    def validation_epoch_end(self, outputs):
        # Calculate average loss over an epoch
        valid_loss = np.nanmean(self.epoch_valid_losses)
        self.avg_val_losses.append(valid_loss)

        with open(self.run_path / "avg_val_losses.txt", 'a') as f:
            f.write(f'{self.current_epoch}: {valid_loss}\n')

        self.log('val_loss', valid_loss, prog_bar=True)

        # Clear list to track next epoch
        self.epoch_valid_losses = []


    def test_epoch_end(self, outputs):
        self.confusion_matrix = self.confusion_matrix.cpu().detach().numpy()

        self.confusion_matrix = self.confusion_matrix[1:, 1:]  # Drop zero label

        # Calculate metrics and confusion matrix
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tp = np.diag(self.confusion_matrix)
        tn = self.confusion_matrix.sum() - (fp + fn + tp)

        # Sensitivity, hit rate, recall, or true positive rate
        tpr = tp / (tp + fn)
        # Specificity or true negative rate
        tnr = tn / (tn + fp)
        # Precision or positive predictive value
        ppv = tp / (tp + fp)
        # Negative predictive value
        npv = tn / (tn + fn)
        # Fall out or false positive rate
        fpr = fp / (fp + tn)
        # False negative rate
        fnr = fn / (tp + fn)
        # False discovery rate
        fdr = fp / (tp + fp)
        # F1-score
        f1 = (2 * ppv * tpr) / (ppv + tpr)

        # Overall accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        # Export metrics in text file
        metrics_file = self.run_path / f"evaluation_metrics_epoch{self.checkpoint_epoch}.csv"

        # Delete file if present
        metrics_file.unlink(missing_ok=True)

        with open(metrics_file, "a") as f:
            row = 'Class'
            for k in sorted(self.linear_encoder.keys()):
                if k == 0: continue
                row += f',{k} ({self.crop_encoding[k]})'
            f.write(row + '\n')

            row = 'tn'
            for i in tn:
                row += f',{i}'
            f.write(row + '\n')

            row = 'tp'
            for i in tp:
                row += f',{i}'
            f.write(row + '\n')

            row = 'fn'
            for i in fn:
                row += f',{i}'
            f.write(row + '\n')

            row = 'fp'
            for i in fp:
                row += f',{i}'
            f.write(row + '\n')

            row = "specificity"
            for i in tnr:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "precision"
            for i in ppv:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "recall"
            for i in tpr:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "accuracy"
            for i in accuracy:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "f1"
            for i in f1:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = 'weighted macro-f1'
            class_samples = self.confusion_matrix.sum(axis=1)
            weighted_f1 = ((f1 * class_samples) / class_samples.sum()).sum()
            f.write(row + f',{weighted_f1:.4f}\n')

        # Normalize each row of the confusion matrix because class imbalance is
        # high and visualization is difficult
        row_mins = self.confusion_matrix.min(axis=1)
        row_maxs = self.confusion_matrix.max(axis=1)
        cm_norm = (self.confusion_matrix - row_mins[:, None]) / (row_maxs[:, None] - row_mins[:, None])

        # Export Confusion Matrix

        # Replace invalid values with 0
        self.confusion_matrix = np.nan_to_num(self.confusion_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(self.confusion_matrix, annot=False, ax=ax, cmap="Blues", fmt="g")

        # Labels, title and ticks
        label_font = {'size':'18'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='vertical')
        ax.set_yticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size':'21'}
        ax.set_title('Confusion Matrix', fontdict=title_font)

        for i in range(len(self.linear_encoder.keys()) - 1):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(self.run_path / f'confusion_matrix_epoch{self.checkpoint_epoch}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)

        np.save(self.run_path / f'cm_epoch{self.checkpoint_epoch}.npy', self.confusion_matrix)


        # Export normalized Confusion Matrix

        # Replace invalid values with 0
        cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(cm_norm, annot=False, ax=ax, cmap="Blues", fmt="g")

        # Labels, title and ticks
        label_font = {'size':'18'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='vertical')
        ax.set_yticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size':'21'}
        ax.set_title('Confusion Matrix', fontdict=title_font)

        for i in range(len(self.linear_encoder.keys()) - 1):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(self.run_path / f'confusion_matrix_norm_epoch{self.checkpoint_epoch}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)

        np.save(self.run_path / f'cm_norm_epoch{self.checkpoint_epoch}.npy', self.confusion_matrix)
        pickle.dump(self.linear_encoder, open(self.run_path / f'linear_encoder_epoch{self.checkpoint_epoch}.pkl', 'wb'))
