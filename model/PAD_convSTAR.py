# Implementation of the convSTAR model.
# Code taken from:
#   https://github.com/0zgur0/ms-convSTAR

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.nn import init
from tensorboardX import SummaryWriter
import pytorch_lightning as pl


class ConvSTARCell(nn.Module):
    """
    Generate a convolutional STAR cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.update.weight)
        init.orthogonal(self.gate.weight)
        init.constant(self.update.bias, 0.)
        init.constant(self.gate.bias, 1.)

        print('convSTAR cell is constructed with h_dim: ', hidden_size)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        gain = torch.sigmoid( self.gate(stacked_inputs) )
        update = torch.tanh( self.update(input_) )
        new_state = gain * prev_state + (1-gain) * update

        return new_state


class ConvSTAR(pl.LightningModule):

    def __init__(self, run_path, linear_encoder, learning_rate=1e-3, parcel_loss=False,
                 class_weights=None, crop_encoding=None, checkpoint_epoch=None,
                 input_size=4, n_layers=3, kernel_sizes=3, hidden_sizes=64):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
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
        input_size: int, default 4
            Depth dimension of input tensors.
        hidden_sizes: int or list, default 64
            Depth dimensions of hidden state. If integer, the same hidden size is used for all cells.
        kernel_sizes: int or list, default 3
            Sizes of Conv2d gate kernels. If integer, the same kernel size is used for all cells.
        n_layers: int, default 3
            Number of chained `ConvSTARCell`.
        '''

        super(ConvSTAR, self).__init__()

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

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvSTARCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvSTARCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

        self.conv2d = nn.Conv2d(in_channels=self.hidden_sizes[-1],
                                out_channels=num_discrete_labels,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.softmax = nn.LogSoftmax(dim=1)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.save_hyperparameters()


    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        The prediction of the last hidden layer.
        '''
        if not hidden:
            hidden = [None] * self.n_layers

        x = x.transpose(1, 2)   # (B, T, C, H, W) -> (B, C, T, H, W)

        batch_size, channels, timesteps, height, width = x.size()

        # retain tensors in list to allow different hidden sizes
        upd_hidden = []

        for timestep in range(timesteps):
            input_ = x[:, :, timestep, :, :]

            for layer_idx in range(self.n_layers):
                cell = self.cells[layer_idx]
                cell_hidden = hidden[layer_idx]

                if layer_idx == 0:
                    upd_cell_hidden = cell(input_, cell_hidden)
                else:
                    upd_cell_hidden = cell(upd_hidden[-1], cell_hidden)
                upd_hidden.append(upd_cell_hidden)
                # update input_ to the last updated hidden layer for next pass
                hidden[layer_idx] = upd_cell_hidden

        # Keep only the last output for an N-to-1 scheme
        x = hidden[-1]   # (L, B, C, H, W) -> (B, C, H, W)
        x = self.softmax(self.conv2d(x))   # (B, K, H, W)

        return x


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        step_lr_scheduler = {
            'scheduler': lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
            'monitor': 'val_loss'
        }
        return [optimizer], [step_lr_scheduler]


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
            for i in sorted(self.linear_encoder.keys()):
                if i == 0: continue
                row += f',{i}'
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


class ConvSTAR_Res(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvSTARCell`.
        '''

        super(ConvSTAR_Res, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            elif i == 2 or i==4:
                input_dim = self.hidden_sizes[i - 1] + self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvSTARCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvSTARCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if not hidden:
            hidden = [None] * self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            if layer_idx==1 or layer_idx==3:
                input_ = torch.cat((upd_cell_hidden, x), 1)
            else:
                input_ = upd_cell_hidden


        # retain tensors in list to allow different hidden sizes
        return upd_hidden
