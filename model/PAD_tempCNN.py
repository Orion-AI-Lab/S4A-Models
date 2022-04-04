'''
Implementation of the model proposed in:
- Charlotte Pelletier, et al. "Temporal Convolutional Neural Network for the
Classification of Satellite Image Time Series." (2018), arXiv:1811.10166.

Code adopted from:
https://github.com/MarcCoru/crop-type-mapping
'''

import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import pytorch_lightning as pl


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class TempCNN(pl.LightningModule):
    def __init__(self, input_dim, nclasses, sequence_length, run_path, linear_encoder,
                 learning_rate=0.001, kernel_size=5, hidden_dims=64, dropout=0.5,
                 weight_decay=1e-6, parcel_loss=False, class_weights=None):

        super(TempCNN, self).__init__()

        self.linear_encoder = linear_encoder
        self.parcel_loss = parcel_loss

        self.epoch_train_losses = []
        self.epoch_valid_losses = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.best_loss = None

        if not parcel_loss:
            if class_weights is None:
                self.lossfunction = nn.NLLLoss()
            else:
                self.lossfunction = nn.NLLLoss(weight=torch.as_tensor(class_weights.values()))
        else:
            if class_weights is None:
                self.lossfunction = nn.NLLLoss(reduction='sum')
            else:
                self.lossfunction = nn.NLLLoss(weight=torch.as_tensor(list(class_weights.values())), reduction='sum')

        self.class_weights = class_weights

        num_discrete_labels = len(set(linear_encoder.values()))
        self.confusion_matrix = torch.zeros([num_discrete_labels, num_discrete_labels])

        self.run_path = run_path

        self.hidden_dims = hidden_dims
        self.sequence_length = sequence_length

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size, drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout)
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims*sequence_length, 4*hidden_dims, drop_probability=dropout)
        self.logsoftmax = nn.Sequential(nn.Linear(4 * hidden_dims, nclasses), nn.LogSoftmax(dim=-1))

        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()


    def forward(self,x):
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.logsoftmax(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=self.weight_decay, lr=self.learning_rate)
        return [optimizer]


    def training_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, S, C, H, W)

        # Reshape inputs (B, S, C, 1, 1) -> (B, S, C)
        inputs = torch.squeeze(inputs)
        # (B, S, C) -> (B, C, S)
        inputs = torch.swapaxes(inputs, 1, 2)

        # Reshape label (B, H, W) -> (B, 1, H, W)
        label = batch['labels'][:, None, :, :]
        #label = torch.squeeze(label)

        pred = self(inputs)

        if self.parcel_loss:
            parcels = batch['parcels'][:, None, :, :]

            mask = (parcels) & (label != 0)
            label = label[mask]
            pred = pred[mask]

            loss = self.lossfunction(pred, label)
            loss = loss / parcels.sum()
        else:
            loss = self.lossfunction(pred, label)

        # Compute total loss for current batch
        loss_aver = loss.item() * inputs.shape[0]
        self.epoch_train_losses.append(loss_aver)

        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, S, C, H, W)

        # Reshape inputs (B, S, C, 1, 1) -> (B, S, C)
        inputs = torch.squeeze(inputs)
        # (B, S, C) -> (B, C, S)
        inputs = torch.swapaxes(inputs, 1, 2)

        # Reshape label (B, H, W) -> (B, 1, H, W)
        label = batch['labels'][:, None, :, :]
        label = torch.squeeze(label)

        pred = self(inputs)

        if self.parcel_loss:
            parcels = batch['parcels'][:, None, :, :]

            mask = (parcels) & (label != 0)
            label = label[mask]
            pred = pred[mask]

            loss = self.lossfunction(pred, label)
            loss = loss / parcels.sum()
        else:
            loss = self.lossfunction(pred, label)

        # Compute total loss for current batch
        loss_aver = loss.item() * inputs.shape[0]

        self.epoch_valid_losses.append(loss_aver)

        return {'val_loss': loss}


    def test_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, S, C, H, W)

        # Reshape inputs (B, S, C, 1, 1) -> (B, S, C)
        inputs = torch.squeeze(inputs)
        # (B, S, C) -> (B, C, S)
        inputs = torch.swapaxes(inputs, 1, 2)

        # Reshape label (B, H, W) -> (B, 1, H, W)
        label = batch['labels'][:, None, :, :]
        #label = torch.squeeze(label)
        label = label.flatten().to(torch.long)

        pred = self(inputs)
        pred = pred.flatten().to(torch.long)

        # Clip predictions larger than the maximum possible label
        pred = torch.clamp(pred, 0, max(self.linear_encoder.values()))

        if self.parcel_loss:
            parcels = batch['parcels'][:, None, :, :]
            parcels = parcels.flatten()

            mask = (parcels) & (label != 0)
            label = label[mask]
            pred = pred[mask]

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

        # Overall accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        # Export metrics in text file
        with open(self.run_path / "evaluation_metrics.txt", "a") as f:
            f.write(f'{"Class":<10}: ')
            for i in self.linear_encoder.keys():
                f.write(f'{i:>10}')
            f.write('\n')

            f.write(f'{"tn":<10}: ')
            for i in tn:
                f.write(f'{i:>10}')
            f.write('\n')

            f.write(f'{"tp":<10}: ')
            for i in tp:
                f.write(f'{i:>10}')
            f.write('\n')

            f.write(f'{"fn":<10}: ')
            for i in fn:
                f.write(f'{i:>10}')
            f.write('\n')

            f.write(f'{"fp":<10}: ')
            for i in fp:
                f.write(f'{i:>10}')
            f.write('\n')

            f.write(f'{"tpr":<10}: ')
            for i in tpr:
                f.write(f'{i:>10}')
            f.write('\n')

            f.write(f'{"tnr":<10}: ')
            for i in tnr:
                f.write(f'{i:>10}')
            f.write('\n')

            f.write(f'{"accuracy":<10}: ')
            for a in accuracy:
                f.write(f'{a:10.4f}')
            f.write('\n')

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(self.confusion_matrix, annot=False, ax=ax, cmap="Blues", fmt="g")

        # Labels, title and ticks
        label_font = {'size':'18'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([self.crop_encoding[k] for k in self.linear_encoder.keys()], fontsize=8, rotation='vertical')
        ax.set_yticklabels([self.crop_encoding[k] for k in self.linear_encoder.keys()], fontsize=8, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size':'21'}
        ax.set_title('Confusion Matrix', fontdict=title_font)

        for i in range(len(self.linear_encoder.keys())):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(self.run_path / 'confusion_matrix_seaborn.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)

        np.save(self.run_path / 'cm.npy', self.confusion_matrix)
        pickle.dump(self.linear_encoder, open(self.run_path / 'linear_encoder.pkl', 'wb'))
