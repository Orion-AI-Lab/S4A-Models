import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x


class OAD_Transformer(pl.LightningModule):

    def __init__(
            self,
            d_model: int = 512,
            num_layers: int = 3,
            dim_feedforward: int = 2048,
            nhead: int = 8,
            dropout: float = 0.1,
            batch_first: bool = True,
            norm_first: bool = False,
            num_classes: int = 14,
            lr: float = 1e-3,
            name_decoder: dict = None,
            id_decoder: dict = None,
            batch_size: int = 4098
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.d_model = d_model
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.nhead = nhead
        self.num_classes = num_classes
        self.name_decoder = name_decoder
        self.id_decoder = id_decoder
        self.batch_size = batch_size

        encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=self.batch_first
            )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers
        )

        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model, dropout=self.dropout, max_len=30
        )

        self.fc = nn.Linear(self.d_model, self.num_classes)

        self.relu = nn.ReLU()

        # Each cost function is modeled through categorical cross entropy
        self.criterion = nn.CrossEntropyLoss()

        metrics = MetricCollection({
            'accuracy': Accuracy(),
            'precision': Precision(num_classes=self.num_classes),
            'precision_weighted': Precision(num_classes=self.num_classes, average='weighted'),
            'recall': Recall(num_classes=self.num_classes),
            'recall_weighted': Recall(num_classes=self.num_classes, average='weighted'),
            'f1-score:': F1(num_classes=self.num_classes),
            'f1-score_weighted:': F1(num_classes=self.num_classes, average='weighted')
        })

        # Only accuracy for training
        self.train_acc = Accuracy()
        self.val_confusion_matrix = ConfusionMatrix(num_classes=num_classes, compute_on_step=False, normalize='true')
        self.test_confusion_matrix = ConfusionMatrix(num_classes=num_classes, compute_on_step=False, normalize='true')

        self.metrics_val = metrics.clone(prefix='val_')
        self.metrics_test = metrics.clone(prefix='test_')

    def forward(self, src: Tensor) -> Tensor:
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        x = self.transformer_encoder(src)
        # Take the mean of the encoded space
        x = x.mean(axis=1)

        # Maybe a relu?!
        x = self.relu(x)

        x = self.fc(x)

        # Final Classification
        #x = self.fc2(x)

        # Normalize
        # x = F.log_softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        data, target = batch

        # Forward
        preds = self(data)

        # Calculate loss
        loss = self.criterion(preds, target)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(preds, target), on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        # Forward
        preds = self(data)

        # Calculate loss
        loss = self.criterion(preds, target)
        self.log('val_loss', loss)

        self.val_confusion_matrix(preds, target)

        self.metrics_val(preds, target)
        self.log_dict(self.metrics_val, on_step=False, on_epoch=True)

        return {'loss': loss, 'preds': preds, 'target': target}

    def test_step(self, batch, batch_idx):
        data, target = batch

        # Forward
        preds = self(data)

        # Calculate loss
        loss = self.criterion(preds, target)
        self.log('test_loss', loss)

        self.test_confusion_matrix(preds, target)

        self.metrics_test(preds, target)
        self.log_dict(self.metrics_test, on_step=False, on_epoch=True)

        return {'loss': loss, 'preds': preds, 'target': target}

    def training_step_end(self, outputs) -> None:
        return None

    def validation_step_end(self, outputs) -> None:
        return None

    def test_step_end(self, outputs) -> None:
        return None

    def training_epoch_end(self, outputs):
        return None

    def validation_epoch_end(self, outputs):
        sns.set(font_scale=1.25)
        confusion_matrix = self.val_confusion_matrix.compute().cpu().numpy()

        indices = [f'{self.name_decoder[str(i)]} ({self.id_decoder[str(i)]})' for i in range(self.num_classes)]

        df_cm = pd.DataFrame(confusion_matrix, index=indices, columns=indices)

        plt.figure(figsize=(15, 15))
        fig = sns.heatmap(df_cm, annot=False, cmap='rocket_r', fmt='.3').get_figure()
        plt.close(fig)

        self.logger.experiment.add_figure('val_confusion_matrix', fig, self.current_epoch)
        self.val_confusion_matrix.reset()

    def test_epoch_end(self, outputs):
        sns.set(font_scale=1.25)
        confusion_matrix = self.test_confusion_matrix.compute().cpu().numpy()

        indices = [f'{self.name_decoder[str(i)]} ({self.id_decoder[str(i)]})' for i in range(self.num_classes)]

        df_cm = pd.DataFrame(confusion_matrix, index=indices, columns=indices)
        plt.figure(figsize=(15, 15))
        fig = sns.heatmap(df_cm, annot=False, cmap='rocket_r', fmt='.3').get_figure()
        plt.close(fig)

        self.logger.experiment.add_figure('test_confusion_matrix', fig, self.current_epoch)
        self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.batch_size * 100 * 5, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
