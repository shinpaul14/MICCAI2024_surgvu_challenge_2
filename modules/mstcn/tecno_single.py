

import logging
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from utils.metric_helper import AccuracyStages, RecallOverClasse, PrecisionOverClasses
from torch import nn
import numpy as np
from argparse import Namespace
from pathlib import Path
from typing import Tuple
import torchmetrics
import json

class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(TeCNO, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.dataset = dataset
        # self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss = FocalLoss()
        self.init_metrics()
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.test_step_outputs = []

    def init_metrics(self):
        self.train_f1 = torchmetrics.F1Score(num_classes=self.hparams.out_features, task="multiclass", average='weighted')
        self.val_f1 = torchmetrics.F1Score(num_classes=self.hparams.out_features, task="multiclass", average='weighted')
        self.test_f1 = torchmetrics.F1Score(num_classes=self.hparams.out_features, task="multiclass", average='weighted')

    def forward(self, x):
        video_fe = x.transpose(2, 1)
        step_classes = self.model.forward(video_fe)
        return step_classes

    def _check_same_shape(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.shape != target.shape:
            raise RuntimeError("Predictions and targets are expected to have the same shape")

    def _input_format_classification(self, preds: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (preds.ndim == target.ndim or preds.ndim == target.ndim + 1):
            raise ValueError("preds and target must have the same number of dimensions, or one additional dimension for preds")

        if preds.ndim == target.ndim + 1:
            preds = torch.argmax(preds, dim=1)

        if preds.ndim == target.ndim and preds.is_floating_point():
            preds = (preds >= threshold).long()

        self._check_same_shape(preds, target)
        return preds, target

    def loss_function(self, y_classes, labels):
        stages = y_classes.shape[0]
        clc_loss = 0
        for j in range(stages):
            p_classes = y_classes[j].squeeze().transpose(1, 0)
            ce_loss = self.ce_loss(p_classes, labels.squeeze().long())
            clc_loss += ce_loss
        clc_loss = clc_loss / (stages * 1.0)
        return clc_loss

    def calc_f1(self, y_pred, y_true, step="val"):
        y_max_pred, y_true = self._input_format_classification(y_pred, y_true, threshold=0.5)
        if step == "train":
            f1 = self.train_f1(y_max_pred, y_true)
        elif step == "val":
            f1 = self.val_f1(y_max_pred, y_true)
        else:
            f1 = self.test_f1(y_max_pred, y_true)
        return f1

    def log_average_f1(self, outputs, step="val"):
        for s in range(self.hparams.mstcn_stages):
            f1_list = [o[f"{step}_S{s+1}_f1"] for o in outputs if f"{step}_S{s+1}_f1" in o]
            if not f1_list:
                print("F1 list is empty.")
                return

            try:
                z = torch.stack(f1_list)
            except RuntimeError as e:
                print("Error stacking tensors:", e)
                return

            if z.ndim == 1:
                phase_avg_f1 = torch.mean(z[~z.isnan()])
            elif z.ndim == 2:
                phase_avg_f1 = [torch.mean(z[~z[:, n].isnan(), n]) for n in range(z.shape[1])]
                phase_avg_f1 = torch.stack(phase_avg_f1)
            else:
                return

            phase_avg_f1_over_video = phase_avg_f1[~phase_avg_f1.isnan()].mean()
            self.log(f"{step}_avg_S{s+1}_f1", phase_avg_f1_over_video, on_epoch=True, on_step=False)

    def training_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        step_pred = self.forward(stem)
        loss = self.loss_function(step_pred, y_true)
        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        step_pred = torch.softmax(step_pred, dim=2)
        f1_stages = []

        for s in range(self.hparams.mstcn_stages):
            f1 = self.calc_f1(step_pred[s], y_true, step="train")
            f1_stages.append(f1)

        metric_dict_f1 = {f"train_S{s + 1}_f1": f1_stages[s] for s in range(len(f1_stages))}

        self.log_dict(metric_dict_f1, on_epoch=True, on_step=False)
        metric_dict = {"loss": loss}
        metric_dict.update(metric_dict_f1)
        self.train_step_outputs.append(metric_dict)
        return metric_dict

    def on_train_epoch_end(self):
        outputs = self.train_step_outputs
        self.log_average_f1(outputs, step="train")

    def validation_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        step_pred = self.forward(stem)
        loss = self.loss_function(step_pred, y_true)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

        step_pred = torch.softmax(step_pred, dim=2)
        f1_stages = []

        for s in range(self.hparams.mstcn_stages):
            f1 = self.calc_f1(step_pred[s], y_true, step="val")
            f1_stages.append(f1)

        metric_dict_f1 = {f"val_S{s + 1}_f1": f1_stages[s] for s in range(len(f1_stages))}
        self.log_dict(metric_dict_f1, on_epoch=True, on_step=False)
        metric_dict = {"val_loss": loss}
        metric_dict.update(metric_dict_f1)
        self.validation_step_outputs.append(metric_dict)
        return metric_dict

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self.log_average_f1(outputs, step="val")

    def test_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        step_pred = self.forward(stem)
        val_loss = self.loss_function(step_pred, y_true)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False)

        step_pred = torch.softmax(step_pred, dim=2)
        f1_stages = []

        for s in range(self.hparams.mstcn_stages):
            f1 = self.calc_f1(step_pred[s], y_true, step="test")
            f1_stages.append(f1)

        metric_dict_f1 = {f"test_S{s + 1}_f1": f1_stages[s] for s in range(len(f1_stages))}
        self.log_dict(metric_dict_f1, on_epoch=True, on_step=False)
        metric_dict = {"val_loss": val_loss}
        metric_dict.update(metric_dict_f1)
        self.test_step_outputs.append(metric_dict)
        return metric_dict

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self.log_average_f1(outputs, step="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        should_shuffle = False
        if split == "train":
            should_shuffle = True
        train_sampler = None
        if self.trainer.strategy and self.trainer.strategy.launcher:
            train_sampler = DistributedSampler(dataset)
            should_shuffle = False
        
        persistent_workers = split != "test"  # Set persistent_workers=True for train/val, False for test
        
        print(f"split: {split} - shuffle: {should_shuffle}")
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers  # Use persistent workers
        )
        return loader


    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(len(dataloader.dataset)))
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called - size: {}".format(len(dataloader.dataset)))
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):
        regressiontcn = parser.add_argument_group(
            title='regression tcn specific args options')
        regressiontcn.add_argument("--learning_rate", default=0.001, type=float)
        regressiontcn.add_argument("--optimizer_name", default="adam", type=str)
        regressiontcn.add_argument("--batch_size", default=1, type=int)
        return parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting easy examples
    and focusing on harder ones. Supports both binary and multi-class classification.

    Args:
        gamma (float): The focusing parameter \gamma. Higher values focus more on hard examples.
        alpha (Union[None, Tensor]): Class weights for addressing class imbalance.
                                     Tensor of size (num_classes) or None. Optional.
        reduction (str): Specifies reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient. Optional.
        eps (float): Smoothing factor to prevent numerical instability when taking the log.
    """
    def __init__(self, gamma: float = 2.0, alpha: Union[None, Tensor] = None, reduction: str = 'mean', ignore_index: int = -100, eps: float = 1e-8) -> None:
        super(FocalLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum'], f"Invalid reduction mode: {reduction}. Choose from 'none', 'mean', or 'sum'."
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.eps = eps

        # Convert alpha to a Tensor if it's a list or float
        if isinstance(alpha, (float, int)):  # For binary classification, can pass a single float as alpha
            self.alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Reshape inputs if necessary (for multi-class tasks with dimension > 2)
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(-1))  # N,H*W,C => N*H*W,C
        else:
            inputs = inputs.view(-1, inputs.size(-1))  # Binary classification

        targets = targets.view(-1)

        # Ignore specified indices (like padding or irrelevant classes)
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            targets = targets[valid_mask]
            inputs = inputs[valid_mask]
        
        # Compute log-softmax for stability
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        # Gather probabilities corresponding to the correct class
        logpt = log_probs.gather(1, targets.unsqueeze(1)).view(-1)
        pt = logpt.exp()

        # Apply the focal loss modulation
        focal_term = (1 - pt) ** self.gamma

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            logpt = logpt * at

        # Final focal loss computation
        loss = -focal_term * logpt

        # Reduction (mean, sum, or none)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss