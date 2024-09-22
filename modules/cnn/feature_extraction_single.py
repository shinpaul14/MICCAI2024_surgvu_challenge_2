import torchmetrics
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import logging
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pycm import ConfusionMatrix
import numpy as np
import pickle
import torchmetrics
from argparse import Namespace
from pathlib import Path

class FeatureExtraction(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(FeatureExtraction, self).__init__()
        self.save_hyperparameters(hparams)

        # Model and Dataset
        self.model = model
        self.dataset = dataset
        self.batch_size = self.hparams.batch_size

        # Initialize loss function with class weights
        weight = torch.from_numpy(self.dataset.class_weights_task).float()
        self.ce_loss = FocalLoss() #nn.CrossEntropyLoss(weight=weight) if self.hparams.weight else nn.CrossEntropyLoss()
        self.len_test_data = len(self.dataset.data["test"])
        # print('self.len_test_data:', self.len_test_data)
        # assert print('stop')
        self.current_video_idx = self.dataset.df["test"].video_idx.min()

        # Initialize metrics for F1-Score (weighted average)
        self.init_metrics()

        # Initialize storage for outputs and buffers for each video
        self.test_outputs = []
        self.current_stems = []
        self.current_step_labels = []
        self.current_p_steps = []

    def init_metrics(self):
        # Initialize Mean Weighted F1-Score for multiclass tasks
        self.train_f1 = torchmetrics.F1Score(num_classes=self.hparams.out_features, task="multiclass", average='weighted')
        self.val_f1 = torchmetrics.F1Score(num_classes=self.hparams.out_features, task="multiclass", average='weighted')
        self.test_f1 = torchmetrics.F1Score(num_classes=self.hparams.out_features, task="multiclass", average='weighted')

    def forward(self, x):
        # Forward pass through the model
        stem, step = self.model.forward(x)
        return stem, step

    def loss_step_task(self, p_step, labels_step):
        # Cross-entropy loss for the step task
        return self.ce_loss(p_step, labels_step)

    def training_step(self, batch, batch_idx):
        # Training step
        x, y_step, _ = batch
        _, p_step = self.forward(x)
        loss = self.loss_step_task(p_step, y_step)
        
        # Update Mean Weighted F1-Score for training
        self.train_f1.update(p_step, y_step)

        # Log training loss
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        
        return loss

    def on_train_epoch_end(self):
        # Compute and log Mean Weighted F1-Score for training
        train_f1 = self.train_f1.compute()
        self.log("train_f1", train_f1, on_epoch=True, on_step=False)

        # Reset the metric
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        # Validation step
        x, y_step, _ = batch
        _, p_step = self.forward(x)

        # Debug: Check for NaN in logits or labels
        if torch.isnan(y_step).any():
            print("NaN detected in labels (y_step)")
        if torch.isnan(p_step).any():
            print("NaN detected in model output (p_step)")

        loss = self.loss_step_task(p_step, y_step)
        if torch.isnan(loss).any():
            print("NaN detected in loss)")
            return None  # Skip this batch if NaN is detected in loss

        # Update Mean Weighted F1-Score for validation
        self.val_f1.update(p_step, y_step)

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        # Compute and log Mean Weighted F1-Score for validation
        val_f1 = self.val_f1.compute()
        self.log("val_f1", val_f1, on_epoch=True, on_step=False)

        # Reset the metric
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        # Unpack the batch
        x, y_step, (vid_idx, img_name, img_index, y_task) = batch
        vid_idx_raw = vid_idx.cpu().numpy()  # Move video index to CPU for NumPy operations

        with torch.no_grad():
            stem, y_hat = self.forward(x)

        # Update Mean Weighted F1-Score for testing
        self.test_f1.update(y_hat, y_step)

        # Find unique video indices and corresponding ranges in the batch
        vid_idxs, indexes = np.unique(vid_idx_raw, return_index=True)
        vid_idxs = [int(v) for v in vid_idxs]  # Convert video indices to integers

        for i in range(len(vid_idxs)):
            vid_idx = vid_idxs[i]
            index = indexes[i]

            # Determine next index for this batch
            if i + 1 < len(indexes):
                index_next = indexes[i + 1]
            else:
                index_next = len(vid_idx_raw)  # End of the batch

            # Check if the current video index has changed
            if vid_idx != self.current_video_idx and self.current_video_idx is not None:
                # Save the current video’s predictions if the video changes
                if len(self.current_stems) > 0:
                    print(f"Saving video {self.current_video_idx} with {len(self.current_stems)} frames.")
                    self.save_to_drive(self.current_video_idx)
                else:
                    print(f"Skipping video {self.current_video_idx} because it has 0 frames.")

                # Reset buffers for the new video
                self.current_stems = []
                self.current_step_labels = []
                self.current_p_steps = []

            # Now update the current video index to the new video
            self.current_video_idx = vid_idx

            # Accumulate the predictions, stems, and labels for this video
            y_hat_slice = y_hat[index:index_next, :]
            self.current_p_steps.append(y_hat_slice)
            self.current_stems.append(stem[index:index_next, :])

            y_step_slice = y_step[index:index_next]
            y_step_slice = y_step_slice if isinstance(y_step_slice, torch.Tensor) else torch.tensor(y_step_slice)
            self.current_step_labels.append(y_step_slice)

            # print(f"Accumulated frames for video {vid_idx}: {len(self.current_stems)}")

        # Handle the final batch and ensure all video predictions are saved
        if (batch_idx + 1) * self.hparams.batch_size >= self.len_test_data:
            if len(self.current_stems) > 0:
                print(f"Final batch. Saving video {self.current_video_idx} with {len(self.current_stems)} frames.")
                self.save_to_drive(self.current_video_idx)
            else:
                print(f"Skipping final video {self.current_video_idx} because it has 0 frames.")
            print(f"Finished extracting all videos.")

        # Store the output for later use in on_test_epoch_end
        self.test_outputs.append({
            'y_hat': y_hat,
            'y_step': y_step,
            'vid_idx': vid_idx,
            'img_name': img_name,
            'img_index': img_index
        })

    def save_to_drive(self, vid_idx):
        if self.current_step_labels:  # Check if the list is not empty
            # Concatenate the lists of tensors into single tensors
            step_labels_tensor = torch.cat(self.current_step_labels)
            p_steps_tensor = torch.cat(self.current_p_steps)
            stems_tensor = torch.cat(self.current_stems)

            # Avoid overwriting by ensuring file is saved only once
            save_path = self.pickle_path / f"case_{int(vid_idx):03d}_predictions.pkl"
            if not save_path.exists():  # Only save if file does not exist
                print(f"Saving video {vid_idx} with {stems_tensor.shape[0]} frames to {save_path}")
                with open(save_path, 'wb') as f:
                    pickle.dump([
                        stems_tensor.cpu().numpy(),
                        p_steps_tensor.cpu().numpy(),
                        step_labels_tensor.cpu().numpy()
                    ], f)
                print(f'Saved predictions for video {vid_idx} to {save_path}')
            else:
                print(f"File for video {vid_idx} already exists, skipping save to avoid overwrite.")

    def on_test_epoch_end(self):
        # Compute and log Mean Weighted F1-Score for testing
        test_f1 = self.test_f1.compute()
        self.log("test_f1", test_f1, on_epoch=True, on_step=False)

        # # Final save for the last video (in case it wasn't saved yet)
        # self.save_to_drive(self.current_video_idx)  # Ensure the last video is saved fully

        # Reset the metric
        self.test_f1.reset()
    


    def configure_optimizers(self):
        # Configure the optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

    def set_export_pickle_path(self):
        # Set the path for saving pickle files
        self.pickle_path = Path(self.hparams.output_path) / "pickle_export"
        self.pickle_path.mkdir(parents=True, exist_ok=True)
        print(f"Setting export_pickle_path: {self.pickle_path}")

    def __dataloader(self, split=None):
        # Initialize dataloaders
        dataset = self.dataset.data[split]
        # Shuffle only for training split
        should_shuffle = True if split == 'train' else False
        
        # Set the number of workers (use the value from hparams for non-test sets)
        worker = self.hparams.num_workers if split != "test" else 0

        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            num_workers=worker,
            pin_memory=True,
            persistent_workers=(worker > 0)
        )
    # def __dataloader(self, split=None):
    #     dataset = self.dataset.data[split]
    #     should_shuffle = False
    #     if split == "train":
    #         should_shuffle = True
    #     else:
    #         should_shuffle = False

    #     train_sampler = None
     
    
        
    #     persistent_workers = split != "test"  # Set persistent_workers=True for train/val, False for test
        
    #     print(f"split: {split} - shuffle: {should_shuffle}")
    #     loader = DataLoader(
    #         dataset=dataset,
    #         batch_size=self.hparams.batch_size,
    #         shuffle=should_shuffle,
    #         sampler=train_sampler,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=True,
    #         persistent_workers=persistent_workers  # Use persistent workers
    #     )
    #     return loader

    def train_dataloader(self):
        # Initialize train dataloader
        return self.__dataloader(split="train")

    def val_dataloader(self):
        # Initialize validation dataloader
        return self.__dataloader(split="val")

    def test_dataloader(self):
        # Initialize test dataloader
        dataloader = self.__dataloader(split="test")
        print(f"starting video idx for testing: {self.current_video_idx}")
        self.set_export_pickle_path()
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):
        # Add module-specific arguments
        group = parser.add_argument_group(title='FeatureExtraction specific args')
        group.add_argument("--learning_rate", default=0.001, type=float)
        group.add_argument("--num_tasks", default=1, type=int, choices=[1, 2])
        group.add_argument("--batch_size", default=32, type=int)
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