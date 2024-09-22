import configargparse
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from argparse import Namespace
from utils.utils import argparse_summary, get_class_by_path
import json
from datetime import datetime
import gc
import os
import time
import torch
import numpy as np
import random
import pytorch_lightning as pl

# Disable warnings from the logging module
logging.disable(logging.WARNING)

# Set seeds for reproducibility
def set_seed(seed):
    """
    Set random seeds for reproducibility in Python, NumPy, and PyTorch.
    Also ensure deterministic operations for PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)

# Main training function
def train(hparams, ModuleClass, ModelClass, DatasetClass, logger):
    """
    Main training routine specific for this project.

    Parameters:
        hparams (Namespace): Hyperparameters and configurations for training.
        ModuleClass (Class): PyTorch Lightning Module for training.
        ModelClass (Class): Model class to be trained.
        DatasetClass (Class): Dataset class to load data.
        logger (List): List of loggers for tracking training progress.
    """
    # Initialize the model, dataset, and training module
    model = ModelClass(hparams=hparams)
    dataset = DatasetClass(hparams=hparams)
    module = ModuleClass(hparams, model, dataset)

    # Configure model checkpoint callback to save the best model based on a specified metric
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.early_stopping_metric,
        mode='max',
        filename=f'{{epoch}}-{{{hparams.early_stopping_metric}:.2f}}'
    )

    # Early stopping callback to stop training when performance on validation plateaus
    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stopping_metric,
        min_delta=0.00,
        patience=3,
        mode='max'
    )

    # Initialize the trainer with relevant settings
    trainer = Trainer(
        devices=1,
        accelerator='gpu',
        logger=logger,
        fast_dev_run=hparams.fast_dev_run,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        log_every_n_steps=hparams.log_every_n_steps,
        deterministic=True
    )

    # Start training and resume from checkpoint if specified
    trainer.fit(module, ckpt_path=hparams.resume_from_checkpoint)

    # Print best model path and score
    print(f"Best: {checkpoint_callback.best_model_score} | "
          f"Monitor: {checkpoint_callback.monitor} | "
          f"Path: {checkpoint_callback.best_model_path}\nTesting...")

    # Test the model using the best checkpoint and validate the model
    test_results = trainer.test(ckpt_path=checkpoint_callback.best_model_path)
    val_results = trainer.validate(ckpt_path=checkpoint_callback.best_model_path)

    # Save test results to a JSON file
    path_name = str(hparams.output_path)
    fold_name = f'fold{hparams.fold}'
    path_name = path_name.replace(fold_name + '_', '')
    txt_file_dir = Path(path_name + '.json')

    # Load existing data if the JSON file exists, otherwise start with an empty dictionary
    if txt_file_dir.exists():
        try:
            with txt_file_dir.open('r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from {txt_file_dir}, starting with an empty dictionary.")
            data = {}
    else:
        data = {}

    # Update the data with the current fold's test results and save it to the JSON file
    data[fold_name] = test_results[0]
    txt_file_dir.parent.mkdir(parents=True, exist_ok=True)

    with txt_file_dir.open('w') as file:
        json.dump(data, file, indent=4)

    print(f"Results saved to {txt_file_dir}")

# Main execution block
if __name__ == "__main__":
    # Initialize parser for configuration files and command-line arguments
    root_dir = Path(__file__).parent
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    # Set seed for reproducibility
    seed = 42
    set_seed(seed)

    # Dynamically load the Module, Model, and Dataset classes based on the configuration
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)

    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)

    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)

    # Parse the arguments and setup experiment name and logging
    hparams = parser.parse_args()
    exp_name = f"{hparams.module.split('.')[-1]}_{hparams.dataset.split('.')[-1]}_{hparams.model.replace('.', '_')}"
    date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
    hparams.name = f"fold{hparams.fold}_{hparams.exp_name}_{exp_name}"

    # Define the output path and setup loggers (TensorBoard and optionally WandB)
    hparams.output_path = Path(hparams.output_path).absolute() /exp_name
    tb_logger = TensorBoardLogger(hparams.output_path, name='tb')
    wandb_logger = WandbLogger(name=hparams.name, project="tecno")

    # Log experiment summary and run training
    argparse_summary(hparams, parser)
    loggers = [tb_logger]
    train(hparams, ModuleClass, ModelClass, DatasetClass, loggers)

    # Clean up and free resources
    del hparams, parser, ModuleClass, ModelClass, DatasetClass, tb_logger, wandb_logger
    torch.cuda.empty_cache()
    gc.collect()