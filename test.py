import configargparse
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from utils.utils import argparse_summary, get_class_by_path
import json
from datetime import datetime
import gc
import torch
import numpy as np
import random
import pytorch_lightning as pl

# Disable warnings from the logging module
logging.disable(logging.WARNING)

# Set seed for reproducibility
def set_seed(seed):
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    This also ensures deterministic behavior for PyTorch operations.
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
    Main training function.

    Parameters:
        hparams (Namespace): Hyperparameters and configuration settings.
        ModuleClass (Class): Lightning module class for training.
        ModelClass (Class): Model class.
        DatasetClass (Class): Dataset class to load training/validation data.
        logger (List): List of loggers for tracking experiment metrics.
    """
    # ------------------------
    # 1. INITIALIZE MODEL, DATASET, AND MODULE
    # ------------------------
    model = ModelClass(hparams=hparams)  # Initialize the model
    dataset = DatasetClass(hparams=hparams)  # Initialize the dataset
    module = ModuleClass(hparams, model, dataset)  # Initialize the Lightning module

    # ------------------------
    # 2. SETUP CHECKPOINT AND EARLY STOPPING CALLBACKS
    # ------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",  # Directory to save checkpoints
        save_top_k=hparams.save_top_k,  # Save only top 'k' models
        verbose=True,  # Enable verbose logging
        monitor=hparams.early_stopping_metric,  # Metric to monitor
        mode='max',  # Mode can be 'min' or 'max' depending on the metric
        filename=f'{{epoch}}-{{{hparams.early_stopping_metric}:.2f}}'  # Filename format
    )

    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stopping_metric,  # Monitor the same metric
        min_delta=0.00,  # Minimum change to be considered improvement
        patience=3,  # Stop after 'patience' epochs with no improvement
        mode='max'  # Stop when metric is maximized
    )

    # ------------------------
    # 3. SETUP THE TRAINER
    # ------------------------
    devices = 1  # Number of devices (GPUs) to use
    accelerator = 'gpu'  # Use GPU acceleration

    trainer = Trainer(
        devices=devices,
        accelerator=accelerator,
        logger=logger,  # Logger to track experiments
        fast_dev_run=hparams.fast_dev_run,  # Whether to run a quick development run
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],  # List of callbacks
        num_sanity_val_steps=hparams.num_sanity_val_steps,  # Number of validation sanity checks before training
        log_every_n_steps=hparams.log_every_n_steps  # Log every 'n' steps
    )

    # ------------------------
    # 4. START TRAINING AND TESTING
    # ------------------------
    trainer.fit(module, ckpt_path=hparams.resume_from_checkpoint)  # Start training

    # Log best checkpoint information
    print(
        f"Best: {checkpoint_callback.best_model_score} | "
        f"Monitor: {checkpoint_callback.monitor} | "
        f"Path: {checkpoint_callback.best_model_path}\nTesting..."
    )

    # Test the model using the best checkpoint
    test_results = trainer.test(module, ckpt_path=hparams.resume_from_checkpoint)

    # ------------------------
    # 5. SAVE RESULTS TO JSON
    # ------------------------
    path_name = str(hparams.output_path)
    fold_name = f'fold{hparams.fold}'  # Define fold name
    path_name = path_name.replace(fold_name + '_', '')  # Remove fold prefix

    # Define the JSON file path
    txt_file_dir = Path(path_name + '.json')

    # Load existing data from the JSON file if it exists, otherwise start with an empty dictionary
    if txt_file_dir.exists():
        try:
            with txt_file_dir.open('r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from {txt_file_dir}, starting with an empty dictionary.")
            data = {}
    else:
        data = {}

    # Update the dictionary with new fold data
    data[fold_name] = test_results[0]

    # Ensure the directory exists
    txt_file_dir.parent.mkdir(parents=True, exist_ok=True)

    # Save updated results to the JSON file
    with txt_file_dir.open('w') as file:
        json.dump(data, file, indent=4)

    print(f"Results saved to {txt_file_dir}")

# Main block to handle training execution
if __name__ == "__main__":
    # ------------------------
    # PARSE ARGUMENTS
    # ------------------------
    root_dir = Path(__file__).parent  # Get the root directory of the script
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')  # Add config file option
    parser, hparams = build_configargparser(parser)  # Build parser from external utility

    # ------------------------
    # SET SEED FOR REPRODUCIBILITY
    # ------------------------
    seed = 42
    set_seed(seed)  # Set seed

    # ------------------------
    # LOAD MODULE, MODEL, AND DATASET CLASSES
    # ------------------------
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)  # Dynamically load module class
    parser = ModuleClass.add_module_specific_args(parser)  # Add module-specific arguments to parser

    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)  # Dynamically load model class
    parser = ModelClass.add_model_specific_args(parser)  # Add model-specific arguments to parser

    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)  # Dynamically load dataset class
    parser = DatasetClass.add_dataset_specific_args(parser)  # Add dataset-specific arguments to parser

    # ------------------------
    # PARSE ARGUMENTS AND SETUP LOGGER
    # ------------------------
    hparams = parser.parse_args()  # Parse the arguments

    # Setup experiment name based on module, dataset, and model
    exp_name = f"{hparams.module.split('.')[-1]}_{hparams.dataset.split('.')[-1]}_{hparams.model.replace('.', '_')}"
    date_str = datetime.now().strftime("%y%m%d-%H%M%S_")  # Timestamp for unique naming
    hparams.name = f"fold{hparams.fold}_{hparams.exp_name}_{exp_name}"  # Full experiment name

    # Define output path and initialize loggers (TensorBoard and WandB)
    hparams.output_path = Path(hparams.output_path).absolute() / exp_name
    tb_logger = TensorBoardLogger(hparams.output_path, name='tb')  # TensorBoard logger
    wandb_logger = WandbLogger(name=hparams.name, project="tecno")  # WandB logger (optional)

    # Log argument summary
    argparse_summary(hparams, parser)

    # ------------------------
    # RUN TRAINING
    # ------------------------
    loggers = [tb_logger]  # Use TensorBoard logger
    train(hparams, ModuleClass, ModelClass, DatasetClass, loggers)  # Start training

    # Clean up to free resources
    del hparams, parser, ModuleClass, ModelClass, DatasetClass, tb_logger, wandb_logger
    torch.cuda.empty_cache()  # Clear GPU memory
    gc.collect()  # Run garbage collection