from pprint import pformat
import sys
from time import strftime
import random
import numpy as np
import gc

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset

from custom_datasets import ImageDatasetFactory
from model_architecture import DeepFakeDetector
from utils.config_loader import parse_yaml_config

def parse_cli_args():
    """parsing cmd config"""
    if len(sys.argv) < 2:
        return "./configs/default_config.yaml"
    return sys.argv[1]

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    config_path = parse_cli_args()
    settings = parse_yaml_config(config_path)
    print(pformat(settings))

    for seed_func in [random.seed, np.random.seed, torch.manual_seed]:
        seed_func(settings['training']['random_seed'])
    torch.set_float32_matmul_precision('high')

    dataset_factory = ImageDatasetFactory()
    train_set, val_set = dataset_factory.create_datasets(settings['dataset'])

    data_loader_args = {
        'batch_size': settings['training']['batch_size'],
        'num_workers': settings['system']['num_workers'],
        'pin_memory': True
    }
    train_loader = DataLoader(train_set, shuffle=True, **data_loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **data_loader_args)

    class_distribution = train_set.get_class_distribution()
    model = DeepFakeDetector(
        settings['model'],
        class_weight=class_distribution['fake'] / class_distribution['real']
    )

    hardware = 'gpu' if torch.cuda.is_available() else 'cpu'
    timestamp = strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{settings['dataset']['name']}_{timestamp}"
    
    wandb_logger = WandbLogger(
        project=settings['logging']['project_name'],
        name=experiment_name,
        id=experiment_name,
        log_model=False
    )
    
    trainer = pl.Trainer(
        accelerator=hardware,
        devices=1,
        precision='16-mixed' if settings['training']['use_mixed_precision'] else 32,
        gradient_clip_algorithm='norm',
        gradient_clip_val=settings['training']['gradient_clip_value'],
        accumulate_grad_batches=settings['training']['gradient_accumulation_steps'],
        max_epochs=settings['training']['epochs'],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=settings['logging']['checkpoint_dir'],
                monitor='val_accuracy',
                save_top_k=1,
                mode='max',
                filename=f'{settings["dataset"]["name"]}_{settings["model"]["backbone"]}_' + '{epoch}-{val_accuracy:.2f}'
            )
        ],
        logger=wandb_logger,
        **settings['training'].get('trainer_kwargs', {})
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)