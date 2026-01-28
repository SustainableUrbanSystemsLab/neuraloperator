"""
Evaluation script for Navier-Stokes equation using neural operators.

This script loads a trained neural operator checkpoint and evaluates it on the test set.
"""

import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.mpu.comm import get_local_rank
from neuralop.training import setup, AdamW


def main():
    # Add project root to sys.path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    
    # Check for manual config file handling first
    import argparse
    import tomllib
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="./ckpt", help="Path to the directory containing model checkpoint")
    args, unknown = parser.parse_known_args()

    # Manual config parsing (same as in train script)
    if args.config:
        with open(args.config, "rb") as f:
            config_dict = tomllib.load(f)

        def flatten_to_args(d, prefix=""):
            args_list = []
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    args_list.extend(flatten_to_args(v, key))
                else:
                    val = str(v)
                    args_list.append(f"--{key}")
                    args_list.append(f"{val}")
            return args_list

        file_args = flatten_to_args(config_dict)
        # Update sys.argv
        sys.argv = [sys.argv[0]] + file_args + unknown

    config_name = "default"
    from zencfg import make_config_from_cli
    from config.navier_stokes_config import Default

    config = make_config_from_cli(Default)
    config = config.to_dict()

    # Distributed setup
    device, is_logger = setup(config)

    # Make sure we only print information when needed
    config.verbose = config.verbose and is_logger

    if config.verbose:
        print(f"##### CONFIG #####\n")
        print(config)

    # Data directory setup
    data_dir = Path(config.data.folder).expanduser()

    # Load the Navier-Stokes dataset
    print("Loading data...")
    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        data_root=data_dir,
        train_resolution=config.data.train_resolution,
        n_train=config.data.n_train,
        batch_size=config.data.batch_size,
        test_resolutions=config.data.test_resolutions,
        n_tests=config.data.n_tests,
        test_batch_sizes=config.data.test_batch_sizes,
        encode_input=config.data.encode_input,
        encode_output=config.data.encode_output,
    )

    # Model initialization
    model = get_model(config)
    model = model.to(device)
    
    # Dataprocessor setup
    if config.patching.levels > 0:
        data_processor = MGPatchingDataProcessor(
            model=model,
            in_normalizer=data_processor.in_normalizer,
            out_normalizer=data_processor.out_normalizer,
            padding_fraction=config.patching.padding,
            stitching=config.patching.stitching,
            levels=config.patching.levels,
            use_distributed=config.distributed.use_distributed,
        )
    data_processor = data_processor.to(device)

    # Distributed data parallel setup
    if config.distributed.use_distributed:
        train_db = train_loader.dataset
        train_sampler = DistributedSampler(train_db, rank=get_local_rank())
        train_loader = DataLoader(
            dataset=train_db, batch_size=config.data.batch_size, sampler=train_sampler
        )
        for (res, loader), batch_size in zip(
            test_loaders.items(), config.data.test_batch_sizes
        ):
            test_db = loader.dataset
            test_sampler = DistributedSampler(test_db, rank=get_local_rank())
            test_loaders[res] = DataLoader(
                dataset=test_db, batch_size=batch_size, shuffle=False, sampler=test_sampler
            )

    # Create dummy optimizer/scheduler just to satisfy Trainer (though we won't train)
    optimizer = AdamW(model.parameters(), lr=config.opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Loss function configuration
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    eval_losses = {"h1": h1loss, "l2": l2loss}

    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        data_processor=data_processor,
        device=device,
        mixed_precision=config.opt.mixed_precision,
        eval_interval=config.opt.eval_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose,
        wandb_log=False, # Disable wandb for testing script
    )

    # Load Checkpoint
    checkpoint_dir = Path(args.checkpoint)
    print(f"Loading checkpoint from {checkpoint_dir}...")
    try:
        trainer.resume_state_from_dir(checkpoint_dir)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Ensure you pointed to a directory containing 'best_model_state_dict.pt' or 'model_state_dict.pt'")
        sys.exit(1)

    # Evaluate
    print("Starting evaluation...")
    metrics = trainer.evaluate_all(
        epoch=0,
        eval_losses=eval_losses,
        test_loaders=test_loaders,
        eval_modes={}
    )

    print("\n### Evaluation Results ###")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Clean up distributed training
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
