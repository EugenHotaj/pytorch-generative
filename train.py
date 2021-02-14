"""Main training script for models."""

import os
import argparse

import torch
import pytorch_generative as pg


MODEL_DICT = {
    "gated_pixel_cnn": pg.models.gated_pixel_cnn,
    "image_gpt": pg.models.image_gpt,
    "made": pg.models.made,
    "nade": pg.models.nade,
    "pixel_cnn": pg.models.pixel_cnn,
    "pixel_snail": pg.models.pixel_snail,
    "vae": pg.models.vae,
    "beta_vae": pg.models.beta_vae,
    "vd_vae": pg.models.vd_vae,
    "vq_vae": pg.models.vq_vae,
    "vq_vae_2": pg.models.vq_vae_2,
}


def _worker(local_rank, *args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=args[-1],
        rank=local_rank,
    )
    model, model_args = args[0], args[1:]
    MODEL_DICT[model].reproduce(*args, device_id=local_rank)


def main(args):
    if args.gpus > 1:
        worker_args = model, args.epochs, args.batch_size, args.logdir, args.gpus
        torch.multiprocessing.spawn(_worker, worker_args, nprocs=args.gpus)
    MODEL_DICT[args.model].reproduce(args.epochs, args.batch_size, args.logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="the available models to train",
        default="nade",
        choices=list(MODEL_DICT.keys()),
    )
    parser.add_argument(
        "--epochs", type=int, help="number of training epochs", default=1
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="the training and evaluation batch_size",
        default=128,
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="the directory where to log model parameters and TensorBoard metrics",
        default="/tmp/run",
    )
    parser.add_argument(
        "--gpus", type=int, help="number of GPUs to run the model on", default=0
    )
    args = parser.parse_args()

    main(args)
