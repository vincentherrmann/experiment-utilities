import torch
import torch.nn.functional as F
import numpy as np
import random
import torchvision

from experiment_utilities.trees import tree_map
from experiment_utilities.meters import MultiMeter
from experiment_utilities.wandb_logging import Logger
from experiment_utilities.misc import fix_seed
import os
import os.path
import argparse


def get_hyperparameters():
    parser = argparse.ArgumentParser()

    # infrastructure
    parser.add_argument('--wandb_logging', default=1, type=int, required=False)
    parser.add_argument('--use_cpu', default=0, type=int, required=False)
    parser.add_argument('--save_model', default=1, type=int, required=False)
    parser.add_argument('--model_name', default="model.p", type=str, required=False)
    parser.add_argument('--num_workers', default=4, type=int, required=False)
    # general
    parser.add_argument('--seed', default=-1, type=int, required=False)
    # training
    parser.add_argument('--batch_size', default=64, type=int, required=False)
    parser.add_argument('--lr', default=3e-4, type=float, required=False)
    parser.add_argument('--patience', default=10, type=float, required=False)
    parser.add_argument('--num_training_steps', default=100_000, type=float, required=False)
    # model
    parser.add_argument('--num_layers', default=4, type=int, required=False)
    parser.add_argument('--hidden_size', default=128, type=int, required=False)
    # logging
    parser.add_argument('--log_every_n_steps', default=100, type=int, required=False)
    parser.add_argument('--evaluate_every_n_steps', default=1000, type=int, required=False)

    args = parser.parse_args()
    return args


def main(args):
    log_with_wandb = args.wandb_logging > 0
    hyperparameters = dict(vars(args))

    logger = Logger(enabled=log_with_wandb,
                    print_logs_to_console=not log_with_wandb,
                    project="test",
                    tags=["MNIST"],
                    config=hyperparameters)

    if args.use_cpu:
        device = 'cpu'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    ### DATA ################################################################
    train_dataset = torchvision.datasets.MNIST(root="/home/vincent/data", train=True,
                                               transform=torchvision.transforms.ToTensor())
    eval_dataset = torchvision.datasets.MNIST(root="/home/vincent/data", train=False,
                                              transform=torchvision.transforms.ToTensor())

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    ### MODEL ################################################################
    model = torch.nn.Sequential(
        torch.nn.Linear(784, args.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_size, 10)
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ### EVALUATION ################################################################
    @torch.no_grad()
    @fix_seed
    def evaluate():
        model.eval()

        total_num_examples = 0
        total_loss = 0
        total_num_correct = 0
        for eval_step, batch in enumerate(eval_dataloader):
            if eval_step >= 10000:
                break
            batch = tree_map(lambda x: x.to(device), batch)
            x, target = batch
            batch_size = x.shape[0]

            model_output = model(x.view(batch_size, 784))

            losses = F.cross_entropy(model_output, target, reduce='none')

            predictions = model_output.argmax(dim=1)
            correct_predictions = predictions == target
            total_num_correct += correct_predictions.sum().item()
            total_loss += losses.sum().item()
            total_num_examples += batch_size
        eval_loss = total_loss / total_num_examples
        eval_accuracy = total_num_correct / total_num_examples
        model.train()
        return eval_loss, eval_accuracy

    meter = MultiMeter()
    step = 0
    epoch = 0
    lowest_eval_loss = float('inf')
    patience_count = 0

    ### TRAIN LOOP ################################################################
    while step < args.num_training_steps:
        for batch in train_dataloader:
            batch = tree_map(lambda x: x.to(device), batch)
            x, target = batch
            batch_size = x.shape[0]

            model_output = model(x.view(batch_size, 784))

            loss = F.cross_entropy(model_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter.update({
                "train loss": loss.item()
            })

            if step % args.log_every_n_steps == 0:
                logger().log({
                    "train loss": meter["train loss"].avg,
                    "epoch": epoch,
                }, step=step)
                meter.reset()

            if step % args.evaluate_every_n_steps == 0:
                eval_loss, eval_acc = evaluate()
                logger().log({
                    "eval loss": eval_loss,
                    "eval accuracy": eval_acc,
                }, step=step)

                if eval_loss < lowest_eval_loss:
                    lowest_eval_loss = eval_loss
                    patience_count = 0
                    if args.save_model and log_with_wandb:
                        torch.save(
                            model.cpu(),
                            os.path.join(logger().run.dir, args.model_name),
                        )
                        model.to(device)
                else:
                    patience_count += 1
                    if patience_count >= args.patience:
                        return
            step += 1
        epoch += 1


if __name__ == '__main__':
    args = get_hyperparameters()
    main(args)