import os
import shutil
import PIL
import numpy as np
import random
from tqdm import tqdm
from argparse import ArgumentParser
from dataset import TextDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import torch
from torch import nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import visdom

import statistics
from torchvision import transforms

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import AveragePrecision
from ignite.handlers import Checkpoint, DiskSaver

from torchvision import models

from model import Model


def fix_seeds():
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)


def create_plot_window(vis, xlabel, ylabel, title, legend):
    return vis.line(X=np.array([[1] * len(legend)]),
                    Y=np.array([[np.nan] * len(legend)]),
                    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, legend=legend))


def compute_class_weights(dataset, c=1.02, num_classes=16):
    """
    Comnpute class weights like in ENet https://arxiv.org/abs/1606.02147
    """
    print('Computing class weights')
    weights = torch.zeros(num_classes)
    for label in tqdm(dataset.labels):
        weights[label] += 1
    weights /= len(dataset)
    weights = 1 / torch.log(c + weights)
    print(f'Class weights:\n{weights}')
    return weights


def main(args):
    fix_seeds()
    # if os.path.exists('./logs'):
    #     shutil.rmtree('./logs')
    # os.mkdir('./logs')
    # writer = SummaryWriter(log_dir='./logs')
    vis = visdom.Visdom()
    val_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Average Loss', legend=['Train', 'Val'])
    val_avg_accuracy_window = create_plot_window(vis, '#Epochs', 'Accuracy', 'Average Accuracy', legend=['Val'])
    size = (args.height, args.width)
    train_transform = transforms.Compose([
        transforms.Resize(size),
        # transforms.RandomResizedCrop(size=size, scale=(0.5, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.8, 1.2), resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = TextDataset(args.data_path, 'train.txt', size=args.train_size, transform=train_transform)
    val_dataset = TextDataset(args.data_path, 'val.txt', size=args.val_size, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 16)

    model.load_state_dict(torch.load(args.resume_from)['model'])

    device = 'cpu'
    if args.cuda:
        device = 'cuda'
    print(device)
    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics, device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def lr_step(engine):
        if model.training:
            scheduler.step()

    global pbar, desc
    pbar, desc = None, None

    @trainer.on(Events.EPOCH_STARTED)
    def create_train_pbar(engine):
        global desc, pbar
        if pbar is not None:
            pbar.close()
        desc = 'Train iteration - loss: {:.4f} - lr: {:.4f}'
        pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0, lr))

    @trainer.on(Events.EPOCH_COMPLETED)
    def create_val_pbar(engine):
        global desc, pbar
        if pbar is not None:
            pbar.close()
        desc = 'Validation iteration - loss: {:.4f}'
        pbar = tqdm(initial=0, leave=False, total=len(val_loader), desc=desc.format(0))

    # desc_val = 'Validation iteration - loss: {:.4f}'
    # pbar_val = tqdm(initial=0, leave=False, total=len(val_loader), desc=desc_val.format(0))

    log_interval = 1
    e = Events.ITERATION_COMPLETED(every=log_interval)

    train_losses = []

    @trainer.on(e)
    def log_training_loss(engine):
        lr = optimizer.param_groups[0]['lr']
        train_losses.append(engine.state.output)
        pbar.desc = desc.format(engine.state.output, lr)
        pbar.update(log_interval)
        # writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
        # writer.add_scalar("lr", lr, engine.state.iteration)

    @evaluator.on(e)
    def log_validation_loss(engine):
        label = engine.state.batch[1].to(device)
        output = engine.state.output[0]
        pbar.desc = desc.format(criterion(output, label))
        pbar.update(log_interval)

    # if args.resume_from is not None:
    #     @trainer.on(Events.STARTED)
    #     def _(engine):
    #         pbar.n = engine.state.iteration

    # @trainer.on(Events.EPOCH_COMPLETED(every=1))
    # def log_train_results(engine):
    #     evaluator.run(train_loader) # eval on train set to check for overfitting
    #     metrics = evaluator.state.metrics
    #     avg_accuracy = metrics['accuracy']
    #     avg_nll = metrics['loss']
    #     tqdm.write(
    #         "Train Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #         .format(engine.state.epoch, avg_accuracy, avg_nll))
    #     pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        pbar.refresh()
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['loss']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll))
        # pbar.n = pbar.last_print_n = 0

        # writer.add_scalars("avg losses", {"train": statistics.mean(train_losses),
        #                                   "valid": avg_nll}, engine.state.epoch)
        # # writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        # writer.add_scalar("avg_accuracy", avg_accuracy, engine.state.epoch)
        vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]),
                 win=val_avg_accuracy_window, update='append')
        vis.line(X=np.column_stack((np.array([engine.state.epoch]), np.array([engine.state.epoch]))),
                 Y=np.column_stack((np.array([statistics.mean(train_losses)]),
                                    np.array([avg_nll]))),
                 win=val_avg_loss_window, update='append',
                 opts=dict(legend=['Train', 'Val']))
        del train_losses[:]

    objects_to_checkpoint = {"trainer": trainer, "model": model, "optimizer": optimizer, "scheduler": scheduler}
    training_checkpoint = Checkpoint(to_save=objects_to_checkpoint,
                                     save_handler=DiskSaver(args.snapshot_dir, require_empty=False))
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), training_checkpoint)
    if args.resume_from not in [None, '']:
        tqdm.write("Resume from a checkpoint: {}".format(args.resume_from))
        checkpoint = torch.load(args.resume_from)
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

    try:
        trainer.run(train_loader, max_epochs=args.epochs)
        pbar.close()
    except Exception as e:
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help="Dataset loaction")
    parser.add_argument('--batch_size', type=int, default=16, help="Training and validation batch size")
    parser.add_argument('--workers', type=int, default=0, help="Number of workers for loading data")
    parser.add_argument('--model_path', type=str, default='./snapshots')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--cuda', action="store_true")
    main(parser.parse_args())
