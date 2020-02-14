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

import itertools


import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import statistics
from torchvision import transforms

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, DiskSaver
from resnet import ResNetASPP, ResNet

from torchvision import models
from aspp import ASPP
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
    for label in dataset.labels:
        weights[label] += 1
    weights /= len(dataset)
    weights = 1 / torch.log(c + weights)
    # weights /= weights.min()
    print(f'Class weights:\n{weights}')
    return weights


def main(args):
    # vis = visdom.Visdom()
    # val_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Average Loss', legend=['Train', 'Val'])
    # val_avg_accuracy_window = create_plot_window(vis, '#Epochs', 'Accuracy', 'Average Accuracy', legend=['Val'])
    size = (args.height, args.width)
    interpolation = PIL.Image.BILINEAR


    # model = models.resnet152(pretrained=args.pretrained)
    # # for param in model.parameters():
    # #     param.requires_grad = False # freeze weights
    # # model.fc = nn.Linear(512, 16) # resnet 18,34
    # model.fc = nn.Linear(2048, 16)  # resnet 50+

    # model = ResNetASPP(pretrained=True)
    # model = Model(16)
    params = {
        # 'layer1': list(itertools.permutations([16, 16, 16, 16], 4)),
        'l1': [4, 2, 1],
        'l2': [4, 2, 1],
        'l3': [8, 6, 4, 2],
        'l4': [4, 2, 1],
        'inplanes': [48, 32, 16, 8],
        'lr': [5e-3, 5e-4]
    }

    # start with largest model to make sure that every model fits on GPU
    keys = params.keys()
    values = (params[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for combination_number, combination in enumerate(combinations):
        # make sure to get the same transforms for every model
        fix_seeds()
        train_transform = transforms.Compose([
            transforms.Resize(size),
            # transforms.RandomResizedCrop(size=size, scale=(0.8, 1), interpolation=interpolation),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.01, 0.01), scale=(0.75, 1.25), resample=interpolation),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(size, interpolation=interpolation),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = TextDataset(args.data_path, 'train.txt', size=args.train_size, transform=train_transform)
        val_dataset = TextDataset(args.data_path, 'val.txt', size=args.val_size, transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                                  drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                                drop_last=False, pin_memory=True)

        snapshot_dir = os.path.join(args.snapshot_dir, f'ResNet_combination_{combination_number}')
        if not os.path.exists(snapshot_dir):
            os.mkdir(snapshot_dir)
        # write info parameters
        with open(os.path.join(snapshot_dir, 'info.txt'), 'w') as f:
            f.write(str(combination))

        writer = SummaryWriter(log_dir=os.path.join(snapshot_dir, 'logs'))

        layers = [combination['l1'], combination['l2'], combination['l3'], combination['l4']]
        # model = models.resnet152()
        model = ResNet(models.resnet.Bottleneck, layers, combination['inplanes'],
                       first_conv_k=3, max_pool_k=2, num_classes=16)
        device = 'cpu'
        if args.cuda:
            device = 'cuda'
        print(device)

        lr = combination['lr']

        class_weights = None # compute_class_weights(train_dataset).to(device) # weights ar almost the same
        criterion = nn.CrossEntropyLoss(class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
        poly_lr = lambda epoch: pow((1 - (epoch / args.epochs)), 0.9)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)
        trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
        metrics = {
            'accuracy': Accuracy(),
            'loss': Loss(criterion)
        }
        evaluator = create_supervised_evaluator(model, metrics, device=device)

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_step(engine):
            scheduler.step()

        global pbar, desc
        pbar,desc = None, None

        @trainer.on(Events.EPOCH_STARTED)
        def create_train_pbar(engine):
            global desc, pbar
            if pbar is not None:
                pbar.close()
            desc = 'Train iteration - loss: {:.4f} - lr: {:.6f}'
            pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0, lr))

        @trainer.on(Events.EPOCH_COMPLETED)
        def create_val_pbar(engine):
            global desc, pbar
            if pbar is not None:
                pbar.close()
            desc = 'Validation iteration - loss: {:.4f}'
            pbar = tqdm(initial=0, leave=False, total=len(val_loader), desc=desc.format(0))

        log_interval = 1
        e = Events.ITERATION_COMPLETED(every=log_interval)

        train_losses = []

        @trainer.on(e)
        def log_training_loss(engine):
            lr = optimizer.param_groups[0]['lr']
            train_losses.append(engine.state.output)
            pbar.desc = desc.format(engine.state.output, lr)
            pbar.update(log_interval)

        @evaluator.on(e)
        def log_validation_loss(engine):
            label = engine.state.batch[1].to(device)
            output = engine.state.output[0]
            pbar.desc = desc.format(criterion(output, label))
            pbar.update(log_interval)

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
            writer.add_scalar("Accuracy", avg_accuracy, engine.state.iteration)
            writer.add_scalars("Loss", {"train": statistics.mean(train_losses),
                                        "valid": avg_nll}, engine.state.epoch)
            # vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_accuracy]),
            #          win=val_avg_accuracy_window, update='append')
            # vis.line(X=np.column_stack((np.array([engine.state.epoch]), np.array([engine.state.epoch]))),
            #          Y=np.column_stack((np.array([statistics.mean(train_losses)]),
            #                             np.array([avg_nll]))),
            #          win=val_avg_loss_window, update='append',
            #          opts=dict(legend=['Train', 'Val']))
            del train_losses[:]

        objects_to_checkpoint = {"trainer": trainer, "model": model, "optimizer": optimizer, "scheduler": scheduler}
        training_checkpoint = Checkpoint(to_save=objects_to_checkpoint,
                                         save_handler=DiskSaver(snapshot_dir, require_empty=False))
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


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help="Dataset loaction")
    parser.add_argument('--batch_size', type=int, default=16, help="Training and validation batch size")
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--workers', type=int, default=0, help="Number of workers for loading data. Keep 0 on Windows.")
    parser.add_argument('--snapshot_dir', type=str, default='./snapshots')
    parser.add_argument('--resume_from', type=str, default=None) #'./snapshots/checkpoint_948.pth')
    parser.add_argument('--train_size', type=int, default=4000)
    parser.add_argument('--val_size', type=int, default=800)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--cuda', action="store_true")
    main(parser.parse_args())
