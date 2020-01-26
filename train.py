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


def main(args):
    fix_seeds()
    train_transform = transforms.Compose([
        transforms.Resize((224, 112)),
        # # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # # transforms.RandomVerticalFlip(),
        # # transforms.RandomRotation(10),
        # transforms.RandomAffine(40, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10, resample=PIL.Image.NEAREST),
        transforms.ToTensor(),
        # # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = TextDataset(args.data_path, 'train.txt', size=1000, transform=train_transform)
    val_dataset = TextDataset(args.data_path, 'val.txt', size=1000, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # model = models.resnet18(pretrained=True)
    # model.fc = nn.Linear(512, 16)
    model = Model(16)

    device = 'cpu'
    if args.cuda:
        device = 'cuda'
    print(device)
    
    # lr = 3e-3 / 10
    lr = 5e-4
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, weight_decay=1e-2, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2, betas=(0.9, 0.99))
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, div_factor=10, steps_per_epoch=len(train_loader), epochs=args.epochs)
    # scheduler = lr_scheduler.StepLR(optimizer, 5, 0.8)
    poly_lr = lambda epoch: pow((1 - ((epoch - 1) / args.epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    metrics = {
        'accuracy': Accuracy(),
        # 'mAP': AveragePrecision(),
        'loss': Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_step(engine):
        scheduler.step()

    desc = "ITERATION - loss: {:.4f} - lr: {:.4f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0, lr)
    )
    
    log_interval = 1
    e = Events.ITERATION_COMPLETED(every=log_interval)

    @trainer.on(e)
    def log_training_loss(engine):
        lr = optimizer.param_groups[0]['lr']
        pbar.desc = desc.format(engine.state.output, lr)
        pbar.update(log_interval)
        # writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
        # writer.add_scalar("lr", lr, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def log_train_results(engine):
        evaluator.run(train_loader) # eval on train set to check for overfitting
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['loss']
        tqdm.write(
            "Train Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))
        pbar.n = pbar.last_print_n = 0

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(engine):
    #     evaluator.run(val_loader)
    #     metrics = evaluator.state.metrics
    #     avg_accuracy = metrics['accuracy']
    #     avg_nll = metrics['loss']
    #     tqdm.write(
    #         "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #         .format(engine.state.epoch, avg_accuracy, avg_nll))
    #     pbar.n = pbar.last_print_n = 0
    #     # writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
    #     # writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    objects_to_checkpoint = {"trainer": trainer, "model": model, "optimizer": optimizer, "scheduler": scheduler}
    training_checkpoint = Checkpoint(to_save=objects_to_checkpoint,
                                     save_handler=DiskSaver(args.snapshot_dir, require_empty=False))
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), training_checkpoint)

    try:
        trainer.run(train_loader, max_epochs=args.epochs)
    except Exception as e:
        import traceback
        print(traceback.format_exc())

    pbar.close()


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', '--d', type=str, default='./data', help="Dataset loaction")
    parser.add_argument('--batch_size', '--b', type=int, default=32, help="Training and validation batch size")
    parser.add_argument('--epochs', '--e', type=int, default=40, help="Number of epochs")
    parser.add_argument('--workers', '--w', type=int, default=0, help="Number of workers for loading data")
    parser.add_argument('--snapshot_dir', type=str, default='./snapshots')
    parser.add_argument('--cuda', action="store_true")
    main(parser.parse_args())
