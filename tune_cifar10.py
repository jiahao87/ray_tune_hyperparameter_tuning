import os
import numpy as np
import argparse
from filelock import FileLock
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import ray
from ray import tune
from ray.tune import track, run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


###################################
##  set main configurations here ##

TRAINING_ITERATION = 64
NUM_SAMPLES = 100
REDUCTION_FACTOR = 4
GRACE_PERIOD = 4
CPU_RESOURCES_PER_TRIAL = 1
GPU_RESOURCES_PER_TRIAL = 0
METRIC = 'accuracy'  # or 'loss'

CONFIG = {
    "out_channels": tune.randint(8, 128),
    "dense2_nodes": tune.choice([a for a in range(20,80)] + [0]*len(range(20,80))),
    "dropout_p": tune.uniform(0.2, 0.8)
    }
###################################


class TrainModel(tune.Trainable):
    """
    Ray Tune's class-based API for hyperparameter tuning

    Note: See https://ray.readthedocs.io/en/latest/_modules/ray/tune/trainable.html#Trainable
    
    """
    def _setup(self, config):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = ConvNet(config).to(self.device)
        self.model_name = 'model_' + '_'.join([str(val) for val in config.values()]) + '.pth'
        self.best_acc = 0
        self.best_loss = np.Infinity
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)  # define optimizer
        self.criterion = nn.CrossEntropyLoss()  # define loss function
        self.epoch = 0

    def _train(self):
        train(self.model, self.optimizer, self.criterion, self.train_loader, self.device)
        acc, loss = eval(self.model, self.criterion, self.test_loader, self.device)
        self.epoch += 1

        # remember best metric and save checkpoint
        if METRIC == 'accuracy':
            is_best = acc > self.best_acc
        else:
            is_best = loss < self.best_loss
        self.best_acc = max(acc, self.best_acc)
        self.best_loss = min(loss, self.best_loss)
        if is_best:
            try:
                torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.model_name)
            except Exception as e:
                logger.warning(e)

        if METRIC == 'accuracy':
            return {"mean_accuracy": acc}
        else:
            return {"mean_loss": loss}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class ConvNet(nn.Module):
    """
    Model architecture in Pytorch
    
    """
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.config['out_channels'], 5)
        self.conv2_bn = nn.BatchNorm2d(self.config['out_channels'])
        self.fc1 = nn.Linear(self.config['out_channels'] * 5 * 5, 120)
        self.fc_dropout = nn.Dropout(self.config['dropout_p'])
        if self.config['dense2_nodes']!=0:
            self.fc2 = nn.Linear(120, self.config['dense2_nodes'])
            self.fc3_2 = nn.Linear(self.config['dense2_nodes'], 10)
        else:
            self.fc3_1 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, self.config['out_channels'] * 5 * 5)
        x = F.relu(self.fc_dropout(self.fc1(x)))
        if self.config['dense2_nodes']!=0:
            x = F.relu(self.fc_dropout(self.fc2(x)))
            x = self.fc3_2(x)
        else:
            x = self.fc3_1(x)
        return x


def train(model, optimizer, criterion, train_loader, device=torch.device("cpu")):
    """
    Model training function

    Parameters
    ---------
    model : Pytorch model
        Model instantiated
    optimizer : Pytorch optimizer
        Optimization algorithm defined
    criterion : Pytorch loss function
        Loss function defined
    train_loader : Pytorch dataloader
        Contains training data
    device : Pytorch device
        cpu or cuda

    Returns
    ---------
    Training for one epoch

    """
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def eval(model, criterion, data_loader, device=torch.device("cpu")):
    """
    Model evaluation function

    Parameters
    ---------
    model : Pytorch model
        Model instantiated
    criterion : Pytorch loss function
        Loss function defined
    data_loader : Pytorch dataloader
        Contains evaluation data
    device : Pytorch device
        cpu or cuda

    Returns
    ---------
    accuracy, loss : tuple
        Accuracy and loss of the evaluated model
    
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, target)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy, loss


def get_data_loaders():
    """
    Retrieve data and load in dataloader

    Returns
    ---------
    trainloader : Pytorch dataloader
        Contains training data
    testloader : Pytorch dataloader
        Contains evaluation data
    
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Cifar10 Example")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--ray-address",
        default="localhost:6379",
        help="Address of Ray cluster for seamless distributed execution.")
    args = parser.parse_args()

    if args.ray_address:
        ray.init(address=args.ray_address)

    if METRIC=='accuracy':
        sched = AsyncHyperBandScheduler(time_attr="training_iteration", 
                                        metric="mean_accuracy", 
                                        mode='max', 
                                        reduction_factor=REDUCTION_FACTOR, 
                                        grace_period=GRACE_PERIOD)
    else:
        sched = AsyncHyperBandScheduler(time_attr="training_iteration", 
                                        metric="mean_loss", 
                                        mode='min', 
                                        reduction_factor=REDUCTION_FACTOR, 
                                        grace_period=GRACE_PERIOD)

    analysis = tune.run(
        TrainModel,
        scheduler=sched,
        queue_trials=True,
        stop={"training_iteration": 1 if args.smoke_test else TRAINING_ITERATION
        },
        resources_per_trial={
            "cpu": CPU_RESOURCES_PER_TRIAL,
            "gpu": GPU_RESOURCES_PER_TRIAL
        },
        num_samples=2 if args.smoke_test else NUM_SAMPLES,
        verbose=1,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        max_failures=3,
        config=CONFIG
        )

    if METRIC=='accuracy':
        print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
    else:
        print("Best config is:", analysis.get_best_config(metric="mean_loss"))
