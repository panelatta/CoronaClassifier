import logging
import torch.distributed as dist
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import DNASequenceClassifier
from preprocess.constants import EPOCH_NUM
from data_loader import get_train_data, get_test_data
from common import get_logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os


# Be sure to change it if you use any other source data
MODULE_SIZE: int = 31000


def train() -> None:
    """
    The main function of the training stage.
    """

    rank: int
    if torch.cuda.is_available():
        # Get the rank of current process
        rank = int(os.environ['RANK'])

    world_size: int
    if torch.cuda.is_available():
        # Get the size of the current process group (i.e. the number of GPUs)
        world_size: int = int(os.environ['WORLD_SIZE'])

    if torch.cuda.is_available():
        # Initialize the process group
        # 'nccl' is the backend for GPU training, optimized for NVIDIA GPUs
        # 'env://' means that the address of the master is stored in the environment variable
        # It will also set torch.distributed.is_initialized() to True
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

        # Set the current process to different GPU
        torch.cuda.set_device(rank)

    logger: logging.Logger = get_logger('trainer', 'train')

    model: nn.Module = DNASequenceClassifier(MODULE_SIZE)

    device: torch.device
    if torch.cuda.is_available():
        logger.info('CUDA is available! Process {rank} will use GPU {rank}.')
        device = torch.device(f'cuda:{rank}')
        model = nn.parallel.DistributedDataParallel(model).to(device)
    else:
        logger.info('CUDA is not available, using CPU...')
        device = torch.device('cpu')

    train_loader: DataLoader
    train_sampler: DistributedSampler
    train_loader, train_sampler = get_train_data()

    test_loader: DataLoader
    test_sampler: DistributedSampler
    test_loader, test_sampler = get_test_data()

    # Use Cross Entropy Loss function, with softmax function so no need to add softmax layer in the model
    loss_function: nn.modules.loss.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCH_NUM):
        logger.info(f'Epoch {epoch + 1}/{EPOCH_NUM}...')

        train_per_epoch(
            logger=logger,
            epoch=epoch,
            train_data_loader=train_loader,
            train_data_sampler=train_sampler,
            device=device,
            loss_function=loss_function,
            model=model,
            optimizer=optimizer,
        )

        val_loss: float
        val_acc: float
        val_loss, val_acc = evaluate_model(
            logger=logger,
            epoch=epoch,
            test_data_loader=test_loader,
            test_data_sampler=test_sampler,
            device=device,
            model=model,
            loss_function=loss_function,
        )

        logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {100. * val_acc:.2f}%')


def train_per_epoch(
    logger: logging.Logger,
    epoch: int,
    train_data_loader: DataLoader,
    train_data_sampler: DistributedSampler,
    device: torch.device,
    loss_function: nn.modules.loss.CrossEntropyLoss,
    model: nn.Module,
    optimizer: optim.Adam
) -> None:
    """
    Trains the model for one epoch.
    :param logger:
    :param epoch:
    :param train_data_loader:
    :param train_data_sampler:
    :param device:
    :param loss_function:
    :param model:
    :param optimizer:
    """

    if train_data_sampler:
        train_data_sampler.set_epoch(epoch)

    model.train()
    train_loss: float = 0.0
    train_correct: int = 0
    train_total: int = 0

    logger.info(f'Training epoch {epoch + 1}/{EPOCH_NUM}...')

    pbar_desc: str = f'Training epoch {epoch + 1}/{EPOCH_NUM}'
    with tqdm(train_data_loader, desc=pbar_desc, leave=False) as pbar:
        for sequences, clades in pbar:
            sequences = sequences.to(device)
            clades = clades.to(device)

            mask: torch.Tensor = (sequences != -1)

            optimizer.zero_grad()
            output = model(sequences)
            output = output.masked_fill(~mask, -1e9)
            loss = loss_function(output, clades)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += clades.size(0)
            train_correct += predicted.eq(clades).sum().item()

            pbar.set_description(f'Train Loss: {train_loss / train_total:.4f}, '
                                 f'Train Accuracy: {100. * train_correct / train_total:.2f}%')
            logger.debug(f'Train Loss: {train_loss / (train_total + 1):.4f}, '
                         f'Train Accuracy: {100. * train_correct / train_total:.2f}%')


def evaluate_model(
    logger: logging.Logger,
    epoch: int,
    test_data_loader: DataLoader,
    test_data_sampler: DistributedSampler,
    device: torch.device,
    model: nn.Module,
    loss_function: nn.modules.loss.CrossEntropyLoss,
) -> tuple[float, float]:
    """
    Evaluates the model in each epoch.
    :param logger:
    :param epoch:
    :param test_data_loader:
    :param test_data_sampler:
    :param device:
    :param model:
    :param loss_function:
    :return tuple[float, float]:
    """

    if test_data_sampler:
        test_data_sampler.set_epoch(epoch)

    model.eval()
    val_loss: int = 0
    correct: int = 0
    total: int = 0

    logger.info(f'Validating epoch {epoch + 1}/{EPOCH_NUM}...')

    with tqdm(test_data_loader, desc='Validating', leave=False) as pbar:
        with torch.no_grad():
            for sequences, clades in pbar:
                sequences = sequences.to(device)
                clades = clades.to(device)

                mask: torch.Tensor = (sequences != -1)

                outputs = model(sequences)
                outputs = outputs.masked_fill(~mask, -1e9)
                loss = loss_function(outputs, clades)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += clades.size(0)
                correct += predicted.eq(clades).sum().item()

                pbar.set_description(f'Val Loss: {val_loss / total:.4f}, Val Accuracy: {100.* correct / total:.2f}%')
                logger.debug(f'Val Loss: {val_loss / total:.4f}, Val Accuracy: {100.* correct / total:.2f}%')

    return val_loss / total, correct / total


if __name__ == '__main__':
    train()

    exit(0)