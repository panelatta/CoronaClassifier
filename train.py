import logging
import torch.distributed as dist
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import DNASequenceClassifier
from data_loader import get_train_data, get_test_data
from common import get_logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from typing import Optional, Any

MODULE_SIZE: int = 31000  # Be sure to change it if you use any other source data
CONV_OUTPUT_SIZE: int = 114  # いいだね
EPOCH_NUM: int = 50  # The number of train epochs.

SAVED_TRAINING_DATA_PATH = 'model_data/model.pth'


def train() -> None:
    """
    The main function of the training stage.
    """

    logger: logging.Logger = get_logger('trainer', 'train')

    if not os.path.exists('model_data'):
        os.makedirs('model_data')

    device, model, optimizer, current_epoch = load_training_data(logger)

    train_loader, train_sampler = get_train_data()
    test_loader, test_sampler = get_test_data()

    # Use Cross Entropy Loss function, with softmax function so no need to add softmax layer in the model
    loss_function: nn.modules.loss.CrossEntropyLoss = nn.CrossEntropyLoss()

    if current_epoch + 1 >= EPOCH_NUM - 1:
        logger.info(f'Model has already been trained for {EPOCH_NUM} epochs. Exiting...')

        val_loss, val_acc = evaluate_model(
            logger=logger,
            epoch=EPOCH_NUM - 1,
            test_data_loader=test_loader,
            test_data_sampler=test_sampler,
            device=device,
            model=model,
            loss_function=loss_function,
        )

        logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {100. * val_acc:.2f}%')

    for epoch in range(current_epoch + 1, EPOCH_NUM):
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

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, SAVED_TRAINING_DATA_PATH)

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

    dist.destroy_process_group()


def load_training_data(logger: logging.Logger) -> tuple[torch.Device, nn.Module, optim.Adam, int]:
    """
    Loads a new model and optimizer.
    :param logger:
    :return tuple[torch.Device, nn.Module, optim.Adam, int]:
    """

    rank = get_rank(logger)
    device = get_device(logger, rank)

    model = DNASequenceClassifier(MODULE_SIZE, CONV_OUTPUT_SIZE)
    if rank is not None:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch: int = -1
    if os.path.exists(SAVED_TRAINING_DATA_PATH):
        logger.info(f'Found saved model. Loading model from {SAVED_TRAINING_DATA_PATH}...')
        checkpoint: dict[str, Any] = torch.load(SAVED_TRAINING_DATA_PATH)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    return device, model, optimizer, epoch


def get_device(logger: logging.Logger, rank: Optional[int]) -> torch.Device:
    """
    Gets the device for training.
    :param logger:
    :param rank:
    :return torch.Device:
    """

    if torch.backends.mps.is_available():
        logger.info(f'MPS is available! Using MPS...')
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info(f'CUDA is available! Process {rank} will use GPU {rank}.')
        return torch.device(f'cuda:{rank}')
    else:
        if is_macos():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                      "and/or you do not have an MPS-enabled device on this machine.")

        logger.info('MPS or CUDA is not available, using CPU...')
        return torch.device('cpu')


def get_rank(logger: logging.Logger) -> Optional[int]:
    """
    Gets the rank of the current process.
    :param logger:
    :return Optional[int]:
    """

    if torch.cuda.is_available():
        # Initialize the process group
        # 'nccl' is the backend for GPU training, optimized for NVIDIA GPUs
        # 'env://' means that the address of the master is stored in the environment variable
        # It will also set torch.distributed.is_initialized() to True
        dist.init_process_group(backend='nccl', init_method='env://')

        # Get the rank of current process
        rank: int = torch.distributed.get_rank()

        # Get the size of the current process group (i.e. the number of GPUs)
        world_size: int = torch.distributed.get_world_size()

        logger.info(f'Process {rank} is running on GPU {rank} of {world_size} GPUs.')

        # Set the current process to different GPU
        torch.cuda.set_device(rank)

    return None


def is_macos() -> bool:
    """
    Checks if the current OS is macOS.
    :return bool:
    """

    return os.uname().sysname == 'Darwin'


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

            optimizer.zero_grad()
            output = model(sequences)
            loss = loss_function(output, clades, ignore_index=-1)
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

                outputs = model(sequences)
                loss = loss_function(outputs, clades, ignore_index=-1)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += clades.size(0)
                correct += predicted.eq(clades).sum().item()

                pbar.set_description(f'Val Loss: {val_loss / total:.4f}, Val Accuracy: {100. * correct / total:.2f}%')
                logger.debug(f'Val Loss: {val_loss / total:.4f}, Val Accuracy: {100. * correct / total:.2f}%')

    return val_loss / total, correct / total


if __name__ == '__main__':
    train()

    exit(0)
