import logging

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import DNASequenceClassifier
from preprocess.constants import EPOCH_NUM
from data_loader import data_chunk_generator
from common import get_logger


# Be sure to change it if you use any other source data
MODULE_SIZE: int = 31000


def train() -> None:
    # Use Cross Entropy Loss function, with softmax function so no need to add softmax layer in the model
    loss_function: nn.modules.loss.CrossEntropyLoss = nn.CrossEntropyLoss()

    model: nn.Module = DNASequenceClassifier(31000)
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=1e-3)

    logger: logging.Logger = get_logger('trainer', 'train')

    for epoch in range(EPOCH_NUM):
        logger.info(f'Epoch {epoch + 1}/{EPOCH_NUM}...')

        train_per_epoch(logger, epoch, loss_function, model, optimizer)

        val_loss: float
        val_acc: float
        val_loss, val_acc = evaluate_model(logger, epoch, model, loss_function)

        logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {100. * val_acc:.2f}%')


def train_per_epoch(
    logger: logging.Logger,
    epoch: int,
    loss_function: nn.modules.loss.CrossEntropyLoss,
    model: nn.Module,
    optimizer: optim.Adam
) -> None:
    """
    Trains the model for one epoch.
    :param logger:
    :param epoch:
    :param loss_function:
    :param model:
    :param optimizer:
    """

    model.train()
    train_loss: float = 0.0
    train_correct: int = 0
    train_total: int = 0

    logger.info(f'Training epoch {epoch + 1}/{EPOCH_NUM}...')

    pbar_desc: str = f'Training epoch {epoch + 1}/{EPOCH_NUM}'
    with tqdm(data_chunk_generator(logger=logger, train=True), desc=pbar_desc, leave=False) as pbar:
        for sequences, clades in pbar:
            optimizer.zero_grad()
            output = model(sequences)
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
    model: nn.Module,
    loss_function: nn.modules.loss.CrossEntropyLoss,
) -> tuple[float, float]:
    """
    Evaluates the model in each epoch.
    :param logger:
    :param epoch:
    :param model:
    :param loss_function:
    :return tuple[float, float]:
    """

    model.eval()
    val_loss: int = 0
    correct: int = 0
    total: int = 0

    logger.info(f'Validating epoch {epoch + 1}/{EPOCH_NUM}...')

    with tqdm(data_chunk_generator(logger=logger, train=False), desc='Validating', leave=False) as pbar:
        with torch.no_grad():
            for sequences, clades in pbar:
                outputs = model(sequences)
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