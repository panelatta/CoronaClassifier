import logging

import torch
from preprocess.constants import TRAIN_DATASET_SIZE, TEST_DATASET_SIZE, BATCH_SIZE
from collections.abc import Iterator


def get_tensor_paths(train: bool = True) -> tuple[list[str], list[str]]:
    """
    Gets the file names of the preprocessed tensors.
    :param train:
    :return tuple[list[str], list[str]]:
    """

    if train:
        return [f'preprocessed_data/train_set/sequence_tensor_{i}.pt' for i in range(1, TRAIN_DATASET_SIZE + 1)], \
            [f'preprocessed_data/train_set/clade_tensor_{i}.pt' for i in range(1, TRAIN_DATASET_SIZE + 1)]
    else:
        return [f'preprocessed_data/test_set/sequence_tensor_{i}.pt' for i in range(1, TEST_DATASET_SIZE + 1)], \
            [f'preprocessed_data/test_set/clade_tensor_{i}.pt' for i in range(1, TEST_DATASET_SIZE + 1)]


def data_chunk_generator(
    logger: logging.Logger,
    chunk_start_index: int = 0,
    batch_start_index: int = 0,
    train: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Generates a batch of data by loading the preprocessed tensors.
    :param logger:
    :param chunk_start_index:
    :param batch_start_index:
    :param train:
    :return Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    if train:
        logger.info("Loading training data...")
    else:
        logger.info("Loading test data...")

    sequence_tensor_paths: list[str]
    clade_tensor_paths: list[str]
    sequence_tensor_paths, clade_tensor_paths = get_tensor_paths(train=train)

    for i in range(chunk_start_index, len(sequence_tensor_paths)):
        sequence_tensor_path: str = sequence_tensor_paths[i]
        clade_tensor_path: str = clade_tensor_paths[i]

        sequence_chunk: torch.Tensor = torch.load(sequence_tensor_path)
        clade_chunk: torch.Tensor = torch.load(clade_tensor_path)

        assert len(sequence_chunk) == len(clade_chunk), "The length of the sequence and clade tensors must match."

        j: int = batch_start_index
        while j < len(sequence_chunk):
            logger.info(f"Loading chunk {i + 1}/{len(sequence_tensor_paths)}...")
            logger.info(f"Loading batch {j // BATCH_SIZE + 1}/{len(sequence_chunk) // BATCH_SIZE}...")

            upper_limit: int = min(j + BATCH_SIZE, len(sequence_chunk))
            sequences: torch.Tensor = sequence_chunk[j: upper_limit]
            clades: torch.Tensor = clade_chunk[j: upper_limit]
            yield sequences, clades

            j += BATCH_SIZE
