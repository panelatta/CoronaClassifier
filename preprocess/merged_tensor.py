import logging
import numpy as np
import pandas as pd
import torch
from collections.abc import Iterator
from preprocess.constants import TRAIN_DATASET_SIZE, TEST_DATASET_SIZE


def tensor_generator(
        logger: logging.Logger,
        metadata_df: pd.DataFrame,
        fasta_df: pd.DataFrame
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate the final PyTorch tensors for training in (TRAIN_DATASET_SIZE + TEST_DATASET_SIZE) chunks.
    :param logger:
    :param metadata_df:
    :param fasta_df:
    :return Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """

    # Divide the dataframe into (TRAIN_DATASET_SIZE + TEST_DATASET_SIZE) chunks
    num_chunks: int = TRAIN_DATASET_SIZE + TEST_DATASET_SIZE
    chunk_size: int = int(np.ceil(len(metadata_df) / num_chunks))
    logger.info(f"Splitting dataframe into {num_chunks} chunks of size {chunk_size}...")

    for i in range(num_chunks):
        logger.info(f"Processing chunk {i + 1}/{num_chunks}...")

        start_idx: int = i * chunk_size
        end_idx: int = min(start_idx + chunk_size, len(metadata_df))

        metadata_chunk: pd.DataFrame = metadata_df.iloc[start_idx:end_idx]
        merged_chunk: pd.DataFrame = pd.merge(metadata_chunk, fasta_df, on='strain', how='inner')
        merged_chunk.dropna(inplace=True)
        merged_chunk.reset_index(drop=True, inplace=True)
        merged_chunk.drop(
            merged_chunk[(merged_chunk['Nextstrain_clade'] == '?') |
                         (merged_chunk['Nextstrain_clade'] == 'recombinant')].index,
            inplace=True
        )

        sequence_chunk: list[list[list[int]]] = merged_chunk['sequence'].to_list()
        clade_chunk: list[int] = merged_chunk['Nextstrain_clade'].to_list()

        logger.info(f"Printing first {min(5, chunk_size)} rows of ({i + 1}/{num_chunks}) chunk:")
        for j in range(min(5, chunk_size)):
            logger.info(f"clade: {clade_chunk[j]}, "
                        f"sequence: {sequence_chunk[j][:5]}{'...' if len(sequence_chunk) > 5 else ''}")

        yield torch.tensor(sequence_chunk, dtype=torch.float32), torch.tensor(clade_chunk, dtype=torch.long)
