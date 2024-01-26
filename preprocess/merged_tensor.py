import logging
import numpy as np
import pandas as pd
import torch
from collections.abc import Iterator
from tqdm import tqdm


def tensor_generator(metadata_df: pd.DataFrame, fasta_df: pd.DataFrame) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate the final PyTorch tensors for training in 20 chunks.
    :param metadata_df:
    :param fasta_df:
    :return Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """

    # Divide the dataframe into 20 chunks
    num_chunks: int = 40
    chunk_size: int = int(np.ceil(len(metadata_df) / num_chunks))
    logging.info(f"Splitting dataframe into {num_chunks} chunks of size {chunk_size}...")

    for i in range(num_chunks):
        logging.info(f"Processing chunk {i + 1}/{num_chunks}...")

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

        logging.info(f"Printing first {min(5, chunk_size)} rows of ({i + 1}/{num_chunks}) chunk:")
        for i in range(min(5, chunk_size)):
            logging.info(f"clade: {clade_chunk[i]}, "
                         f"sequence: {sequence_chunk[i][:5]}{'...' if len(sequence_chunk) > 5 else ''}")

        yield torch.tensor(sequence_chunk, dtype=torch.float32), torch.tensor(clade_chunk, dtype=torch.long)
