import glob
import os.path
import sys
import pandas as pd
import logging
import torch
from tqdm import tqdm
import preprocess.decompress as decompress
import preprocess.fasta as fasta
import preprocess.metadata as metadata
import preprocess.merged_tensor as merged_tensor
from preprocess.constants import TRAIN_DATASET_SIZE, TEST_DATASET_SIZE
from common import get_logger


def decompress_files(logger: logging.Logger) -> None:
    """
    Decompresses all files provided as arguments.
    :param logger:
    """

    for path in sys.argv[1:]:
        output_file_path: str = path[:-3]
        if os.path.exists(output_file_path):
            logger.info(f'File {output_file_path} already exists, skipping decompression...')
            continue

        logger.info(f'Decompressing {path}...')
        decompress.decompress_file(path, output_file_path)


def get_file_names(logger: logging.Logger) -> tuple[list[str], list[str]]:
    """
    Returns a tuple of lists of file names. The first list contains the names of the fasta files, the second list
    contains the names of the metadata files.
    :param logger:
    :return list[str], list[str]:
    """

    fasta_file_names: list[str] = [file[:-3] for file in sys.argv[1:] if file[:-3].endswith('.fasta')]
    if len(fasta_file_names) < 1:
        logger.warning('Please provide at least 1 fasta file, exiting...')
        exit(1)

    metadata_file_names: list[str] = [file[:-3] for file in sys.argv[1:] if file[:-3].endswith('.tsv')]
    if len(metadata_file_names) < 1:
        logger.warning('Please provide at least 1 metadata file, exiting...')
        exit(1)

    return fasta_file_names, metadata_file_names


def parse_fasta_files(logger: logging.Logger, fasta_file_names: list[str]) -> pd.DataFrame:
    """
    Parses the fasta files and returns a dataframe with the sequence IDs and the DNA sequences.
    We require the fasta file includes two columns, which indicate the strain and the sequence
    respectively.
    :param logger:
    :param fasta_file_names:
    :return pd.DataFrame:
    """

    df: pd.DataFrame = pd.DataFrame(columns=['strain', 'sequence'])

    for fasta_file in fasta_file_names:
        with open(fasta_file, 'r') as file:
            # Load the dataframe from the pickle file if it exists
            if os.path.exists(fasta_file + '.pkl'):
                logger.info(f'File {fasta_file + ".pkl"} already exists, skipping parsing...')
                df = pd.concat(
                    [df, pd.read_pickle(fasta_file + '.pkl')],
                    ignore_index=True
                )
                continue

            # Parse the fasta file
            dna_sequences: list[list[list[int]]]
            sequence_ids: list[str]
            dna_sequences, sequence_ids = fasta.parse_fasta_file(logger, fasta_file, file)

            # Construct the dataframe from the parsed data and save it to a pickle file
            logger.info(f'Saving {fasta_file + ".pkl"}...')
            cur_df: pd.DataFrame = pd.DataFrame(list(zip(sequence_ids, dna_sequences)), columns=df.columns)
            cur_df.to_pickle(fasta_file + '.pkl')

            # Merge all dataframes from the fasta files into one
            df = pd.concat([df, cur_df], ignore_index=True)

    return df


def parse_metadata_files(logger: logging.Logger, metadata_file_names: list[str]) -> pd.DataFrame:
    """
    Parses the metadata files and returns a dataframe with the sequence IDs and the metadata.
    As the metafile(s) should be (a) tsv file(s), we require the metadata file includes two columns,
    which indicate the strain and the Nextstrain_clade respectively.
    :param logger:
    :param metadata_file_names:
    :return pd.DataFrame:
    """

    df: pd.DataFrame = pd.DataFrame(columns=['strain', 'Nextstrain_clade'])

    for metadata_file in metadata_file_names:
        with open(metadata_file, 'r', encoding="utf-8") as file:
            # Load the dataframe from the pickle file if it exists
            if os.path.exists(metadata_file + '.pkl'):
                logger.info(f'File {metadata_file + ".pkl"} already exists, skipping parsing...')
                df = pd.concat(
                    [df, pd.read_pickle(metadata_file + '.pkl')],
                    ignore_index=True
                )
                continue

            # Start parsing the metadata file
            logger.info(f'Parsing {metadata_file}...')
            cur_df: pd.DataFrame = pd.read_csv(file, sep='\t', usecols=['strain', 'Nextstrain_clade'])

            # Extract the clade from the Nextstrain_clade column
            clade_encoder: dict[str, int] = {}
            encode_counter: int = 0
            for row in tqdm(cur_df.itertuples(), desc='Extracting'):
                clade_encoder, encode_counter = metadata.extract_clade(cur_df, row, clade_encoder, encode_counter)
            logger.info(f'Total number of all clades: {encode_counter}')
            logger.info("Showing first 5 kind of clades and their encode:")
            for key, value in list(clade_encoder.items())[:5]:
                logger.info(f'clade: {key}, encode: {value}')

            # Save the dataframe to a pickle file
            logger.info(f'Saving {metadata_file + ".pkl"}...')
            cur_df.to_pickle(metadata_file + '.pkl')

            # Merge all dataframes from the metadata files into one
            df = pd.concat([df, cur_df], ignore_index=True)

    return df


def generate_tensors(logger: logging.Logger, fasta_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    """
    Generates the final PyTorch tensors for training in 10 chunks.
    :param logger:
    :param fasta_df:
    :param metadata_df:
    """

    counter: int = 1
    total_chunk_num: int = TRAIN_DATASET_SIZE + TEST_DATASET_SIZE
    with tqdm(total=total_chunk_num, desc='Generating tensors') as pbar:
        for sequence_tensor, clade_tensor in merged_tensor.tensor_generator(logger, metadata_df, fasta_df):
            if counter <= TRAIN_DATASET_SIZE:
                if not os.path.exists('preprocessed_data/train_set'):
                    os.makedirs('preprocessed_data/train_set')

                logger.info(f"Saving training dataset chunk {counter}/{TRAIN_DATASET_SIZE}...")
                torch.save(sequence_tensor, f'preprocessed_data/train_set/sequence_tensor_{counter}.pt')
                torch.save(clade_tensor, f'preprocessed_data/train_set/clade_tensor_{counter}.pt')

            else:
                if not os.path.exists('preprocessed_data/test_set'):
                    os.makedirs('preprocessed_data/test_set')

                logger.info(f"Saving test dataset chunk {counter - TRAIN_DATASET_SIZE}/{TEST_DATASET_SIZE}...")
                torch.save(
                    sequence_tensor,
                    f'preprocessed_data/test_set/sequence_tensor_{counter - TRAIN_DATASET_SIZE}.pt'
                )
                torch.save(
                    clade_tensor,
                    f'preprocessed_data/test_set/clade_tensor_{counter - TRAIN_DATASET_SIZE}.pt'
                )

            counter += 1
            pbar.update(1)


def clean_up() -> None:
    """
    Cleans up the temp files created during preprocessing.
    """

    for ext in ['tsv', 'fasta', 'pkl']:
        for file in glob.glob(f'source_data/*.{ext}'):
            os.remove(file)


def preprocess() -> None:
    """
    The main function of the preprocessing stage.
    """

    logger: logging.Logger = get_logger('preprocessor', 'preprocess')

    decompress_files(logger)

    # retrieve file names from command line arguments
    fasta_file_names: list[str]
    metadata_file_names: list[str]
    fasta_file_names, metadata_file_names = get_file_names(logger)
    logger.info(f"fasta_file_names: {fasta_file_names}, metadata_file_names: {metadata_file_names}")

    # parse fasta files to dataframe
    fasta_df: pd.DataFrame = parse_fasta_files(logger, fasta_file_names)
    logger.info(f"fasta_df.shape: {fasta_df.shape}")

    # parse metadata files to dataframe
    metadata_df: pd.DataFrame = parse_metadata_files(logger, metadata_file_names)
    logger.info(f"metadata_df.shape: {metadata_df.shape}")

    # Generate PyTorch Tensors
    generate_tensors(logger, fasta_df, metadata_df)

    # Clean up any temporary file created during preprocessing
    clean_up()
    logger.info("Preprocessing finished successfully!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python preprocess.py <path_to_data_1> <path_to_data_2> ...')
        exit(1)

    preprocess()

    exit(0)
