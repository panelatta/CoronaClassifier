import csv
import lzma
import os.path
import sys
import pandas as pd
from typing import TextIO
from collections.abc import Iterator
import logging
import re
from typing import Optional

BLOCK_SIZE = 256 * 1024 * 1024  # 256 MB
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def decompress_xz_generator(source_file_path: str, block_size: int) -> Iterator[bytes]:
    """
    Decompresses a file in chunks of block_size bytes. Returns an iterator over the decompressed blocks.
    :param source_file_path:
    :param block_size:
    :return Iterator[bytes]:
    """

    with lzma.open(source_file_path, 'rb') as file:
        while True:
            block: bytes = file.read(block_size)
            if not block:
                break
            yield block


def save_decompressed_blocks(generator: Iterator[bytes], output_file_path: str):
    """
    Appends the decompressed blocks to a file.
    :param generator:
    :param output_file_path:
    """

    with open(output_file_path, 'ab') as file:
        for block in generator:
            file.write(block)


def decompress_file(source_file_path: str, output_file_path: str, block_size: int = BLOCK_SIZE):
    """
    Decompresses a file in chunks of block_size bytes and saves it to output_file_path.
    :param source_file_path:
    :param output_file_path:
    :param block_size:
    """

    generator: Iterator[bytes] = decompress_xz_generator(source_file_path, block_size)
    save_decompressed_blocks(generator, output_file_path)


def decompress_files():
    """
    Decompresses all files provided as arguments.
    """

    for path in sys.argv[1:]:
        output_file_path: str = path[:-3]
        if os.path.exists(output_file_path):
            logging.info(f'File {output_file_path} already exists, skipping decompression...')
            continue

        logging.info(f'Decompressing {path}...')
        decompress_file(path, output_file_path, block_size=BLOCK_SIZE)


def get_file_names() -> tuple[list[str], list[str]]:
    """
    Returns a tuple of lists of file names. The first list contains the names of the fasta files, the second list
    contains the names of the metadata files.
    :return list[str], list[str]:
    """

    fasta_file_names: list[str] = [file[:-3] for file in sys.argv[1:] if file[:-3].endswith('.fasta')]
    if len(fasta_file_names) < 1:
        logging.warning('Please provide at least 1 fasta file, exiting...')
        exit(1)

    metadata_file_names: list[str] = [file[:-3] for file in sys.argv[1:] if file[:-3].endswith('.tsv')]
    if len(metadata_file_names) < 1:
        logging.warning('Please provide at least 1 metadata file, exiting...')
        exit(1)

    return fasta_file_names, metadata_file_names


def parse_fasta_files(fasta_file_names: list[str]) -> pd.DataFrame:
    """
    Parses the fasta files and returns a dataframe with the sequence IDs and the DNA sequences.
    :param fasta_file_names:
    :return pd.DataFrame:
    """

    df: pd.DataFrame = pd.DataFrame(columns=['strain', 'sequence'])

    for fasta_file in fasta_file_names:
        with open(fasta_file, 'r') as file:
            if os.path.exists(fasta_file + '.pkl'):
                logging.info(f'File {fasta_file + ".pkl"} already exists, skipping parsing...')
                df = pd.concat([df, pd.read_pickle(fasta_file + '.pkl')], ignore_index=True)
                continue

            dna_sequences: list[str]
            sequence_ids: list[str]
            dna_sequences, sequence_ids = parse_fasta_file(fasta_file, file)

            logging.info(f'Saving {fasta_file + ".pkl"}...')
            cur_df: pd.DataFrame = pd.DataFrame(list(zip(sequence_ids, dna_sequences)), columns=df.columns)
            cur_df.to_pickle(fasta_file + '.pkl')

            df = pd.concat([df, cur_df], ignore_index=True)

    return df


def parse_fasta_file(fasta_file: str, file: TextIO) -> tuple[list[str], list[str]]:
    """
    Parses a single fasta file.
    :param fasta_file:
    :param file:
    :return list[str], list[str]:
    """

    current_seq_id: Optional[str] = None
    current_seq: list[str] = []

    logging.info(f'Parsing {fasta_file}...')
    sequence_ids: list[str] = []
    dna_sequences: list[str] = []
    for line in file:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if current_seq_id:
                sequence_ids.append(current_seq_id)
                dna_sequences.append(''.join(current_seq))
            current_seq_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line.upper())
    if current_seq_id:
        sequence_ids.append(current_seq_id)
        dna_sequences.append(''.join(current_seq))
    return dna_sequences, sequence_ids


def parse_metadata_files(metadata_file_names: list[str]) -> pd.DataFrame:
    """
    Parses the metadata files and returns a dataframe with the sequence IDs and the metadata.
    :param metadata_file_names:
    :return pd.DataFrame:
    """

    df: pd.DataFrame = pd.DataFrame(columns=['strain', 'Nextstrain_clade'])

    for metadata_file in metadata_file_names:
        with open(metadata_file, 'r', encoding="utf-8") as file:
            if os.path.exists(metadata_file + '.pkl'):
                logging.info(f'File {metadata_file + ".pkl"} already exists, skipping parsing...')
                df = pd.concat([df, pd.read_pickle(metadata_file + '.pkl')], ignore_index=True)
                continue

            logging.info(f'Parsing {metadata_file}...')
            cur_df: pd.DataFrame = pd.read_csv(file, sep='\t', usecols=['strain', 'Nextstrain_clade'])

            # Extract the clade from the Nextstrain_clade column
            for row in cur_df.itertuples():
                match = re.search(r'\d+[A-Z]+', row.Nextstrain_clade)
                if match:
                    cur_df.at[row.Index, 'Nextstrain_clade'] = match.group()

            logging.info(f'Saving {metadata_file + ".pkl"}...')
            cur_df.to_pickle(metadata_file + '.pkl')

            df = pd.concat([df, cur_df], ignore_index=True)

    return df


def merge_df(fasta_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the fasta dataframe and the metadata dataframe.
    :param fasta_df:
    :param metadata_df:
    :return pd.DataFrame:
    """

    if os.path.exists("preprocessed_data_df.pkl"):
        logging.info('File preprocessed_data_df.pkl already exists, skipping merging...')
        return pd.read_pickle("preprocessed_data_df.pkl")

    logging.info('Merging dataframes...')
    df: pd.DataFrame = pd.merge(fasta_df, metadata_df, how='left', on='strain')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(df[(df['Nextstrain_clade'] == '?') | (df['Nextstrain_clade'] == 'recombinant')].index, inplace=True)

    df.to_pickle("preprocessed_data_df.pkl")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python preprocess.py <path_to_data_1> <path_to_data_2> ...')
        exit(1)

    decompress_files()

    fasta_file_names: list[str]
    metadata_file_names: list[str]
    fasta_file_names, metadata_file_names = get_file_names()
    logging.info(f"fasta_file_names: {fasta_file_names}, metadata_file_names: {metadata_file_names}")

    fasta_df: pd.DataFrame = parse_fasta_files(fasta_file_names)
    logging.info(f"fasta_df.shape: {fasta_df.shape}")

    metadata_df: pd.DataFrame = parse_metadata_files(metadata_file_names)
    logging.info(f"metadata_df.shape: {metadata_df.shape}")

    df: pd.DataFrame = merge_df(fasta_df, metadata_df)
    print(df.head(5))
    logging.info(f"df.shape: {df.shape}")

    # save processed data to pickle file
    df.to_pickle('source_data\processed_data.pkl')
    logging.info(f"Processed data saved to processed_data.pkl")
