import lzma
import os.path
import sys
import pandas as pd
from typing import TextIO
from collections.abc import Iterator
import logging
import re
from typing import Optional
from tqdm import tqdm

# The block size for decompressing the files.
# It's for limiting the memory usage, as the file will be decompressed in chunks of block_size bytes.
BLOCK_SIZE = 256 * 1024 * 1024  # 256 MB

# The format and level of the logging messages.
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# The maximum length of the DNA sequences.
# Any DNA sequence shorter than this will be padded.
MAX_SEQUENCE_LENGTH = 31000

BINARY_ENCODING_MAP = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'R': [1, 0, 1, 0],  # A or G
    'Y': [0, 1, 0, 1],  # C or T
    'S': [0, 1, 1, 0],  # G or C
    'W': [1, 0, 0, 1],  # A or T
    'K': [0, 0, 1, 1],  # G or T
    'M': [1, 1, 0, 0],  # A or C
    'B': [0, 1, 1, 1],  # C or G or T
    'D': [1, 0, 1, 1],  # A or G or T
    'H': [1, 1, 0, 1],  # A or C or T
    'V': [1, 1, 1, 0],  # A or C or G
    'N': [1, 1, 1, 1],  # Any nucleotide
}


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
    We require the fasta file includes two columns, which indicate the strain and the sequence
    respectively.
    :param fasta_file_names:
    :return pd.DataFrame:
    """

    df: pd.DataFrame = pd.DataFrame(columns=['strain', 'sequence'])

    for fasta_file in fasta_file_names:
        with open(fasta_file, 'r') as file:
            # Load the dataframe from the pickle file if it exists
            if os.path.exists(fasta_file + '.pkl'):
                logging.info(f'File {fasta_file + ".pkl"} already exists, skipping parsing...')
                df = pd.concat(
                    [df, pd.read_pickle(fasta_file + '.pkl')],
                    ignore_index=True
                )
                continue

            # Parse the fasta file
            dna_sequences: list[str]
            sequence_ids: list[str]
            dna_sequences, sequence_ids = parse_fasta_file(fasta_file, file)

            # Construct the dataframe from the parsed data and save it to a pickle file
            logging.info(f'Saving {fasta_file + ".pkl"}...')
            cur_df: pd.DataFrame = pd.DataFrame(list(zip(sequence_ids, dna_sequences)), columns=df.columns)
            cur_df.to_pickle(fasta_file + '.pkl')

            # Merge all dataframes from the fasta files into one
            df = pd.concat([df, cur_df], ignore_index=True)

    return df


def parse_fasta_file(fasta_file: str, file: TextIO) -> tuple[list[str], list[str]]:
    """
    Parses a single fasta file.
    :param fasta_file:
    :param file:
    :return list[str], list[str]:
    """

    sequence_ids: list[str] = []
    dna_sequences: list[str] = []

    # Start parsing the fasta file
    logging.info(f'Parsing {fasta_file}...')
    current_seq_id: Optional[str] = None
    current_seq: list[str] = []

    for line in tqdm(file, desc='Parsing'):
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
    As the metafile(s) should be (a) tsv file(s), we require the metadata file includes two columns,
    which indicate the strain and the Nextstrain_clade respectively.
    :param metadata_file_names:
    :return pd.DataFrame:
    """

    df: pd.DataFrame = pd.DataFrame(columns=['strain', 'Nextstrain_clade'])

    for metadata_file in metadata_file_names:
        with open(metadata_file, 'r', encoding="utf-8") as file:
            # Load the dataframe from the pickle file if it exists
            if os.path.exists(metadata_file + '.pkl'):
                logging.info(f'File {metadata_file + ".pkl"} already exists, skipping parsing...')
                df = pd.concat(
                    [df, pd.read_pickle(metadata_file + '.pkl')],
                    ignore_index=True
                )
                continue

            # Start parsing the metadata file
            logging.info(f'Parsing {metadata_file}...')
            cur_df: pd.DataFrame = pd.read_csv(file, sep='\t', usecols=['strain', 'Nextstrain_clade'])

            # Extract the clade from the Nextstrain_clade column
            for row in tqdm(cur_df.itertuples(), desc='Extracting'):
                match = re.search(r'\d+[A-Z]+', row.Nextstrain_clade)
                if match:
                    cur_df.at[row.Index, 'Nextstrain_clade'] = match.group()

            # Save the dataframe to a pickle file
            logging.info(f'Saving {metadata_file + ".pkl"}...')
            cur_df.to_pickle(metadata_file + '.pkl')

            # Merge all dataframes from the metadata files into one
            df = pd.concat([df, cur_df], ignore_index=True)

    return df


def merge_df(fasta_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the fasta dataframe and the metadata dataframe.
    :param fasta_df:
    :param metadata_df:
    :return pd.DataFrame:
    """

    if os.path.exists("source_data/preprocessed_data_df.pkl"):
        logging.info('File preprocessed_data_df.pkl already exists, skipping merging...')
        return pd.read_pickle("source_data/preprocessed_data_df.pkl")

    logging.info('Merging dataframes...')
    df: pd.DataFrame = pd.merge(fasta_df, metadata_df, how='left', on='strain')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(
        df[(df['Nextstrain_clade'] == '?') | (df['Nextstrain_clade'] == 'recombinant')].index,
        inplace=True
    )

    # Save the merged dataframe to a pickle file
    df.to_pickle("source_data/preprocessed_data_df.pkl")

    return df


def encode_dna_sequence(sequence: str) -> list[list[int]]:
    """
    Encodes a DNA sequence to a list of binary vectors.
    :param sequence:
    :return list[list[int]]:
    """

    # Remove any '-' in the sequence as it means gap.
    sequence = sequence.replace('-', '')

    # Map all nucleotides to its corresponding binary encoding.
    # For any unknown character, map it to 'N'.
    encoded_sequence: list[list[int]] = [BINARY_ENCODING_MAP.get(nuc, [1, 1, 1, 1]) for nuc in sequence]

    # Padding
    if len(encoded_sequence) < MAX_SEQUENCE_LENGTH:
        encoded_sequence += [[1, 1, 1, 1]] * (MAX_SEQUENCE_LENGTH - len(encoded_sequence))

    # Truncate the sequence if it's too long.
    return encoded_sequence[:MAX_SEQUENCE_LENGTH]


def encoding_dna_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the DNA sequences in the dataframe.
    :param df:
    :return pd.DataFrame:
    """

    # Load the dataframe from the pickle file if it exists
    if os.path.exists("source_data/encoded_data_df.pkl"):
        logging.info('File encoded_data_df.pkl already exists, skipping encoding...')
        return pd.read_pickle("source_data/encoded_data_df.pkl")

    logging.info('Encoding DNA sequences...')

    with tqdm(total=df.shape[0], desc='Encoding') as pbar:
        for row in df.itertuples():
            df.at[row.Index, 'sequence'] = encode_dna_sequence(row.sequence)
            pbar.update(1)

    # Save the encoded dataframe to a pickle file
    logging.info("Saving source_data/encoded_data_df.pkl")
    df.to_pickle("source_data/encoded_data_df.pkl")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python preprocess.py <path_to_data_1> <path_to_data_2> ...')
        exit(1)

    decompress_files()

    # retrieve file names from command line arguments
    fasta_file_names: list[str]
    metadata_file_names: list[str]
    fasta_file_names, metadata_file_names = get_file_names()
    logging.info(f"fasta_file_names: {fasta_file_names}, metadata_file_names: {metadata_file_names}")

    # parse fasta files to dataframe
    fasta_df: pd.DataFrame = parse_fasta_files(fasta_file_names)
    logging.info(f"fasta_df.shape: {fasta_df.shape}")

    # parse metadata files to dataframe
    metadata_df: pd.DataFrame = parse_metadata_files(metadata_file_names)
    logging.info(f"metadata_df.shape: {metadata_df.shape}")

    # merge dataframes
    df: pd.DataFrame = merge_df(fasta_df, metadata_df)
    logging.info("The first 5 rows of merged dataframe:")
    print(df.head(5))
    logging.info(f"df.shape: {df.shape}")

    # encode DNA sequences
    df = encoding_dna_sequences(df)
    logging.info("The first 5 rows of encoded dataframe:")
    print(df.head(5))
    logging.info(f"df.shape: {df.shape}")

    exit(0)