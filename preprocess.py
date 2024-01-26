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
import gc

# The format and level of the logging messages.
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


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
        decompress.decompress_file(path, output_file_path)


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
            dna_sequences: list[list[list[int]]]
            sequence_ids: list[str]
            dna_sequences, sequence_ids = fasta.parse_fasta_file(fasta_file, file)

            # Construct the dataframe from the parsed data and save it to a pickle file
            logging.info(f'Saving {fasta_file + ".pkl"}...')
            cur_df: pd.DataFrame = pd.DataFrame(list(zip(sequence_ids, dna_sequences)), columns=df.columns)
            cur_df.to_pickle(fasta_file + '.pkl')

            # Merge all dataframes from the fasta files into one
            df = pd.concat([df, cur_df], ignore_index=True)

    return df


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
            clade_encoder: dict[str, int] = {}
            encode_counter: int = 0
            for row in tqdm(cur_df.itertuples(), desc='Extracting'):
                clade_encoder, encode_counter = metadata.extract_clade(cur_df, row, clade_encoder, encode_counter)
            logging.info(f'encode_counter: {encode_counter}')
            logging.info("Showing first 5 kind of clades and their encode:")
            for key, value in list(clade_encoder.items())[:5]:
                logging.info(f'clade: {key}, encode: {value}')

            # Save the dataframe to a pickle file
            logging.info(f'Saving {metadata_file + ".pkl"}...')
            cur_df.to_pickle(metadata_file + '.pkl')

            # Merge all dataframes from the metadata files into one
            df = pd.concat([df, cur_df], ignore_index=True)

    return df


def generate_tensors(fasta_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    """
    Generates the final PyTorch tensors for training in 10 chunks.
    :param fasta_df:
    :param metadata_df:
    """

    counter: int = 1
    with tqdm(total=40, desc='Generating tensors') as pbar:
        for sequence_tensor, clade_tensor in merged_tensor.tensor_generator(metadata_df, fasta_df):
            if counter <= 32:
                if not os.path.exists('preprocessed_data/train_set'):
                    os.makedirs('preprocessed_data/train_set')

                logging.info(f"Saving training dataset chunk {counter}/32...")
                torch.save(sequence_tensor, f'preprocessed_data/train_set/sequence_tensor_{counter}.pt')
                torch.save(clade_tensor, f'preprocessed_data/train_set/clade_tensor_{counter}.pt')

            else:
                if not os.path.exists('preprocessed_data/test_set'):
                    os.makedirs('preprocessed_data/test_set')

                logging.info(f"Saving test dataset chunk {counter - 32}/8...")
                torch.save(sequence_tensor, f'preprocessed_data/test_set/sequence_tensor_{counter - 32}.pt')
                torch.save(clade_tensor, f'preprocessed_data/test_set/clade_tensor_{counter - 32}.pt')

            counter += 1
            pbar.update(1)


def clean_up() -> None:
    """
    Cleans up the temp files created during preprocessing.
    """

    if os.path.exists("source_data/*.tsv"):
        os.remove("source_data/*.tsv")
    if os.path.exists("source_data/*.fasta"):
        os.remove("source_data/*.fasta")
    if os.path.exists("source_data/*.pkl"):
        os.remove("source_data/*.pkl")


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

    # Generate PyTorch Tensors
    generate_tensors(fasta_df, metadata_df)

    # Clean up any temporary file created during preprocessing
    clean_up()

    logging.info("Preprocessing finished successfully!")

    exit(0)
