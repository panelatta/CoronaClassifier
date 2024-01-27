from typing import TextIO
import logging
from typing import Optional
from tqdm import tqdm


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


def parse_fasta_file(
    logger: logging.Logger,
    fasta_file: str,
    file: TextIO
) -> tuple[list[list[list[int]]], list[str]]:
    """
    Parses a single fasta file.
    :param logger:
    :param fasta_file:
    :param file:
    :return list[list[list[int]]], list[str]:
    """

    sequence_ids: list[str] = []
    dna_sequences: list[list[list[int]]] = []

    # Start parsing the fasta file
    logger.info(f'Parsing {fasta_file}...')
    current_sequence_id: Optional[str] = None
    current_dna_sequence: list[str] = []

    for line in tqdm(file, desc='Parsing'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            append_dna_sequences(current_dna_sequence, current_sequence_id, dna_sequences, sequence_ids)
            current_sequence_id = line[1:].split()[0]
            current_dna_sequence = []
        else:
            current_dna_sequence.append(line.upper())

    append_dna_sequences(current_dna_sequence, current_sequence_id, dna_sequences, sequence_ids)

    return dna_sequences, sequence_ids


def append_dna_sequences(
    current_dna_sequence: list[str],
    current_sequence_id: Optional[str],
    dna_sequences: list[list[list[int]]],
    sequence_ids: list[str],
) -> None:
    """
    Appends the current sequence to the dna_sequences list and the current sequence ID to the sequence_ids list.
    :param current_dna_sequence:
    :param current_sequence_id:
    :param dna_sequences:
    :param sequence_ids:
    """

    if current_sequence_id:
        sequence_ids.append(current_sequence_id)
        current_seq_str: str = ''.join(current_dna_sequence)
        dna_sequences.append(encode_dna_sequence(current_seq_str))


def encode_nucleotide(nucleotide: str) -> list[int]:
    """
    Encodes a nucleotide to a binary vector.
    :param nucleotide:
    :return list[int]:
    """

    return BINARY_ENCODING_MAP.get(nucleotide, [1, 1, 1, 1])


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
    encoded_sequence: list[list[int]] = list(map(encode_nucleotide, sequence))

    # Padding
    if len(encoded_sequence) < MAX_SEQUENCE_LENGTH:
        encoded_sequence += [BINARY_ENCODING_MAP['N']] * (MAX_SEQUENCE_LENGTH - len(encoded_sequence))

    # Truncate the sequence if it's too long.
    return encoded_sequence[:MAX_SEQUENCE_LENGTH]
