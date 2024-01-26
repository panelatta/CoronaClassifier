import lzma
from collections.abc import Iterator

# The block size for decompressing the files.
# It's for limiting the memory usage, as the file will be decompressed in chunks of block_size bytes.
BLOCK_SIZE = 256 * 1024 * 1024  # 256 MB


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