from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from preprocess.constants import TRAIN_DATASET_SIZE, TEST_DATASET_SIZE, BATCH_SIZE
from torch.nn.utils.rnn import pad_sequence


class TensorDataset(Dataset):
    """
    A dataset class for loading preprocessed tensors.
    """
    sequence_tensor_paths: list[str]
    clade_tensor_paths: list[str]

    def __init__(self, train: bool = True):
        self.sequence_tensor_paths, self.clade_tensor_paths = get_tensor_paths(train=train)

    def __len__(self) -> int:
        return len(self.sequence_tensor_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_tensor_path: str = self.sequence_tensor_paths[index]
        clade_tensor_path: str = self.clade_tensor_paths[index]

        sequence_tensor: torch.Tensor = torch.load(sequence_tensor_path)
        clade_tensor: torch.Tensor = torch.load(clade_tensor_path)

        assert len(sequence_tensor) == len(clade_tensor), "The length of the sequence and clade tensors must match."

        return sequence_tensor, clade_tensor


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


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A function that collates a batch of data.
    :param batch:
    :return tuple[torch.Tensor, torch.Tensor]:
    """

    sequences, clades = zip(*batch)

    # Use -1 as the padding value, since 0 is a valid value in the sequence and clades tensor
    sequences = pad_sequence(sequences, batch_first=True, padding_value=-1)
    clades = pad_sequence(clades, batch_first=True, padding_value=-1)

    # Reshape the sequences and clades tensors to [batch_size，sequence_length, 4] and [batch_size，sequence_length]
    sequences = sequences.reshape(-1, sequences.size(2), sequences.size(3))
    clades = clades.reshape(-1)

    return sequences, clades


def get_train_data() -> tuple[DataLoader, DistributedSampler]:
    """
    Gets the data loader and sampler for training.
    :return tuple[DataLoader, DistributedSampler]:
    """

    train_dataset: TensorDataset = TensorDataset(train=True)
    train_sampler: DistributedSampler = DistributedSampler(train_dataset) \
        if torch.distributed.is_initialized() else None
    train_loader: torch.utils.data.DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn
    )

    return train_loader, train_sampler


def get_test_data() -> tuple[DataLoader, DistributedSampler]:
    """
    Gets the data loader and sampler for testing.
    :return tuple[DataLoader, DistributedSampler]:
    """

    test_dataset: TensorDataset = TensorDataset(train=False)
    test_sampler: DistributedSampler = DistributedSampler(test_dataset) \
        if torch.distributed.is_initialized() else None
    test_loader: torch.utils.data.DataLoader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=(test_sampler is None),
        sampler=test_sampler,
        collate_fn=collate_fn
    )

    return test_loader, test_sampler
