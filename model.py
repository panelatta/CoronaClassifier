import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on our current dataset, we have 40 clades.
# Check the log printed in the preprocess stage to see how many clades our dataset has.
CLADE_NUMBERS: int = 40


class CNNDecoder(nn.Module):
    conv: nn.Conv2d
    fc_conv: nn.Linear

    def __init__(
            self,
            kernel_size: tuple[int, int],
            original_target_length: int,
            conv_target_length: int,
            in_channels: int = 1,
            out_channels: int = 1,
            stride: tuple[int, int] = (1, 1),
            padding: tuple[int, int] = (0, 0),
    ):
        super(CNNDecoder, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Use a fully connected layer to reduce the length of the sequence
        self.fc_conv = nn.Linear(original_target_length, conv_target_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = self.fc_conv(x)
        return x


class Transformer(nn.Module):
    encoder_layer: nn.TransformerEncoderLayer
    transformer_encoder: nn.TransformerEncoder

    def __init__(
            self,
            input_dim: int,
            dim_feedforward: int = 256,   # Number of neurons in the feedforward network model
            num_heads: int = 1,
            dropout: float = 0.1,
            activation: str = "gelu",
            num_layers: int = 3,    # Number of sub-layers in the encoder
    ):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer_encoder(x)
        return x


class MultilayerPerceptron(nn.Module):
    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear

    def __init__(self, input_dim: int, output_dim: int):
        super(MultilayerPerceptron, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.elu(x)
        x = nn.Dropout(p=0.5)(x)

        x = self.fc2(x)
        x = F.elu(x)
        x = nn.Dropout(p=0.3)(x)

        x = self.fc3(x)
        return x


class DNASequenceClassifier(nn.Module):
    cnn1: CNNDecoder
    cnn2: CNNDecoder
    cnn3: CNNDecoder

    batch_norm_1d: nn.BatchNorm1d

    transformer: Transformer

    mlp: MultilayerPerceptron

    def __init__(self, original_target_length: int, conv_target_length: int):
        super(DNASequenceClassifier, self).__init__()

        self.cnn1 = CNNDecoder(
            kernel_size=(3, 4),
            padding=(1, 0),
            original_target_length=original_target_length,
            conv_target_length=conv_target_length
        )
        self.cnn2 = CNNDecoder(
            kernel_size=(2, 1),
            original_target_length=original_target_length - 1,
            conv_target_length=conv_target_length
        )
        self.cnn3 = CNNDecoder(
            kernel_size=(2, 1),
            original_target_length=original_target_length - 2,
            conv_target_length=conv_target_length
        )

        self.batch_norm_1d = nn.BatchNorm1d(conv_target_length * 3)

        self.transformer = Transformer(input_dim=conv_target_length * 3)

        self.mlp = MultilayerPerceptron(input_dim=conv_target_length * 3, output_dim=CLADE_NUMBERS)

    def forward(self, x: torch.Tensor, logger: logging.Logger) -> torch.Tensor:
        """
        The forward function of the model.
        The input tensor should be in the shape of [batch_size, sequence_length, 4].
        :param src:
        :param logger:
        :return torch.Tensor:
        """

        # Reshape the input tensor to [batch_size, 1, sequence_length, channels] to fit the input requirement of conv2d
        x = x.unsqueeze(1)
        logger.info(f'After unsqueeze: {x.shape}')

        cnn1_out: torch.Tensor = self.cnn1(x)
        cnn2_out: torch.Tensor = self.cnn2(x)
        cnn3_out: torch.Tensor = self.cnn3(x)
        logger.info(f'cnn1_out: {cnn1_out.shape}, cnn2_out: {cnn2_out.shape}, cnn3_out: {cnn3_out.shape}')

        x = torch.cat((cnn1_out, cnn2_out, cnn3_out), dim=1)
        logger.info(f'combined cnn output: {x.shape}')

        x = self.batch_norm_1d(x)
        logger.info(f'After batch norm: {x.shape}')

        # Reshape the tensor to [batch_size, combined_channels, sequence_length] to
        # fit the input requirement of Transformer
        x = x.squeeze()
        logger.info(f'After squeeze: {x.shape}')

        x = self.transformer(x)
        logger.info(f'After transformer: {x.shape}')

        x = self.mlp(x)
        logger.info(f'After mlp: {x.shape}')

        return x
