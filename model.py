import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on our current dataset, we have 40 clades.
# Check the log printed in the preprocess stage to see how many clades our dataset has.
CLADE_NUMBERS: int = 40


class CNNDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, target_length: int):
        super(CNNDecoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(target_length)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.conv(src)
        src = F.relu(src)
        src = self.adaptive_pool(src)
        return src


class Transformer(nn.Module):
    def __init__(self, input_dim, nheads, dim_feedforward, num_layers):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        output = self.transformer_encoder(src)
        return output


class DNASequenceClassifier(nn.Module):
    def __init__(self, target_length: int):
        super(DNASequenceClassifier, self).__init__()

        # Three CNN layers with 3*1, 2*1, 2*1 kernels respectively
        self.cnn1 = CNNDecoder(in_channels=4, out_channels=16, kernel_size=3, target_length=target_length)   # 3*1 CNN
        self.cnn2 = CNNDecoder(in_channels=4, out_channels=32, kernel_size=2, target_length=target_length)   # 2*1 CNN
        self.cnn3 = CNNDecoder(in_channels=4, out_channels=64, kernel_size=2, target_length=target_length)   # 2*1 CNN

        self.transformer = Transformer(input_dim=112, nheads=8, dim_feedforward=256, num_layers=3)

        self.classifier = nn.Linear(64, CLADE_NUMBERS)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        The forward function of the model.
        The input tensor should be in the shape of [batch_size, sequence_length, 4].
        :param src:
        :return torch.Tensor:
        """

        # Reshape the input tensor to [batch_size, 4, sequence_length] to fit the input requirement of Conv1d
        src = src.permute(0, 2, 1)
        print(f'src: {src.shape}')

        cnn1_out: torch.Tensor = self.cnn1(src)
        cnn2_out: torch.Tensor = self.cnn2(src)
        cnn3_out: torch.Tensor = self.cnn3(src)
        print(f'cnn1_out: {cnn1_out.shape}, cnn2_out: {cnn2_out.shape}, cnn3_out: {cnn3_out.shape}')

        # [batch_size, combined_channels, sequence_length]
        combined: torch.Tensor = torch.cat((cnn1_out, cnn2_out, cnn3_out), dim=1)

        # Reshape the tensor to [sequence_length, batch_size, combined_channels] to
        # fit the input requirement of Transformer
        combined = combined.permute(2, 0, 1)

        transformer_out: torch.Tensor = self.transformer(combined)
        transformer_out = transformer_out[-1, :, :]     # Only use the last time step output of the transformer

        logits: torch.Tensor = self.classifier(transformer_out)
        return logits
