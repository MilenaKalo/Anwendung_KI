import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Attention(nn.Module):
    """
    A simple attention layer
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, inputs, mask=None):
        # Compute attention scores
        scores = self.attention(inputs)
        scores = scores.squeeze(-1)

        # Mask padding tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        weights = nn.functional.softmax(scores, dim=-1)

        # Apply attention weights to input sequence
        output = torch.bmm(inputs.transpose(1, 2), weights.unsqueeze(-1))
        output = output.squeeze(-1)

        return output, weights


class CLSTM(nn.Module):
    """
    A C-LSTM classifier for text classification
    Reference: A C-LSTM Neural Network for Text Classification
    """

    def __init__(self, config):
        super(CLSTM, self).__init__()
        self.max_length = config.max_length
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.filter_sizes = list(map(int, config.filter_sizes.split(",")))
        self.num_filters = config.num_filters
        self.hidden_size = len(self.filter_sizes) * self.num_filters
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda

        # Word embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # Convolutional layers with different lengths of filters in parallel
        # No max-pooling

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (filter_size, self.embedding_size))
            for filter_size in self.filter_sizes
        ])

        # Input dropout
        self.dropout = nn.Dropout(config.keep_prob)

        # LSTM cell
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.keep_prob
        )

        # Attention layer
        self.attention = Attention(self.hidden_size)

        # Softmax output layer
        self.output = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_classes)
        )

        self.l2_loss = 0

    # input_ids = torch.tensor(input_ids, device="cpu")
    def forward(self, input_ids, lengths):
        # Word embedding
        embed = self.embedding(input_ids)

        # Add channel dimension
        embed = embed.unsqueeze(1)

        # Convolutional layers with different lengths of filters in parallel
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = F.relu(conv_layer(embed))
            conv_output = conv_output.squeeze(3)
            conv_outputs.append(conv_output)



        # Maximale Länge berechnen
        max_length = max([t.shape[2] for t in conv_outputs])

        # Tensoren mit Padding auf gleiche Länge bringen
        padded_tensors = []
        for tensor in conv_outputs:
            padded_tensor = torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[2]), value=0)
            padded_tensors.append(padded_tensor)


        # Concatenate the outputs of the convolutional layers
        concat_output = torch.cat(padded_tensors, 1)

        # Apply input dropout
        concat_output = self.dropout(concat_output)

        # Reshape for LSTM input
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
        sorted_concat_output = concat_output[sorted_idx]
        packed_sequence = pack_padded_sequence(sorted_concat_output, sorted_lengths, batch_first=True)

        # LSTM layer
        packed_lstm_output, _ = self.lstm(packed_sequence)

        # Unpack the packed sequence to retrieve the LSTM output tensor
        lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True)

        # Apply dropout
        lstm_output = self.dropout(lstm_output)

        # Attention layer
        attention_output, attention_weights = self.attention(lstm_output)

        # Softmax output layer
        output = torch.nn.functional.softmax(self.output(attention_output))

        # Compute L2 weight regularization loss
        for param in self.parameters():
            self.l2_loss += torch.norm(param, 2)

        return output
