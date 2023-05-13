import gzip

import torch
import torch.nn as nn
import numpy as np
from gensim.test.utils import datapath, get_tmpfile

import gensim
from gensim.models import Word2Vec,KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


#!pip install wget
import gensim.downloader as api
class CLSTM(nn.Module):
    """
    A C-LSTM classifier for text classification
    Reference: A C-LSTM Neural Network for Text Classification
    """

    def __init__(self, config):
        super(CLSTM, self).__init__()
        wv = api.load('word2vec-google-news-300')
        self.vocab_size, self.vector_size = wv.vectors.shape
        self.max_length = config.max_length
        self.num_classes = config.num_classes
        self.filter_sizes = list(map(int, config.filter_sizes.split(",")))
        self.num_filters = config.num_filters
        self.hidden_size = len(self.filter_sizes) * self.num_filters
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda

        # Word embedding
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.get_normed_vectors()), freeze=True)

        # Convolutional layers with different lengths of filters in parallel
        # No max-pooling

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (filter_size, self.vector_size))
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

        # Softmax output layer
        self.output = nn.Linear(self.hidden_size, self.num_classes)

        self.l2_loss = 0

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
        # change the order of the dimensions
        sorted_concat_output = sorted_concat_output.transpose(1, 2)
        packed_sequence = pack_padded_sequence(sorted_concat_output, sorted_lengths, batch_first=True)

        # LSTM layer
        packed_lstm_output, _ = self.lstm(packed_sequence)
        # Ermittle die Länge der längsten Sequenz in der Batch

        # Unpack the packed sequence to retrieve the LSTM output tensor
        lstm_output, test = pad_packed_sequence(packed_lstm_output, batch_first=True)
        temp = lstm_output[:, -1, :]
        #print("Shape of input_ids:", input_ids.size())
        #print("Shape of lengths:", lengths.size())
        #print("Shape of embed:", embed.size())
        #print("Shape of concat_output:", concat_output.size())
        #print(packed_sequence.shape())
        #print("Shape of packed_lstm_output:", packed_lstm_output.data.size())
        #print("Shape of lstm_output:", lstm_output.size())

        # Softmax output layer
        output = torch.nn.functional.softmax(self.output(temp), dim=1)

        # Reshape output to (batch_size, num_classes)
        #output = output.view(output.size(0), -1)

        # Compute L2 weight regularization loss
        for param in self.parameters():
            self.l2_loss += torch.norm(param, 2)

        #print("Shape of output:", output.size())

        return output
