# -*- coding: utf-8 -*-

# python 3.7!!!! Tensorflow 1.14.0!!!
import os
import time
import torch
import numpy as np
import data_helper
import matplotlib.pyplot as plt
from model2 import CLSTM
from sklearn.model_selection import train_test_split

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================

# Data parameters
datafile = None
stop_word_file = None
language = 'en'
min_frequency = 0
num_classes = 1
vocab_size = 0
# test_size = 0.1
max_length = 0

# Model Hyperparameters
embedding_size = 256  # kann geändert werden
filter_sizes = '3, 4, 5'
num_filters = 18 # fix
hidden_size = 34  # fix
num_layers = 2  # kann geändert werden
keep_prob = 0.5  # aus paper
learning_rate = 0.1
l2_reg_lambda = 0.001
num_classes = 2

# Training parameters
batch_size = 32
num_epochs = 50
decay_rate = 1
decay_steps = 100000
evaluate_every_steps = 10
save_every_steps = 50


# Output files directory
timestamp = str(int(time.time())) + '_' + 'train2'
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(outdir):
    os.makedirs(outdir)

datafile = '/Users/milena/Downloads/aki/C-LSTM-text-classification/data/data.csv'
# Load and save data
# =============================================================================
data, labels, lengths, vocab_processor = data_helper.load_data(file_path=datafile,
                                                               sw_path=stop_word_file,
                                                               min_frequency=min_frequency,
                                                               max_length=max_length,
                                                               language=language,
                                                               shuffle=True)

vocab_processor.save(os.path.join(outdir, 'vocab'))

vocab_size = len(vocab_processor.vocabulary_._mapping)

max_length = vocab_processor.max_document_length

# Split cross validation set
# =============================================================================

x_train, x_valid, y_train, y_valid, train_lengths, valid_lengths = train_test_split(data,
                                                                                    labels,
                                                                                    lengths,
                                                                                    test_size=0.1,
                                                                                    random_state=22)

# Batch iterator
train_data = data_helper.batch_iter(x_train, y_train, train_lengths, batch_size, num_epochs)


class Config:
    def __init__(self):
        # Modellparameter
        self.max_length = max_length
        self.num_classes = num_classes
        # self.vocab_size = vocab_size
        # self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.l2_reg_lambda = 0.0
        self.keep_prob = 0.5


# Training
# =============================================================================
# Initialize model

config = Config()
model = CLSTM(config)

# Loss and optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model

total_step = 0.0
best_accuracy = 0
last_improved_step = 0
flag = False
num_checkpoint = 40
losses = []
std_steps = 10
accuracies = []

with open(os.path.join(outdir, 'log.txt'), 'w') as f:
    for epoch in range(num_epochs):
        f.write('Epoch [{}/{}]\n'.format(epoch + 1, num_epochs))
        for i, (x_batch, y_batch, batch_lengths) in enumerate(train_data):
            model.train()
        total_step += 1
        f.write('total_step: {}\n'.format(total_step))
        # Decay learning rate
        if total_step % decay_steps == 0:
            learning_rate *= decay_rate

        # Forward pass
        x_batch = torch.tensor(x_batch, device="cpu")
        y_batch = torch.tensor(y_batch, device="cpu")
        batch_lengths = torch.tensor(batch_lengths, device="cpu")

        outputs = model(x_batch, batch_lengths)
        loss = criterion(outputs, y_batch)
        losses.append(loss.item())
        f.write('loss: {:.4f}\n'.format(loss.item()))

        l2_loss = torch.tensor(0.)
        for param in model.parameters():
            l2_loss += torch.norm(param)
        loss += l2_reg_lambda * l2_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate model
        if total_step % evaluate_every_steps == 0:
            model.eval()
            with torch.no_grad():
                x_valid = torch.tensor(x_valid, device="cpu")
                y_valid = torch.tensor(y_valid, device="cpu")
                valid_lengths = torch.tensor(valid_lengths, device="cpu")
                outputs = model(x_valid, valid_lengths)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y_valid).sum().item()
                accuracy = correct / y_valid.size(0)
                f.write('Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}\n'.format(total_step, num_epochs * len(x_train),
                                                                                loss.item(), accuracy))

                # Save model if the accuracy is improved
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    last_improved_step = total_step
                    torch.save(model.state_dict(), os.path.join(outdir, 'model.txt'))
                    improved_str = '*'
                else:
                    improved_str = ''

                msg = 'Step: [{}/{}], Validation Accuracy: {:.4f} {}'
                f.write(msg.format(total_step, num_epochs * len(x_train), accuracy, improved_str) + '\n')

        # Save model and optimizer
        if total_step % save_every_steps == 0:
            torch.save(model.state_dict(), os.path.join(outdir, 'model.pt'))
            torch.save(optimizer.state_dict(), os.path.join(outdir, 'optimizer.pt'))

        # Early stop
        if total_step - last_improved_step > num_checkpoint:
            f.write("No optimization for a long time, auto-stopping...\n")
            flag = True
            break
        if flag:
            break

        # Check standard deviation
        if len(losses) >= std_steps:
            std = np.std(losses)
            mean = np.mean(losses)
            if loss.item() > mean + std:
                print("Loss greater than mean + std!")

    # Plot Loss
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel('Loss')
    plt.ylabel('Steps')
    plt.legend()

    plt.savefig(os.path.join(outdir, 'plot.png'))
