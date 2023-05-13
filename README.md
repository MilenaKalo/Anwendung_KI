# Multi-class Text Classification
Implemention of C-LSTM for text classification problem.
## Models
* The proposed C-LSTM model given in below paper is used for text classification.
* Paper: A C-LSTM Neural Network for Text Classification
* Link : https://arxiv.org/abs/1511.08630
## Requirements  
* Python 3.7  
* Tensorflow 1.14.0!pip install tensorflow==1.14.0
* Sklearn > 0.19.0  
## Data Format
Training data should be stored in csv file. The first line of the file should be ["label", "content"] 
## Steps 
* Step 1: First install tensorflow 1.6.0 using the following commad
```
!pip install tensorflow==1.14.0
```
* Step 2 : Verify tensorflow version
```
import tensorflow as ts
print(ts.__version__)
```

git repository for C-LSTM implementation
```
https://github.com/KifayatMsd/C-LSTM-text-classification.git
```


## Train
Run train.py or train2.py to train the models.
Parameters:
```
  --data_file DATA_FILE
                        Data file path
  --stop_word_file STOP_WORD_FILE
                        Stop word file path
  --language LANGUAGE   Language of the data file. 
                        Default: en
  --min_frequency MIN_FREQUENCY
                        Minimal word frequency
  --num_classes NUM_CLASSES
                        Number of classes
  --max_length MAX_LENGTH
                        Max document length
  --vocab_size VOCAB_SIZE
                        Vocabulary size
  --test_size TEST_SIZE
                        Cross validation test size
  --embedding_size EMBEDDING_SIZE
                        Word embedding size. 
  --filter_sizes FILTER_SIZES
                        CNN filter sizes. 
  --num_filters NUM_FILTERS
                        Number of filters per filter size. 
  --hidden_size HIDDEN_SIZE
                        Number of hidden units in the LSTM cell. 
  --num_layers NUM_LAYERS
                        Number of the LSTM cells
  --keep_prob KEEP_PROB
                        Dropout keep probability
  --learning_rate LEARNING_RATE
                        Learning rate
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda
  --batch_size BATCH_SIZE
                        Batch size
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --decay_rate DECAY_RATE
                        Learning rate decay rate
  --decay_steps DECAY_STEPS
                        Learning rate decay steps.
  --evaluate_every_steps EVALUATE_EVERY_STEPS
                        Evaluate the model on validation set after this many
                        steps
  --save_every_steps SAVE_EVERY_STEPS
                        Save the model after this many steps
  --num_checkpoint NUM_CHECKPOINT
                        Number of models to store

```
