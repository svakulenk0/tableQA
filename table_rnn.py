#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Mar 01, 2017

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Based on the babi_rnn Keras implementation:
https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

Trains two recurrent neural networks.

'''

from __future__ import print_function
from functools import reduce
import re
import io

from parse_table import get_tables

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping


RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)


def train_rnn(train, test):
    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', story_maxlen, 'words')
    print('Query max length:', query_maxlen, 'words')
    print('Number of training samples:', len(train))
    print('Number of test samples:', len(test))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(train[0])
    print('-')
    print('Vectorizing the word sequences...')

    X, Xq, Y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tX, tXq, tY = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    print('vocab = {}'.format(vocab))
    print('X.shape = {}'.format(X.shape))
    print('Xq.shape = {}'.format(Xq.shape))
    print('Y.shape = {}'.format(Y.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

    print('Build model...')

    sentrnn = Sequential()
    sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                          input_length=story_maxlen))
    sentrnn.add(Dropout(0.3))

    qrnn = Sequential()
    qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                       input_length=query_maxlen))
    qrnn.add(Dropout(0.3))
    qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
    qrnn.add(RepeatVector(story_maxlen))

    model = Sequential()
    model.add(Merge([sentrnn, qrnn], mode='sum'))
    model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
    print('Training')
    model.fit([X, Xq], Y,
               batch_size=BATCH_SIZE,
               nb_epoch=EPOCHS,
               callbacks=[earlyStopping],
               validation_split=0.05)
    loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


if __name__ == "__main__":
    # data_path = './data/synth_data_{}.txt'
    # data_path = './data/table_data_{}.txt'
    # data_path = './data/sim_data_{}.txt'
    data_path = './data/mix_data_{}.txt'
    train = get_tables(data_path.format('train'))
    test = get_tables(data_path.format('test'))
    # print (test)
    print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))
    train_rnn(train, test)
