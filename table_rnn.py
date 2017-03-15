#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Mar 01, 2017

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Based on babi_rnn Keras implementation:
https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

Trains two recurrent neural networks.

'''

from __future__ import print_function
from functools import reduce
import re
import io
import sys

from parse_table import tokenize

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        try:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = tokenize(q)
                substory = None
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = tokenize(line)
                story.append(sent)
        except:
            e = sys.exc_info()[0]
            print(e)
    return data


def get_tables(path, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(path) as f:
        data = parse_stories(f.readlines(), only_supporting=only_supporting)
        # print(data)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
        return data


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


def load_data(path):
    '''
    path <STRING> local path to the dataset;
    '''
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))
    return train, test


def train_rnn(train, test):
    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

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

    print('Training')
    model.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05)
    loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


if __name__ == "__main__":
    # data_path = './data/synth_data_{}.txt'
    # data_path = './data/table_data_{}.txt'
    data_path = './data/sim_data_{}.txt'
    train = get_tables(data_path.format('train'))
    test = get_tables(data_path.format('test'))
    # print (test)
    print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))
    train_rnn(train, test)
