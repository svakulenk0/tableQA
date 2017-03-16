#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Mar 01, 2017

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Based on the babi_memnn Keras implementation:
https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py

Trains a memory network on a single table.

References:

- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
'''

from __future__ import print_function

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences

from functools import reduce

import tarfile
import numpy as np
import re

from parse_table import get_tables


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
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


def train_memnn(train, test):
    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
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

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inputs_train, queries_train, answers_train = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    print('-')
    print('inputs: integer tensor of shape (samples, max_length)')
    print('inputs_train shape:', inputs_train.shape)
    print('inputs_test shape:', inputs_test.shape)
    print('-')
    print('queries: integer tensor of shape (samples, max_length)')
    print('queries_train shape:', queries_train.shape)
    print('queries_test shape:', queries_test.shape)
    print('-')
    print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    print('answers_train shape:', answers_train.shape)
    print('answers_test shape:', answers_test.shape)
    print('-')
    print('Compiling...')

    # embed the input sequence into a sequence of vectors
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size,
                                  output_dim=64,
                                  input_length=story_maxlen))
    input_encoder_m.add(Dropout(0.3))
    # output: (samples, story_maxlen, embedding_dim)
    # embed the question into a sequence of vectors
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,
                                   output_dim=64,
                                   input_length=query_maxlen))
    question_encoder.add(Dropout(0.3))
    # output: (samples, query_maxlen, embedding_dim)
    # compute a 'match' between input sequence elements (which are vectors)
    # and the question vector sequence
    match = Sequential()
    match.add(Merge([input_encoder_m, question_encoder],
                    mode='dot',
                    dot_axes=[2, 2]))
    match.add(Activation('softmax'))
    # output: (samples, story_maxlen, query_maxlen)
    # embed the input into a single vector with size = story_maxlen:
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size,
                                  output_dim=query_maxlen,
                                  input_length=story_maxlen))
    input_encoder_c.add(Dropout(0.3))
    # output: (samples, story_maxlen, query_maxlen)
    # sum the match vector with the input vector:
    response = Sequential()
    response.add(Merge([match, input_encoder_c], mode='sum'))
    # output: (samples, story_maxlen, query_maxlen)
    response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)

    # concatenate the match vector with the question vector,
    # and do logistic regression on top
    answer = Sequential()
    answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))
    # the original paper uses a matrix multiplication for this reduction step.
    # we choose to use a RNN instead.
    answer.add(LSTM(32))
    # one regularization layer -- more would probably be needed.
    answer.add(Dropout(0.3))
    answer.add(Dense(vocab_size))
    # we output a probability distribution over the vocabulary
    answer.add(Activation('softmax'))

    answer.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                   metrics=['accuracy'])
    # Note: you could use a Graph model to avoid repeat the input twice
    answer.fit([inputs_train, queries_train, inputs_train], answers_train,
               batch_size=32,
               nb_epoch=120,
               validation_data=([inputs_test, queries_test, inputs_test], answers_test))


if __name__ == "__main__":
    data_path = './data/synth_data_{}.txt'
    # data_path = './data/table_data_{}.txt'
    # data_path = './data/sim_data_{}.txt'
    train = get_tables(data_path.format('train'))
    test = get_tables(data_path.format('test'))
    # print (test)
    train_memnn(train, test)