#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Feb 24, 2017

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Generate synthetic dataset from a specific pattern for training a neural network.

For example:
1 City : Linz, Immigration : 200 .
2 City : Aspach, Immigration : 100 .
3 What is the immigration in Linz?  200 1
4 What is the immigration in Aspach?    100 2
'''

import random

CITIES = ['Wien', 'Graz', 'Linz', 'Villach', 'Salzburg', 'Klagenfurt', 'Bludenz', 'Feldkirch']
N_SAMPLES = 500

DATA_FIELDS = ['City', 'Immigration', 'Emmigration']

PATTERN_1 = '''1 {} : {}, {} : {} .
2 {} : {}, {} : {} .
3 What is the {} in {}?\t{}\t{}
'''

PATTERN_2 = '''1 City : {}, Immigration : {}, Emmigration : {}.
2 City : {}, Immigration : {}, Emmigration : {} .
3 What is the immigration in {}?\t{}\t1
4 What is the emmigration in {}?\t{}\t2
5 What is the emmigration in {}?\t{}\t2
'''

path = './data/synth_data.txt'
with open(path, 'w') as file:
    # init data fields for the patterns
    f = DATA_FIELDS
    # generate N_SAMPLES random data samples
    for _ in xrange(N_SAMPLES):
        # place holder values 1st field
        cities = CITIES[:]
        v0 = []
        city = random.choice(cities)  # random string
        v0.append(city)  # random string
        cities.remove(city)
        v0.append(random.choice(cities))  # random string

        # place holder values 2nd field
        v1 = []
        v1.append(random.randrange(10, 20))  # random number
        v1.append(random.randrange(10, 20))  # random number

        # choose data sample to query at random
        q = random.randrange(0, 2)

        # define textual pattern
        pattern = PATTERN_1.format(f[0], v0[0], f[1], v1[0],
                                   f[0], v0[1], f[1], v1[1],
                                   f[1], v0[q], v1[q], q+1)


        file.write(pattern)