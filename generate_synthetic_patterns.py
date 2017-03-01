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
N_TABLES = 500
N_ROWS = 2

DATA_FIELDS = ['City', 'Immigration', 'Emmigration']

PATTERN_1 = '''1 {} : {}, {} : {} .
2 {} : {}, {} : {} .
3 What is the {} in {}?\t{}\t{}
'''
# 3 data fields
PATTERN_2 = '''1 {} : {}, {} : {} , {} : {}.
2 {} : {}, {} : {}, {} : {} .
3 What is the {} in {}?\t{}\t{}
'''

'''1 City : {}, Immigration : {}, Emmigration : {} .
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
    for _ in xrange(N_TABLES):
        # container for the generated sample values
        v = []

        # place holder values 1st field
        cities = CITIES[:]
        v0 = []
        city = random.choice(cities)  # random string
        v0.append(city)  # random string
        cities.remove(city)
        v0.append(random.choice(cities))  # random string
        v.append(v0)

        # place holder values 2nd field
        v1 = []
        for _ in xrange(N_ROWS):
            v1.append(random.randrange(10, 20))  # random number
        v.append(v1)

        # place holder values 3rd field
        v2 = []
        for _ in xrange(N_ROWS):
            v2.append(random.randrange(10, 20))  # random number
        v.append(v2)

        # choose data sample to query at random
        s = random.randrange(0, 2)
        # choose data field to query at random
        q = random.randrange(1, 3)

        # define textual pattern
        # pattern = PATTERN_1.format(f[0], v0[0], f[1], v1[0],
                                   # f[0], v0[1], f[1], v1[1],
                                   # f[1], v0[s], v1[s], s+1)

        pattern = PATTERN_2.format(f[0], v[0][0], f[1], v[1][0], f[2], v[2][0],
                                   f[0], v[0][1], f[1], v[1][1], f[2], v[2][1],
                                   f[q], v[0][s], v[q][s], s+1)


        file.write(pattern)