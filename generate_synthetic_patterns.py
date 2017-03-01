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

PATTERN_1 = '''1 City : {}, Immigration : {} .
2 City : {}, Immigration : {} .
3 What is the immigration in {}?\t{}\t{}
'''

PATTERN_2 = '''1 City : {}, Immigration : {}, Emmigration : {}.
2 City : {}, Immigration : {}, Emmigration : {} .
3 What is the immigration in {}?\t{}\t1
4 What is the emmigration in {}?\t{}\t2
5 What is the emmigration in {}?\t{}\t2
'''

path = './data/synth_data.txt'
with open(path, 'w') as f:
    # generate N_SAMPLES random data samples
    for _ in xrange(N_SAMPLES):
        # place holders 1st field
        cities = CITIES[:]
        ph1 = []
        city = random.choice(cities)  # random string
        ph1.append(city)  # random string
        cities.remove(city)
        ph1.append(random.choice(cities))  # random string

        # place holders 2nd field
        ph2 = []
        ph2.append(random.randrange(10, 20))  # random number
        ph2.append(random.randrange(10, 20))  # random number

        # choose question at random
        q = random.randrange(0, 2)

        # define textual pattern
        pattern = PATTERN_1.format(ph1[0], ph2[0], ph1[1], ph2[1], ph1[q], ph2[q], q+1)


        f.write(pattern)