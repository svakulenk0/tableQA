#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Mar 2, 2017

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

'''
import unittest

import os
import pandas as pd

PATH = './data/'
SAMPLE_TABLE = 'OOE_Wanderungen_Zeitreihe.csv'


def read_tables(fps, delimiter, shuffle=False, limit=False):
    '''
    Input:
    fps <list of strings>  full paths to files to read tables from

    Output:
    tables <dict> {file_path: rows_generator}
    '''
    tables = {}
    for path in fps:
        df = pd.read_csv(path, sep=delimiter)
        if shuffle:
            df_shuffled = df.iloc[np.random.permutation(len(df))]
            df_shuffled.reset_index(drop=True)
            df = df_shuffled
        if limit:
            df = df[:limit]
        tables[path] = df
    return tables


def collect_tables(files):
    # collect file paths
    fps = []
    for file in files:
        fps.append(os.path.join(PATH, file))
    print fps
    return read_tables(fps, delimiter=';')


def generate_data(table, out_path='./data/table_data.txt'):
    with open(out_path, 'w') as file:
        columns = table.columns.values
        print columns
        for row in table.itertuples():
            data_string = str(row[0]+1) + ' '
            # print row
            values = []
            for idx, value in enumerate(row[1:]):
                values.append(str(columns[idx]) + ' : ' + str(value))
            data_string += ', '.join(values) + ' .\n'
            file.write(data_string)


def test_generate_data():
    tables = collect_tables([SAMPLE_TABLE])
    for path, table in tables.items():
        print path
        generate_data(table)


class TestParseTable(unittest.TestCase):
    def test_collect_tables(self):
        tables = collect_tables([SAMPLE_TABLE])
        for path, table in tables.items():
            print path
            print table.columns.values


if __name__ == '__main__':
    # unittest.main()
    test_generate_data()
