#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Mar 2, 2017

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

'''
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


def test_read_tables(files):
    # collect file paths
    fps = []
    for file in files:
        fps.append(os.path.join(PATH, file))
    print fps
    tables = read_tables(fps, delimiter=';')
    for path, table in tables.items():
        print path
        print table.columns.values


def generate_dataset(table):
    columns = table.columns.values
    print columns


def test_generate_dataset(files):
    # collect file paths
    fps = []
    for file in files:
        fps.append(os.path.join(PATH, file))
    print fps
    tables = read_tables(fps, delimiter=';')
    for path, table in tables.items():
        print path
        generate_dataset(table)


if __name__ == '__main__':
    test_generate_dataset([SAMPLE_TABLE])
