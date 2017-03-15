# TableQA
Question answering on tables using deep learning.

## Prerequisites

* keras
* numpy

## Method

* RNN LSTM

## Dataset

1. Syntheticaly generated data samples of T/QA (table/question+answer)

* Training 10k (9500 + 500 validation set)
* Test 500

Patterns:

* Pattern1 

1 City : Graz, Immigration : 13 .
2 City : Wien, Immigration : 18 .
3 What is the immigration in Graz?	13	1

* Pattern2 

1 City : Bludenz, Immigration : 11 , Emmigration : 18.
2 City : Feldkirch, Immigration : 12, Emmigration : 17 .
3 What is the Emmigration in Feldkirch?	17	2

2. Real table data (for training and testing)

* Training (2213 + 117 validation set)
* Test 16


3. Simulated table data + Real table data (simulated data based on the real table data for training and real table data for testing only)

* Training 9002 (8551 + 451 validation set)
* Test ?

4. Simulated table data

* Training 9002 (8551 + 451 validation set)
* Test 500 ?

## Results

1. Synthetic patterns

* Pattern1

Training loss: 0.0181 - acc: 0.9953 - val_loss: 0.0467 - val_acc: 0.9860

Test loss / test accuracy = 0.0245 / 0.9900

* Pattern2

Training loss: 0.0275 - acc: 0.9943 - val_loss: 5.1352e-04 - val_acc: 1.0000

Test loss / test accuracy = 0.0014 / 1.0000

2. Real table data (training and testing on the samples of the original sample table)

Training loss: 5.6899 - acc: 0.0190 - val_loss: 6.5770 - val_acc: 0.0256

Test loss / test accuracy = 7.3377 / 0.0000

3. Simulated table data + Real table data (generating training data based on the sample table
statistics to reduce vocabulary varience: max 10 unique values per column. Testing on the original table.)

!!!

4. Simulated table data

Training loss: 2.2920 - acc: 0.1357 - val_loss: 2.2902 - val_acc: 0.1042
Test loss / test accuracy = 2.2820 / 0.1356

[Finished in 11257.6s]

## License

This project is licensed under the MIT License


## Acknowledgments

* Implementation based on [Keras babi_rnn](https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py)

## References

### Keras

* [Deep Language Modeling for Question Answering using Keras. Benjamin Bolte. 2016](http://ben.bolte.cc/blog/2016/keras-language-modeling.html)

### Tables

* [Neural Enquirer: Learning to Query Tables with Natural Language. IJCAI. 2016](https://arxiv.org/pdf/1512.00965.pdf)

