#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 22:25:27 2017

@author: hemin
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist_data/',one_hot=True)

print('training data size:',mnist.train.num_examples)
print('validating data size',mnist.validation.num_examples)
print('testing data size',mnist.test.num_examples)

batch_size = 100
xs , ys = mnist.train.next_batch(batch_size)
print(xs.shape)
print(ys.shape)