#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:18:46 2017

@author: hemin
"""

import tensorflow as tf

v1 = tf.Variable(0 , dtype=tf.float32)
step = tf.Variable(0 , dtype=tf.float32)

ema = tf.train.ExponentialMovingAverage(0.99 , step)
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print (sess.run([v1 , ema.average(v1)]))
    sess.run(tf.assign(v1 , 5))
    print (sess.run([v1 , ema.average(v1)]))
    sess.run(maintain_average_op)
    
    