#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 21:53:34 2017

@author: hemin
"""

import tensorflow as tf

a = tf.constant([1.0 , 2.0] , name = 'a')
b = tf.constant([1.0 , 2.0] , name = 'b')
print (a.graph is tf.get_default_graph() )


g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v" , initializer=tf.ones(1))
    
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable('b' , initializer=tf.zeros(1))
    
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("" , reuse=True):
        print (sess.run(tf.get_variable('v')))
        
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("" , reuse=True):
        print (sess.run(tf.get_variable("b")))