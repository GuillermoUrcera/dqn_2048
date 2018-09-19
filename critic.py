#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 23:10:01 2018

@author: Guillermo Urcera Martín
"""

import tensorflow as tf

class Critic:
    def __init__(self,sess,state_size,action_size,learning_rate,name,subspace_name,minibatch_size,L2):
        self.sess=sess
        self.learning_rate=learning_rate
        self.name=name
        self.state_input_tensor=tf.placeholder(tf.float32, shape=(None, state_size),name="state_input_tensor")
        self.action_input_tensor=tf.placeholder(tf.float32, shape=(None, action_size),name="action_input_tensor")
        self.output=self.createCritic(action_size)
        self.target_Q=tf.placeholder(tf.float32,shape=(None,4),name="target_Q")
        self.loss=tf.reduce_mean(tf.square(self.output-self.target_Q),name="loss")
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=(subspace_name+"/"+name+"_network"))
#        for weight in self.weights:###COMMENT THIS!
#            if not 'bias' in weight.name:
#                self.loss+=L2*tf.nn.l2_loss(weight)
        self.train=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,var_list=self.weights)
    def createCritic(self,action_size):
        with tf.variable_scope(self.name+"_network"):
            critic_input=self.state_input_tensor
            h1=tf.layers.dense(critic_input,256,activation=tf.nn.relu,name="hidden_layer_1",reuse=False)
            h2=tf.layers.dense(h1,256,activation=tf.nn.relu,name="hidden_layer_2",reuse=False)
            h3=tf.layers.dense(h2,256,activation=tf.nn.relu,name="hidden_layer_3",reuse=False)
            output=tf.layers.dense(h3,4,activation=None,name="output_layer")
        return output
    def predict(self,state):
        feed_dict={self.state_input_tensor:state}
        return self.sess.run(self.output,feed_dict)
    def trainModel(self,state,targetQ):
        feed_dict={self.state_input_tensor:state,self.target_Q:targetQ}
        self.sess.run(self.train,feed_dict)
        return self.sess.run(self.loss,feed_dict)
