#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:42:13 2018

@author: guille
"""

import tensorflow as tf
import numpy as np
import gym

import replayMemory

# MAKE ENVIRONMENT
env=gym.make('CartPole-v0')

# CONSTANTS
LOGS_PATH="/tmp/simple_logs/"
STATE_SIZE=env.observation_space.shape[0]
ACTION_RANGE=env.action_space.shape[0]
ACTION_SIZE=1
LEARNING_RATE=1e-5
EPSILON=0.1
NUM_EPISODES=50000
MINIBATCH_SIZE=64
MEMORY_MAX_SIZE=int(1e5)
DISCOUNT_FACTOR=0.99
INDEX_STATE=0
INDEX_REWARD=1
INDEX_DONE=2
INDEX_LAST_STATE=3
INDEX_ACTION=4
VAR_SIZE_DIC={INDEX_STATE:STATE_SIZE,
              INDEX_REWARD:1,
              INDEX_DONE:1,
              INDEX_LAST_STATE:STATE_SIZE,
              INDEX_ACTION:ACTION_SIZE}

replayMemory=replayMemory.replayMemory(MINIBATCH_SIZE,MEMORY_MAX_SIZE,VAR_SIZE_DIC)

tf.reset_default_graph()
sess=tf.Session()

writer=tf.summary.FileWriter(LOGS_PATH,sess.graph)
loss_summary=tf.placeholder('float',name='Critic_loss_value')
reward_summary=tf.placeholder('float',name='Reward_value')
loss_sum=tf.summary.scalar("Critic_loss", loss_summary)
re_sum=tf.summary.scalar("reward", reward_summary)

# Make network
state_input_tensor=tf.placeholder(tf.float32,shape=(None,STATE_SIZE),name="state_input_tensor")
target_Q_tensor=tf.placeholder(tf.float32,shape=(None,ACTION_RANGE),name="state_input_tensor")

h1=tf.layers.dense(state_input_tensor,32,activation=tf.nn.relu,name="hidden_layer_1",reuse=False)
h2=tf.layers.dense(h1,32,activation=tf.nn.relu,name="hidden_layer_2",reuse=False)
h3=tf.layers.dense(h2,32,activation=tf.nn.relu,name="hidden_layer_3",reuse=False)
output=tf.layers.dense(h3,ACTION_RANGE,activation=None,name="output_layer",reuse=False)

loss=tf.reduce_mean(tf.square(output-target_Q_tensor),name="loss")
train=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

summaryMerged=tf.summary.merge_all()
init_op=tf.global_variables_initializer()
tf.get_default_graph().finalize()
sess.run(init_op)

for episode in range(NUM_EPISODES):
    state=np.reshape(env.reset(),(1,STATE_SIZE))
    done=False
    my_loss=0
    epoch=0
    acc_reward=0
    while not done:
        # Select action
        if np.random.random()<EPSILON:
            action=np.random.randint(0,2)
        else:
            Qs=sess.run(output,feed_dict={state_input_tensor:state})
            action=np.argmax(Qs)
        # Execute action
        new_state,reward,done,_=env.step(action)
        new_state=np.reshape(new_state,(1,STATE_SIZE))
        # Record interaction
        replayMemory.add(new_state,reward,done,state,action)
        state=new_state
        epoch+=1
        acc_reward+=reward
        if episode>100:
            minibatch=replayMemory.get_batch()
            S=replayMemory.get_from_minibatch(minibatch,INDEX_STATE)
            St0=replayMemory.get_from_minibatch(minibatch,INDEX_LAST_STATE)
            A=replayMemory.get_from_minibatch(minibatch,INDEX_ACTION)
            D=replayMemory.get_from_minibatch(minibatch,INDEX_DONE)
            R=replayMemory.get_from_minibatch(minibatch,INDEX_REWARD)
            #TODO add target network
            #TODO add proper target_Q_tensor
            next_states_Q=R+DISCOUNT_FACTOR*np.reshape(np.max(sess.run(output,feed_dict={state_input_tensor:S}),axis=-1),(MINIBATCH_SIZE,1))*(1-D)
            target_Q=sess.run(output,feed_dict={state_input_tensor:St0})
            target_Q[np.arange(MINIBATCH_SIZE),A.flatten()]=np.transpose(next_states_Q)
            
#            target_Q2=R+DISCOUNT_FACTOR*sess.run(output,feed_dict={state_input_tensor:S})
#            target_F=sess.run(output,feed_dict={state_input_tensor:St0})
#            for e in range(MINIBATCH_SIZE):
#                if D[e][0]:
#                    target_F[e][A[e]]=R[e]
#                else:
#                    target_F[e][A[e]]=np.amax(target_Q2,axis=-1)[e]
                          
            
            my_loss+=sess.run(loss,feed_dict={state_input_tensor:St0,target_Q_tensor:target_Q})
            sess.run(train,feed_dict={state_input_tensor:St0,target_Q_tensor:target_Q})
            if done:
                mean_reward=float(acc_reward)
                sumOut=sess.run(re_sum,feed_dict={reward_summary:mean_reward})
                writer.add_summary(sumOut,episode)
                acc_reward=0
                # record loss
                mean_loss=float(my_loss)/epoch
                summary_loss=sess.run(loss_sum,feed_dict={loss_summary:mean_loss})
                writer.add_summary(summary_loss,episode)
                my_loss=0
#                if episode>2000 and episode%100==0:
#                    done=False
#                    state=np.reshape(env.reset(),(1,STATE_SIZE))
#                    while not done:
#                        Qs=sess.run(output,feed_dict={state_input_tensor:state})
#                        action=np.argmax(Qs)
#                        new_state,reward,done,_=env.step(action)
#                        new_state=np.reshape(new_state,(1,STATE_SIZE))
#                        env.render()
#                        state=new_state
        
