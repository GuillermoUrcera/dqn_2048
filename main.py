#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 22:16:39 2018

@author: Guillermo Urcera Martín
"""
import sys
sys.path.insert(0,'gym-2048')
import gym
import gym_2048.envs.game_2048
import tensorflow as tf
import numpy as np
import replayMemory
import critic

# CONSTANTS
UP=0
RIGHT=1
DOWN=2
LEFT=3
LOGS_PATH="/tmp/dqn2048_logs"
STATE_SIZE=16
ACTION_SIZE=1
MINIBATCH_SIZE=32
MEMORY_MAX_SIZE=int(1e5)
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
LEARNING_RATE=1e-3
NUM_EPISODES=1000
DISCOUNT_FACTOR=0.99
EPSILON=0.05
TAU=1e-2

# INITIALISATION
tf.reset_default_graph()
sess=tf.Session()
replayMemory=replayMemory.replayMemory(MINIBATCH_SIZE,MEMORY_MAX_SIZE,VAR_SIZE_DIC)
with tf.variable_scope("CRITIC_OPS"):
    my_critic=critic.Critic(sess,STATE_SIZE,ACTION_SIZE,LEARNING_RATE,"critic",MINIBATCH_SIZE)
with tf.variable_scope("TARGET_CRITIC_OPS"):
    my_target_critic=critic.Critic(sess,STATE_SIZE,ACTION_SIZE,LEARNING_RATE,"target_critic",MINIBATCH_SIZE)
with tf.variable_scope("TARGET_UPDATE"):
    for i in range(len(my_critic.weights)):
        update_target_op=my_critic_target.weights[i].assign(TAU*my_critic.weights[i]+(1-TAU)*my_critic_target.weights[i])
        update_target_ops.append(update_target_op)
# TENSORBOARD
writer=tf.summary.FileWriter(LOGS_PATH,sess.graph)
loss_summary=tf.placeholder('float',name='Critic_loss_value')
reward_summary=tf.placeholder('float',name='Reward_value')
loss_sum=tf.summary.scalar("Critic_loss", loss_summary)
re_sum=tf.summary.scalar("reward", reward_summary)
summaryMerged=tf.summary.merge_all()
saver = tf.train.Saver()

# MAKE ENVIRONMENT
env=gym.make('game-2048-v0')

# START 
done=False
epoch=0
for episode in range(NUM_EPISODES):
    state=np.asarray(env.reset()).flatten()
    acc_reward=0
    loss=0
    while not done:
        # select action
        if np.random.random()>EPSILON:
            # random action
            action=np.random.randint(0,4)
        else:
            # let critic decide best action
            action=critic.predict(state)
        # execute action
        new_state,reward,done,_=env.step(action)
        # store transition
        replayMemory.add(new_state,reward,done,state,action)
        acc_reward+=reward
        state=new_state
        epoch+=1
        # learn
        # sample minibatch
        minibatch=replayMemory.get_batch()
        S=replayMemory.get_from_minibatch(minibatch,INDEX_STATE)
        St0=replayMemory.get_from_minibatch(minibatch,INDEX_LAST_STATE)
        A=replayMemory.get_from_minibatch(minibatch,INDEX_ACTION)
        D=replayMemory.get_from_minibatch(minibatch,INDEX_DONE)
        R=replayMemory.get_from_minibatch(minibatch,INDEX_REWARD)
        # calculate target Q with target network
        target_Q=R+DISCOUNT*my_critic_target.predict(S,my_actor_target.predict(S))
        for element in range(len(target_Q)):
            if D[element][0]==True:
                target_Q[element]=R[element]
        # update critic
        loss+=my_critic.trainModel(St0,A,target_Q)
        # update target networks
        sess.run(update_target_ops)
        if done: # log progress
            # record Reward
            mean_reward=float(acc_reward)/epoch
            sumOut=sess.run(re_sum,feed_dict={reward_summary:mean_reward})
            writer.add_summary(sumOut,episode)
            # record loss
            mean_loss=float(loss)/epoch
            summary_loss=sess.run(loss_sum,feed_dict={loss_summary:mean_loss})
            writer.add_summary(summary_loss,episode)

    