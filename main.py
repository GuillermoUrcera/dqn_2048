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
MINIBATCH_SIZE=64
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
LEARNING_RATE=1e-5
NUM_EPISODES=100000
DISCOUNT_FACTOR=0.99
EPSILON=0.2
TAU=1e-5
CRITIC_SUBSPACE_NAME="CRITIC_OPS"
CRITIC_TARGET_SUBSPACE_NAME="TARGET_CRITIC_OPS"
CRITIC_L2_WEIGHT_DECAY=1e-2
WARMUP=10000

# INITIALISATION
tf.reset_default_graph()
sess=tf.Session()
replayMemory=replayMemory.replayMemory(MINIBATCH_SIZE,MEMORY_MAX_SIZE,VAR_SIZE_DIC)
with tf.variable_scope(CRITIC_SUBSPACE_NAME):
    my_critic=critic.Critic(sess,STATE_SIZE,ACTION_SIZE,LEARNING_RATE,"critic",CRITIC_SUBSPACE_NAME,MINIBATCH_SIZE,CRITIC_L2_WEIGHT_DECAY)
with tf.variable_scope(CRITIC_TARGET_SUBSPACE_NAME):
    my_target_critic=critic.Critic(sess,STATE_SIZE,ACTION_SIZE,LEARNING_RATE,"target_critic",CRITIC_TARGET_SUBSPACE_NAME,MINIBATCH_SIZE,CRITIC_L2_WEIGHT_DECAY)
update_target_ops=[]
with tf.variable_scope("TARGET_UPDATE"):
    for i in range(len(my_critic.weights)):
        update_target_op=my_target_critic.weights[i].assign(TAU*my_critic.weights[i]+(1-TAU)*my_target_critic.weights[i])
        update_target_ops.append(update_target_op)
# TENSORBOARD
writer=tf.summary.FileWriter(LOGS_PATH,sess.graph)
loss_summary=tf.placeholder('float',name='Critic_loss_value')
reward_summary=tf.placeholder('float',name='Reward_value')
loss_sum=tf.summary.scalar("Critic_loss", loss_summary)
re_sum=tf.summary.scalar("reward", reward_summary)
summaryMerged=tf.summary.merge_all()
saver = tf.train.Saver()
init_op=tf.global_variables_initializer()
tf.get_default_graph().finalize()
sess.run(init_op)

# MAKE ENVIRONMENT
env=gym.make('game-2048-v0')

def reward_tiles_merged(state,new_state):
    return -(np.count_nonzero(new_state)-1-np.count_nonzero(state))

# START
training_has_started=False
acc_reward=0
loss=0
for episode in range(NUM_EPISODES):
    np.seterr(divide='ignore')
    state=np.log2(np.reshape(np.asarray(env.reset()),(1,16)))
    np.seterr(divide='warn')
    state[np.isneginf(state)]=0
    state=state/11.0
    done=False
    epoch=0
    EPSILON*=0.999
    if episode%100==0:
        print "Episode",episode,"of",NUM_EPISODES
    while not done:
        no_change=True
        no_change2=False
        while no_change:
            # select action
            if (np.random.random()<EPSILON or no_change2):
                # random action
                action=np.random.randint(0,4)
            else:
                # let critic decide best action
                qs=my_critic.predict(state)
                action=np.argmax(qs)
            # execute action
            new_state,reward,done,_=env.step(action)
            np.seterr(divide='ignore')
            new_state=np.log2(np.reshape(np.asarray(new_state),(1,16)))
            np.seterr(divide='warn')
            new_state[np.isneginf(new_state)]=0
            new_state=new_state/11.0
            no_change=np.array_equal(new_state,state)
            no_change2=no_change
        reward=reward_tiles_merged(state,new_state)
        replayMemory.add(new_state,reward,done,state,action)
        acc_reward+=reward
        state=new_state
        epoch+=1
        # learn
        if len(replayMemory.memory)>MINIBATCH_SIZE*10:
            if not training_has_started:
                print "Starting training!"
            training_has_started=True
            # sample minibatch
            minibatch=replayMemory.get_batch()
            S=replayMemory.get_from_minibatch(minibatch,INDEX_STATE)
            St0=replayMemory.get_from_minibatch(minibatch,INDEX_LAST_STATE)
            A=replayMemory.get_from_minibatch(minibatch,INDEX_ACTION)
            D=replayMemory.get_from_minibatch(minibatch,INDEX_DONE)
            R=replayMemory.get_from_minibatch(minibatch,INDEX_REWARD)
            # calculate target Q with target network
            next_states_Q=R+DISCOUNT_FACTOR*np.transpose(np.max(my_target_critic.predict(S),axis=-1)*np.transpose(1-D))
            target_Q=my_critic.predict(St0)
            target_Q[np.arange(MINIBATCH_SIZE),A.flatten()]=np.transpose(next_states_Q)
            # update critic
            loss+=my_critic.trainModel(St0,target_Q)
            # update target networks
            sess.run(update_target_ops)
            if done: # log progress
                if episode%10==0:
                    # record reward
                    mean_reward=float(acc_reward)/10
                    sumOut=sess.run(re_sum,feed_dict={reward_summary:mean_reward})
                    writer.add_summary(sumOut,episode)
                    acc_reward=0
                # record loss
                mean_loss=float(loss)/epoch
                summary_loss=sess.run(loss_sum,feed_dict={loss_summary:mean_loss})
                writer.add_summary(summary_loss,episode)
                loss=0
