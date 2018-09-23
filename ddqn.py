#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:42:13 2018

@author: guille
"""
import sys
sys.path.insert(0,'gym-2048')
import gym
import gym_2048.envs.game_2048
import tensorflow as tf
import numpy as np

import replayMemory

# MAKE ENVIRONMENT
env=gym.make('game-2048-v0')

# CONSTANTS
LOGS_PATH="/tmp/ddqn_logs/"
SAVE_PATH="/tmp/ddqn_weights/"
STATE_SIZE=16
ACTION_RANGE=env.action_space.shape[0]
ACTION_SIZE=1
LEARNING_RATE=1e-5
EPSILON=0.2
TAU=1e-5
NUM_EPISODES=100000
MINIBATCH_SIZE=256
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
LEARNING_HAS_STARTED=False

replayMemory=replayMemory.replayMemory(MINIBATCH_SIZE,MEMORY_MAX_SIZE,VAR_SIZE_DIC)

tf.reset_default_graph()
sess=tf.Session()

# Make network
state_input_tensor=tf.placeholder(tf.float32,shape=(None,STATE_SIZE),name="state_input_tensor")
target_Q_tensor=tf.stop_gradient(tf.placeholder(tf.float32,shape=(None,ACTION_RANGE),name="target_Q_input_tensor"))

with tf.variable_scope("critic"):
    h1=tf.layers.dense(state_input_tensor,300,activation=tf.nn.relu,name="hidden_layer_1",reuse=False)
    h2=tf.layers.dense(h1,300,activation=tf.nn.relu,name="hidden_layer_2",reuse=False)
    h3=tf.layers.dense(h2,200,activation=tf.nn.relu,name="hidden_layer_3",reuse=False)
    h4=tf.layers.dense(h3,200,activation=tf.nn.relu,name="hidden_layer_4",reuse=False)
    h5=tf.layers.dense(h4,100,activation=tf.nn.relu,name="hidden_layer_5",reuse=False)
    
    # Advantage
    A1=tf.layers.dense(h5,100,activation=tf.nn.relu,name="advantage_layer_1",reuse=False)
    A_out=tf.layers.dense(A1,4,activation=None,name="advantage_output",reuse=False)
    
    #Value
    V1=tf.layers.dense(h5,60,activation=tf.nn.relu,name="value_layer_1",reuse=False)
    V_out=tf.layers.dense(V1,1,activation=None,name="value_output",reuse=False)
    
    output=V_out+A_out-tf.reduce_mean(A_out,axis=1, keepdims=True)

loss=tf.reduce_mean(tf.square(output-target_Q_tensor),name="loss")
train=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.variable_scope("target_critic"):
    h1_target=tf.layers.dense(state_input_tensor,300,activation=tf.nn.relu,name="hidden_layer_1",reuse=False)
    h2_target=tf.layers.dense(h1_target,300,activation=tf.nn.relu,name="hidden_layer_2",reuse=False)
    h3_target=tf.layers.dense(h2_target,200,activation=tf.nn.relu,name="hidden_layer_3",reuse=False)
    h4_target=tf.layers.dense(h3_target,200,activation=tf.nn.relu,name="hidden_layer_4",reuse=False)
    h5_target=tf.layers.dense(h4_target,100,activation=tf.nn.relu,name="hidden_layer_5",reuse=False)
    
    # Advantage
    A1_target=tf.layers.dense(h5_target,100,activation=tf.nn.relu,name="advantage_layer_1",reuse=False)
    A_out_target=tf.layers.dense(A1_target,4,activation=None,name="advantage_output",reuse=False)
    
    #Value
    V1_target=tf.layers.dense(h5_target,60,activation=tf.nn.relu,name="value_layer_1",reuse=False)
    V_out_target=tf.layers.dense(V1_target,1,activation=None,name="value_output",reuse=False)
    
    output_target=V_out_target+A_out_target-tf.reduce_mean(A_out_target,axis=1, keepdims=True)

critic_weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=("critic"))
target_critic_weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=("target_critic"))

update_target_ops=[]
with tf.variable_scope("TARGET_UPDATE"):
    for i in range(len(critic_weights)):
        update_target_op=target_critic_weights[i].assign(TAU*critic_weights[i]+(1-TAU)*target_critic_weights[i])
        update_target_ops.append(update_target_op)

summaryMerged=tf.summary.merge_all()

writer=tf.summary.FileWriter(LOGS_PATH,sess.graph)
loss_summary=tf.placeholder('float',name='Critic_loss_value')
reward_summary=tf.placeholder('float',name='Reward_value')
Q_summary=tf.placeholder('float',name='Q_value')
epoch_summary=tf.placeholder('float',name='epochs_per_episode')
loss_sum=tf.summary.scalar("Critic_loss", loss_summary)
re_sum=tf.summary.scalar("reward", reward_summary)
Q_sum=tf.summary.scalar("Q values",Q_summary)
epoch_sum=tf.summary.scalar("Epoch",epoch_summary)

init_op=tf.global_variables_initializer()
saver = tf.train.Saver()
tf.get_default_graph().finalize()
sess.run(init_op)

# Make target_network and network have same weights
TAU=1
sess.run(update_target_ops)
TAU=1e-4

def reward_tiles_merged(state,new_state):
    return -(np.count_nonzero(new_state)-1-np.count_nonzero(state))

def preprocess_state(state):
    np.seterr(divide='ignore')
    state=np.log2(np.reshape(np.asarray(state),(1,STATE_SIZE)))
    np.seterr(divide='warn')
    state[np.isneginf(state)]=0
    return state/11.0

for episode in range(NUM_EPISODES):
    if episode%100==0:
        print "Episode",episode,"of",NUM_EPISODES
    EPSILON*=0.999
    state=preprocess_state(env.reset())
    done=False
    my_loss=0
    epoch=0
    acc_reward=0
    acc_Q=0
    while not done:
        no_change=True
        no_change2=False
        while no_change:
            # Select action
            if (np.random.random()<EPSILON or no_change2):
                action=np.random.randint(0,ACTION_RANGE)
            else:
                Qs=sess.run(output,feed_dict={state_input_tensor:state})
                action=np.argmax(Qs)
            # Execute action
            new_state,reward,done,_=env.step(action)
            new_state=preprocess_state(new_state)
            no_change=np.array_equal(new_state,state)
            no_change2=no_change
        reward=reward_tiles_merged(state,new_state)
        # Record interaction
        replayMemory.add(new_state,reward,done,state,action)
        state=new_state
        epoch+=1
        acc_reward+=reward
        if episode>500:
            if not LEARNING_HAS_STARTED:
                LEARNING_HAS_STARTED=True
                print "Warmup phase over, started learning!"
            # Sample minibatch
            minibatch=replayMemory.get_batch()
            S=replayMemory.get_from_minibatch(minibatch,INDEX_STATE)
            St0=replayMemory.get_from_minibatch(minibatch,INDEX_LAST_STATE)
            A=replayMemory.get_from_minibatch(minibatch,INDEX_ACTION)
            D=replayMemory.get_from_minibatch(minibatch,INDEX_DONE)
            R=replayMemory.get_from_minibatch(minibatch,INDEX_REWARD)
            # Create targets
            next_states_Q=R+DISCOUNT_FACTOR*np.reshape(np.max(sess.run(output_target,feed_dict={state_input_tensor:S}),axis=-1),(MINIBATCH_SIZE,1))*(1-D)
            target_Q=sess.run(output,feed_dict={state_input_tensor:St0})
            target_Q[np.arange(MINIBATCH_SIZE),A.flatten()]=np.transpose(next_states_Q)
            acc_Q+=np.mean(np.amax(target_Q,axis=1))
            # Train network
            my_loss+=sess.run(loss,feed_dict={state_input_tensor:St0,target_Q_tensor:target_Q})
            sess.run(train,feed_dict={state_input_tensor:St0,target_Q_tensor:target_Q})
            # Update target network
            sess.run(update_target_ops)
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
                # record Q
                mean_Q=acc_Q/epoch
                summary_Q=sess.run(Q_sum,feed_dict={Q_summary:mean_Q})
                writer.add_summary(summary_Q,episode)
                acc_Q=0
                # record epoch
                summary_epoch=sess.run(epoch_sum,feed_dict={epoch_summary:epoch})
                writer.add_summary(summary_epoch,episode)
                if episode%1000==0:
                    saver.save(sess,SAVE_PATH)
                    print "Model saved in path: ",SAVE_PATH
done=False
for example in range(20):
    state=preprocess_state(env.reset())
    while not done:
        Qs=sess.run(output,feed_dict={state_input_tensor:state})
        action=np.argmax(Qs)
        new_state,reward,done,_=env.step(action)
        new_state=np.reshape(new_state,(1,STATE_SIZE))
        env.render()
        state=new_state
