import gym
import random
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam



def discount_rewards(r, gamma):
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    # we go from last reward to first one so we don't have to do exponentiations
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # if the game ended (in Pong), reset the reward sum
        running_add = running_add * gamma + r[
            t]  # the point here is to use Horner's method to compute those rewards efficiently
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)  # normalizing the result
    discounted_r /= np.std(discounted_r)  # idem

    #print(discounted_r)
    return discounted_r



########## Initialization ##########
# initializing our environment
env = gym.make('CartPole-v0')

nb_actions = env.action_space.n

# beginning of an episode
observation = env.reset()


# initialization of variables used in the main loop
x_train, y_train, rewards = [],[],[]

# Hyperparameters
gamma = 0.99

########## Model creation ##########

# Next, we build a very simple model.
model = Sequential()
model.add(Dense(24, input_shape=(4,)))
model.add(Activation('relu'))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
model.compile(loss="mse", optimizer=Adam(lr=0.001))

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())


movements = 0
episode_nb = 0
reward_sum = 0
########## Main Loop ##########
while(True):
    # render a frame
    #env.render()

    proba = model.predict(np.expand_dims(observation, axis=1).T)[0]

    action = proba.argmax(axis=-1)
    # log the input and label to train later
    x_train.append(observation)
    y_train.append(proba)

    #print(action)
    movements += 1

    # run one step
    observation, reward, done, info = env.step(action)

    reward_sum += reward
    # if the episode is over, reset the environment
    if done:
        rewards.append((-1) * reward)
        #print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)
        # increment episode number
        episode_nb += 1
        #print("X: " + str(x_train))
        #print("Y: " + str(y_train))
        #print("rewards: " + str(rewards))
        model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=0, sample_weight=discount_rewards(rewards, gamma))
        time.sleep(0.1)
        print("Movements: " + str(movements))
        movements = 0
        #time.sleep(1)

        observation = env.reset()
    else:
        rewards.append(reward)
    #time.sleep(0.1)

env.close()


