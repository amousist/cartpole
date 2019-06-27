# import numpy as np
# r= [1.0,1.0,1.0,-1.0,1.0,1.0]
# gamma = 0.5
#
# r = np.array(r)
# discounted_r = np.zeros_like(r)
# running_add = 0
# # we go from last reward to first one so we don't have to do exponentiations
# for t in reversed(range(0, r.size)):
#     if r[t] != 0:
#         running_add = 0 # if the game ended (in Pong), reset the reward sum
#     running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
#     discounted_r[t] = running_add
# discounted_r -= np.mean(discounted_r) #normalizing the result
# discounted_r /= np.std(discounted_r) #idem
# print (discounted_r)
#
# print ('{0:02b}'.format(0))

from gym import envs
envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)


import gym

# initializing our environment
env = gym.make('BipedalWalker-v2')

nb_actions = env.action_space.n

# beginning of an episode
observation = env.reset()
