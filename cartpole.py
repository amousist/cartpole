import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        # Do nothing if less experience than BATCH_SIZE
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Pick BATCH_SIZE samples from experience
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, finished in batch:

            if finished:
                q_update = reward
            else:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))

            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    # Size of observations
    observation_space = env.observation_space.shape[0]
    # Size of actions
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    game_nb = 0
    while True:
        game_nb += 1
        # Game creation/reset
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        finished = False
        # While game does not end
        while not finished:
            step += 1
            env.render()

            # Decide which move
            action = dqn_solver.act(state)

            # Perform the move
            state_next, reward, finished, info = env.step(action)
            state_next = np.reshape(state_next, [1, observation_space])

            # If the game has finished (we failed) invert the reward
            reward = reward if not finished else -reward

            # Save the movement and its reward
            dqn_solver.remember(state, action, reward, state_next, finished)

            state = state_next

            # Learn from experience
            dqn_solver.experience_replay()

            if finished:
                # Save result
                print ("Game number: " + str(game_nb) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, game_nb)


if __name__ == "__main__":
    cartpole()
