#!/usr/bin/env python

import numpy as np
import kinpy as kp

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

class RobotKinematics:

    def __init__(self):
        self.chain = kp.build_chain_from_urdf(open("modello.urdf").read())
        self.njoints = len( self.chain.get_joint_parameter_names() )
        self.end_link = "link{}".format(self.njoints)
        print("End link is: ", self.end_link)

    def calculate_direct_kinematic(self, joint_angles):

        assert len(joint_angles) == self.njoints

        joint_state = {}
        for i in range(0, len(joint_angles)):
            joint_state["joint{}".format(i)] = joint_angles[i]

        all_link_positions = self.chain.forward_kinematics(joint_state)
        return all_link_positions[self.end_link].pos

    def get_joints_number(self):
        return self.njoints

class SnakeGym(gym.Env):

    class NoInfo:
        def items(self):
            return []

    # Spaces: https://ai-mrkogao.github.io/reinforcement%20learning/openaigymtutorial/
    # Own enironment: https://mc.ai/creating-a-custom-openai-gym-environment-for-stock-trading/

    metadata = {'render.modes': ['human']}

    JOINT_LIMIT = np.pi/4

    def __init__(self, kinematics, MOVE_DELTA=0.1):
        self.N_JOINTS = kinematics.get_joints_number()
        self.MOVE_DELTA = MOVE_DELTA
        self.action_space = spaces.Discrete(self.N_JOINTS * 2)
        self.joint_limits = np.array(self.N_JOINTS * [np.pi/4])
        self.observation_space = spaces.Box(low=-self.joint_limits, high=self.joint_limits)
        self.kinematics = kinematics
        self.reset()     

    def __parse_action(self, action):
        joint = int(action // 2)
        direction = int(action % 2)
        if joint >= self.N_JOINTS: raise ValueError("Invalid action: {} | NJOINTS: {}, joint: {}".format(action, self.N_JOINTS, joint))
        if direction == 0: direction = -1
        return (joint, direction)

    def __render_state(self, state):
        state = self.kinematics.calculate_direct_kinematic(state)
        return "DIST {0:3.3f} | XYZ {1: 1.6f} {2: 1.6f} {3: 1.6f}".format(self.distance, state[0], state[1], state[2])

    def reset(self):
        self.current_state = np.array(self.N_JOINTS * [0]).astype(float)
        self.goal_state = np.random.uniform(-1 * self.JOINT_LIMIT, self.JOINT_LIMIT, self.N_JOINTS).astype(float)
        self.goal = self.kinematics.calculate_direct_kinematic(self.goal_state)
        self.distance = np.linalg.norm(self.goal - self.kinematics.calculate_direct_kinematic(self.current_state))

        self.best_distance = self.distance
        self.best_state = self.current_state

        print("Setting GOAL to ", self.__render_state(self.goal_state))
        return self.current_state

    def render(self, mode='human', close=False):
        print(self.__render_state(self.current_state))

    def step(self, action):

        joint, direction = self.__parse_action(action)
        next_state = np.copy(self.current_state)
        next_state[joint] = next_state[joint] + direction * self.MOVE_DELTA
        next_state[joint] = min(next_state[joint], self.JOINT_LIMIT)
        next_state[joint] = max(next_state[joint], -1*self.JOINT_LIMIT)

        next_position = self.kinematics.calculate_direct_kinematic(next_state)
        self.distance = np.linalg.norm(self.goal - next_position)
        reward = -1 * self.distance

        if self.distance < self.best_distance:
            self.best_distance = self.distance
            self.best_state = next_state

        self.current_state = next_state
        self.done = reward > -0.01
        self.info = SnakeGym.NoInfo()

        return self.current_state, reward, self.done, self.info


if __name__ == '__main__':

    kin = RobotKinematics()
    tf.compat.v1.disable_eager_execution()

    # Get the environment and extract the number of actions.
    env = SnakeGym(kin)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                target_model_update=1e-2, policy=policy)


    adam = Adam(lr=1e-3)
    # setattr(adam, "_name", "Adam")
    dqn.compile(adam, metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2, nb_max_episode_steps=2000)

    print("Best distance found: ", env.best_distance)
    print("Best state found: ", env.best_state)

    # After training is done, we save the final weights.
    #dqn.save_weights('dqn_{}_weights.h5f'.format("snakev1"), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    #dqn.test(env, nb_episodes=5, visualize=True)
