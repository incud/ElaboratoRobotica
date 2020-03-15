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

class SnakeGymContinuous(gym.Env):

    class NoInfo:
        def items(self):
            return []

    # Spaces: https://ai-mrkogao.github.io/reinforcement%20learning/openaigymtutorial/
    # Own enironment: https://mc.ai/creating-a-custom-openai-gym-environment-for-stock-trading/

    metadata = {'render.modes': ['human']}

    JOINT_LIMIT = np.pi/4

    def __init__(self, kinematics, MOVE_DELTA=1e-2):
        self.N_JOINTS = kinematics.get_joints_number()
        self.MOVE_DELTA = MOVE_DELTA
        self.joint_limits = np.array(self.N_JOINTS * [np.pi/4], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.joint_limits, high=self.joint_limits, dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.joint_limits, high=self.joint_limits, dtype=np.float32)
        self.kinematics = kinematics
        self.reset()

    def __parse_action(self, action):
        joint = int(action // 2)
        direction = int(action % 2)
        if joint >= self.N_JOINTS: raise ValueError("Invalid action: {} | NJOINTS: {}, joint: {}".format(action, self.N_JOINTS, joint))
        if direction == 0: direction = -1
        return (joint, direction)

    def __render_state(self, state):
        position = self.kinematics.calculate_direct_kinematic(state)
        return "DIST {0:3.3f} | XYZ {1: 1.6f} {2: 1.6f} {3: 1.6f}".format(self.distance, position[0], position[1], position[2])
        #return "DIST {0:3.3f} | XYZ {1: 1.6f} {2: 1.6f} {3: 1.6f} | state {4}".format(self.distance, position[0], position[1], position[2], state)

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

        next_state = np.clip(action, -1 * self.joint_limits, self.joint_limits)

        next_position = self.kinematics.calculate_direct_kinematic(next_state)

        old_distance = self.distance
        self.distance = np.linalg.norm(self.goal - next_position)
        reward = 0 if old_distance > self.distance else -1 
        # -1 * self.distance 
        # 1.0/self.distance if self.distance < old_distance else -1 * self.distance

        if self.distance < self.best_distance:
            self.best_distance = self.distance
            self.best_state = next_state

        self.current_state = next_state
        self.done = False
        self.info = SnakeGymContinuous.NoInfo()

        return self.current_state, reward, self.done, self.info


if __name__ == '__main__':

    kin = RobotKinematics()

    tf.compat.v1.disable_eager_execution()

    # Get the environment and extract the number of actions.
    env = SnakeGymContinuous(kin)
    np.random.seed(123)
    env.seed(123)
    print(env.action_space)
    print(env.action_space.shape)
    nb_actions = env.action_space.shape[0]

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                    memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                    random_process=random_process, gamma=.99, target_model_update=1e-3)
    #agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    agent.compile('adam', metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    agent.fit(env, nb_steps=50000, visualize=True, verbose=2, nb_max_episode_steps=400)

    print("Best distance found: ", env.best_distance)
    print("Best state found: ", env.best_state)

    # Finally, evaluate our algorithm for 5 episodes.
    # agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)

