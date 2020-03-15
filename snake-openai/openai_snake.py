from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent, SARSAAgent
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.policy import BoltzmannQPolicy

import tensorflow as tf

from gyms import SnakeGymContinuous, SnakeGymDiscrete
import numpy as np

N_NODE_NETWORK = 16

tf.compat.v1.disable_eager_execution()

def run_sarsa():

    global N_NODE_NETWORK

    env = SnakeGymDiscrete()
    nb_actions = env.action_space.n

    # initialize randomness
    np.random.seed(123)
    env.seed(123)

    # create model
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(N_NODE_NETWORK))
    model.add(Activation('relu'))
    model.add(Dense(N_NODE_NETWORK))
    model.add(Activation('relu'))
    model.add(Dense(N_NODE_NETWORK))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    # SARSA does not require a memory.
    policy = BoltzmannQPolicy()
    sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
    sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

    sarsa.fit(env, nb_steps=50000, visualize=False, verbose=2)
    sarsa.save_weights('sarsa_SnakeGymDiscrete_weights.h5f', overwrite=True)

    sarsa.test(env, nb_episodes=5, visualize=True)


def run_dqn():
    
    global N_NODE_NETWORK

    env = SnakeGymDiscrete()
    nb_actions = env.action_space.n

    # initialize randomness
    np.random.seed(123)
    env.seed(123)

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(N_NODE_NETWORK))
    model.add(Activation('relu'))
    model.add(Dense(N_NODE_NETWORK))
    model.add(Activation('relu'))
    model.add(Dense(N_NODE_NETWORK))
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
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
    dqn.save_weights('dqn_SnakeGymDiscrete_weights.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)


def run_ddpg():

    global N_NODE_NETWORK

    env = SnakeGymContinuous()
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # initialize randomness
    np.random.seed(123)
    env.seed(123)

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(N_NODE_NETWORK))
    actor.add(Activation('relu'))
    actor.add(Dense(N_NODE_NETWORK))
    actor.add(Activation('relu'))
    actor.add(Dense(N_NODE_NETWORK))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(N_NODE_NETWORK*2)(x)
    x = Activation('relu')(x)
    x = Dense(N_NODE_NETWORK*2)(x)
    x = Activation('relu')(x)
    x = Dense(N_NODE_NETWORK*2)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                    memory=memory, nb_steps_warmup_critic=500, nb_steps_warmup_actor=500,
                    random_process=random_process, gamma=.99, target_model_update=1e-3)
    
    agent.compile('adam', metrics=['mae'])

    agent.fit(env, nb_steps=50000, visualize=True, verbose=2, nb_max_episode_steps=200)
    agent.save_weights('ddpg_SnakeGymContinuous_weights.h5f', overwrite=True)

    agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)


