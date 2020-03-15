import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import kinpy as kp

# Spaces: https://ai-mrkogao.github.io/reinforcement%20learning/openaigymtutorial/
# Own enironment: https://mc.ai/creating-a-custom-openai-gym-environment-for-stock-trading/

class SnakeGym(gym.Env):

    metadata = {'render.modes': ['human']}

    JOINT_LIMIT = (np.pi - 0.000001)/4

    URDF_MODEL_PATH = "modello.urdf"

    URDF_ENDEFFECTOR_NAME = "end_effector"

    class NoInfo:
        def items(self):
            return []

    def __init__(self):
        self.chain = kp.build_chain_from_urdf(open(SnakeGym.URDF_MODEL_PATH).read())
        self.N_JOINTS = len( self.chain.get_joint_parameter_names() )
        self.JOINT_LIMITS = np.array(self.N_JOINTS * [ SnakeGym.JOINT_LIMIT ])
        self.reset()

    # ===================== UTILITIES ======================

    def regenerate_goal(self):
        goal_state = np.random.uniform(-1 * self.JOINT_LIMIT, self.JOINT_LIMIT, self.N_JOINTS).astype(float)
        self.goal_position = self.calculate_direct_kinematic(goal_state)
        return self.goal_position

    def calculate_direct_kinematic(self, joint_angles):
        assert len(joint_angles) == self.N_JOINTS
        joint_state = { "joint{}".format(i) : joint_angles[i] for i in range(0, len(joint_angles)) }
        all_link_positions = self.chain.forward_kinematics(joint_state)
        return all_link_positions["end_effector"].pos

    def calculate_distance_from_goal(self, joint_angles):
        return np.linalg.norm(self.goal_position - self.current_position)

    def calculate_next_state(self, action):
        raise NotImplementedError

    # =================== OPEN-AI GYM API ===================

    def reset(self):
        self.current_state = np.array(self.N_JOINTS * [0]).astype(float)
        self.current_position = self.calculate_direct_kinematic(self.current_state)
        self.regenerate_goal()
        return self.current_position

    def render(self):
        distance = self.calculate_distance_from_goal(self.current_state)
        render_str = "{0:2.4f} | {1: 1.4f} {2: 1.4f} {3: 1.4f}".format(distance, self.current_position[0], self.current_position[1], self.current_position[2])
        print(render_str)

    def step(self, action):

        # update state
        self.current_state = self.calculate_next_state(action)
        self.current_position = self.calculate_direct_kinematic(self.current_state)

        # calculate reward and done flag
        distance = self.calculate_distance_from_goal(self.current_state)
        reward = -1 * distance
        done = distance < 1e-2

        return self.current_state, reward, done, SnakeGym.NoInfo()    


class SnakeGymDiscrete(SnakeGym):
    
    MOVE_DELTA = 1e-2

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(self.N_JOINTS * 2)
        self.observation_space = spaces.Box(low=-self.JOINT_LIMITS, high=self.JOINT_LIMITS)

    def calculate_next_state(self, action):
        action_joint, action_direction = int(action // 2), (-1)**int(action % 2)
        assert action_joint < self.N_JOINTS and action_direction in {1, -1}
        next_state = np.copy(self.current_state)
        next_state[action_joint] = next_state[action_joint] + action_direction * self.MOVE_DELTA
        next_state[action_joint] = np.clip(next_state[action_joint], -1 * self.JOINT_LIMIT, self.JOINT_LIMIT)
        return next_state


class SnakeGymContinuous(gym.Env):

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-self.JOINT_LIMITS, high=self.JOINT_LIMITS, dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.JOINT_LIMITS, high=self.JOINT_LIMITS, dtype=np.float32)

    def calculate_next_state(self, action):
        next_state = np.clip(action, -1 * self.JOINT_LIMIT, self.JOINT_LIMIT)
        return next_state
