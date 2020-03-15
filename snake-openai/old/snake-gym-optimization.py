#!/usr/bin/env python

import numpy as np
import kinpy as kp
from scipy.optimize import minimize

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


# create kinematics
kin = RobotKinematics()
N_JOINTS = kin.get_joints_number()
JOINT_LIMIT = (np.pi - 0.0000001)/4.0

initial_state = np.array(N_JOINTS * [0.0]).astype(float)
goal_state = np.random.uniform(-JOINT_LIMIT, JOINT_LIMIT, N_JOINTS).astype(float)
goal_position = kin.calculate_direct_kinematic(goal_state)

# create function to minimize
def distance_from_goal(joint_state):
    global kin, goal_state
    current_position = kin.calculate_direct_kinematic(joint_state)
    return np.linalg.norm(goal_position - current_position)

# create bounds
bounds = ( (-JOINT_LIMIT, JOINT_LIMIT), ) * N_JOINTS

# run minimization
solution = minimize(distance_from_goal, initial_state, method='SLSQP', bounds=bounds)

# get results
final_state = solution.x
final_position = kin.calculate_direct_kinematic(final_state)
actual_distance = distance_from_goal(final_state)

print(goal_position)
print(final_position)
print(actual_distance)