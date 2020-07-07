#!/usr/bin/python3.7

# Direct kinematics
import kinpy as kp

# Inverse kinematics through optimization
import numpy as np
import time
from math import cos, sin
from scipy.optimize import minimize
import pandas as pd
import re

# drawing
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


class RobotKinematics:
    """Calculate direct and inverse kinematics through different methods"""

    def __init__(self, urdf_mode_path):
        urdf = open(urdf_mode_path).read()        
        urdf = urdf.replace('encoding="utf-8"', '').replace('encoding=\'utf-8\'', '')
        self.chain = kp.build_chain_from_urdf(urdf)
        self.njoints = len( self.chain.get_joint_parameter_names() )
        self.end_link = "end_effector" # "link{}".format(self.njoints)

    @staticmethod
    def point_to_point_distance(p1, p2):
        """distance between two points"""
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def segment_to_point_distance(p, a, b):
        """perp. distance between point `p` and segment [a,b]"""
        # normalized tangent vector
        d = np.divide(b - a, np.linalg.norm(b - a))
        # signed parallel distance components
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)
        # clamped parallel distance
        h = np.maximum.reduce([s, t, 0])
        # perpendicular distance component
        c = np.cross(p - a, d)
        return np.hypot(h, np.linalg.norm(c))

    @staticmethod
    def manipulator_to_point_distance(p, points):
        distances = map(lambda i: RobotKinematics.segment_to_point_distance(p, points[i], points[i+1]), range(0, len(points)-1))
        return min(distances)

    def angle_to_positions(self, joint_angles):
        # check input
        assert len(joint_angles) == self.njoints, "Model needs exactly {} angles ({} currently given)".format(self.njoints, len(joint_angles))
        # from angles construct a `joint state` structure
        joint_state = {}
        for i in range(0, len(joint_angles)):
            joint_state["joint{}".format(i)] = joint_angles[i]
        # apply direct kinematics
        all_link_positions = self.chain.forward_kinematics(joint_state)
        # get positions
        manipulator_positions = []
        for i in range(0, self.njoints+1):
            joint_name = "link{}".format(i)
            manipulator_positions.append(all_link_positions[joint_name].pos)
        manipulator_positions.append(all_link_positions[self.end_link].pos)
        return manipulator_positions

    def zero_angles_to_position(self):
        return self.angle_to_positions(np.array([0] * self.njoints))

    def search_goal_only(self, goal_position, method):
        
        def cost_function(joint_angles):
            # cost function = distance from goal
            endeffector_position = self.angle_to_positions(joint_angles)[-1]
            return np.linalg.norm(goal_position - endeffector_position)

        return self.search(cost_function, method)

    def search_goal_and_intermediates(self, goal_position, intermediate_positions, method):
        
        def cost_function(joint_angles):
            # cost function = distance from goal
            cost = 0
            GOAL_DISTANCE_COST = 10
            INTERMEDIATE_DISTANCE_COST = 3
            # add endeffector position cost 
            positions = self.angle_to_positions(joint_angles)
            endeffector_position = positions[-1]
            distance_from_goal = np.linalg.norm(goal_position - endeffector_position)
            cost = cost + distance_from_goal * GOAL_DISTANCE_COST
            # add each intermediate point cost
            for point in intermediate_positions:
                cost = cost + RobotKinematics.manipulator_to_point_distance(point, positions) * INTERMEDIATE_DISTANCE_COST
            return cost

        return self.search(cost_function, method)

    def search(self, cost_function, method):
        JOINT_LIMIT = (np.pi - 0.0000001)/2.0
        bounds = ( (-JOINT_LIMIT, JOINT_LIMIT), ) * self.njoints
        initial_state = np.array(self.njoints * [0.0]).astype(float)
        solution = minimize(cost_function, initial_state, method=method, bounds=bounds)
        return solution


class Tester:

    METHODS = [ 'SLSQP', 'COBYLA' ] # [ 'L-BFGS-B', 'TNC', 'SLSQP', 'COBYLA' ]

    MODELS = [ "modello_8.urdf", "modello_16.urdf", "modello_24.urdf", "modello_32.urdf", "modello_40.urdf", "modello_48.urdf" ]

    COLUMNS = [ 'main_point_given', 'main_point_inferred', 'main_point_error (%)',
                'middle_point_given', 'middle_point_error (%)', 'time']

    NPOINTS = 200

    @staticmethod
    def pick_a_point(radius):
        r = radius * np.sqrt(np.random.uniform())
        theta = np.random.uniform() * np.pi/2
        x, y = r*cos(theta), r*sin(theta)
        xy2 = x**2 + y**2
        z = np.sqrt(radius**2 - xy2)
        z = np.random.uniform() * z
        return np.array([x, y, z])

    def __init__(self):
        # generate empty dataframe
        self.dataframe_collection = {}
        for model in self.MODELS:
            for method in self.METHODS:
                self.dataframe_collection[(model, method)] = pd.DataFrame(columns=self.COLUMNS)
        # generate point set
        self.point_collection = {}
        for model in self.MODELS:
            rk = RobotKinematics(model)
            max_radius = np.linalg.norm(rk.zero_angles_to_position()[-1])
            POINTS = [ Tester.pick_a_point(max_radius) for i in range(0,self.NPOINTS) ]
            MIDWAY_POINTS = [ Tester.pick_a_point(max_radius/2) for i in range(0,self.NPOINTS) ]
            self.point_collection[model] = pd.DataFrame(list(zip(POINTS, MIDWAY_POINTS)), columns=["main_point", "middle_point"])


    def run(self, consider_middle_point=False):
        print("\nTester::run")
        for model in self.MODELS:
            rk = RobotKinematics(model)
            for method in self.METHODS:
                df = self.dataframe_collection[(model, method)]
                for i in range(self.NPOINTS):
                    main_point = self.point_collection[model]["main_point"][i]
                    middle_point = self.point_collection[model]["middle_point"][i]
                    start = time.time()
                    if not consider_middle_point:
                        sol = rk.search_goal_and_intermediates(main_point, [middle_point], method)
                        inferred_main_point = rk.angle_to_positions(sol.x)[-1]
                        # errore percentuale
                        main_point_error = np.linalg.norm(main_point - inferred_main_point)*100/np.linalg.norm(main_point)
                        middle_point_error = RobotKinematics.manipulator_to_point_distance(middle_point, rk.angle_to_positions(sol.x))*100/np.linalg.norm(middle_point)
                    else:
                        sol = rk.search_goal_only(main_point, method)
                        inferred_main_point = rk.angle_to_positions(sol.x)[-1]
                        # errore percentuale
                        main_point_error = np.linalg.norm(main_point - inferred_main_point)*100/np.linalg.norm(main_point)
                        middle_point_error = -1
                    end = time.time()
                    # save result
                    df.loc[len(df)] = [ main_point, main_point_error, inferred_main_point, middle_point, middle_point_error, (end-start) ]
                    print(".", end="", flush=True)
        print()

    def write_to_files(self):
        for model in self.MODELS:
            for method in self.METHODS:
                self.dataframe_collection[(model, method)].to_csv("dataframe_output/df_{}_{}.csv".format(re.findall(r'\d+', model)[0], method))
        print("Wrote correctly to file")

    def plot_points(self, model, max_radius):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = list( item[0] for item in self.point_collection[model]["main_point"] )
        y = list( item[1] for item in self.point_collection[model]["main_point"] )
        z = list( item[2] for item in self.point_collection[model]["main_point"] )
        ax.scatter(x, y, z, c='r', marker='o')
        x = list( item[0] for item in self.point_collection[model]["middle_point"] )
        y = list( item[1] for item in self.point_collection[model]["middle_point"] )
        z = list( item[2] for item in self.point_collection[model]["middle_point"] )
        ax.scatter(x, y, z, c='b', marker='o')
        # Make data.
        X = np.arange(0, max_radius, 0.001)
        Y = np.arange(0, max_radius, 0.001)
        X, Y = np.meshgrid(X, Y)
        XY2 = X**2 + Y**2
        Z = np.sqrt(np.abs((max_radius**2 - XY2).clip(min=0)))
        # Plot the surface.
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.5, linewidth=0, antialiased=False)
        plt.show()


tester = Tester()
tester.plot_points(tester.MODELS[0], 0.3)
tester.run(consider_middle_point=False)
tester.write_to_files()














# Method: L-BFGS-B
# Model:       modello_4.urdf | mean error= 0.0573 | mean time= 0.0057
# Model:       modello_8.urdf | mean error= 0.0172 | mean time= 0.0933
# Model:      modello_16.urdf | mean error= 0.0183 | mean time= 0.7823
# Model:      modello_32.urdf | mean error= 0.0142 | mean time= 9.2236

# Method: TNC
# Model:       modello_4.urdf | mean error= 0.0438 | mean time= 0.0190
# Model:       modello_8.urdf | mean error= 0.0303 | mean time= 0.2682
# Model:      modello_16.urdf | mean error= 0.0281 | mean time= 1.6415
# Model:      modello_32.urdf | mean error= 0.0481 | mean time= 12.8617

# Method: COBYLA
# /home/incud/.local/lib/python3.7/site-packages/scipy/optimize/_minimize.py:528: RuntimeWarning: Method COBYLA cannot handle bounds.
#   RuntimeWarning)
# Model:       modello_4.urdf | mean error= 0.0518 | mean time= 0.0699
# Model:       modello_8.urdf | mean error= 0.0270 | mean time= 0.2918
# Model:      modello_16.urdf | mean error= 0.0298 | mean time= 0.6613
# Model:      modello_32.urdf | mean error= 0.0849 | mean time= 1.2893

# Method: SLSQP
# Model:       modello_4.urdf | mean error= 0.0337 | mean time= 0.0055
# Model:       modello_8.urdf | mean error= 0.0185 | mean time= 0.0489
# Model:      modello_16.urdf | mean error= 0.0151 | mean time= 0.2597
# Model:      modello_32.urdf | mean error= 0.0212 | mean time= 1.5779
