#!/usr/bin/env python3

# Compatibility with unicode
from __future__ import print_function, unicode_literals

# ROS libraries
import rospy
from sensor_msgs.msg import JointState

# Direct kinematics
import kinpy as kp

# Inverse kinematics library
from trac_ik_python.trac_ik import IK

# Inverse kinematics through optimization
import numpy as np
from scipy.optimize import minimize

# XML library
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

# get timestamp
from datetime import datetime

# Command-line interface
from PyInquirer import prompt, print_json, Validator, ValidationError


class RobotMover:
    """ Interface with the robot. Get and set its joint states"""

    PUBLISHER_TOPIC = "move_group/fake_controller_joint_states"
    SUBSCRIBER_TOPIC = "joint_states"
    joint_positions = []

    def __init__(self):
        self.pub = rospy.Publisher(self.PUBLISHER_TOPIC, JointState, queue_size=1)
        rospy.Subscriber(self.SUBSCRIBER_TOPIC, JointState, self.reader_callback, queue_size=1)

    def reader_callback(self, message):
        self.joint_positions = []
        for i,name in enumerate(message.name):
            elem = (name, message.position[i])
            self.joint_positions.append(elem)

    def get_joint_name_state(self):
        return self.joint_positions

    def read_joint(self, joint_name):
        return self.joint_positions[joint_name]

    def write_joint(self, joint_name, joint_value):
        message = JointState()
        message.name.append(joint_name)
        message.position.append(joint_value)
        message.velocity.append(0.0)
        self.pub.publish(message)

    def read_joints(self):
        return list(map(lambda elem: elem[1], self.joint_positions))

    def write_joints(self, joint_angles):
        message = JointState()
        for i in range(0, len(self.joint_positions)):
            joint_name = self.joint_positions[i][0]
            joint_value = joint_angles[i]
            message.name.append(joint_name)
            message.position.append(joint_value)
            message.velocity.append(0.0)
        self.pub.publish(message)


class RobotKinematics:
    """Calculate direct and inverse kinematics through different methods"""

    def __init__(self, from_rosparam=False):
        if from_rosparam:
            try:
                urdf = rospy.get_param('/robot_description')
            except:
                urdf = open("modello.urdf").read()
        else:
            urdf = open("modello.urdf").read()
        
        urdf = urdf.replace('encoding="utf-8"', '').replace('encoding=\'utf-8\'', '')
        self.chain = kp.build_chain_from_urdf(urdf)
        self.njoints = len( self.chain.get_joint_parameter_names() )
        self.end_link = "end_effector" # "link{}".format(self.njoints)
        self.ik_solver = IK("link0", self.end_link, urdf_string=urdf)

    def calculate_direct_kinematic(self, joint_angles):
        assert len(joint_angles) == self.njoints

        joint_state = {}
        for i in range(0, len(joint_angles)):
            joint_state["joint{}".format(i)] = joint_angles[i]

        all_link_positions = self.chain.forward_kinematics(joint_state)
        return all_link_positions[self.end_link].pos

    def calculate_inverse_kinematics_ik(self, x, y, z):
        initial_state = np.array(self.njoints * [0.0]).astype(float)
        return self.ik_solver.get_ik(initial_state, x, y, z, 0, 0, 0, 1.0, 1e-8, 1e-8, 1e-8)

    def calculate_inverse_kinematics_optimization(self, x, y, z):
        
        goal_position = np.array([x, y, z])
        print("Optimize ", goal_position)

        def distance_from_goal(joint_angles):
            current_position = self.calculate_direct_kinematic(joint_angles)
            return np.linalg.norm(goal_position - current_position)

        JOINT_LIMIT = (np.pi - 0.0000001)/4.0
        N_JOINTS = self.njoints
        bounds = ( (-JOINT_LIMIT, JOINT_LIMIT), ) * N_JOINTS

        initial_state = np.array(N_JOINTS * [0.0]).astype(float)
        print("From initial state ", initial_state)
        solution = minimize(distance_from_goal, initial_state, method='SLSQP', bounds=bounds)
        return solution.x

    def get_joints_number(self):
        return self.njoints


class SnakeBuilder:
    """Create XACRO model of our snake"""
    SIZE_SCALE = 0.1
    DIAMETER_PROPERTY = 'diameter'
    LENGTH_PROPERTY = 'length'

    @staticmethod
    def __property(root, name, value):
        SubElement(root, 'xacro:property', { 'name' : str(name), 'value': str(value) })

    @staticmethod
    def __material(root, name, r, g, b):
        material = SubElement(root, 'material', { 'name' : str(name) })
        SubElement(material, 'color', { 'rgba' : "{} {} {} 1".format(r, g, b) })

    @staticmethod
    def __link(root, name, xyz, material, is_endeffector=False):
        link = SubElement(root, 'link', { 'name' : str(name) })
        visual = SubElement(link, 'visual', {})
        SubElement(visual, 'origin', { 'xyz': str(xyz)  })
        geometry = SubElement(visual, 'geometry', {})
        if not is_endeffector:
            SubElement(geometry, 'cylinder', {
                'length' : "${{{}}}".format(SnakeBuilder.LENGTH_PROPERTY),
                'radius': "${{{}/2}}".format(SnakeBuilder.DIAMETER_PROPERTY)
            })
        else:
            SubElement(geometry, 'sphere', {
                'radius': "0.1"
            })
        if material is not None:
            SubElement(visual, 'material', { 'name' : str(material) })

    @staticmethod
    def __joint(root, joint_name, parent_link_name, child_link_name, joint_xyz, joint_axis, joint_type='revolute'):
        joint = SubElement(root, 'joint', { 'name': str(joint_name), 'type': joint_type })
        SubElement(joint, 'parent', { 'link': str(parent_link_name) })
        SubElement(joint, 'child', { 'link': str(child_link_name) })
        if joint_axis is not None:
            SubElement(joint, 'axis', { 'xyz': str(joint_axis) })
        if joint_xyz is not None:
            SubElement(joint, 'origin', { 'xyz': str(joint_xyz) })
        SubElement(joint, 'limit', {
            'effort': '100',
            'velocity': '100',
            'lower': '-${pi/4}',
            'upper': '${pi/4}'
        })

    @staticmethod
    def __end_effector(root, last_link_name):
        SnakeBuilder.__material(root, "end_effector_material", 1, 1, 1)
        SnakeBuilder.__link(root, "end_effector", "0 0 ${{{}}}".format(SnakeBuilder.LENGTH_PROPERTY), "end_effector_material", is_endeffector=True)
        SnakeBuilder.__joint(root, "joint_end_effector", last_link_name, "end_effector", None, None, "fixed")

    @staticmethod
    def build(n_pieces, diameter, length):
        root = Element('robot', { 'name': 'snake', 'xmlns:xacro' : 'http://ros.org/wiki/xacro' })
        SnakeBuilder.__property(root, SnakeBuilder.DIAMETER_PROPERTY, diameter*SnakeBuilder.SIZE_SCALE)
        SnakeBuilder.__property(root, SnakeBuilder.LENGTH_PROPERTY, length*SnakeBuilder.SIZE_SCALE)
        # create materials
        MATERIALS = [ ( 'color_{}'.format(chr(ord('a')+i)), (i >> 2) % 2, (i >> 1) % 2, i % 2) for i in range(0, 8) ]
        for (name, r, g, b) in MATERIALS:
            SnakeBuilder.__material(root, name, r, g, b)
        # create links
        for i in range(0, n_pieces):
            material_name = MATERIALS[i % len(MATERIALS)][0]
            SnakeBuilder.__link(root, "link{}".format(i), "0 0 ${{{}/2}}".format(SnakeBuilder.LENGTH_PROPERTY), material_name)
        # create joints
        for i in range(0, n_pieces-1):
            axis = ['1 0 0', '0 1 0'][i % 2]
            SnakeBuilder.__joint(root, "joint{}".format(i), "link{}".format(i), "link{}".format(i+1), "0 0 ${{{}}}".format(SnakeBuilder.LENGTH_PROPERTY), axis)
        # create end effector
        SnakeBuilder.__end_effector(root, "link{}".format(n_pieces-1))
        # print to string
        rough_string = ElementTree.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")
        return pretty


class App:

    class NumberValidator(Validator):
        def validate(self, document):
            try:
                float(document.text)
            except ValueError:
                raise ValidationError(
                    message='Please enter a correct float',
                    cursor_position=len(document.text))  # Move cursor to end

    class NaturalValidator(Validator):
        def validate(self, document):
            try:
                if int(document.text) <= 0:
                    raise ValueError()                
            except ValueError:
                raise ValidationError(
                    message='Please enter a correct float',
                    cursor_position=len(document.text))  # Move cursor to end

    NEXT_MOVE_SEE_STATE = 'See joint state'

    NEXT_MOVE_DIRECT_KIN = 'Direct kinematics - calculate end effector position'

    NEXT_MOVE_INVERSE_KIN_IK = 'Inverse kinematics (TRAC-IK) - move end effector'

    NEXT_MOVE_INVERSE_KIN_OPT = 'Inverse kinematics (optimization) - move end effector'

    NEXT_MOVE_GENERATE_MODEL = 'Generate a new XACRO snake module'

    NEXT_MOVE_EXIT = 'Exit'

    QUESTION_NEXT_MOVE = [{
        'type': 'list',
        'name': 'move',
        'message': 'What\'s your next move?',
        'choices': [NEXT_MOVE_SEE_STATE, NEXT_MOVE_DIRECT_KIN, NEXT_MOVE_INVERSE_KIN_IK, NEXT_MOVE_INVERSE_KIN_OPT, NEXT_MOVE_GENERATE_MODEL, NEXT_MOVE_EXIT]
    }]

    QUESTION_GET_POSITION = [
        {
            'type': 'input',
            'name': 'x',
            'message': 'X coordinate?',
            'validate': NumberValidator,
            'filter': lambda x: float(x)
        },
        {
            'type': 'input',
            'name': 'y',
            'message': 'Y coordinate?',
            'validate': NumberValidator,
            'filter': lambda x: float(x)
        },
        {
            'type': 'input',
            'name': 'z',
            'message': 'Z coordinate?',
            'validate': NumberValidator,
            'filter': lambda x: float(x)
        }
    ]

    QUESTION_BUILD_URDF = [
        {
            'type': 'input',
            'name': 'n_pieces',
            'message': 'How many pieces does the shake have?',
            'validate': NaturalValidator,
            'filter': lambda x: int(x)
        },
        {
            'type': 'input',
            'name': 'diameter',
            'message': 'How is the diameter of each piece?',
            'validate': NumberValidator,
            'filter': lambda x: float(x)
        },
        {
            'type': 'input',
            'name': 'length',
            'message': 'How long is each piece?',
            'validate': NumberValidator,
            'filter': lambda x: float(x)
        },
        {
            'type': 'confirm',
            'name': 'overwrite',
            'message': 'Do you want to overwrite "modello.xacro"?',
            'default': False
        }
    ]

    def __init__(self, mover, kin):
        self.mover = mover
        self.kin = kin


    def print_table_row(self, name, value):
        print("{0:>7s} : {1: 1.4f}".format(name, value))

    def print_joint_state(self, values=None):
        print("Joint state:")
        if values is None:
            for (name, value) in self.mover.get_joint_name_state():
                self.print_table_row(name, value)
        else:
            for i in range(0, len(values)):
                self.print_table_row("joint{}".format(i), values[i])

    def print_position(self, position): 
        print("Position:")
        self.print_table_row("x", position[0])
        self.print_table_row("y", position[1])
        self.print_table_row("z", position[2])

    def ask_for_something(self):

        answer = prompt(self.QUESTION_NEXT_MOVE)

        try:
            move = answer['move']
        except KeyError:
            print("Please use arrow keys to choose an action, not the mouse pointer")
            return True

        if move == self.NEXT_MOVE_SEE_STATE:
            self.print_joint_state()

        elif move == self.NEXT_MOVE_DIRECT_KIN:
            joint_state = self.mover.read_joints()
            endeffector_position = self.kin.calculate_direct_kinematic(joint_state)
            self.print_position(endeffector_position)

        elif move in [ self.NEXT_MOVE_INVERSE_KIN_IK, self.NEXT_MOVE_INVERSE_KIN_OPT]:
            coord = prompt(self.QUESTION_GET_POSITION)
            
            if move == self.NEXT_MOVE_INVERSE_KIN_IK:
                new_joint_state = self.kin.calculate_inverse_kinematics_ik(coord['x'], coord['y'], coord['z'])
            else:
                new_joint_state = self.kin.calculate_inverse_kinematics_optimization(coord['x'], coord['y'], coord['z'])
            
            if new_joint_state is None:
                print("Cannot reach given point")
            else:
                self.mover.write_joints(new_joint_state)
                self.print_joint_state(values=new_joint_state)
                new_position = self.kin.calculate_direct_kinematic(new_joint_state)
                self.print_position(new_position)

        elif move == self.NEXT_MOVE_GENERATE_MODEL:
            robot_spec = prompt(self.QUESTION_BUILD_URDF)
            n_pieces, diameter, length, ts = robot_spec["n_pieces"], robot_spec["diameter"], robot_spec["length"], datetime.now().strftime('%Y%m%d%H%M%S')
            overwrite = robot_spec["overwrite"]
            output_urdf = SnakeBuilder.build(n_pieces, diameter, length)
            output_filename = "/root/catkin_ws/src/snake/modello_n{}_diam{}_len{}_ts{}.xacro".format(n_pieces, diameter, length, ts)
            open(output_filename, mode="w+").write(output_urdf)
            if overwrite:
                open("/root/catkin_ws/src/snake/modello.xacro", mode="w+").write(output_urdf)

        elif move == self.NEXT_MOVE_EXIT:
            return False
        
        else:
            raise AssertionError("Cannot understand move: {}".format(move))

        return True


if __name__ == '__main__':
    try:
        rospy.init_node('ros_snake', anonymous=True)
        rate = rospy.Rate(1)
        app = App(RobotMover(), RobotKinematics(from_rosparam=True))
        go_on = True
        while not rospy.is_shutdown() and go_on:
            go_on = app.ask_for_something()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
