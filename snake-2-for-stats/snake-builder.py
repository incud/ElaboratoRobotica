# XML library
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

# get timestamp
from datetime import datetime


class SnakeBuilderEnhanced: 

    def __init__(self, n_joints, length, diameter):
        self.n_joints = n_joints
        self.length = length
        self.diameter = diameter

    @staticmethod
    def tag_material(root, name, r, g, b):
        material = SubElement(root, 'material', { 'name' : str(name) })
        SubElement(material, 'color', { 'rgba' : "{} {} {} 1".format(r, g, b) })

    @staticmethod
    def tag_link(root, name, xyz, material, length, diameter, is_endeffector=False):
        # root > link
        link = SubElement(root, 'link', { 'name' : str(name) })
        # root > link > visual
        visual = SubElement(link, 'visual', {})
        # root > link > visual > origin
        SubElement(visual, 'origin', { 'xyz': str(xyz) })
        # root > link > visual > material
        if material is not None: SubElement(visual, 'material', { 'name' : str(material) })
        # root > link > visual > geometry
        geometry = SubElement(visual, 'geometry', {})
        # root > link > visual > geometry > {cylinder,sphere}
        if not is_endeffector:
            SubElement(geometry, 'cylinder', { 'length' : str(length), 'radius': str(diameter/2) })
        else:
            SubElement(geometry, 'sphere', { 'radius': "0.1" })

    @staticmethod
    def tag_joint(root, joint_name, parent_link_name, child_link_name, joint_xyz, joint_axis, joint_type='revolute'):
        JOINT_LIMIT = 1.570 # approx = PI/4
        # root > joint
        joint = SubElement(root, 'joint', { 'name': str(joint_name), 'type': joint_type })
        # root > joint > parent
        SubElement(joint, 'parent', { 'link': str(parent_link_name) })
        # root > joint > child
        SubElement(joint, 'child', { 'link': str(child_link_name) })
        # root > joint > axis
        if joint_axis is not None: SubElement(joint, 'axis', { 'xyz': str(joint_axis) })
        # root > joint > origin
        if joint_xyz is not None:  SubElement(joint, 'origin', { 'xyz': str(joint_xyz) })
        # root > joint > limit
        SubElement(joint, 'limit', { 'effort': '100', 'velocity': '100', 'lower': str(-JOINT_LIMIT), 'upper': str(JOINT_LIMIT) })

    def struct_endeffector(self, root, last_link_name):
        SnakeBuilderEnhanced.tag_material(root, "end_effector_material", 1, 1, 1)
        xyz = "0 0 {}".format(self.length)
        SnakeBuilderEnhanced.tag_link(root, "end_effector", xyz, "end_effector_material", self.length, self.diameter, is_endeffector=True)
        SnakeBuilderEnhanced.tag_joint(root, "joint_end_effector", last_link_name, "end_effector", None, None, "fixed")

    def build(self):
        root = Element('robot', { 'name': 'snake', 'xmlns:xacro' : 'http://ros.org/wiki/xacro' })
        # create materials
        MATERIALS = [ ( 'color_{}'.format(chr(ord('a')+i)), (i >> 2) % 2, (i >> 1) % 2, i % 2) for i in range(0, 8) ]
        for (name, r, g, b) in MATERIALS:
            self.tag_material(root, name, r, g, b)
        # create links
        LENGTHS = [0.00001, self.length]
        for i in range(0, self.n_joints):
            length = LENGTHS[i % len(LENGTHS)]
            link_name = "link{}".format(i)
            xyz = "0 0 {}".format(length/2)
            material_name = MATERIALS[i % len(MATERIALS)][0]
            self.tag_link(root, link_name, xyz, material_name, length, self.diameter, is_endeffector=False)
        # create joints
        AXES = ['1 0 0', '0 1 0']
        for i in range(0, self.n_joints-1):
            axis = AXES[i % 2]
            length = LENGTHS[i % len(LENGTHS)]
            # tag_joint(root, joint_name, parent_link_name, child_link_name, joint_xyz, joint_axis, joint_type='revolute')
            joint_name = "joint{}".format(i)
            parent_link_name = "link{}".format(i)
            child_link_name = "link{}".format(i+1)
            joint_xyz = "0 0 {}".format(length)
            self.tag_joint(root, joint_name, parent_link_name, child_link_name, joint_xyz, axis)
        # create end effector
        last_link_name = "link{}".format(self.n_joints-1)
        self.struct_endeffector(root, last_link_name)
        # print to string
        rough_string = ElementTree.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")
        return pretty


if __name__ == "__main__":
    print("Starting...")
    for i in [8, 16, 24, 32, 40, 48]:
        sb = SnakeBuilderEnhanced(i, 0.1, 0.02)
        urdf = sb.build()
        filename = "modello_{}.urdf".format(i)
        with open(filename, 'w') as out:
            out.write(urdf + "\n")
    print("Ended.")
