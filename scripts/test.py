# from owlready2 import *

# onto_path.append("/home/bass/ur5_ws/src/iai_maps/iai_kitchen/owl/iai-kitchen-objects.owl")
# onto = get_ontology("http://knowrob.org/kb/knowrob.owl").load()
# print(onto.get_parents_of(onto.Spoon))
# print(onto.Spoon.instances())
# Ontology.get_parents_of()

# import re

# s = "teapacket"
# res = re.sub( r"([A-Z])", r" \1", s).split()
# res = "-".join(res)
# print(res)

import subprocess
import rospy
import os


def terminate_ros_node(s='/record'):
        # Adapted from http://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
        list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
        list_output = list_cmd.stdout.read()
        retcode = list_cmd.wait()
        assert retcode == 0, "List command returned %d" % retcode
        # print(list_output.decode("utf-8").split("\\n"))
        for string in list_output.decode("utf-8").split("\n"):
            if (string.startswith(s)):
                os.system("rosnode kill " + string)
      
rospy.init_node('test', anonymous=True)
command = ' '.join(['rosbag', 'record', '-O', 'name' + '.bag',
                            '/camera/depth/color/points',
                            '/camera/color/image_raw',
                            '/camera/aligned_depth_to_color/image_raw',
                            '/camera/depth/image_rect_raw',
                            '/camera/color/camera_info',
                            '/camera/depth/camera_info',
                            '/camera/aligned_depth_to_color/camera_info',
                            '/camera/extrinsics/depth_to_color',
                            '/joint_states',
                            '/joint_group_velocity_controller/command',
                            '/wrench',
                            '/wrench/filtered',
                            '/scaled_pos_joint_traj_controller/command',
                            '/scaled_pos_joint_traj_controller/follow_joint_trajectory/cancel',
                            '/scaled_pos_joint_traj_controller/follow_joint_trajectory/goal',
                            '/scaled_pos_joint_traj_controller/follow_joint_trajectory/result',
                            '/scaled_pos_joint_traj_controller/follow_joint_trajectory/status',
                            '/scaled_pos_joint_traj_controller/state',
                            '/tf',
                            '/tf_static',
                            '/Robotiq3FGripperRobotInput',
                            '/Robotiq3FGripperRobotOutput'])
a = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)
rospy.sleep(1)
# a.send_signal(subprocess.signal.SIGINT)
# rospy.sleep(10)
terminate_ros_node()
print(command)