#!/usr/bin/env python
from butler_brain.utils import text_to_speech, speach_to_text, get_top_matching_candidate, tell_me_one_of
import rospy
from butler_perception.segment_pcl import PCLProcessor
from butler_action.butler_actions import ButlerActions
from math import pi
import numpy as np
import open3d as o3d
from copy import deepcopy
from trac_ik_python.trac_ik import IK
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply, quaternion_inverse
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotInput
import pickle
import os
from sensor_msgs.msg import JointState
from datetime import datetime
import subprocess


class RobotActionPrimitives:
    def __init__(self, verbose=True, save_data=False, load_data=False):
        self.verbose = verbose
        self.pcl_proc = PCLProcessor()
        
        self.positive_answers = lambda obj_name:["done", "here it is", "here", "take it", "take",
                                                    "i brought it",
                                                    "i brought you the {}".format(obj_name),
                                                    "continue",
                                                    "i brought you a {}".format(obj_name)]
        self.negative_answers = lambda obj_name:["i don't have it", "i don't have a {}".format(obj_name), "skip it", "ignore it"]
        self.ba = ButlerActions()
        self.pinch_pos_shift = (-0.03, 0.05, 0.003)
        self.power_pos_shift = (-0.015, 0.03, 0.01)
        self.pinch_ori_shift = (0, 25*pi/180, 0)
        self.power_ori_shift = (0, 0, 0)
        self.tea_placing_waypoints = [[-0.03, 0.12, 0.16], [-0.03, 0.02, 0.06]]
        self.tea_placing_orientations = [self.pinch_ori_shift, self.pinch_ori_shift]
        self.kobuki_placing_orientations = [self.pinch_ori_shift, self.pinch_ori_shift, self.pinch_ori_shift]
        self.obj_pick_data = {"tea-packet": {'pos':self.pinch_pos_shift,
                                        'ori':self.pinch_ori_shift,
                                        'grip_pos':110,
                                        'loc':'_top'},
                            "cup": {'pos':self.power_pos_shift,
                                    'ori':self.power_ori_shift,
                                    'grip_pos':75,
                                    'loc':'_center'},
                            "sugar": {'pos':self.pinch_pos_shift,
                                    'ori':self.pinch_ori_shift,
                                    'grip_pos':75,
                                    'loc':'_center'},
                            "coffee": { 'pos':self.pinch_pos_shift,
                                        'ori':self.pinch_ori_shift,
                                        'grip_pos':75,
                                        'loc':'_top'}}
        self.obj_place_data = {("tea-packet", "cup"): {'pos':[[-0.03, 0.12, 0.16], [-0.03, 0.02, 0.06]],
                                                        'ori':[[0, 25*pi/180, 0], [0, 25*pi/180, 0]],
                                                        'grip_pos':[None, 0],
                                                        'loc':['_top', '_top'],
                                                        'place_force':[None, 4],
                                                        'touch':[False, True],
                                                        'use_tool_orient':[False, False]},
                                ("cup", "kobuki"): {'pos':[[-0.01, 0, 0.2], [-0.01, 0, -0.06], [-0.01, 0, 0.3]],
                                         'touch':[False, True, False],
                                         'ori':[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                         'grip_pos':[None, 0, None],
                                         'place_force':[None, 4, None],
                                         'use_tool_orient':[False, False, False],
                                         'loc':['_top', '_bottom', '_top']}}
        self.obj_pick_data["kobuki"] = {'pos':self.power_pos_shift, 'ori':self.power_ori_shift}
        # self.kobuki_limits = {'x_min': -2.0, 'x_max': 0.6920000000000002, 'y_min': -0.5700000000000001, 'y_max': -0.3999999999999999, 'z_min': -0.901, 'z_max': -0.42799999999999994}
        self.kobuki_limits = {'x_min': -2.0, 'x_max': 0.6920000000000002, 'y_min': -0.5700000000000001, 'y_max': -0.3999999999999999, 'z_min': -1.1560000000000001, 'z_max': -0.42799999999999994}
        self.obj_data = {} # {obj_name: {'center':(x,y,z), 'top':(x,y,z), 'bottom':(x,y,z)}}
        self.save_data = save_data
        self.load_data = load_data
        self.cwd = os.getcwd()
        self.pick_data_path = os.path.join(os.getcwd(), 'pick_data')
        self.place_data_path = os.path.join(os.getcwd(), 'place_data')
        place_obj_shift = deepcopy(self.obj_place_data)
        for key, val in place_obj_shift.items():
            orientation = deepcopy(val['ori'])
            if type(orientation[0]) not in [tuple, list]:
                self.obj_place_data[key]['ori'] = [orientation for _ in range(len(val['pos']))]
        if self.load_data:
            with open(self.pick_data_path + '.pkl', 'rb') as f:
                self.obj_pick_data = pickle.load(f)
            with open(self.place_data_path + '.pkl', 'rb') as f:
                place_obj_shift = pickle.load(f)
                for key, val in place_obj_shift.items():
                    if key in self.obj_place_data:
                        place_obj_shift[key] = self.obj_place_data[key]
                    orientation = deepcopy(val['ori'])
                    if (type(orientation[0]) not in [tuple, list]) and (type(val['pos'][0]) in [tuple, list]):
                        place_obj_shift[key]['ori'] = [orientation for _ in range(len(val['pos']))]
                self.obj_place_data = place_obj_shift
        self.record_joint_states = False
        self.recorded_joint_states = []
        self.recorded_motion_data = None
        self.process = None
        self.context = None
    
    def terminate_ros_node(self, s='/record'):
        # Adapted from http://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
        list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
        list_output = list_cmd.stdout.read()
        retcode = list_cmd.wait()
        assert retcode == 0, "List command returned %d" % retcode
        # print(list_output.decode("utf-8").split("\\n"))
        for string in list_output.decode("utf-8").split("\n"):
            if (string.startswith(s)):
                os.system("rosnode kill " + string)
    
    def record_bag(self, bag_name):
        command = ' '.join(['rosbag', 'record', '-O', bag_name,
                            '/camera/depth/color/points',
                            # '/camera/color/image_raw',
                            # '/camera/aligned_depth_to_color/image_raw',
                            # '/camera/depth/image_rect_raw',
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
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)

    def find(self, object_name, return_all=True, verbose=False):
        if object_name == "kobuki":
            obj_data = self.find_kobuki()
        else:
            all_objects_names = list(object_name) if type(object_name) == list else [object_name]
            print("Finding the following objects {}".format(all_objects_names))
            obj_data = self.pcl_proc.get_object_location(n_trials=10, object_names=all_objects_names, number=True)  
        if obj_data is None:
            text_to_speech(f"I don't see {object_name}. if it is not there, please bring it to me or tell me to skip it", verbose=verbose)
            obj_data = self.wait_for(object_name, verbose=verbose)
            return obj_data
        if return_all:
            return obj_data
        else:
            obj_name = list(obj_data.keys())[0]
            obj_loc = obj_data[obj_name]['center']
            return {obj_name: obj_loc}
    
    def register_pos_request(self, object_name, object_loc, pick=True, serve_to=None, verbose=False):
        curr_pose = self.ba.ts.lookup_transform("base_link", "gripper_tip_link")
        pos_shift = []
        pos_shift.append(curr_pose.position.x - object_loc.position.x)
        pos_shift.append(curr_pose.position.y - object_loc.position.y)
        pos_shift.append(curr_pose.position.z - object_loc.position.z)
        ori_shift = []
        old_quat = [object_loc.orientation.x, object_loc.orientation.y, object_loc.orientation.z, object_loc.orientation.w]
        new_quat = [curr_pose.orientation.x, curr_pose.orientation.y, curr_pose.orientation.z, curr_pose.orientation.w]
        # new_quat = quaternion_multiply(old_quat, diff_quat)
        diff_quat = quaternion_multiply(quaternion_inverse(old_quat), new_quat)
        euler = euler_from_quaternion(diff_quat)
        ori_shift = [euler[0], euler[1], euler[2]]
        grip_pos_msg = rospy.wait_for_message("/Robotiq3FGripperRobotInput", Robotiq3FGripperRobotInput)
        grip_pos = grip_pos_msg.gPOC
        if pick:
            touch = True if tell_me_one_of("touch or no?", verbose=self.verbose) == 'yes' else False
            if touch:
                text_to_speech("please enter the touch force")
                force = int(input("touch force = "))
                text_to_speech("now, please enter the force axis")
                axis = input("force axis = ")
                text_to_speech("now, please enter the extra distance in z")
                extra_dist = float(input("extra distance in z = "))
                
            self.obj_pick_data[object_name] = {'pos':pos_shift,
                                            'ori':ori_shift,
                                            'grip_pos':grip_pos,
                                            'loc':'_top',
                                            'touch':touch,
                                            'force':force,
                                            'axis':axis,
                                            'extra_dist':extra_dist}
        else:
            key = (object_name, serve_to)
            touch = True if tell_me_one_of("touch or no?", verbose=self.verbose) == 'yes' else False
            if key not in self.obj_place_data.keys():
                self.obj_place_data[key] = {'pos':[pos_shift],
                                                'ori':[ori_shift],
                                                'grip_pos':[grip_pos],
                                                'loc':['_top'],
                                                'place_force':[4],
                                                'use_tool_orient':[False],
                                                'touch':[touch]}
            else:
                self.obj_place_data[key]['pos'].append(pos_shift)
                self.obj_place_data[key]['ori'].append(ori_shift)
                self.obj_place_data[key]['touch'].append(touch)
                self.obj_place_data[key]['grip_pos'].append(grip_pos)
                self.obj_place_data[key]['loc'].append('_top')
                self.obj_place_data[key]['place_force'].append(4)
                self.obj_place_data[key]['use_tool_orient'].append(False)
    
    def handle_new_object(self, object_name, object_loc, pick=True, serve_to=None, verbose=False):
        if pick:
            candidates = list(self.obj_pick_data.keys())
            original_candidates = candidates
            speech_candidates = original_candidates
            speech = f"Ok I haven't picked {object_name} before, which of the following is manipulated similarly:"
        else:
            original_candidates = list(self.obj_place_data.keys())
            candidates = [a for a,b in original_candidates]
            speech_candidates = ['from ' + a + ' to ' + b for a,b in original_candidates]
            speech = f"Ok I haven't placed {object_name} to {serve_to} before, which of the following is manipulated similarly:"
            
        for i, c in enumerate(speech_candidates):
            if i == len(speech_candidates)-1:
                speech += f" or {c}?"
            else:
                speech += f"{c}, "
        text_to_speech(speech, verbose=self.verbose)
        # c_idx = int(input("Please enter the number of your choice (-1 for none): "))
        # if c_idx == -1:
        #     similar_object_name = None
        # else:
        #     similar_object_name = original_candidates[c_idx-1]
        similar_object_name = None
        speech_candidates.append('none')
        while not rospy.is_shutdown():
            txt = speach_to_text(verbose=self.verbose)
            c_idx, m_idx, tr = get_top_matching_candidate(speech_candidates, txt, bert=True, verbose=self.verbose)
            if tr >= 0.8:
                print('c_idx', c_idx, speech_candidates)
                print(self.obj_place_data)
                if c_idx == len(speech_candidates)-1:
                    similar_object_name = None
                else:
                    similar_object_name = original_candidates[c_idx]
                break
            else:
                text_to_speech(f"Sorry, I didn't understand your answer. Please try again", verbose=self.verbose)
        moved = False
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        bag_name = ''
        if similar_object_name is None:
            bag_name = 'pick' if pick else 'place'
            bag_name += '_' + object_name
            if not pick:
                bag_name += '_to_' + serve_to
            bag_name += '_' + self.context['name']
            context_path = os.path.join(os.getcwd(), bag_name + '_context_' + time_str + '.pkl')
            bag_path = os.path.join(os.getcwd(), bag_name + '_' + time_str)
            with open(context_path, 'wb') as f:
                pickle.dump(self.context, f)
            self.record_bag(bag_path)
            
            text_to_speech(f"Ok, please move me to correct pose, then tell me ok", verbose=self.verbose)
            while not rospy.is_shutdown():
                txt = speach_to_text(verbose=self.verbose)
                if "ok" in txt:
                    break
            frame_name = object_name+'_top' if pick else serve_to+'_top'
            self.register_pos_request(object_name, object_loc[frame_name], pick=pick, serve_to=serve_to, verbose=verbose)
            if not pick:
                text_to_speech(f"Ok, please move me to the second pose, then tell me ok", verbose=self.verbose)
                while not rospy.is_shutdown():
                    # publish a msg to record joint states of motion
                    txt = speach_to_text(verbose=self.verbose)
                    if "ok" in txt:
                        # publish a msg to stop recording to joint states
                        break
                self.register_pos_request(object_name, object_loc[frame_name], pick=pick, serve_to=serve_to, verbose=verbose)
            moved = True
            print("Process PID: ", self.process.pid)
            # self.process.kill()
            self.terminate_ros_node()
        else:
            # New Knowledge about an object
            if pick:
                self.obj_pick_data[object_name] = self.obj_pick_data[similar_object_name]
            else:
                self.obj_place_data[(object_name, serve_to)] = self.obj_place_data[similar_object_name]
            moved = False
        if self.save_data:
            with open(self.pick_data_path + '.pkl', 'wb') as f:
                pickle.dump(self.obj_pick_data, f)
            with open(self.place_data_path + '.pkl', 'wb') as f:
                pickle.dump(self.obj_place_data, f)
            with open(os.path.join(os.getcwd(), bag_name + '_pick_data_' + time_str + '.pkl'), 'wb') as f:
                pickle.dump(self.obj_pick_data, f)
            with open(os.path.join(os.getcwd(), bag_name + '_place_data_' + time_str + '.pkl'), 'wb') as f:
                pickle.dump(self.obj_place_data, f)
        return moved
        
    def pick(self, object_name, object_loc, step_axis=1, step=0, frame_name=None):
        if frame_name is None:
            frame_name = object_name
        print(f"Picking the following object {object_name}")
        # CREATE FRAMES FOR OBJECTS
        object_loc = self.ba.create_frames_for_objects({frame_name:object_loc})
        self.ba.grip_control.move_gripper(0)
        rospy.sleep(1.0)
        moved = False
        if object_name not in self.obj_pick_data:
            moved = self.handle_new_object(frame_name, object_loc)
        touch = True if 'touch' not in list(self.obj_pick_data[object_name].keys()) else self.obj_pick_data[object_name]['touch']
        if not moved:
            pos_shift = self.obj_pick_data[object_name]['pos']
            ori_shift = self.obj_pick_data[object_name]['ori']
            adjusted_pos_shift = (pos_shift[0], pos_shift[1], pos_shift[2]+0.1)
            # MOVE TO ABOVE OBJECT
            self.ba.move_to_frame_pose_oriented(frame_name+"_top",
                                            position_shift=adjusted_pos_shift,
                                            orientation_shift=ori_shift,
                                            step_axis=step_axis,
                                            step=step,
                                            straight=True,
                                            touch=False)
            
            if object_name == 'bottle':
                self.ba.move_to_frame_pose_oriented(frame_name+self.obj_pick_data[object_name]['loc'],
                                            position_shift=pos_shift,
                                            orientation_shift=ori_shift,
                                            step_axis=step_axis,
                                            step=step,
                                            straight=True,
                                            touch=False)
            elif touch:
                # MOVE DOWN TO GRIPPING POSITION WHILE CHECKING FOR COLLISIONS VERTICALLY
                pos_shift= list(pos_shift)
                pos_shift[2] += 0.01
                axis = 'xy' if 'axis' not in list(self.obj_pick_data[object_name].keys()) else self.obj_pick_data[object_name]['axis']
                force = 6 if 'force' not in list(self.obj_pick_data[object_name].keys()) else self.obj_pick_data[object_name]['force']
                extra_dist = 0.03 if 'extra_dist' not in list(self.obj_pick_data[object_name].keys()) else self.obj_pick_data[object_name]['extra_dist']
                self.ba.move_to_touch(frame_name=frame_name+self.obj_pick_data[object_name]['loc'],
                                    position_shift=pos_shift,
                                    orientation_shift=ori_shift,
                                    extra_dist=extra_dist,
                                    force_thresh=force,
                                    axis=axis,
                                    avoid_collisions=False)
            
            # GRIP
            if self.obj_pick_data[object_name]['grip_pos'] is not None:
                self.ba.grip_control.move_gripper(self.obj_pick_data[object_name]['grip_pos'])
        
    def place(self, object_name, object_loc, object_to_place, frame_name=None, touch=False, constraints_name=None):
        if frame_name is None:
            frame_name = object_name
        print(f"Placing the following object {object_name}")
        # CREATE FRAMES FOR OBJECTS
        object_loc = self.ba.create_frames_for_objects({frame_name: object_loc})
        rospy.sleep(1.0)
        key = (object_to_place, object_name)
        if key == ('bowl', 'kobuki'):
            self.obj_place_data.pop(key)
        moved = False
        if key not in self.obj_place_data:
            moved = self.handle_new_object(object_to_place, object_loc, pick=False, serve_to=object_name)
        if not moved:
            for i in range(len(self.obj_place_data[key]['pos'])):
                rospy.sleep(2.0)
                self.ba.move_to_frame_pos(frame_name+self.obj_place_data[key]['loc'][i],
                                        position_shift=self.obj_place_data[key]['pos'][i],
                                        orientation_shift=self.obj_place_data[key]['ori'][i],
                                        touch=self.obj_place_data[key]['touch'][i],
                                        straight=True,
                                        force_thresh=self.obj_place_data[key]['place_force'][i],
                                        constraints_name=None,
                                        joint_goal=False,
                                        use_tool_orientation=self.obj_place_data[key]['use_tool_orient'][i])
                
                if self.obj_place_data[key]['grip_pos'][i] is not None:
                    self.ba.grip_control.move_gripper(self.obj_place_data[key]['grip_pos'][i])
            if object_to_place == 'bottle':
                self.ba.move_to_named_location('home')
        
        # OPEN GRIPPER
        if key not in [('bottle', 'cup'), ('bottle', 'bowl')]:
            self.ba.grip_control.move_gripper(0)

    def pour(self, object_location):
        print("Pouring an object at {}".format(object_location))
        pass

    def open(self, object_location):
        print("Opening an object at {}".format(object_location))
        pass

    def push_button(object_location):
        print("Pushing a button at {}".format(object_location))
        pass

    def wait_for(self, object_name, verbose=False):
        while not rospy.is_shutdown():
            text = speach_to_text(verbose=verbose)
            found = False
            possible_answers = self.positive_answers(object_name) + self.negative_answers(object_name)
            c_idx, _, m_ratio = get_top_matching_candidate(possible_answers, text)
            if m_ratio >= 0.7:
                if c_idx < len(self.positive_answers(object_name)):
                    found = True
                else:
                    text_to_speech(f"Ok, skipping {object_name}", verbose=verbose)
                    found = False
            else:
                text_to_speech(f"I didn't get you, please say again", verbose=verbose)
                continue
            # for answer in self.positive_answers(object_name):
            #     if answer in text:
            #         found = True
            if found:
                object_loc = self.find(object_name)
                if object_loc is None:
                    text_to_speech(f"I still don't see {object_name}. Please bring me {object_name}, or select it from image window", verbose=verbose)
                    # TODO: Add a way to select the object from the image window
                    continue
                return object_loc
            return None
    
    def find_kobuki(self, visualize=False):
        pcd = self.pcl_proc.get_pcd()
        _, plane_pcd, pcd = self.pcl_proc.constrain_and_segment_plane(pcd, self.kobuki_limits, visualize=False)
        if plane_pcd is None:
            return None
        cl, ind = plane_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
        plane_pcd = plane_pcd.select_by_index(ind)
        if np.asarray(plane_pcd.points).shape[0] < 100:
            return None
        center = np.asarray(plane_pcd.get_center())
        points = np.asarray(plane_pcd.points)
        top_idx = np.argmax(points[:, 1])
        bottom_idx = np.argmin(points[:, 1])
        top = points[top_idx]
        bottom = points[bottom_idx]
        # center[2] -= 0.05
        if visualize:
            new_cluster = deepcopy(plane_pcd)
            new_cluster.points.extend([center])
            new_cluster.colors.extend([[0, 0, 1]])
            o3d.visualization.draw_geometries([new_cluster])
        # self.ba.create_frames_for_objects({"kobuki": center})
        return {"kobuki": {'center':center, 'top':top, 'bottom':bottom}}

if __name__ == '__main__':
    rospy.init_node("robot_action_primitives")
    rob_act_prim = RobotActionPrimitives()
    kobuki_loc = rob_act_prim.find("kobuki")
    rob_act_prim.place("kobuki", kobuki_loc["kobuki"])
    # rob_act_prim.find("cup")
    # while not rospy.is_shutdown():
    #     rob_act_prim.find_kobuki(visualize=False)