#!/usr/bin/env python
from owl_test.utils import text_to_speech, speach_to_text, get_top_matching_candidate
import rospy
from butler_perception.segment_pcl import PCLProcessor
from butler_action.butler_actions import ButlerActions
from math import pi
import numpy as np
import open3d as o3d
from copy import deepcopy
from trac_ik_python.trac_ik import IK



class RobotActionPrimitives:
    def __init__(self, verbose=True):
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
        self.tea_placing_waypoints = [(-0.03, 0.12, 0.16), (-0.03, 0.02, 0.06)]
        self.kobuki_placing_waypoints = [(0, 0, 0.2), (0.0, 0, -0.06)]
        self.obj_shift = {"tea-packet": {'pos':self.pinch_pos_shift,
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
        self.place_obj_shift = {("tea-packet", "cup"): {'pos':self.tea_placing_waypoints,
                                                        'ori':self.pinch_ori_shift,
                                                        'grip_pos':110,
                                                        'loc':'_top',
                                                        'place_force':2.5,
                                                        'use_tool_orient':False},
                                ("cup", "kobuki"): {'pos':self.kobuki_placing_waypoints,
                                                    'ori':self.power_ori_shift,
                                                    'grip_pos':75,
                                                    'loc':'_bottom',
                                                    'place_force':4,
                                                    'use_tool_orient':True}}
        self.obj_shift["kobuki"] = {'pos':self.power_pos_shift, 'ori':self.power_ori_shift}
        self.kobuki_limits = {'x_min': -2.0, 'x_max': 0.6920000000000002, 'y_min': -0.5700000000000001, 'y_max': -0.3999999999999999, 'z_min': -0.901, 'z_max': -0.42799999999999994}
        self.obj_data = {} # {obj_name: {'center':(x,y,z), 'top':(x,y,z), 'bottom':(x,y,z)}}

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
        
    def pick(self, object_name, object_loc, step_axis=1, step=0, frame_name=None):
        if frame_name is None:
            frame_name = object_name
        print(f"Picking the following object {object_name}")
        # CREATE FRAMES FOR OBJECTS
        self.ba.create_frames_for_objects({frame_name:object_loc})
        rospy.sleep(1.0)
        
        similar_object_name = None
        if object_name not in self.obj_shift:
            candidates = list(self.obj_shift.keys())
            c_idx, _, _ = get_top_matching_candidate(candidates, object_name)
            text_to_speech(f"I don't know how to pick {object_name}. Is it similar to {candidates[c_idx]}?", verbose=self.verbose)
            
            while True:
                txt = speach_to_text(verbose=self.verbose)
                yes_no_idx, _, m_ratio =get_top_matching_candidate(["yes", "no"], txt)
                if m_ratio >= 0.7:
                    break
                text_to_speech(f"I didn't get you, please say again", verbose=self.verbose)
                    
            if yes_no_idx == 0:
                text_to_speech(f"Ok, I will pick the {object_name} the same way as I pick a {candidates[c_idx]}, please stop me if I do something stupid!", verbose=self.verbose)
                similar_object_name = candidates[c_idx]
            else:
                speech = f"Ok then, which of the following is picked similarly:"
                for i, c in enumerate(candidates):
                    if i == c_idx:
                        continue
                    if i == len(candidates)-1:
                        speech += f" or {c}?"
                    else:
                        speech += f"{c}, "
                text_to_speech(speech, verbose=self.verbose)
                while similar_object_name is None:
                    txt = speach_to_text(verbose=self.verbose)
                    for c in candidates:
                        if c in txt:
                            similar_object_name = c
                            break
                    if similar_object_name is None:
                        if "none" in txt:
                            text_to_speech(f"Ok, lets do it together then", verbose=self.verbose)
                            break
                        text_to_speech(f"What you said does not indicate any of the choices, please say again", verbose=self.verbose)
        
            if similar_object_name is not None:
                # New Knowledge about an object
                self.obj_shift[object_name] = self.obj_shift[similar_object_name]
            else:
                # Take Commands from the user
                text_to_speech(f"Please tell me how to pick the {object_name}", verbose=self.verbose)
                txt = speach_to_text(verbose=self.verbose)
                exit()
        
        pos_shift = self.obj_shift[object_name]['pos']
        ori_shift = self.obj_shift[object_name]['ori']
        adjusted_pos_shift = (pos_shift[0], pos_shift[1], pos_shift[2]+0.1)
        # MOVE TO ABOVE OBJECT
        self.ba.move_to_frame_pose_oriented(frame_name+"_top",
                                        position_shift=adjusted_pos_shift,
                                        orientation_shift=ori_shift,
                                        step_axis=step_axis,
                                        step=step,
                                        straight=True,
                                        touch=False)
        
        # MOVE DOWN TO GRIPPING POSITION WHILE CHECKING FOR COLLISIONS VERTICALLY
        self.ba.move_to_touch(frame_name=frame_name+self.obj_shift[object_name]['loc'],
                              position_shift=pos_shift,
                              orientation_shift=ori_shift,
                              extra_dist=0,
                              force_thresh=4,
                              axis='xy',
                              avoid_collisions=False)
        
        # GRIP
        self.ba.grip_control.move_gripper(self.obj_shift[object_name]['grip_pos'])
        
    def place(self, object_name, object_loc, object_to_place, frame_name=None, touch=False, constraints_name=None):
        if frame_name is None:
            frame_name = object_name + "_place"
        print(f"Placing the following object {object_name}")
        # CREATE FRAMES FOR OBJECTS
        self.ba.create_frames_for_objects({frame_name: object_loc})
        rospy.sleep(1.0)
        key = (object_to_place, object_name)
        # MOVE TO OBJECT LOCATION # TODO: MAKE THIS MORE ROBUST BY COMPINING THESE TWO into one
        self.ba.move_to_frame_pos(frame_name+"_top", position_shift=self.place_obj_shift[key]['pos'][0], straight=True,
                                  joint_goal=False, approximated=False,
                                  constraints_name=constraints_name,
                                  use_tool_orientation=self.place_obj_shift[key]['use_tool_orient'])
        rospy.sleep(2.0)
        self.ba.move_to_frame_pos(frame_name+self.place_obj_shift[key]['loc'],
                                  position_shift=self.place_obj_shift[key]['pos'][1],
                                  touch=True,
                                  force_thresh=self.place_obj_shift[key]['place_force'],
                                  constraints_name=constraints_name,
                                  use_tool_orientation=self.place_obj_shift[key]['use_tool_orient'])
        
        # OPEN GRIPPER
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