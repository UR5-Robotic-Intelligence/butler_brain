#!/usr/bin/env python
from owl_test.utils import text_to_speech, speach_to_text
import rospy
from butler_perception.segment_pcl import PCLProcessor
from butler_action.butler_actions import ButlerActions
from math import pi


class RobotActionPrimitives:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.pcl_proc = PCLProcessor()
        
        self.possible_answers = lambda obj_name:["done", "here it is", "here", "take it", "take",
                                                    "i brought it",
                                                    "i brought you the {}".format(obj_name),
                                                    "continue",
                                                    "i brought you a {}".format(obj_name)]
        self.ba = ButlerActions()
        self.pinch_pos_shift = (-0.03, 0.05, 0.003)
        self.power_pos_shift = (0, 0.05, 0)
        self.pinch_ori_shift = (0, 25*pi/180, 0)
        self.power_ori_shift = (0, 0, 0)
        self.obj_shift = {"teapacket": {'pos':self.pinch_pos_shift, 'ori':self.pinch_ori_shift, 'grip_pos':110},
                          "cup": {'pos':self.power_pos_shift, 'ori':self.power_ori_shift, 'grip_pos':75},
                          "sugar": {'pos':self.pinch_pos_shift, 'ori':self.pinch_ori_shift, 'grip_pos':75},
                          "coffee": {'pos':self.pinch_pos_shift, 'ori':self.pinch_ori_shift, 'grip_pos':75}}
        

    def find(self, object_name):
        all_objects_names = list(object_name) if type(object_name) == list else [object_name]
        print("Finding the following objects {}".format(all_objects_names))
        return self.pcl_proc.get_object_location(n_trials=10, object_names=all_objects_names, number=True)

    def pick(self, object_name, object_loc, step_axis=1, step=0, frame_name=None):
        if frame_name is None:
            frame_name = object_name
        print(f"Picking the following object {object_name}")
        # CREATE FRAMES FOR OBJECTS
        self.ba.create_frames_for_objects({frame_name: object_loc})
        rospy.sleep(1.0)

        # MOVE TO TEA PACKET
        self.ba.move_to_frame_pose_oriented(frame_name,
                                        position_shift=self.obj_shift[object_name]['pos'],
                                        orientation_shift=self.obj_shift[object_name]['ori'],
                                        step_axis=step_axis,
                                        step=step)
        
        # GRIP
        self.ba.grip_control.move_gripper(self.obj_shift[object_name]['grip_pos'])
        
    def place(self, object_name, object_loc, frame_name=None):
        if frame_name is None:
            frame_name = object_name
        print(f"Placing the following object {object_name}")
        # CREATE FRAMES FOR OBJECTS
        self.ba.create_frames_for_objects({frame_name: object_loc})
        rospy.sleep(1.0)
        
        # MOVE TO OBJECT LOCATION
        self.ba.move_to_frame_pos(frame_name, position_shift=(-0.03, 0.12, 0.16))
        self.ba.move_to_frame_pos(frame_name, position_shift=(-0.03, 0.05, 0.08), touch=False)
        
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
            for answer in self.possible_answers(object_name):
                if answer in text:
                    found = True
            if found:
                object_loc = self.find(object_name)
                if object_loc is None:
                    text_to_speech("I still don't see a cup. Please bring me a cup.", verbose=verbose)
                    continue
                return object_loc
                    

if __name__ == '__main__':
    rospy.init_node("robot_action_primitives")
    rob_act_prim = RobotActionPrimitives()
    rob_act_prim.find("cup")