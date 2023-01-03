#!/usr/bin/env python
from owl_test.utils import text_to_speech, speach_to_text, get_top_matching_candidate
import rospy
from butler_perception.segment_pcl import PCLProcessor
from butler_action.butler_actions import ButlerActions
from math import pi


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
        self.power_pos_shift = (0, 0.05, 0)
        self.pinch_ori_shift = (0, 25*pi/180, 0)
        self.power_ori_shift = (0, 0, 0)
        self.obj_shift = {"teapacket": {'pos':self.pinch_pos_shift, 'ori':self.pinch_ori_shift, 'grip_pos':110},
                          "cup": {'pos':self.power_pos_shift, 'ori':self.power_ori_shift, 'grip_pos':75},
                          "sugar": {'pos':self.pinch_pos_shift, 'ori':self.pinch_ori_shift, 'grip_pos':75},
                          "coffee": {'pos':self.pinch_pos_shift, 'ori':self.pinch_ori_shift, 'grip_pos':75}}
        

    def find(self, object_name, return_top=False):
        all_objects_names = list(object_name) if type(object_name) == list else [object_name]
        print("Finding the following objects {}".format(all_objects_names))
        obj_data = self.pcl_proc.get_object_location(n_trials=10, object_names=all_objects_names, number=True)
        if return_top:
            return obj_data
        else:
            return obj_data['center']
    def pick(self, object_name, object_loc, step_axis=1, step=0, frame_name=None):
        if frame_name is None:
            frame_name = object_name
        print(f"Picking the following object {object_name}")
        # CREATE FRAMES FOR OBJECTS
        self.ba.create_frames_for_objects({frame_name: object_loc})
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
        self.ba.move_to_frame_pos(frame_name, position_shift=[(-0.03, 0.12, 0.16), (-0.03, 0.05, 0.08)])
        # self.ba.move_to_frame_pos(frame_name, position_shift=(-0.03, 0.05, 0.08), touch=False)
        
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
                    text_to_speech(f"I still don't see {object_name}. Please bring me {object_name}.", verbose=verbose)
                    continue
                return object_loc
            return None
                    

if __name__ == '__main__':
    rospy.init_node("robot_action_primitives")
    rob_act_prim = RobotActionPrimitives()
    rob_act_prim.find("cup")