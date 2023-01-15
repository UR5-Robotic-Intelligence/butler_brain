#!/usr/bin/env python

from owl_test.robot_action_primitives import RobotActionPrimitives
from owl_test.utils import text_to_speech, speach_to_text, get_top_matching_candidate, tell_me_one_of
import rospy
import re
from std_msgs.msg import Float64, Empty
from geometry_msgs.msg import PoseArray
import numpy as np
from std_srvs.srv import Empty as EmptySrv
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply
from socket import socket, AF_INET, SOCK_DGRAM


class RobotActivities:
  def __init__(self, load_data=False, save_data=False, verbose=True):
    self.verbose = verbose
    self.ap = RobotActionPrimitives(load_data=load_data, save_data=save_data, verbose=verbose)
    self.serving_places = ["kobuki", "table", "human hand"]
    self.pointing_req_pub = rospy.Publisher('/pointed_object/request', Empty, queue_size=1)
    self.SERVER_IP   = '192.168.1.10'
    self.PORT_NUMBER = 6000
    self.SIZE = 1024
    self.mySocket = socket( AF_INET, SOCK_DGRAM )


  def prepare_food_or_drink(self, activity, container=None, verbose=True):
    self.ap.context = activity
    if container is None:
      container = 'cup' if activity['type'] == 'Drink' else 'bowl'
    text_to_speech("Preparing {}".format(activity['name']), verbose=verbose)
    
    # get components of the drink
    components = activity['steps']

    # do I see a cup?
    cup_loc_all = self.ap.find(container, verbose=verbose)
    if cup_loc_all is None:
      return

    cup_name = container
    cup_loc = cup_loc_all[cup_name+str(1)]
    container_loc = None
    components_loc = {}
    skip = False
    for c in range(len(components)):
      comp1 , comp2 = components[c][0], components[c][1]
      comps = [comp1, comp2]
      print("components are comp1: ", comp1, "comp2: ", comp2)
      for i, component in enumerate(comps):
        component = re.sub( r"([A-Z])", r" \1", component).split()
        component = "-".join(component)
        component = component.lower()
        if component == "water":
          component = "bottle"
        comps[i] = component
        
        components_loc[component] = self.ap.find(component)
        if components_loc[component] is None:
          skip = True
          break
        
        min_component = None
        if (component == container):
          if container_loc is not None:
            components_loc[component] = {comps[i] + '1':container_loc}
            min_component = components_loc[component]
          elif len(components_loc[component]) > 1:
            choices = ['pointing', 'near', 'any']
            choice = tell_me_one_of(f"I see multiple instances of {component}. Please tell me what to do", answers=choices, loop_stop_cond=rospy.is_shutdown, verbose=verbose)
            if choice != 'pointing':
              min_dist = 100000
              for loc_name, loc in components_loc[component].items():
                centre = loc['center']
                dist = np.linalg.norm(centre)
                if dist < min_dist:
                  min_component = {comps[i]+'1':loc}
                  min_dist = dist
            else:
              rospy.wait_for_service("/pointed_object/request")
              response = rospy.ServiceProxy("/pointed_object/request", EmptySrv)
              response()
              centroid = rospy.wait_for_message("/pointed_object/centroid", PoseArray)
              centroid = [centroid.poses[0].position.x, centroid.poses[0].position.y, centroid.poses[0].position.z]
              print("centroid: ", centroid)
              min_dist = 100000
              for loc_name, loc in components_loc[component].items():
                loc_arr = np.array(loc['center'])
                print(loc_arr)
                centroid_arr = np.array(centroid)
                print(centroid_arr)
                dist = np.linalg.norm(loc_arr-centroid_arr)
                if dist < min_dist:
                  min_component = {comps[i]+'1':loc}
                  min_dist = dist
          else:
            min_component = components_loc[component]
          if i == 1:
            cup_loc = min_component
            container_loc = list(min_component.values())[0]
          components_loc[component] = min_component
          print("min_component: ", min_component)
        # if sugar is one of the drink componets, ask how much sugar to put in.
        # if component == "sugar":
        #   text_to_speech("How much sugar do you want?", verbose=verbose)
        #   n_sugar = int(input("Enter the number of sugar cubes: "))
        #   # if there's not enough sugar, ask for more sugar.
        #   if len(components_loc["sugar"]) < n_sugar:
        #     text_to_speech("I don't have enough sugar. Please bring me more sugar.", verbose=verbose)
        #     components_loc["sugar"] = self.ap.find("sugar")
        #   for i in range(1, n_sugar):
        #     self.transport_object(component, cup_name,
        #                           object_loc=components_loc["sugar"][component+str(i)], serve_loc=cup_loc, verbose=verbose)
      # elif component['type'] != "liquid":
      if skip:
        continue
      self.transport_object(comps[0], comps[1],
                            object_loc=components_loc[comps[0]][comps[0]+'1'], serve_loc=components_loc[comps[1]][comps[1]+'1'], verbose=verbose)
    
    while True:
      text_to_speech("Please tell me where to serve the drink.", verbose=verbose)
      txt = speach_to_text(verbose=verbose)
      top_candidate_index, top_match_index, top_ratio = get_top_matching_candidate(txt, self.serving_places, bert=True, verbose=verbose)
      print("top_candidate_index: {}, top_match_index: {}, top_ratio: {}".format(top_candidate_index, top_match_index, top_ratio))
      if top_ratio < 0.85:
        # TODO: add a a wat for user to show the location to the robot.
        text_to_speech("Please tell me a plausible location to serve the drink.", verbose=verbose)
        to_say = "Currently, I can serve the drink to "
        for i, serving_place in enumerate(self.serving_places):
          to_say += serving_place
          if i < len(self.serving_places)-1:
            to_say += ", or "
        text_to_speech(to_say, verbose=verbose)
        txt = speach_to_text(verbose=verbose)
      serve_to = self.serving_places[top_match_index]
      text_to_speech("I will serve the drink to {}, right?".format(serve_to), verbose=verbose)
      txt = speach_to_text(verbose=verbose)
      while txt != "yes" and txt != "no":
        text_to_speech("Please tell me yes or no.", verbose=verbose)
        txt = speach_to_text(verbose=verbose)
      if txt == "yes":
        break
    # top_match_index = 0
    serve_to = self.serving_places[top_match_index]
    
    self.transport_object(container, serve_to, object_loc=container_loc,
                          verbose=verbose, touch=True, constraints_name="drink_serving_constraints",
                          go_home=True, serve=True)
    
    self.mySocket.sendto(bytes('Drink Served', 'utf-8'),(self.SERVER_IP, self.PORT_NUMBER))
    
    return
    # if there's a liquid component, pour it into the cup.
    for component in components:
      if component['type'] == "liquid": # TODO: check if it is a liquid using the ontology
        # pour water from the kettle into the cup.
        pour_from_loc = self.ap.find("bottle")
        if 'temperature' in activity.keys():
          if activity['temperature'] == "hot":
            pour_from_loc = self.ap.find("kettle")
        self.ap.pour(pour_from_loc, cup_loc)
  
  def transport_object(self, object_name, serve_to,
                       object_loc=None, serve_loc=None,
                       verbose=True, touch=False, constraints_name=None,
                       go_home=True, serve=False):
    if serve_loc is None:
      serve_loc = self.ap.find(serve_to)
      if serve_loc is None:
        return
      serve_loc = [loc for loc in serve_loc.values()][0]
      
    if object_loc is None:
      object_loc = self.ap.find(object_name)
      if object_loc is None:
        return
      object_loc = [loc for loc in object_loc.values()][0]
    print(f"{object_name} has loc {object_loc}")
    self.ap.pick(object_name, object_loc)
    rospy.sleep(1.0)
    if go_home or serve:
      self.ap.ba.move_to_named_location("home")
      rospy.sleep(1.0)
    if serve:
      # self.ap.ba.move_to_named_location("pre_serving")
      pos_shift = (0.03512445861759707, -0.8339168536235313, 0.3024274275995986)
      angle = np.arctan2(pos_shift[1], pos_shift[0])
      self.ap.ba.move_to_frame_pos("base_link",
                          position_shift=pos_shift,
                          orientation_shift=(0, 0, angle),
                          straight=True,  
                          use_tool_orientation=True)
      rospy.sleep(1.0)
    self.ap.place(serve_to, serve_loc, object_name, touch=touch, constraints_name=constraints_name)
    if object_name == "bottle":
      self.ap.ba.move_to_named_location("home")
      rospy.sleep(1.0)
    if object_name == "bottle":
      self.ap.place(object_name, object_loc, object_name, touch=touch, constraints_name=constraints_name)
      rospy.sleep(1.0)
    if go_home or (object_name == "bottle"):
      self.ap.ba.move_to_named_location("home")


  def prepareAMeal(self, meal_activity, container='bowl', verbose=True):
    pass
  
  def prepareGeneric(self, generic_activity, container, verbose=True):
    text_to_speech("Preparing an activity named {}".format(generic_activity['name']),verbose=verbose)
    

if __name__ == '__main__':
  rospy.init_node('robot_activities')
  ra = RobotActivities()
  ra.serve_drink()