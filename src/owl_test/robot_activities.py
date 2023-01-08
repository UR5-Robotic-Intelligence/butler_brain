#!/usr/bin/env python

from owl_test.robot_action_primitives import RobotActionPrimitives
from owl_test.utils import text_to_speech, speach_to_text, get_top_matching_candidate
import rospy
import re


class RobotActivities:
  def __init__(self, verbose=True):
    self.verbose = verbose
    self.ap = RobotActionPrimitives(verbose=True)
    self.serving_places = ["kobuki", "table", "human hand"]


  def prepareADrink(self, drink_activity, verbose=True):
    text_to_speech("Preparing a drink named {}".format(drink_activity['name']), verbose=verbose)
    # get components of the drink
    components = drink_activity['objectActedOn']
    # if it is a hot drink, press the button on the kettle.
    if 'temperature' in drink_activity.keys():
      if drink_activity['temperature'] == "hot":
        kettle_loc = self.ap.find("kettle")
        self.ap.push_button(kettle_loc)
    # do I see a cup?
    cup_loc = self.ap.find("cup")
    if cup_loc is None:
      return

    # if yes, and I see many, ask which one to use
    idx = 1
    if len(cup_loc) > 1:
      # text_to_speech("I see multiple cups. Please tell me which one to use.", verbose=verbose)
      # idx = int(input("Enter the index of the cup to use: "))
      idx = 1
    cup_name = "cup"
    cup_loc = cup_loc[cup_name+str(idx)]
    # do I see all the drink components?
    components_loc = {}
    for component in components:
      component = re.sub( r"([A-Z])", r" \1", component).split()
      component = "-".join(component)
      component = component.lower()
      if component == "drinking-mug": # TODO: Find a better way to handle this using the ontology and reasoning
        continue
      if component == "water":
        continue
      components_loc[component] = self.ap.find(component)
      if components_loc[component] is None:
        continue
      # if sugar is one of the drink componets, ask how much sugar to put in.
      if component == "sugar":
        text_to_speech("How much sugar do you want?", verbose=verbose)
        n_sugar = int(input("Enter the number of sugar cubes: "))
        # if there's not enough sugar, ask for more sugar.
        if len(components_loc["sugar"]) < n_sugar:
          text_to_speech("I don't have enough sugar. Please bring me more sugar.", verbose=verbose)
          components_loc["sugar"] = self.ap.find("sugar")
        for i in range(1, n_sugar):
          self.transport_object(component, cup_name,
                                object_loc=components_loc["sugar"][component+str(i)], serve_loc=cup_loc, verbose=verbose)
      # elif component['type'] != "liquid":
      else:
        self.transport_object(component, cup_name,
                              object_loc=components_loc[component][component+'1'], serve_loc=cup_loc, verbose=verbose)
    
    while True:
      text_to_speech("Please tell me where to serve the drink.", verbose=verbose)
      txt = speach_to_text(verbose=verbose)
      top_candidate_index, top_match_index, top_ratio = get_top_matching_candidate(txt, self.serving_places, bert=True, verbose=verbose)
      print("top_candidate_index: {}, top_match_index: {}, top_ratio: {}".format(top_candidate_index, top_match_index, top_ratio))
      if top_ratio < 0.9:
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
      text_to_speech("I will serve the drink to {} right?".format(serve_to), verbose=verbose)
      txt = speach_to_text(verbose=verbose)
      while txt != "yes" and txt != "no":
        text_to_speech("Please tell me yes or no.", verbose=verbose)
        txt = speach_to_text(verbose=verbose)
      if txt == "yes":
        break
    # top_match_index = 0
    serve_to = self.serving_places[top_match_index]
    self.transport_object("cup", serve_to, object_loc=cup_loc,
                          verbose=verbose, touch=True, constraints_name="drink_serving_constraints",
                          go_home=True, serve=True)
    return
    # if there's a liquid component, pour it into the cup.
    for component in components:
      if component['type'] == "liquid": # TODO: check if it is a liquid using the ontology
        # pour water from the kettle into the cup.
        pour_from_loc = self.ap.find("bottle")
        if 'temperature' in drink_activity.keys():
          if drink_activity['temperature'] == "hot":
            pour_from_loc = self.ap.find("kettle")
        self.ap.pour(pour_from_loc, cup_loc)
  
  def transport_object(self, object_name, serve_to,
                       object_loc=None, serve_loc=None,
                       verbose=True, touch=False, constraints_name=None,
                       go_home=False, serve=False):
    if serve_loc is None:
      serve_loc = self.ap.find(serve_to)
      if serve_loc is None:
        return
      serve_loc = [loc for loc in serve_loc.values()][0]
      
    if object_loc is None:
      object_loc = self.ap.find(object_name)
      if object_loc is None:
        return
      object_loc = [loc for loc in serve_loc.values()][0]
    self.ap.pick(object_name, object_loc)
    rospy.sleep(1.0)
    if serve:
      self.ap.ba.move_to_named_location("pre_serving")
      rospy.sleep(1.0)
    self.ap.place(serve_to, serve_loc, object_name, touch=touch, constraints_name=constraints_name)
    if go_home:
      self.ap.ba.move_to_named_location("home")


  def prepareAMeal(self, meal_activity, verbose=True):
    text_to_speech("Preparing a meal named {}".format(meal_activity['name']),verbose=verbose)
    

if __name__ == '__main__':
  rospy.init_node('robot_activities')
  ra = RobotActivities()
  ra.serve_drink()