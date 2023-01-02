from owl_test.robot_action_primitives import RobotActionPrimitives
from owl_test.utils import text_to_speech, speach_to_text
import rospy


class RobotActivities:
  def __init__(self, verbose=True):
    self.verbose = verbose
    self.ab = RobotActionPrimitives(verbose=True)


  def prepareADrink(self, drink_activity, verbose=True):
    text_to_speech("Preparing a drink named {}".format(drink_activity['name']), verbose=verbose)
    # get components of the drink
    components = drink_activity['objectActedOn']
    # if it is a hot drink, press the button on the kettle.
    if 'temperature' in drink_activity.keys():
      if drink_activity['temperature'] == "hot":
        kettle_loc = self.ab.find("kettle")
        self.ab.push_button(kettle_loc)
    # do I see a cup?
    cup_loc = self.ab.find("cup")
    # if not, ask for a cup
    if cup_loc is None:
      text_to_speech("I don't see a cup. Please bring me a cup.", verbose=verbose)
      cup_loc = self.ab.wait_for("cup", verbose=verbose)

    # if yes, and I see many, ask which one to use
    if len(cup_loc) > 1:
      text_to_speech("I see multiple cups. Please tell me which one to use.", verbose=verbose)
      idx = int(input("Enter the index of the cup to use: "))
      cup_loc = cup_loc["cup"+str(idx)]
    # do I see all the drink components?
    components_loc = {}
    for component in components:
      if component == "DrinkingMug":
        continue
      components_loc[component] = self.ab.find(component.lower())
      if components_loc[component] is None:
        text_to_speech("I don't see {}. Please bring me {}.".format(component, component), verbose=verbose)
        self.ab.wait_for(component.lower(), verbose=verbose)
        components_loc[component] = self.ab.find(component.lower())
      # if sugar is one of the drink componets, ask how much sugar to put in.
      if component == "sugar":
        text_to_speech("How much sugar do you want?", verbose=verbose)
        n_sugar = int(input("Enter the number of sugar cubes: "))
        # if there's not enough sugar, ask for more sugar.
        if len(components_loc["sugar"]) < n_sugar:
          text_to_speech("I don't have enough sugar. Please bring me more sugar.", verbose=verbose)
          components_loc["sugar"] = self.ab.find("sugar")
        for i in range(n_sugar):
          self.ab.pick(components_loc["sugar"][i])
          self.ab.place(cup_loc)
      # elif component['type'] != "liquid":
      else:
        self.ab.pick(component.lower(), components_loc[component][component.lower()+'1'])
        rospy.sleep(1.0)
        self.ab.place("cup", cup_loc)
    
    # if there's a liquid component, pour it into the cup.
    for component in components:
      if component['type'] == "liquid":
        # pour water from the kettle into the cup.
        pour_from_loc = self.ab.find("bottle")
        if 'temperature' in drink_activity.keys():
          if drink_activity['temperature'] == "hot":
            pour_from_loc = self.ab.find("kettle")
        self.ab.pour(pour_from_loc, cup_loc)


  def prepareAMeal(self, meal_activity, verbose=True):
    text_to_speech("Preparing a meal named {}".format(meal_activity['name']),verbose=verbose)

  def bringObject(self, object_name, verbose=True):
    text_to_speech("Bringing an object named {}".format(object_name), verbose=verbose)
    ob_loc = self.ab.find(object_name)
    self.ab.pick(object_name, ob_loc)
    self.ab.place(object_name, self.ab.find("table"))